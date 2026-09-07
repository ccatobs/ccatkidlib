"""Module for analyzing kinetic inductance detector (KID) composite data

Authors:
    - Darshan Patel <dp649@cornell.edu>
"""

import sys

import copy
import numpy as np
import polars as pl
import pathlib
import concurrent.futures
import lmfit

from collections.abc import Iterable
from typing import Any, TypeAlias, Literal
from pathlib import Path
from functools import cached_property

# local imports
import ccatkidlib.io as io
import ccatkidlib.log as log
import ccatkidlib.analysis.utils.pair as pair
import ccatkidlib.analysis.fit.fit as ccat_fit
import ccatkidlib.analysis.routines.properties as property_routines
import ccatkidlib.analysis.utils.multiprocess as ccat_mp
import ccatkidlib.analysis.utils.dataframe as ccat_df
import ccatkidlib.analysis.viz.viz_utils as viz_utils

from ccatkidlib.analysis.core.timestream import Timestream
from ccatkidlib.analysis.core.vna import VNA
from ccatkidlib.analysis.core.target import Target

# Plotting functions
import holoviews as hv
import hvplot.polars
from holoviews import opts

Format: TypeAlias = Literal["png", "jpeg", "pdf"]


class Detector:
    """Class representing kinetic inductance detectors (KIDs). Used for KID analyses requiring fitting and/or multiple types of data files (e.g., timestream and target sweep data).

    Attributes:
        bid  (str): RFSoC board that took the detector data
        drid (str): RFSoC drone that took the detector data
        tones (list[int]): List of detectors

        stream (Timestream | None): ``Timestream`` object of detector timestream
        targ (Target): ``Target`` object of detector target sweep
        vna (VNA | None): ``VNA`` object of detector VNA sweep

        analysis_cfg (dict): Config file with parameters used for data analysis
        viz_cfg (dict): Config file with paramateres used for data visualization

        cable_delay (float | None): Cable delay of the network (in nanoseconds)
        properties  (pl.DataFrame): Polars dataframe with detector properties
    """

    def __init__(
        self,
        com_to: str,
        cfg_path: str = str(Path(__file__).parents[1] / "analysis_config.yaml"),
        dets: int | list[int] = -1,
        noise_tones: int | list[int] | None = None,
        cable_delay: float | None = None,
        stream: Timestream | None = None,
        stream_path: str
        | pathlib.PosixPath
        | list[str]
        | list[pathlib.PosixPath]
        | None = None,
        stream_timestamp: int | str | None = None,
        targ: Target | None = None,
        targ_path: str | pathlib.PosixPath | None = None,
        targ_timestamp: int | str | None = None,
        vna: VNA | None = None,
        vna_path: str | pathlib.PosixPath | None = None,
        vna_timestamp: int | str | None = None,
        analysis_cfg: dict | None = None,
        viz_cfg: dict | None = None,
        **kwargs,
    ):
        """
        Constructor for Detector. Creates *ccatkidlib* data objects (``VNA``, ``Target``, ``Timsetream``)

        Note:
            - A Detector object can be initialized without a ``Timestream`` object but must **always** have a ``Target`` object.
            - If only the information to load a timestream is provided, an attempt will be made to find the target sweep file corresponding to the timestream

        Args:
            com_to (str): Drone that took the data. In form *'\<board>.\<drone>'*
            analysis_cfg (str, optional): Path to analysis configuration file. Defaults to analysis configuration file in *ccatkidlib/ccatkidlib/analysis*
            dets (int | list[int], optional): Which detectors to load; -1 to load all detectors. Defaults to -1
            noise_tones (int | list[int] | None, optional): Indicies of noise tones (tones not placed on detectors). Defaults to *None*
            cable_delay (float | None, optional): Cable delay of full network. Defaults to *None*

            stream (Timestream | None, optional): Detector ``Timestream`` object. Defaults to *None*
            stream_path (str | pathlib.PosixPath | list[str] | list[pathlib.PosixPath] | None, optional): Data path to detector timestream files. Defaults to *None*
            stream_timestamp (int | str | None, optional): Timestamp of detector timestream files. Defaults to *None*

            targ (Target | None, optional): Detector ``Target`` object. Defaults to *None*
            targ_path (str | pathlib.PosixPath | None, optional): Data path to detector target sweep file. Defaults to *None*
            targ_timestamp (int | str | None, optional): Timestamp of detector target sweep file. Defaults to *None*

            vna (VNA | None, optional): Detector ``VNA`` object. Defaults to *None*
            vna_path (str | pathlib.PosixPath | None, optional): Data path to detector VNA sweep file. Defaults to *None*
            vna_timestamp (int | str | None, optional): Timestamp of detector VNA sweep file. Defaults to *None*

            **kwargs: Key word arguments for finding data files. See below:
            root_data_dir (str, optional): Root directory where data is stored. Defaults to that specified in analysis config
            data_dir (str, optional): Directory where data is stored
            date (str, optional): Date data was taken
            sess_id (str, optional): ccatkidlib session ID of data

        """

        self.analysis_cfg, self.viz_cfg = io.load_config(cfg_path)
        if analysis_cfg is not None:
            self.analysis_cfg = analysis_cfg
        if viz_cfg is not None:
            self.viz_cfg = viz_cfg

        # Create Timestream, Target, and VNA data objects based provided arguments
        # ------------------------------------------------------------------------
        if not isinstance(stream, Timestream):
            stream = Detector._load_data(
                Timestream,
                com_to,
                cfg_path,
                self.analysis_cfg,
                self.viz_cfg,
                dets,
                noise_tones,
                stream_timestamp,
                stream_path,
                **kwargs,
            )
        if not isinstance(targ, Target):
            targ = Detector._load_data(
                Target,
                com_to,
                cfg_path,
                self.analysis_cfg,
                self.viz_cfg,
                dets,
                noise_tones,
                targ_timestamp,
                targ_path,
                **kwargs,
            )
        if not isinstance(vna, VNA):
            vna = Detector._load_data(
                VNA,
                com_to,
                cfg_path,
                self.analysis_cfg,
                self.viz_cfg,
                None,
                None,
                vna_timestamp,
                vna_path,
                **kwargs,
            )

        # Must have a sweep to do meaningful data analysis
        # ------------------------------------------------
        if not isinstance(targ, Target):
            if isinstance(
                stream, Timestream
            ):  # If timestream provided, try to find associated sweep
                self.tones = stream.tones
                vna_path, targ_path = pair.get_sweep(stream.data_path[0], **kwargs)

                # Load found sweep
                if Path(vna_path).exists():
                    vna = Detector._load_data(
                        VNA,
                        com_to,
                        cfg_path,
                        self.analysis_cfg,
                        self.viz_cfg,
                        None,
                        None,
                        vna_timestamp,
                        vna_path,
                        **kwargs,
                    )
                if Path(targ_path).exists():
                    targ = Detector._load_data(
                        Target,
                        com_to,
                        cfg_path,
                        self.analysis_cfg,
                        self.viz_cfg,
                        dets,
                        noise_tones,
                        targ_timestamp,
                        targ_path,
                        **kwargs,
                    )

                if not isinstance(targ, Target):  # and not isinstance(vna, VNA):
                    error = "Failed to find target sweep associated with timestream. If there is no target sweep, create a Timestream object instead."
                    log.log("CRITICAL", error)
                    raise RuntimeError(error)

            else:  # Error of no sweep or timestream provided
                error = "A timestream, target sweep or both need to be specified!"
                log.log("CRITICAL", error)
                raise RuntimeError(error)
        else:
            self.tones = targ.tones
            if vna is None:
                vna_path, _ = pair.get_sweep(targ.data_path[0], **kwargs)
                if Path(vna_path).exists():
                    vna = Detector._load_data(
                        VNA,
                        com_to,
                        cfg_path,
                        self.analysis_cfg,
                        self.viz_cfg,
                        None,
                        None,
                        vna_timestamp,
                        vna_path,
                        **kwargs,
                    )

        self.bid, self.drid = com_to.split(".")

        self.stream = stream
        self.targ = targ
        self.vna = vna

        self.timestamp = (
            self.stream.timestamp if self.stream is not None else self.targ.timestamp
        )

        # Create internal attributes corresponding to lazily loaded attributes
        # --------------------------------------------------------------------
        self._cable_delay = cable_delay
        self._properties_df = None
        self.fit_result = {}

        log_dir = io.add_dir(
            "log",
            str(self.targ.data_path[0]),
            save_root=self.analysis_cfg["io"]["file_logging"]["logging_root_dir"],
            data_root=self.targ._root_dir,
            sub_dirs=[""],
        )
        log.setup_logging(
            Path(log_dir) / self.analysis_cfg["io"]["file_logging"]["logging_fname"],
            self.analysis_cfg["io"]["file_logging"]["detector_level"],
            self.analysis_cfg["io"]["terminal_logging"]["detector_level"],
            name="analysis.detector",
        )

    # =========================#
    # Lazily Loaded Attributes #
    # =========================#

    @property
    def properties(self) -> pl.DataFrame:
        """ """

        def _merge_properties(new_properties_df: pl.DataFrame) -> None:
            """
            Merge two ``properties`` Polars DataFrames

            Args:
                new_properties_df (pl.DataFrame):
            """
            shared_cols = (
                set(self._properties_df.columns) & set(new_properties_df.columns)
            ) - {"det"}
            self._properties_df = ccat_df.coalesce_join(
                self._properties_df, new_properties_df, "det", shared_cols
            )

        if isinstance(self._properties_df, pl.LazyFrame):
            self._properties_df = self._properties_df.collect()
        elif self._properties_df is None:
            self._properties_df = self.targ.comb
        if self.targ.properties is not None:
            _merge_properties(self.targ._properties_df)
        if self.stream is not None and self.stream.properties is not None:
            _merge_properties(self.stream._properties_df)

        return self._properties_df

    @properties.setter
    def properties(self, value):
        if isinstance(value, (pl.DataFrame, pl.LazyFrame)):
            self._properties_df = value

    @property
    def cable_delay(self):
        self._cable_delay = (
            self.vna.cable_delay
            if self._cable_delay is None and isinstance(self.vna, VNA)
            else self._cable_delay
        )
        if not isinstance(self._cable_delay, Iterable):
            self._properties_df = self.properties.with_columns(
                pl.lit(self._cable_delay).alias(
                    f"vna_{self.analysis_cfg['convention']['name']['cable_delay']}"
                )
            )
            # Get cable delays for individual detectors using target sweep data. The target sweep cable delays tend to be too large so average with the overall network cable delay
            # self.targ._properties = {det: {'det_cable_delay': 0.4*delay + 0.6*self._cable_delay} for det, delay in self.targ.cable_delay.items()}

            # Replace cable delays that are far from the overall network cable delay with the network cable delay
            # threshold = pl.lit(100) # TODO: Make this accessible
            # self._properties_df = (self.properties.lazy()#.with_columns(pl.when((pl.col('det_cable_delay') - self._cable_delay).abs() > threshold)
            #                                             #                .then(pl.lit(self._cable_delay))
            #                                             #                .otherwise(pl.col('det_cable_delay'))
            #                                             #               .alias('det_cable_delay'))
            #                                            .with_columns(pl.lit(self._cable_delay).alias('network_cable_delay'))
            #                                            .collect())
        return self._cable_delay

    @cached_property
    def fig_dir(self) -> str:
        """
        Directory where figures should be saved. Create if it does not already exist.
        """

        return io.add_dir(
            "fig",
            str(self.targ.data_path[0]),
            save_root=self.viz_cfg["save"]["fig_root_dir"],
            data_root=self.targ._root_dir,
            sub_dirs=["detector"],
            timestamp=str(self.timestamp),
        )

    @cached_property
    def pickle_dir(self) -> str:
        """
        Directory where pickle files should be saved. Create if it does not already exist.
        """

        pickle_dir = io.add_dir(
            "pickle",
            str(self.targ.data_path[0]),
            save_root=self.analysis_cfg["io"]["pickle"]["pickle_root_dir"],
            data_root=self.targ._root_dir,
            sub_dirs=["detector"],
            timestamp=str(self.timestamp),
        )
        io.create_dir(Path(pickle_dir) / "dataframe")
        return pickle_dir

    @cached_property
    def fit_dir(self):
        return self.analysis_cfg["file_paths"]["fit_dir"]

    # =====================#
    # Data Getter Methods #
    # =====================#

    def get_properties(
        self,
        col_name: str | list[str] = ".*",
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        strict: bool = False,
    ):
        """Get the specified data columns and rows from the ``properties`` Polars DataFrame

        Args:
            col_name (str | list[str], optional): Defaults to all columns
            include (int | list[int] | None, optional): Defaults to *None*
            exclude (int | list[int] | None, optional): Defaults to *None*
            strict (bool, optional): Defaults to *False*

        """
        return ccat_df.get_properties(
            self, col_name=col_name, include=include, exclude=exclude, strict=strict
        )

    # Fitting
    # -------
    def complex_fit(
        self,
        prefix: str | list[str] = "",
        nonlinear: bool | list[bool] = False,
        asymm: bool | list[bool] = False,
        fix_cable: bool | list[bool] = False,
        fix_thetaQ: bool | list[bool] = False,
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        recalc: bool = False,
        max_workers: int = 1,
        ex=None,
    ) -> pl.DataFrame:
        """
        Fit target sweep using complex forward transmission data (*z = I + iQ*)

        Args:
            nonlinear (bool, optional): Whether to perform a nonlinear fit. Defaults to *False*
            asymm (bool, optional): Whether to perform a asymmetric fit. Defaults to *False*
            fix_cable (bool, optional): Whether to vary cable parameters for fit. Defaults to *False*: parameters are varied
            fix_thetaQ (bool, optional): Whether to vary impedance mismatch angle and coupling quality factor for fit. Defaults to *False*: parameters are varied
        Returns:
            return (pl.DataFrame): Polars DataFrame with fit I and Q data
        """

        if self.fit_dir not in sys.path:
            sys.path.append(self.fit_dir)
        import resonator_model_v3

        globals()["resonator_model_v3"] = resonator_model_v3

        name_enums, prefix_enums = ccat_df.create_enums(
            [
                "frequency",
                "in_phase",
                "quadrature",
                "internal_quality_factor",
                "coupling_quality_factor",
                "real_external_quality_factor",
                "imag_external_quality_factor",
                "total_quality_factor",
                "cable_delay",
            ],
            prefix,
            ["complex_fit", "complex_fit_cable"],
            self.analysis_cfg,
            no_prefix=["frequency"],
        )

        num_prefix = len(name_enums)
        nonlinear = ccat_df.check_args(nonlinear, num_prefix, bool)
        asymm = ccat_df.check_args(asymm, num_prefix, bool)
        fix_cable = ccat_df.check_args(fix_cable, num_prefix, bool)
        fix_thetaQ = ccat_df.check_args(fix_thetaQ, num_prefix, bool)

        args = [
            [
                self,
                nonlin,
                asym,
                cable,
                thetaQ,
                ccat_mp.check_max_workers(max_workers),
                ex,
            ]
            for nonlin, asym, cable, thetaQ in zip(
                nonlinear, asymm, fix_cable, fix_thetaQ
            )
        ]
        self.properties
        self.targ.transform(
            Detector._calc_complex_fit,
            *args,
            include=include,
            exclude=exclude,
            recalc=recalc,
            col_enum=name_enums,
            prefix_enum=prefix_enums,
        )

        col_names = []
        for prefix_enum, name_enum in zip(prefix_enums, name_enums):
            col_names += [
                f"{prefix_enum.COMPLEX_FIT.value}_{name_enum.IN_PHASE.value}",
                f"{prefix_enum.COMPLEX_FIT.value}_{name_enum.QUADRATURE.value}",
            ]

            # Calculate Q_c and Q_i. Convert cable delay into nanaseconds
            self.targ._properties_df = (
                self.targ.properties.lazy()
                .with_columns(
                    (
                        (
                            pl.col(
                                f"{prefix_enum.COMPLEX_FIT.value}_{name_enum.REAL_EXTERNAL_QUALITY_FACTOR.value}"
                            )
                            / (
                                pl.col(
                                    f"{prefix_enum.COMPLEX_FIT.value}_{name_enum.REAL_EXTERNAL_QUALITY_FACTOR.value}"
                                )
                                ** 2
                                + pl.col(
                                    f"{prefix_enum.COMPLEX_FIT.value}_{name_enum.IMAG_EXTERNAL_QUALITY_FACTOR.value}"
                                )
                                ** 2
                            )
                        )
                        ** -1
                    ).alias(
                        f"{prefix_enum.COMPLEX_FIT.value}_{name_enum.COUPLING_QUALITY_FACTOR.value}"
                    )
                )  # Calculate Q_c
                .with_columns(
                    [
                        (
                            (
                                1
                                / pl.col(
                                    f"{prefix_enum.COMPLEX_FIT.value}_{name_enum.TOTAL_QUALITY_FACTOR.value}"
                                )
                                - 1
                                / pl.col(
                                    f"{prefix_enum.COMPLEX_FIT.value}_{name_enum.COUPLING_QUALITY_FACTOR.value}"
                                )
                            )
                            ** -1
                        ).alias(
                            f"{prefix_enum.COMPLEX_FIT.value}_{name_enum.INTERNAL_QUALITY_FACTOR.value}"
                        ),  # Calculate Q_i
                        (
                            -1e9
                            * pl.col(
                                f"{prefix_enum.COMPLEX_FIT.value}_{name_enum.CABLE_DELAY.value}"
                            )
                        ).alias(
                            f"ns_{prefix_enum.COMPLEX_FIT.value}_{name_enum.CABLE_DELAY.value}"
                        ),
                    ]
                )  # Convert cable delay to nanoseconds
                .collect()
            )

        self.targ.data = ccat_df.unnest(
            self.targ, [f"struct_{name}" for name in col_names]
        )
        return self.targ.get_data(
            col_name=col_names,
            include=include,
            exclude=exclude,
            strict=True,
        )

    def phase_fit(
        self,
        prefix: str
        | list[str] = "mismatch_rotate_origin_shift_origin_rotate_unwind_rotate",
        circle_fit_prefix: str = "circle_fit_unwind_rotate",
        nonlinear: bool = False,
        method: str = "least_squares",
        params: lmfit.Parameters = None,
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        recalc: bool = False,
        max_workers: int = 1,
        ex=None,
    ) -> pl.DataFrame:
        """
        Fit target sweep using phase data (*arctan(Q/I)*)

        """

        name_enums, prefix_enums = ccat_df.create_enums(
            [
                "frequency",
                "in_phase",
                "quadrature",
                "phase",
                "IQ_circle_radius",
                "internal_quality_factor",
                "coupling_quality_factor",
                "total_quality_factor",
                "phase_fit_beta",
                #"phase_fit_gamma",
                #"phase_fit_delta",
                "phase_fit_theta",
                "fit_result",
                "resonant_frequency",
                "ki_nonlinearity_parameter",
                #"qp_nonlinearity_parameter",
            ],
            prefix,
            ["phase_fit"],
            self.analysis_cfg,
            no_prefix=["frequency"],
        )

        num_prefix = len(name_enums)
        params = ccat_df.check_args(params, num_prefix, lmfit.parameter.Parameters)
        nonlinear = ccat_df.check_args(nonlinear, num_prefix, bool)
        method = ccat_df.check_args(method, num_prefix, str)
        circle_fit_prefix = ccat_df.check_args(circle_fit_prefix, num_prefix, str)

        circle_name_enums, _ = ccat_df.create_enums(
            ["IQ_circle_radius", "IQ_circle_center", "magnitude"],
            circle_fit_prefix,
            [],
            self.analysis_cfg,
            no_prefix=["magnitude"],
        )

        radii = [
            self.get_properties(
                col_name=name_enum.IQ_CIRCLE_RADIUS.value,
                include=include,
                exclude=exclude,
                strict=True,
            )
            .to_numpy()
            .T[1]
            for name_enum in circle_name_enums
        ]

        args = [
            [
                self,
                radius,
                nonlin,
                meth,
                param,
                ccat_mp.check_max_workers(max_workers),
                ex,
            ]
            for radius, nonlin, param, meth in zip(
                radii, nonlinear, params, method
            )
        ]
        self.targ.transform(
            [Detector._calc_phase_fit] * num_prefix,
            *args,
            include=include,
            exclude=exclude,
            recalc=recalc,
            col_enum=name_enums,
            prefix_enum=prefix_enums,
        )

        col_names = []
        for prefix_enum, name_enum, circle_enum in zip(
            prefix_enums, name_enums, circle_name_enums
        ):
            col_names += [
                f"{prefix_enum.PHASE_FIT.value}_{name_enum.PHASE.value}",
            ]

            self.targ._properties_df = (
                self.targ.properties.lazy()
                .with_columns(
                    [
                        (
                            1e6
                            * (
                                pl.col(
                                    f"{prefix_enum.PHASE_FIT.value}_{name_enum.PHASE_FIT_BETA.value}"
                                )
                                * (
                                    2
                                    * pl.col(
                                        f"{prefix_enum.PHASE_FIT.value}_{name_enum.IQ_CIRCLE_RADIUS.value}"
                                    )
                                )
                                ** 2
                            )
                            / (
                                pl.col(
                                    f"{prefix_enum.PHASE_FIT.value}_{name_enum.RESONANT_FREQUENCY.value}"
                                )
                                / pl.col(
                                    f"{prefix_enum.PHASE_FIT.value}_{name_enum.TOTAL_QUALITY_FACTOR.value}"
                                )
                            )
                        ).alias(
                            f"{prefix_enum.PHASE_FIT.value}_{name_enum.KI_NONLINEARITY_PARAMETER.value}"
                        ),
                        #(
                        #    1e6
                        #    * (
                        #        pl.col(
                        #            f"{prefix_enum.PHASE_FIT.value}_{name_enum.PHASE_FIT_GAMMA.value}"
                        #        )
                        #        * (
                        #            2
                        #            * pl.col(
                        #                f"{prefix_enum.PHASE_FIT.value}_{name_enum.IQ_CIRCLE_RADIUS.value}"
                        #            )
                        #        )
                        #        ** pl.col(f"{prefix_enum.PHASE_FIT.value}_{name_enum.PHASE_FIT_DELTA.value}")
                        #    )
                        #    / (
                        #        pl.col(
                        #            f"{prefix_enum.PHASE_FIT.value}_{name_enum.RESONANT_FREQUENCY.value}"
                        #        )
                        #        / pl.col(
                        #            f"{prefix_enum.PHASE_FIT.value}_{name_enum.TOTAL_QUALITY_FACTOR.value}"
                        #        )
                        #    )
                        #).alias(
                        #    f"{prefix_enum.PHASE_FIT.value}_{name_enum.QP_NONLINEARITY_PARAMETER.value}"
                        #),
                        (
                            pl.col(
                                f"{prefix_enum.PHASE_FIT.value}_{name_enum.TOTAL_QUALITY_FACTOR.value}"
                            )
                            * (
                                pl.col(
                                    f"{circle_enum.IQ_CIRCLE_CENTER.value}_{circle_enum.MAGNITUDE.value}"
                                )
                                + pl.col(
                                    f"{prefix_enum.PHASE_FIT.value}_{name_enum.IQ_CIRCLE_RADIUS.value}"
                                )
                            )
                            / (
                                2
                                * pl.col(
                                    f"{prefix_enum.PHASE_FIT.value}_{name_enum.IQ_CIRCLE_RADIUS.value}"
                                )
                            )
                        ).alias(
                            f"{prefix_enum.PHASE_FIT.value}_{name_enum.COUPLING_QUALITY_FACTOR.value}"
                        ),
                    ]
                )
                .with_columns(
                    (
                        (
                            1
                            / pl.col(
                                f"{prefix_enum.PHASE_FIT.value}_{name_enum.TOTAL_QUALITY_FACTOR.value}"
                            )
                            - 1
                            / pl.col(
                                f"{prefix_enum.PHASE_FIT.value}_{name_enum.COUPLING_QUALITY_FACTOR.value}"
                            )
                        )
                        ** -1
                    ).alias(
                        f"{prefix_enum.PHASE_FIT.value}_{name_enum.INTERNAL_QUALITY_FACTOR.value}"
                    )
                )
                .collect()
            )

        self.targ.data = ccat_df.unnest(
            self.targ, [f"struct_{name}" for name in col_names]
        )
        return self.targ.get_data(
            col_name=col_names,
            include=include,
            exclude=exclude,
        )

    def IQ_circle_fit(
        self,
        prefix: str | list[str] = "unwind_rotate",
        bounds=None,
        loss: str = "soft_l1",
        f_scale: float = 1,
        method: str = "trf",
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        recalc: bool = False,
        max_workers=1,
        ex=None,
    ) -> pl.DataFrame:
        """
        Fit the target sweep circle in the IQ plane

        """

        name_enums, prefix_enums = ccat_df.create_enums(
            ["in_phase", "quadrature", "IQ_circle_radius", "IQ_circle_center"],
            prefix,
            ["IQ_circle_fit"],
            self.analysis_cfg,
        )

        num_prefix = len(name_enums)
        method = ccat_df.check_args(method, num_prefix, str)
        loss = ccat_df.check_args(loss, num_prefix, str)
        f_scale = ccat_df.check_args(f_scale, num_prefix, float)

        args = [
            [
                self,
                bounds,
                los,
                scale,
                meth,
                ccat_mp.check_max_workers(max_workers),
                ex,
            ]
            for los, scale, meth in zip(loss, f_scale, method)
        ]
        self.properties
        self.targ.transform(
            [Detector._calc_IQ_circle_fit] * num_prefix,
            *args,
            include=include,
            exclude=exclude,
            recalc=recalc,
            col_enum=name_enums,
            prefix_enum=prefix_enums,
        )

        col_names = []
        for prefix_enum, name_enum in zip(prefix_enums, name_enums):
            col_names += [
                f"{prefix_enum.IQ_CIRCLE_FIT.value}_{name_enum.IN_PHASE.value}",
                f"{prefix_enum.IQ_CIRCLE_FIT.value}_{name_enum.QUADRATURE.value}",
            ]

        self.targ.data = ccat_df.unnest(
            self.targ, [f"struct_{name}" for name in col_names]
        )
        return self.targ.get_data(
            col_name=col_names,
            include=include,
            exclude=exclude,
        )

    def nonlinearity_parameter(self,
                               prefix: str | list[str] = "",
                               m: str | float | list[float],
                               b: str | float | list[float],
                               f_shift: str | float | list[float] = 0,
                               include: int | list[int] | None = None,
                               exclude: int | list[int] | None = None,
                               recalc: bool = False,
                               **kwargs,):
        name_enums, prefix_enums = ccat_df.create_enums(
            ["linewidth_shift", "ki_nonlinearity_parameter"],
            prefix,
            [],
            self.analysis_cfg,
            no_prefix=[],
        )

        num_prefix = len(name_enums)
        m = ccat_df.check_args(f_shift, num_prefix, str)
        b = ccat_df.check_args(f_shift, num_prefix, str)
        f_shift = ccat_df.check_args(f_shift, num_prefix, str)

        for i, f_0 in enumerate(ref_f):
            if isinstance(f_0, str):
                ref_f[i] = (
                    self.get_properties(
                        f_0, include=include, exclude=exclude, strict=True
                    )
                    .to_numpy()
                    .T[1]
                )  # Get reference frequencies from .properties DataFrame

        frac_f_dfs, args = [], [[f] for f in ref_f]
        for data_obj in data_objs:
            data_obj.transform(
                [Detector._calc_frac_f] * num_prefix,
                *args,
                include=include,
                exclude=exclude,
                recalc=recalc,
                col_enum=name_enums,
                prefix_enum=prefix_enums,
            )

            frac_f_dfs.append(
                data_obj.get_data(
                    col_name=[name_enum.FRACTIONAL_FREQUENCY.value for name_enum in name_enums],
                    include=include,
                    exclude=exclude,
                    strict=True
                    ))

        return frac_f_dfs

    # IQ transformations
    # ------------------

    def IQ_unwind(
        self,
        delay_col: str,
        prefix: str | list[str] = "",
        data: str = "both",
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        recalc: bool = False,
    ) -> list[pl.DataFrame]:
        """
        Remove cable delay from target sweep and/or timestream I & Q data

        Args:
            prefix (str | list[str]): Prefix of I & Q data to remove cable delay from
            data (str, optional): Which type of data to remove cable delay from. Options are 'targ' or 'timestream'. Defaults to 'targ'.
            delay_col (str): Column in ``properties`` DataFrame with detector cable delays
            include (int | list[int] | None, optional): Detector(s) for which to perform calculation
            exclude (int | list[int] | None, optional): Detector(s) for which not to perform calculation
            recalc (bool): Whether to recalculate if data is already in ``data`` DataFrame. Defaults to False

        Returns:
            pl.Dataframe: Polars DataFrame with unwound I & Q data
        """
        name_enums, prefix_enums = ccat_df.create_enums(
            [
                "frequency",
                "in_phase",
                "quadrature",
                "tone_frequency"
            ],
            prefix,
            ["rotate", "remove_cable"],
            self.analysis_cfg,
            no_prefix=["frequency", "tone_frequency"],
        )

        # TODO: Allow different delay cols for each prefix
        data_objs, data_types = self._get_data_obj(data)
        if not data_objs:
            error = (
                f"Invalid data type {data}, must be 'targ', 'timestream', or 'both'."
            )
            log.log("ERROR", error)
            raise ValueError(error)

        # Ensure that the cable delay has been calculated
        delays = (
            self.get_properties(
                delay_col, include=include, exclude=exclude, strict=True
            )
            .to_numpy()
            .T[1]
        )

        delays = -2 * np.pi * 1e-9 * delays
        unwind_dfs = []
        for data_obj, data_type in zip(data_objs, data_types):
            angle = (
                [
                    delay * tone_freq
                    for tone_freq, delay in zip(
                        self.targ.get_data(
                            name_enums[0].FREQUENCY.value,
                            include=include,
                            exclude=exclude,
                            strict=True,
                        )
                        .to_numpy()
                        .T,
                        delays,
                    )
                ]
                if data_type == "targ"
                else [
                    delay * tone_freq
                    for tone_freq, delay in zip(
                        self.stream.comb[name_enums[0].TONE_FREQUENCY.value].to_numpy().T, delays
                    )
                ]
            )
            data_obj.IQ_rotate(
                prefix=prefix,
                angle=angle,
                name=[prefix_enum.REMOVE_CABLE.value for prefix_enum in prefix_enums],
                include=include,
                exclude=exclude,
                recalc=recalc,
            )

            col_names = []
            for name_enum, prefix_enum in zip(name_enums, prefix_enums):
                col_names += [
                    f"{prefix_enum.REMOVE_CABLE.value}_{prefix_enum.ROTATE.value}_{name_enum.IN_PHASE.value}",
                    f"{prefix_enum.REMOVE_CABLE.value}_{prefix_enum.ROTATE.value}_{name_enum.QUADRATURE.value}",
                ]

            unwind_dfs.append(
                data_obj.get_data(
                    col_name=col_names,
                    include=include,
                    exclude=exclude,
                    strict=True,
                )
            )
        return unwind_dfs

    def IQ_norm(
        self,
        norm_prefix: str,
        prefix: str | list[str] = "",
        data: str = "both",
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        recalc: bool = False,
    ) -> list[pl.DataFrame]:
        """
        Divide out cable baseline from target sweep and/or timestream I & Q data

        """

        name_enums, prefix_enums = ccat_df.create_enums(
            ["frequency", "in_phase", "quadrature", "magnitude", 'tone_frequency'],
            prefix,
            ["scale", "normalize"],
            self.analysis_cfg,
            no_prefix=["frequency", "magnitude", 'tone_frequency'],
        )

        data_objs, data_types = self._get_data_obj(data)
        if not data_objs:
            error = (
                f"Invalid data type {data}, must be 'targ', 'timestream', or 'both'."
            )
            log.log("ERROR", error)
            raise ValueError(error)

        tone_freqs = self.get_properties(
            name_enums[0].TONE_FREQUENCY.value, include=include, exclude=exclude, strict=True
        )

        norm_dfs = []
        for data_obj, data_type in zip(data_objs, data_types):
            # TODO: Allow different norm_prefix for each prefix
            cable_mags = self.targ.get_data(
                f"{norm_prefix}{'_' if norm_prefix else ''}{name_enums[0].MAGNITUDE.value}",
                include=include,
                exclude=exclude,
                strict=True,
            )
            if data_type == "targ":
                scale = 1 / cable_mags.to_numpy().T
            else:
                # TODO: Probably unnecessary computation to find data point corresponding to tone. Should always be sample = num_samples/2 (unless timestream tone frequency is changed without retaking target sweep)
                f_col = name_enums[0].FREQUENCY.value
                f_df = (
                    self.targ.get_data(
                        f_col, include=include, exclude=exclude, strict=True
                    )
                    .unpivot(variable_name="det", value_name=f_col)
                    .with_columns(pl.col("det").str.strip_prefix(f"{f_col}_").cast(int))
                )
                cable_df = cable_mags.unpivot(
                    variable_name="temp", value_name="cable"
                ).drop("temp")
                f_cable_df = pl.concat([f_df, cable_df], how="horizontal")
                f_cable_df = f_cable_df.join(
                    tone_freqs, on="det", how="left", coalesce=True
                )
                scale = (
                    1
                    / (
                        f_cable_df.lazy()
                        .sort((pl.col(f_col) - pl.col(name_enums[0].TONE_FREQUENCY.value)).abs())
                        .select(pl.col("cable").first().over("det"))
                        .collect()
                    )
                    .to_numpy()
                    .T
                )

            data_obj.IQ_scale(
                prefix=prefix,
                scale=scale,
                name=[prefix_enum.NORMALIZE.value for prefix_enum in prefix_enums],
                include=include,
                exclude=exclude,
                recalc=recalc,
            )

            col_names = []
            for name_enum, prefix_enum in zip(name_enums, prefix_enums):
                col_names += [
                    f"{prefix_enum.NORMALIZE.value}_{prefix_enum.SCALE.value}_{name_enum.IN_PHASE.value}",
                    f"{prefix_enum.NORMALIZE.value}_{prefix_enum.SCALE.value}_{name_enum.QUADRATURE.value}",
                ]

            norm_dfs.append(
                data_obj.get_data(
                    col_name=col_names,
                    include=include,
                    exclude=exclude,
                    strict=True,
                )
            )
        return norm_dfs

    def IQ_trim(
        self,
        prefix: str | list[str] = "",
        window: float | list[float] = 1.5,
        use_fit: bool = False,
        f_0_col: str = "complex_fit_f_0",
        Q_col: str = "complex_fit_Q",
        mean_points: int = 10,
        mag_prefix: str = "",
        invert: bool = False,
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        recalc: bool = False,
    ) -> pl.DataFrame:
        """
        Trim off-resonance target sweep I & Q data

        Args:
            prefix:
            window (float | list[float], optional): How many linewidths of data to include around the magnitude minimum. Defaults to 1.5
            use_fit (bool): Whether to use resonant frequency and total quality factor from a fit to determine linewidth. Defaults to False
            f_0_col (str): Name of column in ``properties`` DataFrame with resonant frequencies to use if ``use_fit`` is True.
            Q_col (str): Name of column in ``properties`` DataFrame with total quality factors to use if ``use_fit`` is True.
            mean_points (int): Number of points to average to determine max magnitude of resonator profile. Used if ``use_fit`` is False. Defaults to 10
        Returns:
            return (pl.DataFrame): Polars DataFrame with trimmed I & Q data
        """

        name_enums, prefix_enums = ccat_df.create_enums(
            [
                "sample",
                "in_phase",
                "quadrature",
            ],
            prefix,
            ["trim", "trim_tail"],
            self.analysis_cfg,
            no_prefix=["sample"],
        )

        num_prefix = len(name_enums)
        use_fit = ccat_df.check_args(use_fit, num_prefix, bool)
        window = ccat_df.check_args(window, num_prefix, int)
        mean_points = ccat_df.check_args(mean_points, num_prefix, int)
        mag_prefix = ccat_df.check_args(mag_prefix, num_prefix, str)

        lower_bounds, upper_bounds = [[]] * num_prefix, [[]] * num_prefix
        for i, (fit, mag_pre, mean_point, win) in enumerate(
            zip(use_fit, mag_prefix, mean_points, window)
        ):
            if fit:
                if f_0_col in self.properties.schema:
                    pass
                    # TODO: Add trimming based on fit FWHM
            else:
                HM_low, HM_mid, HM_high = (
                    property_routines.fwhm(
                        self.targ,
                        mag_pre,
                        mean_points=mean_point,
                        include=include,
                        exclude=exclude,
                        recalc=recalc,
                    )
                    .to_numpy()
                    .T[1:4]
                )

            lower_bounds[i] = np.maximum((HM_mid - (HM_mid - HM_low) * win).astype(int), 0)
            upper_bounds[i] = (HM_mid + (HM_high - HM_mid) * win).astype(int)

        self.targ.IQ_trim(
            prefix=prefix,
            lower_index=lower_bounds,
            upper_index=upper_bounds,
            name=[prefix_enum.TRIM_TAIL.value for prefix_enum in prefix_enums],
            invert=invert,
            include=include,
            exclude=exclude,
            recalc=recalc,
        )

        col_names = []
        for name_enum, prefix_enum in zip(name_enums, prefix_enums):
            col_names += [
                f"{prefix_enum.TRIM_TAIL.value}_{prefix_enum.TRIM.value}_{name_enum.IN_PHASE.value}",
                f"{prefix_enum.TRIM_TAIL.value}_{prefix_enum.TRIM.value}_{name_enum.QUADRATURE.value}",
            ]

        return self.targ.get_data(
            col_name=col_names,
            include=include,
            exclude=exclude,
            strict=True,
        )

    def IQ_circle_origin(
        self,
        prefix: str | list[str] = "unwind_rotate",
        circle_fit_prefix="",
        data: str = "both",
        use_fit: bool | list[bool] = True,
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        recalc: bool = False,
    ) -> list[pl.DataFrame]:
        """
        Rotate and center the target sweep circle in the IQ plane onto the real axis
        """
        if isinstance(prefix, str):
            prefix = [prefix]

        name_enums, prefix_enums = ccat_df.create_enums(
            ["in_phase", "quadrature", "magnitude", "phase"],
            prefix,
            ["center_origin", "rotate", "translate"],
            self.analysis_cfg,
            no_prefix=[],
        )

        num_prefix = len(name_enums)
        use_fit = ccat_df.check_args(use_fit, num_prefix, bool)
        circle_fit_prefix = ccat_df.check_args(circle_fit_prefix, num_prefix, str)

        circle_name_enums, _ = ccat_df.create_enums(
            ["IQ_circle_center", "in_phase", "quadrature", "magnitude", "phase"],
            circle_fit_prefix,
            [],
            self.analysis_cfg,
            no_prefix=["in_phase", "quadrature", "magnitude", "phase"],
        )

        shifts, angles = [[]] * num_prefix, [[]] * num_prefix
        for i, (pre, name_enum, prefix_enum, circle_enum, fit) in enumerate(
            zip(prefix, name_enums, prefix_enums, circle_name_enums, use_fit)
        ):
            if not fit:  # Use median I & Q values of target sweep IQ circle if not using center from circle fit
                # Calculate target sweep median I & Q values if not in ``properties`` DataFrame already
                (
                    properties_I_col,
                    properties_Q_col,
                    properties_mag_col,
                    properties_phase_col,
                ) = (
                    f"median_targ_{name_enum.IN_PHASE.value}",
                    f"median_targ_{name_enum.QUADRATURE.value}",
                    f"median_targ_{name_enum.MAGNITUDE.value}",
                    f"median_targ_{name_enum.PHASE.value}",
                )
                ccat_df.agg(
                    self.targ,
                    "median",
                    circle_enum.IN_PHASE.value,
                    prefix=pre,
                    include=include,
                    exclude=exclude,
                    recalc=recalc,
                )
                ccat_df.agg(
                    self.targ,
                    "median",
                    circle_enum.QUADRATURE.value,
                    prefix=pre,
                    include=include,
                    exclude=exclude,
                    recalc=recalc,
                )
            else:
                (
                    properties_I_col,
                    properties_Q_col,
                    properties_mag_col,
                    properties_phase_col,
                ) = (
                    f"{circle_enum.IQ_CIRCLE_CENTER.value}_{circle_enum.IN_PHASE.value}",
                    f"{circle_enum.IQ_CIRCLE_CENTER.value}_{circle_enum.QUADRATURE.value}",
                    f"{circle_enum.IQ_CIRCLE_CENTER.value}_{circle_enum.MAGNITUDE.value}",
                    f"{circle_enum.IQ_CIRCLE_CENTER.value}_{circle_enum.PHASE.value}",
                )

            include_subset = ccat_df.check_properties(
                self,
                properties_mag_col,
                include=include,
                exclude=exclude,
                recalc=recalc,
            )
            if not len(include_subset) == 0:
                circle_fit_df = self.get_properties(
                    [properties_I_col, properties_Q_col],
                    include=include_subset,
                    strict=True,
                ).select(
                    [
                        "det",
                        (
                            pl.arctan2(
                                pl.col(properties_Q_col), pl.col(properties_I_col)
                            )
                        ).alias(properties_phase_col),  # Calculate angle
                        (
                            (
                                pl.col(properties_Q_col) ** 2
                                + pl.col(properties_I_col) ** 2
                            ).sqrt()
                        ).alias(properties_mag_col),
                    ]
                )  # Calculate magnitude
                shared_cols = (
                    [properties_mag_col, properties_phase_col]
                    if properties_mag_col in self._properties_df.schema
                    else []
                )
                self.targ._properties_df = ccat_df.coalesce_join(
                    self.targ._properties_df,
                    circle_fit_df,
                    on="det",
                    shared_cols=shared_cols,
                )

            center_angle, center_mag = (
                self.get_properties(
                    [properties_phase_col, properties_mag_col],
                    include=include,
                    exclude=exclude,
                    strict=True,
                )
                .to_numpy()
                .T[1:3]
            )
            angles[i] = np.pi - center_angle
            shifts[i] = center_mag

        data_objs, _ = self._get_data_obj(data)
        if not data_objs:
            error = (
                f"Invalid data type {data}, must be 'targ', 'timestream', or 'both'."
            )
            log.log("ERROR", error)
            raise ValueError(error)

        shift_dfs = []
        for data_obj in data_objs:
            data_obj.IQ_rotate(
                prefix=prefix,
                angle=angles,
                name=[prefix_enum.CENTER_ORIGIN.value for prefix_enum in prefix_enums],
                include=include,
                exclude=exclude,
                recalc=recalc,
            )
            data_obj.IQ_shift(
                prefix=[
                    f"{prefix_enum.CENTER_ORIGIN.value}_{prefix_enum.ROTATE.value}_{pre}"
                    for pre, prefix_enum in zip(prefix, prefix_enums)
                ],
                shift_I=shifts,
                name=[prefix_enum.CENTER_ORIGIN.value for prefix_enum in prefix_enums],
                include=include,
                exclude=exclude,
                recalc=recalc,
            )

            col_names = []
            for name_enum, prefix_enum in zip(name_enums, prefix_enums):
                col_names += [
                    f"{prefix_enum.CENTER_ORIGIN.value}_{prefix_enum.TRANSLATE.value}_{prefix_enum.CENTER_ORIGIN.value}_{prefix_enum.ROTATE.value}_{name_enum.IN_PHASE.value}",
                    f"{prefix_enum.CENTER_ORIGIN.value}_{prefix_enum.TRANSLATE.value}_{prefix_enum.CENTER_ORIGIN.value}_{prefix_enum.ROTATE.value}_{name_enum.QUADRATURE.value}",
                ]

            shift_dfs.append(
                data_obj.get_data(
                    col_name=col_names,
                    include=include,
                    exclude=exclude,
                    strict=True,
                )
            )
        return shift_dfs

    def IQ_circle_rotate(
        self,
        prefix: str | list[str] = "origin_shift_origin_rotate_unwind_rotate",
        data: str = "both",
        rotation: Literal["mismatch", "timestream"]
        | list[Literal["mismatch", "timestream"]] = "mismatch",
        mean_points: int = 10,
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        recalc: bool = False,
        **kwargs,
    ) -> list[pl.DataFrame]:
        """
        Rotate the target sweep circle in the IQ plane around its center
        """

        if isinstance(prefix, str):
            prefix = [prefix]

        name_enums, prefix_enums = ccat_df.create_enums(
            ["in_phase", "quadrature", "phase"],
            prefix,
            ["remove_impedance_mismatch", "center_timestream", "rotate"],
            self.analysis_cfg,
            no_prefix=[],
        )

        num_prefix = len(name_enums)
        mean_points = ccat_df.check_args(mean_points, num_prefix, int)
        rotation = ccat_df.check_args(rotation, num_prefix, str)

        angles, rot_names = [[]] * num_prefix, [""] * num_prefix
        for i, (pre, name_enum, prefix_enum, rot, mean_point) in enumerate(
            zip(prefix, name_enums, prefix_enums, rotation, mean_points)
        ):
            if rot == "mismatch":
                angles[i] = (
                    property_routines.mismatch_angle(
                        self.targ,
                        pre,
                        mean_points=mean_point,
                        include=include,
                        exclude=exclude,
                        recalc=recalc,
                    )
                    .to_numpy()
                    .T[1]
                )
                rot_names[i] = prefix_enum.REMOVE_IMPEDANCE_MISMATCH.value
            elif rot == "timestream":
                angles[i] = (
                    property_routines.timestream_angle(
                        self.stream,
                        pre,
                        include=include,
                        exclude=exclude,
                        recalc=recalc,
                    )
                    .to_numpy()
                    .T[1]
                )
                rot_names[i] = prefix_enum.CENTER_TIMESTREAM.value
            else:
                error = f"Invalid rotation '{rot}' specified; Must be one of 'mismatch' or 'timestream'."
                log.log("ERROR", error)
                raise ValueError(error)

        data_objs, _ = self._get_data_obj(data)
        if not data_objs:
            error = (
                f"Invalid data type {data}, must be 'targ', 'timestream', or 'both'."
            )
            log.log("ERROR", error)
            raise ValueError(error)

        rotation_dfs = []
        for data_obj in data_objs:
            data_obj.IQ_rotate(
                prefix=prefix,
                angle=angles,
                name=rot_names,
                include=include,
                exclude=exclude,
                recalc=recalc,
            )

            col_names = []
            for name_enum, prefix_enum, rot_name in zip(
                name_enums, prefix_enums, rot_names
            ):
                col_names += [
                    f"{rot_name}_{prefix_enum.ROTATE.value}_{name_enum.IN_PHASE.value}",
                    f"{rot_name}_{prefix_enum.ROTATE.value}_{name_enum.QUADRATURE.value}",
                ]

            rotation_dfs.append(
                data_obj.get_data(
                    col_name=col_names,
                    include=include,
                    exclude=exclude,
                    strict=True,
                )
            )
        return rotation_dfs

    def IQ_circle_diss_corr(
        self,
        prefix: str | list[str] = "",
        circle_fit_prefix: str | list[str] = "circle_fit_unwind_rotate",
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        recalc: bool = False,
        max_workers=1,
        ex=None,
        **kwargs):
        if isinstance(prefix, str):
            prefix = [prefix]

        name_enums, prefix_enums = ccat_df.create_enums(
            ["in_phase", "quadrature", "magnitude"],
            prefix,
            ['dissipation_correction'],
            self.analysis_cfg,
            no_prefix=[],
        )
        num_prefix = len(name_enums)

        circle_fit_prefix = ccat_df.check_args(circle_fit_prefix, num_prefix, str)
        radii = [
            self.get_properties(
                col_name=f"{pre}_{self.analysis_cfg['convention']['name']['IQ_circle_radius']}",
                include=include,
                exclude=exclude,
                strict=True,
            )
            .to_numpy()
            .T[1]
            for pre in circle_fit_prefix
        ]

        args = [[radius, ccat_mp.check_max_workers(max_workers), ex] for radius in radii]
        self.targ.transform(
            [Detector._calc_dissipation_correction] * num_prefix,
            *args,
            include=include,
            exclude=exclude,
            recalc=recalc,
            col_enum=name_enums,
            prefix_enum=prefix_enums,
        )

        col_names = []
        for prefix_enum, name_enum in zip(prefix_enums, name_enums):
            col_names += [
                f"{prefix_enum.DISSIPATION_CORRECTION.value}_{name_enum.IN_PHASE.value}",
                f"{prefix_enum.DISSIPATION_CORRECTION.value}_{name_enum.QUADRATURE.value}",
            ]

        self.targ.data = ccat_df.unnest(
            self.targ, [f"struct_{name}" for name in col_names]
        )
        return self.targ.get_data(
            col_name=col_names,
            include=include,
            exclude=exclude,
        )

    # Timestream conversion
    # ---------------------
    def phase_spline(
        self,
        prefix: str
        | list[str] = "mismatch_rotate_origin_shift_origin_rotate_unwind_rotate",
        phase_low: float = -3.14,
        phase_up: float = 3.14,
        k: int = 3,
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        recalc: bool = False,
        max_workers=1,
        ex=None,
        **kwargs,
    ) -> pl.DataFrame:
        """Interpolate target sweep phase vs. frequency data and add interpolating splines to ``properties`` attribute

        Args:
            phase_low (float): Lower bound of phase to use for interpolation. Defaults to -pi
            phase_up (float): Upper bound of phase to use for interpolation. Defaults to +pi
            k (int): Degree of polynomials to use for interpolation. Defaults to degree 3.
        """

        name_enums, prefix_enums = ccat_df.create_enums(
            ["frequency", "phase"],
            prefix,
            ["spline"],
            self.analysis_cfg,
            no_prefix=["frequency"],
        )

        num_prefix = len(name_enums)
        phase_low = ccat_df.check_args(phase_low, num_prefix, float)
        phase_up = ccat_df.check_args(phase_up, num_prefix, float)
        k = ccat_df.check_args(k, num_prefix, int)

        args = [
            [
                self,
                low,
                up,
                kk,
                self.stream.timestamp,
                ccat_mp.check_max_workers(max_workers),
                ex,
            ]
            for low, up, kk in zip(phase_low, phase_up, k)
        ]
        self.properties
        self.targ.transform(
            [Detector._calc_phase_spline] * num_prefix,
            *args,
            include=include,
            exclude=exclude,
            recalc=recalc,
            col_enum=name_enums,
            prefix_enum=prefix_enums,
        )

        col_name = []
        for prefix_enum, name_enum in zip(prefix_enums, name_enums):
            col_name += [
                f"{self.stream.timestamp}_{prefix_enum.SPLINE.value}_{name_enum.FREQUENCY.value}_to_{name_enum.PHASE.value}",
                f"{self.stream.timestamp}_{prefix_enum.SPLINE.value}_{name_enum.PHASE.value}_to_{name_enum.FREQUENCY.value}",
            ]

        self.targ.data = ccat_df.unnest(
            self.targ, [f"struct_{name}" for name in col_name]
        )

        return self.targ.get_data(
            col_name=col_name,
            include=include,
            exclude=exclude,
            strict=True
        )

    def phase_to_f(
        self,
        prefix: str | list[str] = "mismatch_rotate_origin_shift_origin_rotate_unwind_rotate",
        spline_prefix: str | list[str] = '',
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        recalc: bool = False,
        max_workers=1,
        ex=None,
        **kwargs,
    ) -> pl.DataFrame:
        """
        Convert timestream phase data to frequency using target sweep phase vs. frequency interpolating spline

        Args:
            prefix (str | list[str]): Prefix of phase data to convert to frequency
            spline_prefix (str): Prefix of phase data used to construct spline
            phase_bounds (float): Amount to add to min and max timestream phase to determine phase vs. frequency spline bounds (i.e., ``phase_low = min_stream_phase - phase_bounds`` & ``phase_up = max_stream_phase + phase_bounds``)
            k (int): Order of polynomials to use for phase vs. frequency spline
            include (int | list[int] | None, optional): Detector(s) for which to perform calculation
            exclude (int | list[int] | None, optional): Detector(s) for which not to perform calculation
            recalc (bool): Whether to recalculate if data is already in ``data`` DataFrame. Defaults to False
            max_workers (int): Number of processor cores to use for calculation. Defaults to 1.
        """
        if not spline_prefix: spline_prefix = prefix

        name_enums, prefix_enums = ccat_df.create_enums(
            ["frequency", "phase"],
            prefix,
            [],
            self.analysis_cfg,
            no_prefix=[],
        )

        num_prefix = len(name_enums)
        spline_prefix = ccat_df.check_args(spline_prefix, num_prefix, str)

        spline_name_enums, spline_prefix_enums = ccat_df.create_enums(
            ["spline", "frequency", "phase"],
            spline_prefix,
            [],
            self.analysis_cfg,
            no_prefix=["spline"],
        )

        spline_dict = self.stream.spline_dict
        y_to_x_spline = [
            self.get_properties(
                col_name=f"{spline_name_enum.PHASE.value}_to_{spline_name_enum.FREQUENCY.value.split('_')[-1]}_{spline_name_enum.SPLINE.value}",
                include=include,
                exclude=exclude,
                strict=True,
            )
            .to_numpy()
            .T[1]
            for spline_name_enum in spline_name_enums
        ]
        x_to_y_spline = [
            self.get_properties(
                col_name=f"{spline_name_enum.FREQUENCY.value.split('_')[-1]}_to_{spline_name_enum.PHASE.value}_{spline_name_enum.SPLINE.value}",
                include=include,
                exclude=exclude,
                strict=True,
            )
            .to_numpy()
            .T[1]
            for spline_name_enum in spline_name_enums
        ]

        args = [
            [
                [spline_dict[spline] for spline in y_to_x],
                [spline_dict[spline] for spline in x_to_y],
                ccat_mp.check_max_workers(max_workers),
                ex,
            ]
            for y_to_x, x_to_y in zip(y_to_x_spline, x_to_y_spline)
        ]
        self.stream.transform(
            [Detector._calc_phase_to_f] * num_prefix,
            *args,
            include=include,
            exclude=exclude,
            recalc=recalc,
            col_enum=name_enums,
            prefix_enum=prefix_enums,
        )

        col_name = [name_enum.FREQUENCY.value for name_enum in name_enums]

        self.stream.data = ccat_df.unnest(
            self.stream, [f"struct_{name}" for name in col_name]
        )
        return self.stream.get_data(
            col_name=col_name,
            include=include,
            exclude=exclude,
            strict=True
        )

    def frac_f(
        self,
        prefix: str
        | list[str] = "mismatch_rotate_origin_shift_origin_rotate_unwind_rotate",
        ref_f: str | list[float] = "",
        data='timestream',
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        recalc: bool = False,
        **kwargs,
    ) -> pl.DataFrame:
        """
        Convert timestream frequency data to fractional frequency shift
        """
        if not ref_f: ref_f = self.analysis_cfg['convention']['name']['tone_frequency']

        name_enums, prefix_enums = ccat_df.create_enums(
            ["frequency", "fractional_frequency"],
            prefix,
            [],
            self.analysis_cfg,
            no_prefix=[],
        )

        data_objs, _ = self._get_data_obj(data)
        if not data_objs:
            error = (
                f"Invalid data type {data}, must be 'targ', 'timestream', or 'both'."
            )
            log.log("ERROR", error)
            raise ValueError(error)

        num_prefix = len(name_enums)
        ref_f = ccat_df.check_args(ref_f, num_prefix, str)

        for i, f_0 in enumerate(ref_f):
            if isinstance(f_0, str):
                ref_f[i] = (
                    self.get_properties(
                        f_0, include=include, exclude=exclude, strict=True
                    )
                    .to_numpy()
                    .T[1]
                )  # Get reference frequencies from .properties DataFrame

        frac_f_dfs, args = [], [[f] for f in ref_f]
        for data_obj in data_objs:
            data_obj.transform(
                [Detector._calc_frac_f] * num_prefix,
                *args,
                include=include,
                exclude=exclude,
                recalc=recalc,
                col_enum=name_enums,
                prefix_enum=prefix_enums,
            )

            frac_f_dfs.append(
                data_obj.get_data(
                    col_name=[name_enum.FRACTIONAL_FREQUENCY.value for name_enum in name_enums],
                    include=include,
                    exclude=exclude,
                    strict=True
                    ))

        return frac_f_dfs

    def linear_frac_f(self,
                      prefix: str | list[str] = "",
                      circle_fit_prefix: str | list[str] = "circle_fit_unwind_rotate",
                      mag_prefix: str = "",
                      mean_points: int = 10,
                      include: int | list[int] | None = None,
                      exclude: int | list[int] | None = None,
                      recalc: bool = False,
                      max_workers=1,
                      ex=None,
                      **kwargs):
        if isinstance(prefix, str):
            prefix = [prefix]

        name_enums, prefix_enums = ccat_df.create_enums(
            ["in_phase", "quadrature", "fractional_frequency"],
            prefix,
            ['linear'],
            self.analysis_cfg,
            no_prefix=[],
        )
        num_prefix = len(name_enums)
        mean_points = ccat_df.check_args(mean_points, num_prefix, int)
        mag_prefix = ccat_df.check_args(mag_prefix, num_prefix, str)
        circle_fit_prefix = ccat_df.check_args(circle_fit_prefix, num_prefix, str)

        # Get IQ circle radius
        # --------------------
        radii = [
            self.get_properties(
                col_name=f"{pre}_{self.analysis_cfg['convention']['name']['IQ_circle_radius']}",
                include=include,
                exclude=exclude,
                strict=True,
            )
            .to_numpy()
            .T[1]
            for pre in circle_fit_prefix
        ]

        # Estimate detector total quality factors
        # ---------------------------------------
        Qs = [[]]*num_prefix
        for i, (mag_pre, mean_point) in enumerate(zip(mag_prefix, mean_points)):
            f_low, f_mid, f_high = (
                property_routines.fwhm(
                    self.targ,
                    mag_pre,
                    mean_points=mean_point,
                    include=include,
                    exclude=exclude,
                    recalc=recalc,
                )
                .to_numpy()
                .T[4:]
            )
            fwhm = f_high - f_low
            Qs[i] = f_mid/fwhm

        args = [[radius, Q, ccat_mp.check_max_workers(max_workers), ex] for radius, Q in zip(radii, Qs)]
        self.targ.transform(
            [Detector._calc_linear_frac_f] * num_prefix,
            *args,
            include=include,
            exclude=exclude,
            recalc=recalc,
            col_enum=name_enums,
            prefix_enum=prefix_enums,
        )

        col_names = [f"{prefix_enum.LINEAR.value}_{name_enum.FRACTIONAL_FREQUENCY.value}" for prefix_enum, name_enum in zip(prefix_enums, name_enums)]

        self.targ.data = ccat_df.unnest(
            self.targ, [f"struct_{name}" for name in col_names]
        )
        return self.targ.get_data(
            col_name=col_names,
            include=include,
            exclude=exclude,
        )

    # ==================#
    # Analysis Methods #
    # ==================#

    @staticmethod
    def _calc_complex_fit(
        schema,
        *args,
        tones: list[int],
        padding: int = 4,
        recalc: bool = False,
        col_enum=None,
        prefix_enum=None,
    ):
        """Fit using resonator_model_v3"""

        def _complex_fit(df):
            data = ccat_mp.struct_batches(df, 3, batch_len, max_workers)

            results_dict = {}
            with ccat_mp.optional_executor(max_workers, ex=ex) as executor:
                future_to_batch = {
                    executor.submit(
                        ccat_mp.process_batches,
                        resonator_model_v3.full_fit,  # noqa: F821
                        data[i][0],
                        data[i][1],
                        data[i][2],
                        nonlinear=nonlinear[inds],
                        asymm=asymm[inds],
                        fix_cable=fix_cable[inds],
                        #fix_thetaQ=fix_thetaQ[inds],
                    ): (i, all_tones[inds])
                    for i, inds in enumerate(calc_ind)
                }

                for future in concurrent.futures.as_completed(future_to_batch):
                    i, tones = future_to_batch[future]
                    f_cols, fit_cols = data[i][0], future.result()
                    for tone, f_col, fit_col in zip(tones, f_cols, fit_cols):
                        if isinstance(fit_col, Exception):
                            log.log(
                                "DEBUG",
                                "Fit failed for tone %s with exception: %s",
                                tone,
                                fit_col,
                            )
                            best_fit, cable_fit = np.zeros(df.len()), np.zeros(df.len())
                            best_vals_dict = {}
                        else:
                            best_fit = fit_col.best_fit
                            cable_fit = resonator_model_v3.fine_s21_model(  # noqa: F821
                                f_col, fit_col.params, cable=True
                            )
                            best_vals_dict = {
                                f"{fit_name_col}_{k}": float(v)
                                for k, v in fit_col.best_values.items()
                            }
                        det.targ._properties[ccat_df.add_tone("det", tone, padding)] = (
                            best_vals_dict
                        )

                        for name, val in zip(
                            return_col,
                            [
                                best_fit.real,
                                best_fit.imag,
                                cable_fit.real,
                                cable_fit.imag,
                            ],
                        ):
                            results_dict[ccat_df.add_tone(name, tone, padding)] = val
            return ccat_mp.package_results(results_dict)

        if not len(args) == 7:
            log.log(
                "ERROR",
                "det, nonlinear, asymm, fix_cable, and fix_thetaQ, max_workers, and ex are required arguments.",
            )

        det, nonlinear, asymm, fix_cable, fix_thetaQ, max_workers, ex = np.array(args)
        det, max_workers, ex = det[0], int(max_workers[0]), ex[0]
        all_tones = np.array(tones)

        f_col, I_col, Q_col, fit_name_col, cable_col = (
            col_enum.FREQUENCY.value,
            col_enum.IN_PHASE.value,
            col_enum.QUADRATURE.value,
            prefix_enum.COMPLEX_FIT.value,
            prefix_enum.COMPLEX_FIT_CABLE.value,
        )

        return_col = [
            f"{fit_name_col}_{I_col}",
            f"{fit_name_col}_{Q_col}",
            f"{cable_col}_{fit_name_col}_{I_col}",
            f"{cable_col}_{fit_name_col}_{Q_col}",
        ]
        return_type = [pl.Float64, pl.Float64, pl.Float64, pl.Float64]
        expr, calc_ind, batch_len = ccat_mp.create_batches(
            _complex_fit,
            tones,
            schema,
            input_col=[f_col, I_col, Q_col],
            return_col=return_col,
            return_type=return_type,
            padding=padding,
            max_workers=max_workers,
            recalc=recalc,
        )
        return expr

    @staticmethod
    def _calc_phase_fit(
        schema,
        *args,
        tones: list[int],
        padding: int = 4,
        recalc: bool = False,
        col_enum=None,
        prefix_enum=None,
    ):
        def _phase_fit(df):
            data = ccat_mp.struct_batches(df, 4, batch_len, max_workers)

            results_dict = {}
            with ccat_mp.optional_executor(max_workers, ex=ex) as executor:
                future_to_batch = {
                    executor.submit(
                        ccat_mp.process_batches,
                        ccat_fit.phase_fit,
                        data[i][0],
                        data[i][1],
                        I=data[i][2],
                        Q=data[i][3],
                        nonlinear=nonlinear[inds],
                        method=method[inds],
                        params=params[inds],
                        R=radius[inds],
                    ): all_tones[inds]
                    for i, inds in enumerate(calc_ind)
                }

                fit_result = det.fit_result.get(f"{fit_name_col}_{fit_result_col}", [None]*det.targ.num_tones)
                for future in concurrent.futures.as_completed(future_to_batch):
                    tones = future_to_batch[future]
                    fit_cols = future.result()
                    for tone, fit_col in zip(tones, fit_cols):
                        if isinstance(fit_col, Exception):
                            log.log(
                                "INFO",
                                "Fit failed for tone %s with exception: %s",
                                tone,
                                fit_col,
                            )
                            best_fit = np.full(df.len(), np.nan)
                            properties_dict = {}
                        else:
                            best_fit = np.full(df.len(), np.nan)
                            mask = fit_col.mask
                            best_fit[mask] = fit_col.best_fit
                            best_vals_dict = {
                                f"{fit_name_col}_{param_names[k]}": float(v)
                                for k, v in fit_col.best_values.items()
                            }
                            init_vals_dict = {
                                f"init_{fit_name_col}_{param_names[k]}": float(v)
                                for k, v in fit_col.init_values.items()
                            }
                            fit_result[tone] = fit_col

                            properties_dict = best_vals_dict | init_vals_dict
                        det.targ._properties[ccat_df.add_tone("det", tone, padding)] = (
                            properties_dict
                        )
                        results_dict[ccat_df.add_tone(return_col[0], tone, padding)] = (
                            best_fit
                        )
                det.fit_result[f"{fit_name_col}_{fit_result_col}"] = fit_result
            return ccat_mp.package_results(results_dict)

        if not len(args) == 7:
            log.log(
                "ERROR",
                "nonlinear, params, window, and max_workers are required arguments.",
            )

        (
            det,
            radius,
            nonlinear,
            method,
            params_list,
            max_workers,
            ex,
        ) = args
        params = np.empty(len(params_list), dtype=object)
        for i in range(len(params_list)):
            params[i] = params_list[i]
        radius, nonlinear, method = (
            np.array(radius),
            np.array(nonlinear),
            np.array(method),
        )
        det, max_workers, ex = (
            det[0],
            int(max_workers[0]),
            ex[0],
        )
        all_tones = np.array(tones)

        f_col, I_col, Q_col, phase_col, fit_name_col, fit_result_col = (
            col_enum.FREQUENCY.value,
            col_enum.IN_PHASE.value,
            col_enum.QUADRATURE.value,
            col_enum.PHASE.value,
            prefix_enum.PHASE_FIT.value,
            col_enum.FIT_RESULT.value
        )

        param_names = {
            "f_0": col_enum.RESONANT_FREQUENCY.value,
            "Qr": col_enum.TOTAL_QUALITY_FACTOR.value,
            "theta_0": col_enum.PHASE_FIT_THETA.value,
            "beta": col_enum.PHASE_FIT_BETA.value,
            #"gamma": col_enum.PHASE_FIT_GAMMA.value,
            #"delta": col_enum.PHASE_FIT_DELTA.value,
            "R": col_enum.IQ_CIRCLE_RADIUS.value,
        }

        return_col, return_type = [f"{fit_name_col}_{phase_col}"], [pl.Float64]
        expr, calc_ind, batch_len = ccat_mp.create_batches(
            _phase_fit,
            tones,
            schema,
            input_col=[f_col, phase_col, I_col, Q_col],
            return_col=return_col,
            return_type=return_type,
            padding=padding,
            max_workers=max_workers,
            recalc=recalc,
        )
        return expr

    @staticmethod
    def _calc_IQ_circle_fit(
        schema,
        *args,
        tones: list[int],
        padding: int = 4,
        recalc: bool = False,
        col_enum=None,
        prefix_enum=None,
    ):
        def _circle_fit(df):
            data = ccat_mp.struct_batches(df, 2, batch_len, max_workers)

            angles = np.linspace(0, 2 * np.pi, df.len())
            sin = np.sin(angles)
            cos = np.cos(angles)

            results_dict = {}
            with ccat_mp.optional_executor(max_workers, ex=ex) as executor:
                future_to_batch = {
                    executor.submit(
                        ccat_mp.process_batches,
                        ccat_fit.circle_fit,
                        data[i][0],
                        data[i][1],
                        full_output=[True] * len(all_tones[inds]),
                        bounds=bounds[inds],
                        loss=loss[inds],
                        f_scale=f_scale[inds],
                        method=method[inds],
                    ): all_tones[inds]
                    for i, inds in enumerate(calc_ind)
                }

                for future in concurrent.futures.as_completed(future_to_batch):
                    tones = future_to_batch[future]
                    fit_cols = future.result()
                    for tone, fit_col in zip(tones, fit_cols):
                        properties_dict = {}
                        if isinstance(fit_col, Exception):
                            log.log(
                                "DEBUG",
                                "Fit failed for tone %s with exception: %s",
                                tone,
                                fit_col,
                            )
                            fit_I, fit_Q = (
                                np.full(df.len(), np.nan),
                                np.full(df.len(), np.nan),
                            )
                        else:
                            I_c, Q_c, R, result = fit_col
                            fit_I, fit_Q = R * cos + I_c, R * sin + Q_c
                            properties_dict[f"{fit_name_col}_{center_I_col}"] = I_c
                            properties_dict[f"{fit_name_col}_{center_Q_col}"] = Q_c
                            properties_dict[f"{fit_name_col}_{R_col}"] = R

                        det.targ._properties[ccat_df.add_tone("det", tone, padding)] = (
                            properties_dict
                        )
                        results_dict[ccat_df.add_tone(return_col[0], tone, padding)] = (
                            fit_I
                        )
                        results_dict[ccat_df.add_tone(return_col[1], tone, padding)] = (
                            fit_Q
                        )
            return ccat_mp.package_results(results_dict)

        if not len(args) == 7:
            log.log(
                "ERROR",
                "det, bounds, loss, f_scale, method, and max_workers are required arguments.",
            )
        det, bounds, loss, f_scale, method, max_workers, ex = np.array(args)
        det, max_workers, ex = (
            det[0],
            int(max_workers[0]),
            ex[0],
        )
        all_tones = np.array(tones)

        I_col, Q_col, fit_name_col = (
            col_enum.IN_PHASE.value,
            col_enum.QUADRATURE.value,
            prefix_enum.IQ_CIRCLE_FIT.value,
        )

        center_I_col, center_Q_col, R_col = (
            f"{col_enum.IQ_CIRCLE_CENTER.value}_{I_col.split('_')[-1]}",
            f"{col_enum.IQ_CIRCLE_CENTER.value}_{Q_col.split('_')[-1]}",
            col_enum.IQ_CIRCLE_RADIUS.value,
        )

        return_col, return_type = (
            [f"{fit_name_col}_{I_col}", f"{fit_name_col}_{Q_col}"],
            [pl.Float64, pl.Float64],
        )
        expr, calc_ind, batch_len = ccat_mp.create_batches(
            _circle_fit,
            tones,
            schema,
            input_col=[I_col, Q_col],
            return_col=return_col,
            return_type=return_type,
            padding=padding,
            max_workers=max_workers,
            recalc=recalc,
        )
        return expr

    @staticmethod
    def _calc_phase_spline(
        schema,
        *args,
        tones: list[int],
        padding: int = 4,
        recalc: bool = False,
        col_enum=None,
        prefix_enum=None,
    ):
        def _phase_spline(df):
            data = ccat_mp.struct_batches(df, 2, batch_len, max_workers)

            results_dict = {}
            with ccat_mp.optional_executor(max_workers, ex=ex) as executor:
                future_to_batch = {
                    executor.submit(
                        ccat_mp.process_batches,
                        ccat_fit.y_to_x_spline,
                        data[i][0],
                        data[i][1],
                        k=k[inds],
                        y_low=phase_low[inds],
                        y_up=phase_up[inds],
                    ): (i, all_tones[inds])
                    for i, inds in enumerate(calc_ind)
                }

                for future in concurrent.futures.as_completed(future_to_batch):
                    i, tones = future_to_batch[future]
                    spline_cols = future.result()
                    for j, (tone, spline_col) in enumerate(zip(tones, spline_cols)):
                        property_dict = {name: "None" for name in property_col}
                        spline_data = 2 * [np.full(df.len(), np.nan)]
                        if isinstance(spline_col, Exception):
                            log.log(
                                "DEBUG",
                                "Spline calculation for tone %s failed with exception: %s",
                                tone,
                                spline_col,
                            )
                        else:
                            for l, (name, spline, data_index) in enumerate(
                                zip(property_col, spline_col, [1, 0])
                            ):
                                if spline is not None:
                                    dat = data[i][data_index][j]
                                    spline.extrapolate = False
                                    spline_data[l] = spline(dat)
                                    det.stream.spline_dict[str(spline)] = spline
                                    property_dict[name] = str(spline)
                        det.stream._properties[
                            ccat_df.add_tone("det", tone, padding)
                        ] = property_dict
                        results_dict[ccat_df.add_tone(return_col[0], tone, padding)] = spline_data[0]
                        results_dict[ccat_df.add_tone(return_col[1], tone, padding)] = spline_data[1]
            return ccat_mp.package_results(results_dict)

        if not len(args) == 7:
            log.log(
                "ERROR",
                "det, phase_low, phase_up,, k, stream_timestam, max_workers, and ex are required arguments.",
            )

        det, phase_low, phase_up, k, stream_timestamp, max_workers, ex = np.array(args)
        det, stream_timestamp, max_workers, ex = (
            det[0],
            stream_timestamp[0],
            int(max_workers[0]),
            ex[0],
        )
        all_tones = np.array(tones)

        f_col, phase_col, spline_prefix = (
            col_enum.FREQUENCY.value,
            col_enum.PHASE.value,
            prefix_enum.SPLINE.value,
        )
        phase_to_f, f_to_phase = (
            f"{phase_col}_to_{f_col}",
            f"{f_col}_to_{phase_col}",
        )

        property_col = [
            f"{phase_to_f}_{spline_prefix}",
            f"{f_to_phase}_{spline_prefix}",
        ]

        return_col = [
            f"{stream_timestamp}_{spline_prefix}_{phase_to_f}",
            f"{stream_timestamp}_{spline_prefix}_{f_to_phase}",
        ]
        return_type = [pl.Float64, pl.Float64]
        expr, calc_ind, batch_len = ccat_mp.create_batches(
            _phase_spline,
            tones,
            schema,
            input_col=[f_col, phase_col],
            return_col=return_col,
            return_type=return_type,
            padding=padding,
            max_workers=max_workers,
            recalc=recalc,
        )
        return expr

    @staticmethod
    def _calc_phase_to_f(
        schema,
        *args,
        tones: list[int],
        padding: int = 4,
        recalc: bool = False,
        col_enum=None,
        prefix_enum=None,
    ):
        def _phase_to_f(df):
            data = ccat_mp.struct_batches(df, 1, batch_len, max_workers)

            results_dict = {}
            with ccat_mp.optional_executor(max_workers, ex=ex) as executor:
                future_to_batch = {
                    executor.submit(
                        ccat_mp.process_batches,
                        ccat_fit.y_to_x_interp,
                        data[i][0],
                        y_to_x_spline=y_to_x_spline[inds],
                        x_to_y_spline=x_to_y_spline[inds],
                    ): all_tones[inds]
                    for i, inds in enumerate(calc_ind)
                }

                for future in concurrent.futures.as_completed(future_to_batch):
                    tones = future_to_batch[future]
                    f_cols = future.result()
                    for tone, f_col in zip(tones, f_cols):
                        if isinstance(f_col, Exception):
                            log.log(
                                "DEBUG",
                                "Interpolation for tone %s failed with exception: %s",
                                tone,
                                f_col,
                            )
                            f_col = np.full(df.len(), np.nan)
                        results_dict[ccat_df.add_tone(return_col[0], tone, padding)] = (
                            f_col
                        )
            return ccat_mp.package_results(results_dict)

        if not len(args) == 4:
            log.log(
                "ERROR",
                "'y_to_x_spline', 'x_to_y_spline', 'max_workers', and 'ex' are required arguments",
            )

        y_to_x_spline, x_to_y_spline, max_workers, ex = np.array(args)
        max_workers, ex = int(max_workers[0]), ex[0]
        all_tones = np.array(tones)

        phase_col, f_col = col_enum.PHASE.value, col_enum.FREQUENCY.value
        return_col, return_type = [f_col], [pl.Float64]
        expr, calc_ind, batch_len = ccat_mp.create_batches(
            _phase_to_f,
            tones,
            schema,
            input_col=[phase_col],
            return_col=return_col,
            return_type=return_type,
            padding=padding,
            max_workers=max_workers,
            recalc=recalc,
        )
        return expr

    @staticmethod
    def _calc_frac_f(
        schema,
        *args,
        tones: list[int],
        padding: int = 4,
        recalc: bool = False,
        col_enum=None,
        prefix_enum=None,
    ):
        if not len(args) == 1:
            log.log("ERROR", "'f_0' is a required argument")
        f_0 = args[0]

        f_col, ff_col = col_enum.FREQUENCY.value, col_enum.FRACTIONAL_FREQUENCY.value

        exprs = []
        for tone, f in zip(tones, f_0):
            if (
                ff_tone_col := ccat_df.add_tone(ff_col, tone, padding)
            ) not in schema or recalc:
                f_tone_col = ccat_df.add_tone(f_col, tone, padding)
                exprs.append(((pl.col(f_tone_col) - f) / f).alias(ff_tone_col))
        return exprs

    @staticmethod
    def _calc_linear_frac_f(
        schema,
        *args,
        tones: list[int],
        padding: int = 4,
        recalc: bool = False,
        col_enum=None,
        prefix_enum=None,
    ):
        def _linear_frac_f(df):
            data = ccat_mp.struct_batches(df, 2, batch_len, max_workers)

            results_dict = {}
            with ccat_mp.optional_executor(max_workers, ex=ex) as executor:
                future_to_batch = {
                    executor.submit(
                        ccat_mp.process_batches,
                        ccat_fit.linear_frac_f,
                        np.array(data[i][0]),
                        np.array(data[i][1]),
                        R[inds],
                        Qr[inds]
                    ): all_tones[inds]
                    for i, inds in enumerate(calc_ind)
                }

                for future in concurrent.futures.as_completed(future_to_batch):
                    tones = future_to_batch[future]
                    frac_f_cols = future.result()
                    for tone, frac_f_col in zip(tones, frac_f_cols):
                        if isinstance(frac_f_col, Exception):
                            log.log(
                                "DEBUG",
                                "Linear fractional frequency shift calculation failed for tone %s with exception: %s",
                                tone,
                                frac_f_col,
                            )
                            frac_f = np.full(df.len(), np.nan)
                        else:
                            frac_f = frac_f_col
                            
                        results_dict[ccat_df.add_tone(return_col[0], tone, padding)] = frac_f
            return ccat_mp.package_results(results_dict)

        if not len(args) == 4:
            log.log(
                "ERROR",
                "R, Qr, max_workers, and ex are required arguments.",
            )
        R, Qr, max_workers, ex = np.array(args)
        max_workers, ex = (
            int(max_workers[0]),
            ex[0],
        )
        all_tones = np.array(tones)
        I_col, Q_col, frac_f_col, linear_prefix = col_enum.IN_PHASE.value, col_enum.QUADRATURE.value, col_enum.FRACTIONAL_FREQUENCY.value, prefix_enum.LINEAR.value

        return_col, return_type = (
            [f"{linear_prefix}_{frac_f_col}"],
            [pl.Float64],
        )
        expr, calc_ind, batch_len = ccat_mp.create_batches(
            _linear_frac_f,
            tones,
            schema,
            input_col=[I_col, Q_col],
            return_col=return_col,
            return_type=return_type,
            padding=padding,
            max_workers=max_workers,
            recalc=recalc,
        )
        return expr

    @staticmethod
    def _calc_dissipation_correction(
        schema,
        *args,
        tones: list[int],
        padding: int = 4,
        recalc: bool = False,
        col_enum=None,
        prefix_enum=None,
    ):
        def _dissipation_correction(df):
            data = ccat_mp.struct_batches(df, 3, batch_len, max_workers)

            results_dict = {}
            with ccat_mp.optional_executor(max_workers, ex=ex) as executor:
                future_to_batch = {
                    executor.submit(
                        ccat_mp.process_batches,
                        ccat_fit.dissipation_correction,
                        np.array(data[i][0]),
                        np.array(data[i][1]),
                        np.array(data[i][2]),
                        R[inds]
                    ): all_tones[inds]
                    for i, inds in enumerate(calc_ind)
                }

                for future in concurrent.futures.as_completed(future_to_batch):
                    tones = future_to_batch[future]
                    proj_cols = future.result()
                    for tone, proj_col in zip(tones, proj_cols):
                        if isinstance(proj_col, Exception):
                            log.log(
                                "DEBUG",
                                "Dissipation correction projection failed for tone %s with exception: %s",
                                tone,
                                proj_col,
                            )
                            proj_I, proj_Q = (
                                np.full(df.len(), np.nan),
                                np.full(df.len(), np.nan),
                            )
                        else:
                            proj_I, proj_Q = proj_col
                            
                        results_dict[ccat_df.add_tone(return_col[0], tone, padding)] = (
                            proj_I
                        )
                        results_dict[ccat_df.add_tone(return_col[1], tone, padding)] = (
                            proj_Q
                        )
            return ccat_mp.package_results(results_dict)

        if not len(args) == 3:
            log.log(
                "ERROR",
                "R, max_workers, and ex are required arguments.",
            )
        R, max_workers, ex = np.array(args)
        max_workers, ex = (
            int(max_workers[0]),
            ex[0],
        )
        all_tones = np.array(tones)
        I_col, Q_col, mag_col, corr_prefix = col_enum.IN_PHASE.value, col_enum.QUADRATURE.value, col_enum.MAGNITUDE.value, prefix_enum.DISSIPATION_CORRECTION.value

        return_col, return_type = (
            [f"{corr_prefix}_{I_col}", f"{corr_prefix}_{Q_col}"],
            [pl.Float64, pl.Float64],
        )
        expr, calc_ind, batch_len = ccat_mp.create_batches(
            _dissipation_correction,
            tones,
            schema,
            input_col=[I_col, Q_col, mag_col],
            return_col=return_col,
            return_type=return_type,
            padding=padding,
            max_workers=max_workers,
            recalc=recalc,
        )
        return expr

    def properties_histogram(
        self,
        col_name,
        bins=None,
        mad_filter=True,
        num_mads=10,
        filter_exprs=[],
        recalc=False,
        **kwargs,
    ):
        """Calculate histogram of a detector property

        Args:
            col_name (str): Name of property
            bins (int | None, optional): Number of bins to use for histogram. Defaults to number of tones used in histogram divided by five
            mad_filter (bool, optional): Whether to use the median absolute deviation (MAD) to filter outliers before creating histogram. Defaults to True
            num_mads (int, optional): Number of MADs data must be within to keep. Defaults to 10
            recalc (bool, optional): Whether to recalculate histogram if data already exists. Defaults to False
        Returns:
            return (pl.DataFrame): Polars DataFrame with histogram counts and bin edges
        """
        hist_col_name = ["counts", "edges"]
        hist_col_name = [f"hist_{name}_{col_name}" for name in hist_col_name]

        properties = self.properties
        if recalc or hist_col_name[0] not in properties.schema:
            num_tones = properties.height
            df = properties.select(col_name).filter(
                ~pl.col(col_name).is_nan()
            )  # Filter out NaN values
            if filter_exprs:
                df = df.filter(filter_exprs)

            if mad_filter:
                df = (
                    df.with_columns(pl.col(col_name).median().alias("median"))
                    .with_columns(
                        (pl.col(col_name) - pl.col("median"))
                        .abs()
                        .median()
                        .alias("MAD")
                    )
                    .filter(
                        (
                            pl.col(col_name)
                            > (pl.col("median") - num_mads * pl.col("MAD"))
                        )
                        & (
                            pl.col(col_name)
                            < (pl.col("median") + num_mads * pl.col("MAD"))
                        )
                    )
                )

            bins = df.height // 8 if bins is None else min(bins, num_tones)
            data = df[col_name].to_numpy()

            counts, edges = np.histogram(data, bins)
            counts, edges = (
                np.pad(
                    np.array(counts, dtype=float),
                    (0, int(num_tones - len(counts))),
                    constant_values=None,
                ),
                np.pad(
                    np.array(edges, dtype=float),
                    (0, int(num_tones - len(edges))),
                    constant_values=None,
                ),
            )

            self.properties = properties.with_columns(
                [
                    pl.Series(name, value)
                    for name, value in zip(hist_col_name, [counts, edges])
                ]
            )
        return self.properties.select(hist_col_name)

    # ==================#
    # Plotting Methods #
    # ==================#

    @staticmethod
    def _plot_histogram(df, plot_opts, *args, **kwargs):
        dynamic = kwargs["dynamic"] if "dynamic" in kwargs else True
        by, plot_median = args

        if by is None:
            df = df.with_columns(pl.lit(True).alias("tmp"))

        hist_dict = {}
        for *vals, counts, edges, median in (
            df.group_by(by if by is not None else ["tmp"])
            .agg("counts", "edges", "median")
            .iter_rows()
        ):
            label = (
                ",".join([f"{name}={value}" for name, value in zip(by, vals)])
                if by is not None
                else ""
            )
            hist = hv.Histogram((edges, counts), label=label).relabel(group="Detector")
            if plot_median:
                median = median[0]
                vline = hv.VLine(median, label="Median").relabel(group="Detector")
                spike = hv.Curve(
                    ([median, median], [0, 1]), label=f"Median: {median:0.2e}"
                ).relabel(group="Detector")
                hist = hist * spike * vline
            hist_dict[tuple(vals)] = hist

        hist = hv.HoloMap(hist_dict, kdims=by)
        hist.opts(*plot_opts)

        if dynamic and by is not None:
            hist = hv.util.Dynamic(hist)
        return hist

    def properties_histogram_plot(
        self,
        col_name,
        plot_median=None,
        xlabel="",
        title="",
        filter_exprs=[],
        save_fig: bool | None = None,
        figs_per_file: int | None = None,
        overwrite: bool | None = None,
        save_dir: str | Path | None = None,
        save_name: str = None,
        save_fmt: Format | None = None,
        return_fig=True,
        return_df=False,
        df=None,
        by=None,
        **kwargs,
    ):
        """Plot histogram of a detector property

        Args:
            col_name (str): Name of property
            plot_median (bool): Whether to plot a vertical line at the median
        Returns:
            return (hv.Histogram | hv.Overlay): Holoviews histogram figure
        """

        if df is None:
            df = self.properties_histogram(col_name, **kwargs)
            df = df.rename(
                {col: name for col, name in zip(df.columns, ["counts", "edges"])}
            ).with_columns(pl.lit(self.properties[col_name].median()).alias("median"))
            if filter_exprs:
                df = df.filter(filter_exprs)
        if not return_fig:
            return df, None

        if plot_median is None:
            plot_median = self.viz_cfg["static_plot"]["histogram"]["plot_median"]
        if isinstance(by, str):
            by = [by]
        args = [by, plot_median]

        cfg = self.targ.drone_cfg["det_config"]
        title = title if title else rf"${cfg['detector_type']}\ {cfg['network']}$"

        linewidth = (
            kwargs["linewidth"]
            if "linewidth" in kwargs
            else self.viz_cfg["static_plot"]["histogram"]["linewidth"]
        )
        linestyle = (
            kwargs["linestyle"]
            if "linestyle" in kwargs
            else self.viz_cfg["static_plot"]["histogram"]["linestyle"]
        )
        plot_opts = (
            opts.Histogram(
                xlabel=xlabel if xlabel else col_name,
                ylabel="Count",
                title=title,
                aspect=kwargs["aspect"]
                if "aspect" in kwargs
                else self.viz_cfg["static_plot"]["histogram"]["aspect"],
                fig_size=kwargs["fig_size"]
                if "aspect" in kwargs
                else self.viz_cfg["static_plot"]["histogram"]["fig_size"],
                show_grid=True,
                show_legend=True,
            ),
            opts.Curve(linewidth=linewidth, linestyle=linestyle),
            opts.VLine(linewidth=linewidth, linestyle=linestyle),
        )

        # Create plot for immediate visualization
        # ---------------------------------------
        plot = Detector._plot_histogram(df, plot_opts, *args, **kwargs)

        # Save plot in background
        # -----------------------
        if save_name is None:
            save_name = f"hist_{col_name}"
        viz_utils.save_fig(
            self,
            Detector._plot_histogram,
            df,
            plot_opts,
            *args,
            save_fig=save_fig,
            figs_per_file=figs_per_file,
            overwrite=overwrite,
            save_dir=save_dir,
            save_name=save_name,
            save_fmt=save_fmt,
            **kwargs,
        )

        if return_df:
            return plot, df
        else:
            return plot

    # ================#
    # Helper Methods #
    # ================#

    def _get_data_obj(self, data: str):
        """
        Get data object (Target, Timestream, or both) corresponding to string

        data (

        """
        data_objs = []
        data_types = []
        if data == "targ" or data == "both":
            data_objs.append(self.targ)
            data_types.append("targ")
        if data == "timestream" or data == "both":
            data_objs.append(self.stream)
            data_types.append("stream")
        return data_objs, data_types

    @staticmethod
    def _load_data(
        data_class,
        com_to,
        cfg_path,
        analysis_cfg,
        viz_cfg,
        dets,
        noise_tones,
        timestamp,
        data_path,
        **kwargs,
    ):
        """
        Load *ccatkidlib* data file into VNA, Target, or Timestream data object

        Args:
            data_class (VNA | Target | Timestream): Class corresponding to the type of data to load. Must be one of VNA, Target, or Timestream
            com_to (str):
            analysis_cfg (str): Path to analysis configuration file
            dets (int | list[int]): Subset of detectors to load
            timestamp (int | str | None): Timestamp of data file
            data_path (str | list[str] | pathlib.PosixPath | list[pathlib.PosixPath] | None): Path of data file. Can pass a list of file paths for a G3 timestream split into multiple files.
        """

        data = None
        if data_path is not None or timestamp is not None:
            try:
                data = data_class(
                    com_to=com_to,
                    cfg_path=cfg_path,
                    analysis_cfg=analysis_cfg,
                    viz_cfg=viz_cfg,
                    tones=dets,
                    noise_tones=noise_tones,
                    timestamp=timestamp,
                    data_path=data_path,
                    **kwargs,
                )
            except Exception as e:
                log.log(
                    "ERROR",
                    "Failed to load %s with exception: %s.",
                    data_class.__name__,
                    e,
                )
                data = None
        return data

    def join(self, other, in_place=False):
        def _join_consts(
            left_const: Any | list[Any], right_const: Any | list[Any]
        ) -> list[Any]:
            if not isinstance(left_const, list):
                left_const = [left_const]
            if not isinstance(right_const, list):
                right_const = [right_const]
            return left_const + right_const

        if not isinstance(other, Detector):
            error = f"Cannot join with object of type {type(other)}. Must be of type Detector."
            log.log("ERROR", error)
            raise ValueError(error)

        # Create a copy of the Detector object
        new_data = self if in_place else copy.deepcopy(self)
        new_data._cable_delay = _join_consts(self.cable_delay, other.cable_delay)

        # Join Target, Timestream, and VNA objects
        # ----------------------------------------
        new_data.targ = self.targ.join(other.targ, in_place=in_place)
        new_data.vna = _join_consts(self.vna, other.vna)

        left_stream, right_stream = self.stream, other.stream

        if not (left_stream is None or right_stream is None):
            new_data.stream = self.stream.join(other.stream, in_place=in_place)
        elif bool(left_stream is None) ^ bool(right_stream is None):
            error = "Cannot join Detector objects where one has a Timestream and the other does not. Either both or neither Detector objects must have a Timestream."
            log.log("ERROR", error)
            raise ValueError(error)

        # Join properties
        # ---------------
        left_prop, right_prop = self.properties, other.properties
        new_data._properties_df = pl.concat(
            [left_prop, right_prop], how="diagonal"
        ).with_columns(pl.Series("det", new_data.targ.tones))

        return new_data

    # ============= #
    # Magic Methods #
    # ============= #

    def __str__(self):
        return f"detector_{self.timestamp}"

    # =============================== #
    # Define Custom Pickling Behavior #
    # =============================== #

    def __getstate__(self):
        state = self.__dict__.copy()
        if (
            not self.analysis_cfg["io"]["pickle"]["pickle_dataframes"]
            and (properties := self._properties_df) is not None
        ):
            del state["_properties_df"]
            save_path, file_count = io.increment_file(
                Path(self.pickle_dir) / "dataframe",
                "properties_",
                ".parquet",
                overwrite=self.analysis_cfg["io"]["pickle"]["overwrite"],
            )
            state["pickle_count"] = file_count
            if isinstance(properties, pl.DataFrame):
                properties.write_parquet(save_path)
            elif isinstance(properties, pl.LazyFrame):
                properties.sink_parquet(save_path)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not self.analysis_cfg["io"]["pickle"]["pickle_dataframes"] and getattr(
            self, "_properties_df", True
        ):
            file_name = (
                "properties"
                if (pickle_count := self.pickle_count) is None
                else f"properties_{pickle_count}"
            )

            analysis_cfg, _ = io.load_config(
                str(Path(__file__).parents[1] / "analysis_config.yaml")
            )
            if curr_dir := analysis_cfg["io"]["pickle"]["curr_pickle_root_dir"]:
                self.analysis_cfg["io"]["pickle"]["pickle_root_dir"] = curr_dir

            self.properties = pl.scan_parquet(
                Path(self.pickle_dir) / "dataframe" / f"{file_name}.parquet"
            )
