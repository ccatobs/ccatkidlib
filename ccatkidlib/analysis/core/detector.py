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
from concurrent.futures import ProcessPoolExecutor
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

from ccatkidlib.analysis.utils.config import return_prefix
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
        """Lazily loaded *Polars* DataFrame of per-|tone| detector properties.

        Collects any ``pl.LazyFrame``, merges in target-sweep and timestream
        per-tone properties, and returns the unified ``_properties_df``.

        Returns:
            pl.DataFrame: Per-|tone| detector properties.
        """

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

    def cable_delay(
        self,
        prefix: str = "",
        mag_prefix: str = "",
        window: int | list[int] = 5,
        mean_points: int | list[int] = 10,
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        recalc: bool = False,
        stitch_workers: int = 1,
        linear_fit_workers: int = 1,
        ex: ProcessPoolExecutor | None = None,
    ) -> pl.DataFrame:
        """Calculate the cable delay of the detector network.

        Delegates to ``Target.cable_delay()`` for the target sweep, then
        uses a VNA-derived delay as a fallback for |tones| where the
        target-sweep estimate is NaN.

        Args:
            prefix (str, optional): Prefix of |I| & |Q| data. Defaults to
                ``''``.
            mag_prefix (str, optional): Prefix of magnitude data. Defaults to
                ``''``.
            window (int | list[int], optional): Smoothing window for stitch.
                Defaults to 5.
            mean_points (int | list[int], optional): Points to average at
                cable-delay line endpoints. Defaults to 10.
            include (int | list[int] | None, optional): |tones| to include.
                Defaults to *None* (all).
            exclude (int | list[int] | None, optional): |tones| to exclude.
                Defaults to *None*.
            recalc (bool, optional): Recalculate even if result already exists.
                Defaults to *False*.
            stitch_workers (int, optional): Worker processes for stitching.
                Defaults to 1.
            linear_fit_workers (int, optional): Worker processes for linear fit.
                Defaults to 1.
            ex (Any | None, optional): Existing executor to reuse. Defaults to
                *None*.
        Returns:
            pl.DataFrame: Sub-frame of ``properties`` with VNA and target
            cable-delay columns.
        """
        delay_col = self.analysis_cfg["convention"]["name"]["cable_delay"]
        vna_col, targ_col = "_".join(["vna", delay_col]), "_".join(["targ", delay_col])

        self.targ.cable_delay(
            prefix=prefix,
            mag_prefix=mag_prefix,
            window=window,
            mean_points=mean_points,
            include=include,
            exclude=exclude,
            recalc=recalc,
            stitch_workers=stitch_workers,
            linear_fit_workers=linear_fit_workers,
            ex=ex,
        )
        vna_delay = self.vna.cable_delay() if isinstance(self.vna, VNA) else 0
        self._properties_df = self.properties.with_columns(
            pl.lit(vna_delay).alias(vna_col)
        ).with_columns(
            pl.when(pl.col(targ_col).is_nan())
            .then(pl.col(vna_col))
            .otherwise(pl.col(targ_col))
            .alias(targ_col)
        )

        return self.get_properties(
            [vna_col, targ_col], strict=True, include=include, exclude=exclude
        )

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
    def fit_dir(self) -> str:
        """Directory path where fit files are stored, from analysis config."""
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
    ) -> pl.DataFrame:
        """Get the specified data columns and rows from the ``properties`` Polars DataFrame.

        Args:
            col_name (str | list[str], optional): Column name pattern or list of
                names to select. Defaults to all columns.
            include (int | list[int] | None, optional): |tones| to include.
                Defaults to *None* (all).
            exclude (int | list[int] | None, optional): |tones| to exclude.
                Defaults to *None*.
            strict (bool, optional): Raise if a requested column is absent.
                Defaults to *False*.
        Returns:
            pl.DataFrame: Sub-frame of ``properties`` matching ``col_name`` and
            detector filters.
        """
        return ccat_df.get_properties(
            self, col_name=col_name, include=include, exclude=exclude, strict=strict
        )

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
        if isinstance(prefix, str):
            prefix = [prefix]
        name_enums, prefix_enums = ccat_df.create_enums(
            ["frequency", "in_phase", "quadrature", "tone_frequency"],
            prefix,
            ["rotate", "remove_cable"],
            self.analysis_cfg,
            no_prefix=["frequency", "tone_frequency"],
        )

        # TODO: Allow different delay cols for each prefix
        data_objs = self._get_data_obj(data)
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

        with return_prefix(self.analysis_cfg) as ret_prefix:
            unwind_dfs = []
            for data_obj in data_objs:
                angle = (
                    [
                        delay * tone_freq
                        for tone_freq, delay in zip(
                            data_obj.get_data(
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
                    if data_obj.data_type == "targ"
                    else [
                        delay * tone_freq
                        for tone_freq, delay in zip(
                            data_obj.comb[name_enums[0].TONE_FREQUENCY.value]
                            .to_numpy()
                            .T,
                            delays,
                        )
                    ]
                )
                unwind_df, unwind_prefix = data_obj.IQ_rotate(
                    prefix=prefix,
                    angle=angle,
                    name=[
                        prefix_enum.REMOVE_CABLE.value for prefix_enum in prefix_enums
                    ],
                    include=include,
                    exclude=exclude,
                    recalc=recalc,
                )
                unwind_dfs.append(unwind_df)

            ret_pre = unwind_prefix
            if len(ret_pre) == 1:
                ret_pre = ret_pre[0]

            return unwind_dfs, ret_pre if ret_prefix else unwind_dfs

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
        if isinstance(prefix, str):
            prefix = [prefix]
        name_enums, prefix_enums = ccat_df.create_enums(
            ["frequency", "in_phase", "quadrature", "magnitude", "tone_frequency"],
            prefix,
            ["scale", "normalize"],
            self.analysis_cfg,
            no_prefix=["frequency", "magnitude", "tone_frequency"],
        )

        data_objs = self._get_data_obj(data)
        if not data_objs:
            error = (
                f"Invalid data type {data}, must be 'targ', 'timestream', or 'both'."
            )
            log.log("ERROR", error)
            raise ValueError(error)

        tone_freqs = self.get_properties(
            name_enums[0].TONE_FREQUENCY.value,
            include=include,
            exclude=exclude,
            strict=True,
        )

        with return_prefix(self.analysis_cfg) as ret_prefix:
            norm_dfs = []
            for data_obj in data_objs:
                # TODO: Allow different norm_prefix for each prefix
                cable_mags = self.targ.get_data(
                    f"{norm_prefix}{'_' if norm_prefix else ''}{name_enums[0].MAGNITUDE.value}",
                    include=include,
                    exclude=exclude,
                    strict=True,
                )
                if data_obj.data_type == "targ":
                    scale = 1 / cable_mags.to_numpy().T
                else:
                    # TODO: Probably unnecessary computation to find data point corresponding to tone. Should always be sample = num_samples/2 (unless timestream tone frequency is changed without retaking target sweep)
                    f_col = name_enums[0].FREQUENCY.value
                    f_df = (
                        self.targ.get_data(
                            f_col, include=include, exclude=exclude, strict=True
                        )
                        .unpivot(variable_name="det", value_name=f_col)
                        .with_columns(
                            pl.col("det").str.strip_prefix(f"{f_col}_").cast(int)
                        )
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
                            .with_columns(
                                (
                                    pl.col(f_col)
                                    - pl.col(name_enums[0].TONE_FREQUENCY.value)
                                )
                                .abs()
                                .alias("diff")
                            )
                            .filter(pl.col("diff") == pl.col("diff").min().over("det"))
                            .select("cable")
                            .collect()
                        )
                        .to_numpy()
                        .T[0]
                    )

                norm_df, norm_prefix = data_obj.IQ_scale(
                    prefix=prefix,
                    scale=scale,
                    name=[prefix_enum.NORMALIZE.value for prefix_enum in prefix_enums],
                    include=include,
                    exclude=exclude,
                    recalc=recalc,
                )

                norm_dfs.append(norm_df)
            ret_pre = norm_prefix
            if len(ret_pre) == 1:
                ret_pre = ret_pre[0]

            return norm_dfs, ret_pre if ret_prefix else norm_dfs

    def IQ_circle_origin(
        self,
        prefix: str | list[str] = "unwind_rotate",
        circle_fit_prefix: str | list[str] = "",
        data: str = "both",
        use_fit: bool | list[bool] = True,
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        recalc: bool = False,
    ) -> list[pl.DataFrame]:
        """Rotate and centre the target-sweep |IQ| circle onto the real axis at the origin.

        Args:
            prefix (str | list[str], optional): Prefix of |I| & |Q| data.
                Defaults to ``'unwind_rotate'``.
            circle_fit_prefix (str | list[str], optional): Prefix of circle-fit
                results used to retrieve the circle centre. Defaults to ``''``.
            data (str, optional): Which data objects to transform; ``'targ'``,
                ``'timestream'``, or ``'both'``. Defaults to ``'both'``.
            use_fit (bool | list[bool], optional): Use circle-fit centre rather
                than median |I|/|Q|. Defaults to *True*.
            include (int | list[int] | None, optional): |tones| to include.
                Defaults to *None* (all).
            exclude (int | list[int] | None, optional): |tones| to exclude.
                Defaults to *None*.
            recalc (bool, optional): Recalculate even if result already exists.
                Defaults to *False*.
        Returns:
            list[pl.DataFrame]: Transformed data DataFrames (one per data
            object).
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

        with return_prefix(self.analysis_cfg) as ret_prefix:
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

            data_objs = self._get_data_obj(data)
            if not data_objs:
                error = f"Invalid data type {data}, must be 'targ', 'timestream', or 'both'."
                log.log("ERROR", error)
                raise ValueError(error)

            shift_dfs = []
            for data_obj in data_objs:
                _, rotate_prefix = data_obj.IQ_rotate(
                    prefix=prefix,
                    angle=angles,
                    name=[
                        prefix_enum.CENTER_ORIGIN.value for prefix_enum in prefix_enums
                    ],
                    include=include,
                    exclude=exclude,
                    recalc=recalc,
                )
                shift_df, shift_prefix = data_obj.IQ_shift(
                    prefix=rotate_prefix,
                    shift_I=shifts,
                    name=[
                        prefix_enum.CENTER_ORIGIN.value for prefix_enum in prefix_enums
                    ],
                    include=include,
                    exclude=exclude,
                    recalc=recalc,
                )
                shift_dfs.append(shift_df)

            ret_pre = shift_prefix
            if len(ret_pre) == 1:
                ret_pre = ret_pre[0]

            return shift_dfs, ret_pre if ret_prefix else shift_dfs

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

        with return_prefix(self.analysis_cfg) as ret_prefix:
            angles, rot_names = [[]] * num_prefix, [""] * num_prefix
            for i, (pre, name_enum, prefix_enum, rot, mean_point) in enumerate(
                zip(prefix, name_enums, prefix_enums, rotation, mean_points)
            ):
                if rot.lower() == "mismatch":
                    mismatch_df, _ = self.targ.mismatch_angle(
                        pre,
                        mean_points=mean_point,
                        include=include,
                        exclude=exclude,
                        recalc=recalc,
                    )
                    angles[i] = mismatch_df.to_numpy().T[1]
                    rot_names[i] = prefix_enum.REMOVE_IMPEDANCE_MISMATCH.value
                elif rot.lower() == "timestream":
                    stream_df, _ = self.stream.timestream_angle(
                        pre,
                        include=include,
                        exclude=exclude,
                        recalc=recalc,
                    )
                    angles[i] = stream_df.to_numpy().T[1]
                    rot_names[i] = prefix_enum.CENTER_TIMESTREAM.value
                else:
                    error = f"Invalid rotation '{rot}' specified; Must be one of 'mismatch' or 'timestream'."
                    log.log("ERROR", error)
                    raise ValueError(error)

            data_objs = self._get_data_obj(data)
            rotation_dfs = []
            for data_obj in data_objs:
                rotate_df, rotate_prefix = data_obj.IQ_rotate(
                    prefix=prefix,
                    angle=angles,
                    name=rot_names,
                    include=include,
                    exclude=exclude,
                    recalc=recalc,
                )
                rotation_dfs.append(rotate_df)

            ret_pre = rotate_prefix
            if len(ret_pre) == 1:
                ret_pre = ret_pre[0]

            return rotation_dfs, ret_pre if ret_prefix else rotation_dfs

    def IQ_circle_natural(
        self,
        prefix: str | list[str] = "",
        data: str = "both",
        mean_points: int | list[int] = 10,
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        recalc: bool = False,
    ) -> list[pl.DataFrame]:
        """Shift an origin-centred |IQ| circle to the natural resonator coordinates.

        In natural coordinates the off-resonance point sits at ``(1, 0)``.
        The shift distance is computed as ``1 − mismatch_dist``.

        Args:
            prefix (str | list[str], optional): Prefix of |I| & |Q| data.
                Defaults to ``''``.
            data (str, optional): Which data objects to transform; ``'targ'``,
                ``'timestream'``, or ``'both'``. Defaults to ``'both'``.
            mean_points (int | list[int], optional): Points averaged to estimate
                the off-resonance position. Defaults to 10.
            include (int | list[int] | None, optional): |tones| to include.
                Defaults to *None* (all).
            exclude (int | list[int] | None, optional): |tones| to exclude.
                Defaults to *None*.
            recalc (bool, optional): Recalculate even if result already exists.
                Defaults to *False*.
        Returns:
            list[pl.DataFrame]: Transformed data DataFrames (one per data
            object).
        """
        if isinstance(prefix, str):
            prefix = [prefix]

        name_enums, prefix_enums = ccat_df.create_enums(
            ["in_phase", "quadrature"],
            prefix,
            ["center_natural", "rotate"],
            self.analysis_cfg,
            no_prefix=[],
        )

        num_prefix = len(name_enums)
        mean_points = ccat_df.check_args(mean_points, num_prefix, int)

        with return_prefix(self.analysis_cfg) as ret_prefix:
            dists = [None] * num_prefix
            for i, (pre, mean_point) in enumerate(zip(prefix, mean_points)):
                dist, _ = self.targ.mismatch_dist(
                    prefix=pre,
                    mean_points=mean_point,
                    include=include,
                    exclude=exclude,
                    recalc=recalc,
                )
                dists[i] = 1 - dist.to_numpy().T[1]

            data_objs = self._get_data_obj(data)
            nat_dfs = [None] * len(data_objs)
            for i, data_obj in enumerate(data_objs):
                _, rotate_prefix = data_obj.IQ_rotate(
                    prefix=prefix,
                    angle=np.pi,
                    name="half",
                    include=include,
                    exclude=exclude,
                    recalc=recalc,
                )
                nat_df, nat_prefix = data_obj.IQ_shift(
                    prefix=rotate_prefix,
                    shift_I=dists,
                    shift_Q=0,
                    name=prefix_enums[0].CENTER_NATURAL.value,
                    include=include,
                    exclude=exclude,
                    recalc=recalc,
                )
                nat_dfs[i] = nat_df

            ret_pre = nat_prefix
            if len(ret_pre) == 1:
                ret_pre = ret_pre[0]

            return nat_dfs, ret_pre if ret_prefix else nat_dfs

    def IQ_circle_tail(
        self,
        prefix: str | list[str] = "",
        data: str = "both",
        mean_points: int | list[int] = 10,
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        recalc: bool = False,
    ) -> list[pl.DataFrame]:
        """Shift an origin-centred |IQ| circle so the on-resonance tail sits at the origin.

        The shift distance equals ``mismatch_dist`` (versus ``1 − mismatch_dist``
        used by ``IQ_circle_natural``), placing the resonance tail at ``(0, 0)``.

        Args:
            prefix (str | list[str], optional): Prefix of |I| & |Q| data.
                Defaults to ``''``.
            data (str, optional): Which data objects to transform; ``'targ'``,
                ``'timestream'``, or ``'both'``. Defaults to ``'both'``.
            mean_points (int | list[int], optional): Points averaged to estimate
                the on-resonance position. Defaults to 10.
            include (int | list[int] | None, optional): |tones| to include.
                Defaults to *None* (all).
            exclude (int | list[int] | None, optional): |tones| to exclude.
                Defaults to *None*.
            recalc (bool, optional): Recalculate even if result already exists.
                Defaults to *False*.
        Returns:
            list[pl.DataFrame]: Transformed data DataFrames (one per data
            object).
        """
        if isinstance(prefix, str):
            prefix = [prefix]

        name_enums, prefix_enums = ccat_df.create_enums(
            ["in_phase", "quadrature"],
            prefix,
            ["center_tail"],
            self.analysis_cfg,
            no_prefix=[],
        )

        num_prefix = len(name_enums)
        mean_points = ccat_df.check_args(mean_points, num_prefix, int)

        with return_prefix(self.analysis_cfg) as ret_prefix:
            dists = [None] * num_prefix
            for i, (pre, mean_point) in enumerate(zip(prefix, mean_points)):
                dist, _ = self.targ.mismatch_dist(
                    prefix=pre,
                    mean_points=mean_point,
                    include=include,
                    exclude=exclude,
                    recalc=recalc,
                )
                dists[i] = dist.to_numpy().T[1]

            data_objs = self._get_data_obj(data)
            tail_dfs = [None] * len(data_objs)
            for i, data_obj in enumerate(data_objs):
                tail_df, tail_prefix = data_obj.IQ_shift(
                    prefix=prefix,
                    shift_I=dists,
                    shift_Q=0,
                    name=prefix_enums[0].CENTER_TAIL.value,
                    include=include,
                    exclude=exclude,
                    recalc=recalc,
                )
                tail_dfs[i] = tail_df

            ret_pre = tail_prefix
            if len(ret_pre) == 1:
                ret_pre = ret_pre[0]

            return tail_dfs, ret_pre if ret_prefix else tail_dfs

    def IQ_circle_diss_corr(
        self,
        prefix: str | list[str] = "",
        circle_fit_prefix: str | list[str] = "circle_fit_unwind_rotate",
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        recalc: bool = False,
        max_workers: int = 1,
        ex: ProcessPoolExecutor | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        """Apply dissipation correction to origin-centred |IQ| circle data.

        Projects |I| & |Q| onto the dissipation axis using the circle radius
        from a prior circle fit.

        Args:
            prefix (str | list[str], optional): Prefix of |I| & |Q| data.
                Defaults to ``''``.
            circle_fit_prefix (str | list[str], optional): Prefix under which the
                |IQ| circle radius was stored. Defaults to
                ``'circle_fit_unwind_rotate'``.
            include (int | list[int] | None, optional): |tones| to include.
                Defaults to *None* (all).
            exclude (int | list[int] | None, optional): |tones| to exclude.
                Defaults to *None*.
            recalc (bool, optional): Recalculate even if result already exists.
                Defaults to *False*.
            max_workers (int, optional): Worker processes for computation.
                Defaults to 1.
            ex (Any | None, optional): Existing executor to reuse. Defaults to
                *None*.
        Returns:
            pl.DataFrame: Target-sweep DataFrame with dissipation-corrected
            columns.
        """
        if isinstance(prefix, str):
            prefix = [prefix]

        name_enums, prefix_enums = ccat_df.create_enums(
            ["in_phase", "quadrature", "magnitude"],
            prefix,
            ["dissipation_correction"],
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

        args = [
            [radius, ccat_mp.check_max_workers(max_workers), ex] for radius in radii
        ]
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

        ret_df = self.targ.get_data(
            col_name=col_names,
            include=include,
            exclude=exclude,
        )

        ret_pre = [
            f"{prefix_enum.DISSIPATION_CORRECTION.value}{'_' if pre else ''}{pre}"
            for pre, prefix_enum in zip(prefix, prefix_enums)
        ]
        if len(ret_pre) == 1:
            ret_pre = ret_pre[0]

        return ret_df, ret_pre if self.analysis_cfg["runtime"][
            "return_prefix"
        ] else ret_df

    # Timesteream conversion
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
        max_workers: int = 1,
        ex: ProcessPoolExecutor | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        """Interpolate target-sweep phase vs. frequency and store splines in ``properties``.

        Fits bidirectional interpolating splines (phase→frequency and
        frequency→phase) over the unwind-rotated target sweep for each |tone|,
        then writes the spline objects to ``stream.spline_dict`` and their
        string keys to ``properties``.

        Args:
            prefix (str | list[str], optional): Prefix of phase and frequency
                data. Defaults to
                ``'mismatch_rotate_origin_shift_origin_rotate_unwind_rotate'``.
            phase_low (float, optional): Lower phase bound for spline fit.
                Defaults to -3.14 (approx. −π).
            phase_up (float, optional): Upper phase bound for spline fit.
                Defaults to 3.14 (approx. +π).
            k (int, optional): Polynomial degree for spline. Defaults to 3.
            include (int | list[int] | None, optional): |tones| to include.
                Defaults to *None* (all).
            exclude (int | list[int] | None, optional): |tones| to exclude.
                Defaults to *None*.
            recalc (bool, optional): Recalculate even if result already exists.
                Defaults to *False*.
            max_workers (int, optional): Worker processes. Defaults to 1.
            ex (Any | None, optional): Existing executor to reuse. Defaults to
                *None*.
        Returns:
            pl.DataFrame: Target-sweep DataFrame with spline-interpolated
            columns.
        """

        if isinstance(prefix, str):
            prefix = [prefix]
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

        ret_df = self.targ.get_data(
            col_name=col_name, include=include, exclude=exclude, strict=True
        )

        ret_pre = [
            (
                f"{self.stream.timestamp}_{prefix_enum.SPLINE.value}_{name_enum.FREQUENCY.value}_to{'_' if pre else ''}{pre}",
                f"{self.stream.timestamp}_{prefix_enum.SPLINE.value}_{name_enum.PHASE.value}_to",
            )
            for pre, prefix_enum, name_enum in zip(prefix, prefix_enums, name_enums)
        ]
        if len(ret_pre) == 1:
            ret_pre = ret_pre[0]

        return ret_df, ret_pre if self.analysis_cfg["runtime"][
            "return_prefix"
        ] else ret_df

    def phase_to_f(
        self,
        prefix: str
        | list[str] = "mismatch_rotate_origin_shift_origin_rotate_unwind_rotate",
        spline_prefix: str | list[str] = "",
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        recalc: bool = False,
        max_workers: int = 1,
        ex: ProcessPoolExecutor | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        """Convert timestream phase data to frequency via a target-sweep spline.

        Looks up pre-built phase→frequency splines from ``stream.spline_dict``
        (keyed by their string representation stored in ``properties``) and
        applies them to the timestream phase columns.

        Args:
            prefix (str | list[str], optional): Prefix of phase data to convert.
                Defaults to
                ``'mismatch_rotate_origin_shift_origin_rotate_unwind_rotate'``.
            spline_prefix (str | list[str], optional): Prefix of phase data used
                to construct the spline (matches the ``prefix`` passed to
                ``phase_spline``). Defaults to the value of ``prefix``.
            include (int | list[int] | None, optional): |tones| to include.
                Defaults to *None* (all).
            exclude (int | list[int] | None, optional): |tones| to exclude.
                Defaults to *None*.
            recalc (bool, optional): Recalculate even if result already exists.
                Defaults to *False*.
            max_workers (int, optional): Worker processes for computation.
                Defaults to 1.
            ex (Any | None, optional): Existing executor to reuse. Defaults to
                *None*.
        Returns:
            pl.DataFrame: Timestream DataFrame with frequency columns derived
            from phase via the target-sweep spline.
        """
        if not spline_prefix:
            spline_prefix = prefix

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
        ret_df = self.stream.get_data(
            col_name=col_name, include=include, exclude=exclude, strict=True
        )

        return ret_df, prefix if self.analysis_cfg["runtime"][
            "return_prefix"
        ] else ret_df

    def frac_f(
        self,
        prefix: str
        | list[str] = "mismatch_rotate_origin_shift_origin_rotate_unwind_rotate",
        ref_f: str | list[float] = "",
        data: str = "timestream",
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        recalc: bool = False,
        **kwargs,
    ) -> pl.DataFrame:
        r"""Convert frequency data to fractional frequency shift :math:`\delta f / f_0`.

        Args:
            prefix (str | list[str], optional): Prefix of frequency data to
                convert. Defaults to
                ``'mismatch_rotate_origin_shift_origin_rotate_unwind_rotate'``.
            ref_f (str | list[float], optional): Reference frequency: either
                a ``properties`` column name (str) or per-|tone| float values.
                Defaults to the ``tone_frequency`` config column.
            data (str, optional): Which data objects to convert; ``'targ'``,
                ``'timestream'``, or ``'both'``. Defaults to ``'timestream'``.
            include (int | list[int] | None, optional): |tones| to include.
                Defaults to *None* (all).
            exclude (int | list[int] | None, optional): |tones| to exclude.
                Defaults to *None*.
            recalc (bool, optional): Recalculate even if result already exists.
                Defaults to *False*.
        Returns:
            list[pl.DataFrame]: Data DataFrames with fractional-frequency
            columns (one per data object).
        """
        if not ref_f:
            ref_f = self.analysis_cfg["convention"]["name"]["tone_frequency"]

        name_enums, prefix_enums = ccat_df.create_enums(
            ["frequency", "fractional_frequency"],
            prefix,
            [],
            self.analysis_cfg,
            no_prefix=[],
        )

        data_objs = self._get_data_obj(data)
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
                    col_name=[
                        name_enum.FRACTIONAL_FREQUENCY.value for name_enum in name_enums
                    ],
                    include=include,
                    exclude=exclude,
                    strict=True,
                )
            )

        return frac_f_dfs, prefix if self.analysis_cfg["runtime"][
            "return_prefix"
        ] else frac_f_dfs

    def linear_frac_f(
        self,
        prefix: str | list[str] = "",
        circle_fit_prefix: str | list[str] = "circle_fit_unwind_rotate",
        mag_prefix: str = "",
        mean_points: int = 10,
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        recalc: bool = False,
        max_workers: int = 1,
        ex: ProcessPoolExecutor | None = None,
        **kwargs,
    ) -> pl.DataFrame:
        r"""Estimate fractional frequency shift linearly from origin-centred |IQ| position.

        Uses the circle radius and total quality factor Q to convert |I|/|Q|
        displacement to :math:`\delta f / f_0` without requiring a phase spline.

        Args:
            prefix (str | list[str], optional): Prefix of |I| & |Q| data.
                Defaults to ``''``.
            circle_fit_prefix (str | list[str], optional): Prefix of circle-fit
                results used to retrieve the radius. Defaults to
                ``'circle_fit_unwind_rotate'``.
            mag_prefix (str, optional): Prefix of magnitude data used to
                estimate Q from FWHM. Defaults to ``''``.
            mean_points (int, optional): Points averaged for FWHM estimation.
                Defaults to 10.
            include (int | list[int] | None, optional): |tones| to include.
                Defaults to *None* (all).
            exclude (int | list[int] | None, optional): |tones| to exclude.
                Defaults to *None*.
            recalc (bool, optional): Recalculate even if result already exists.
                Defaults to *False*.
            max_workers (int, optional): Worker processes. Defaults to 1.
            ex (Any | None, optional): Existing executor to reuse. Defaults to
                *None*.
        Returns:
            pl.DataFrame: Target-sweep DataFrame with linear fractional-
            frequency columns.
        """
        if isinstance(prefix, str):
            prefix = [prefix]

        name_enums, prefix_enums = ccat_df.create_enums(
            ["in_phase", "quadrature", "fractional_frequency"],
            prefix,
            ["linear"],
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
        Qs, return_prefix = (
            [[]] * num_prefix,
            self.analysis_cfg["runtime"]["return_prefix"],
        )
        for i, (mag_pre, mean_point) in enumerate(zip(mag_prefix, mean_points)):
            fwhm_df = self.targ.fwhm(
                mag_pre,
                mean_points=mean_point,
                include=include,
                exclude=exclude,
                recalc=recalc,
            )
            if return_prefix:
                fwhm_df = fwhm_df[0]
            f_low, f_mid, f_high = fwhm_df.to_numpy().T[4:]
            fwhm = f_high - f_low
            Qs[i] = f_mid / fwhm

        args = [
            [radius, Q, ccat_mp.check_max_workers(max_workers), ex]
            for radius, Q in zip(radii, Qs)
        ]
        self.targ.transform(
            [Detector._calc_linear_frac_f] * num_prefix,
            *args,
            include=include,
            exclude=exclude,
            recalc=recalc,
            col_enum=name_enums,
            prefix_enum=prefix_enums,
        )

        col_names = [
            f"{prefix_enum.LINEAR.value}_{name_enum.FRACTIONAL_FREQUENCY.value}"
            for prefix_enum, name_enum in zip(prefix_enums, name_enums)
        ]

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
    def _calc_phase_spline(
        schema,
        *args,
        tones: list[int],
        padding: int = 4,
        recalc: bool = False,
        col_enum=None,
        prefix_enum=None,
    ):
        """Polars expression factory: build phase-to-frequency splines for a batch of |tones|."""

        def _phase_spline(df):
            """Execute spline fitting over a batch of |tones| and return packed result struct."""
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
                        results_dict[ccat_df.add_tone(return_col[0], tone, padding)] = (
                            spline_data[0]
                        )
                        results_dict[ccat_df.add_tone(return_col[1], tone, padding)] = (
                            spline_data[1]
                        )
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
        """Polars expression factory: convert phase to frequency via pre-built splines."""

        def _phase_to_f(df):
            """Interpolate phase-to-frequency for a batch of |tones| and return packed results."""
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
        """Return Polars expressions for fractional frequency shift ``(f - f0) / f0``."""
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
        """Polars expression factory: build linear fractional-frequency expressions."""

        def _linear_frac_f(df):
            """Compute linear fractional frequency shift for a batch of |tones|."""
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
                        Qr[inds],
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

                        results_dict[ccat_df.add_tone(return_col[0], tone, padding)] = (
                            frac_f
                        )
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
        I_col, Q_col, frac_f_col, linear_prefix = (
            col_enum.IN_PHASE.value,
            col_enum.QUADRATURE.value,
            col_enum.FRACTIONAL_FREQUENCY.value,
            prefix_enum.LINEAR.value,
        )

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
        """Polars expression factory: build dissipation-correction projection expressions."""

        def _dissipation_correction(df):
            """Project |I| & |Q| onto the dissipation axis for a batch of |tones|."""
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
                        R[inds],
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
        I_col, Q_col, mag_col, corr_prefix = (
            col_enum.IN_PHASE.value,
            col_enum.QUADRATURE.value,
            col_enum.MAGNITUDE.value,
            prefix_enum.DISSIPATION_CORRECTION.value,
        )

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
        col_name: str,
        bins: int | None = None,
        mad_filter: bool = True,
        num_mads: int = 10,
        filter_exprs: list = [],
        recalc: bool = False,
        **kwargs,
    ) -> pl.DataFrame:
        """Calculate histogram of a detector property.

        Args:
            col_name (str): Name of the ``properties`` column to histogram.
            bins (int | None, optional): Number of bins. Defaults to number of
                non-NaN |tones| divided by eight.
            mad_filter (bool, optional): Filter outliers using the median
                absolute deviation before binning. Defaults to *True*.
            num_mads (int, optional): Number of MADs a value must fall within
                to be kept. Defaults to 10.
            filter_exprs (list, optional): Additional *Polars* filter
                expressions applied before binning. Defaults to ``[]``.
            recalc (bool, optional): Recalculate even if result already exists.
                Defaults to *False*.
        Returns:
            pl.DataFrame: *Polars* DataFrame with histogram counts and bin
            edges.
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
        """Build a *Holoviews* histogram (optionally with median spike) from a binned DataFrame."""
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
        col_name: str,
        plot_median: bool | None = None,
        xlabel: str = "",
        title: str = "",
        filter_exprs: list = [],
        save_fig: bool | None = None,
        figs_per_file: int | None = None,
        overwrite: bool | None = None,
        save_dir: str | Path | None = None,
        save_name: str = None,
        save_fmt: Format | None = None,
        return_fig: bool = True,
        return_df: bool = False,
        df: pl.DataFrame | None = None,
        by: str | list[str] | None = None,
        **kwargs,
    ) -> hv.Histogram:
        """Plot a histogram of a detector property.

        Args:
            col_name (str): Name of the ``properties`` column to histogram.
            plot_median (bool | None, optional): Plot a vertical line at the
                median. Defaults to the value in ``viz_cfg``.
            xlabel (str, optional): X-axis label. Defaults to ``col_name``.
            title (str, optional): Plot title. Defaults to the detector-type
                label from the drone config.
            filter_exprs (list, optional): Additional *Polars* filter
                expressions applied before plotting. Defaults to ``[]``.
            save_fig (bool | None, optional): Save the figure to disk. Defaults
                to the value in ``viz_cfg``.
            figs_per_file (int | None, optional): Figures per output file.
                Defaults to the value in ``viz_cfg``.
            overwrite (bool | None, optional): Overwrite existing figure files.
                Defaults to the value in ``viz_cfg``.
            save_dir (str | Path | None, optional): Directory for saved figures.
                Defaults to ``fig_dir``.
            save_name (str, optional): Base filename for saved figures. Defaults
                to *None*.
            save_fmt (Format | None, optional): Output file format
                (``'png'``, ``'jpeg'``, or ``'pdf'``). Defaults to the value in
                ``viz_cfg``.
            return_fig (bool, optional): Return the *Holoviews* figure object.
                Defaults to *True*.
            return_df (bool, optional): Also return the histogram DataFrame.
                Defaults to *False*.
            df (pl.DataFrame | None, optional): Pre-computed histogram DataFrame
                to plot directly (skips ``properties_histogram``). Defaults to
                *None*.
            by (str | list[str] | None, optional): Column(s) to group the
                histogram by. Defaults to *None*.
        Returns:
            hv.Histogram: *Holoviews* histogram figure (or ``hv.Overlay`` when
            plotting the median line).
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

    def _get_data_obj(self, data: Literal["targ", "timestream", "both"]):
        """
        Get data object (Target, Timestream, or both) corresponding to string

        data (

        """

        data_objs = []
        if data == "targ" or data == "both":
            data_objs.append(self.targ)
        if data == "timestream" or data == "both":
            data_objs.append(self.stream)
        if not data_objs:
            error = (
                f"Invalid data type {data}, must be 'targ', 'timestream', or 'both'."
            )
            log.log("ERROR", error)
            raise ValueError(error)
        return data_objs

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

    def join(self, other: "Detector", in_place: bool = False) -> "Detector":
        """Join two ``Detector`` objects into one, concatenating their data.

        Merges ``Target``, ``Timestream`` (if both present), ``VNA``, and
        ``properties`` DataFrames.  Both objects must either have a
        ``Timestream`` or neither must.

        Args:
            other (Detector): The ``Detector`` to join with this one.
            in_place (bool, optional): Modify this object in place instead of
                returning a deep copy. Defaults to *False*.
        Returns:
            Detector: Combined ``Detector`` with concatenated data.
        Raises:
            ValueError: If ``other`` is not a ``Detector``.
            ValueError: If one ``Detector`` has a ``Timestream`` and the other
                does not.
        """

        def _join_consts(
            left_const: Any | list[Any], right_const: Any | list[Any]
        ) -> list[Any]:
            """Merge two scalar-or-list constants into a single combined list."""
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
        """Return string identifier for the ``Detector``."""
        return f"detector_{self.timestamp}"

    # =============================== #
    # Define Custom Pickling Behavior #
    # =============================== #

    def __getstate__(self):
        """Serialise state, optionally writing ``properties`` to parquet before pickling."""
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
        """Restore state, lazily scanning ``properties`` parquet when not pickled inline."""
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
