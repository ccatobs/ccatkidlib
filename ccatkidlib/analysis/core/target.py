import gc
import sys
import numpy as np
import polars as pl
import concurrent.futures
import time
import lmfit
from numba import njit

from functools import cached_property
from collections.abc import Iterable
from pathlib import Path
from enum import Enum

# Local Imports
import ccatkidlib.log as log
import ccatkidlib.utils as utils
import ccatkidlib.analysis.utils.pair as pair
import ccatkidlib.analysis.utils.dataframe as ccat_df
import ccatkidlib.analysis.fit.fit as ccat_fit
import ccatkidlib.analysis.utils.multiprocess as ccat_mp

from ccatkidlib.analysis.core.sweep import Sweep
from ccatkidlib.analysis.fit.fit import linear_fit


class Target(Sweep):
    """Class representing a target sweep taken with a Radio Frequency System on a Chip (RFSoC)

    Subclasses Sweep.
    """

    def __init__(
        self,
        com_to: str,
        tones: int | list[int] | None = None,
        noise_tones: int | list[int] | None = None,
        **kwargs,
    ):
        """Subclass of Sweep with additional arguments

        Args:
            tones (int | list[int] | None, optional): Which tones to load. None for loading all data without splitting into individual tones. -1 for all data split into individual tones. Defaults to None
        """
        kwargs["data_type"] = "targ"
        super().__init__(com_to, **kwargs)

        # Parse 'tone' argument specifying which tones should be loaded
        # -------------------------------------------------------------
        if isinstance(tones, int):
            if tones >= 0:
                tones = [tones]
            else:
                tones = list(range(self.num_tones))

        if not (
            tones is None
            or (
                isinstance(tones, Iterable)
                and all([isinstance(tone, int) for tone in tones])
            )
        ):
            error = f"Invalid type {type(tones)} for argument 'tones'. Should be int, list[int], or None."
            log.log("CRITICAL", error)
            raise ValueError(error)
        self.tones = tones

        # Define list of noise tones
        # --------------------------
        if noise_tones is not None:
            if isinstance(noise_tones, int):
                noise_tones = [noise_tones]
            elif not isinstance(noise_tones, Iterable) or not all(
                [isinstance(noise_tone, int) for noise_tone in noise_tones]
            ):
                noise_tones = None
                log.log(
                    "CRITICAL",
                    f"Invalid type {type(noise_tones)} for argument 'noise_tones'. Should be int, list[int], or None.",
                )
        else:
            noise_tones = utils.dict_get(self.drone_cfg, ["tones", "noise_tones"])
        self.noise_tones = noise_tones

        self._properties = {}
        self._properties_df = None
        self.fit_result = {}

    # ======================== #
    # Lazily Loaded Attributes #
    # ======================== #

    @property
    def data(self):
        if self._data is None:
            data = super().data  # Get DataFrame of sweep data
            tones = self.tones  # Define local variable since it will be used often

            if tones is not None:  # Run if a specific tone(s) is specified
                data = data.to_numpy().T
                try:
                    sweep_steps = self.drone_cfg["tones"]["sweep_steps"]
                except KeyError:
                    sweep_steps = self.drone_cfg["tones"]["N_step"]

                data = data[1:].reshape((3, -1, sweep_steps))
                num_tones = len(tones)

                try:
                    fs, Is, Qs = (
                        [None] * num_tones,
                        [None] * num_tones,
                        [None] * num_tones,
                    )
                    for i, t in enumerate(tones):
                        f, I, Q = data[:, t, :]
                        fs[i] = f
                        Is[i] = I
                        Qs[i] = Q
                    fs, Is, Qs = np.array(fs), np.array(Is), np.array(Qs)
                except Exception as e:
                    fs, Is, Qs = (
                        [None] * num_tones,
                        [None] * num_tones,
                        [None] * num_tones,
                    )
                    log.log("ERROR", "Failed to reshape data array with error %s.", e)

                name_mapping = self.analysis_cfg["convention"]["name"]
                data_dict = {name_mapping["sample"]: range(sweep_steps)}
                for t, f, I, Q in zip(tones, fs, Is, Qs):
                    data_dict[
                        ccat_df.add_tone(name_mapping["frequency"], t, self.padding)
                    ] = f
                    data_dict[
                        ccat_df.add_tone(name_mapping["in_phase"], t, self.padding)
                    ] = I
                    data_dict[
                        ccat_df.add_tone(name_mapping["quadrature"], t, self.padding)
                    ] = Q
                df = pl.DataFrame(data_dict)
            else:
                df = data
            self._data = df
        elif isinstance(self._data, pl.LazyFrame):
            self._data = self._data.collect()
        return self._data

    @data.setter
    def data(self, value: pl.lazyframe.frame.LazyFrame | None):
        if value is None or isinstance(value, (pl.DataFrame, pl.LazyFrame)):
            self._data = value
        else:
            log.log(
                "ERROR", "Cannot set data with type %s. Must be a Polars DataFrame!"
            )

    @property
    def properties(self):
        if isinstance(self._properties_df, pl.LazyFrame):
            self._properties_df = self._properties_df.collect()
        elif self._properties_df is None:
            self._properties_df = self.comb.with_columns(
                pl.lit(f"{self.bid}.{self.drid}").alias("com_to"),
                pl.lit(self.timestamp).alias("timestamp"),
            )

        if self.tones is None:
            return self._properties_df

        # Reshape properties dictionary to have resonator properties as primary keys
        new_dict = {"det": []}

        props_dict = self._properties
        self._properties = {
            ccat_df.add_tone("det", tone, self.padding): {} for tone in self.tones
        }

        all_props = set(
            [prop for props in props_dict.values() for prop in props.keys()]
        )
        if len(all_props) == 0:
            return self._properties_df

        for det, props in props_dict.items():
            new_dict["det"].append(int(det.split("_")[-1]))
            for prop in all_props:
                curr = new_dict.get(prop, [])
                value = props.get(prop, None)
                if curr:
                    curr.append(value)
                else:
                    new_dict[prop] = [value]

        new_df = pl.DataFrame(new_dict)
        shared_cols = set(self._properties_df.columns) & set(new_df.columns) - {"det"}
        self._properties_df = ccat_df.coalesce_join(
            self._properties_df, new_df, "det", shared_cols
        )
        return self._properties_df

    @properties.setter
    def properties(self, value):
        if isinstance(value, (pl.DataFrame, pl.LazyFrame)):
            self._properties_df = value

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

    # ====================== #
    # Transformation Methods #
    # ====================== #

    def stitch_phase(self, 
                     prefix: str = '',
                     threshold: float = 2*np.pi, 
                     include: int | list[int] | None = None, 
                     exclude: int | list[int] | None = None, 
                     recalc: bool =False,
                     max_workers: int = 1, 
                     ex = None):
        '''
        '''

        name_enums, prefix_enums = ccat_df.create_enums(
            ["phase"],
            prefix,
            ["stitch"],
            self.analysis_cfg,
        )

        num_prefix = len(name_enums)
        threshold = ccat_df.check_args(threshold, num_prefix, float)

        args = [
            [
                thresh,
                ccat_mp.check_max_workers(max_workers),
                ex,
            ]
            for thresh in threshold
        ]
        self.transform(
            [Target._calc_stitch_phase] * num_prefix,
            *args,
            include=include,
            exclude=exclude,
            recalc=recalc,
            col_enum=name_enums,
            prefix_enum=prefix_enums,
        )

        col_name = ["_".join([prefix_enum.STITCH.value, name_enum.PHASE.value]) for name_enum, prefix_enum in zip(name_enums, prefix_enums)]

        self.data = ccat_df.unnest(
            self, [f"struct_{name}" for name in col_name]
        )
        return self.get_data(
            col_name=col_name,
            include=include,
            exclude=exclude,
            strict=True
        )

    def res_trim(self,
                    col_name: str,
                    prefix: str | list[str] = "",
                    window: float | list[float] = 1.5,
                    mean_points: int = 10,
                    mag_prefix: str = "",
                    invert: bool = False,
                    include: int | list[int] | None = None,
                    exclude: int | list[int] | None = None,
                    recalc: bool = False,) -> pl.DataFrame:
        """
        Trim target sweep IQ data based on resonator linewidths. 

        Args:
            prefix: Prefix of the IQ data to trim.
            window (float | list[float], optional): How many linewidths of data to include around the magnitude minimum. Defaults to 1.5
            use_fit (bool): Whether to use resonant frequency and total quality factor from a fit to determine linewidth. Defaults to False
            f_0_col (str): Name of column in ``properties`` DataFrame with resonant frequencies to use if ``use_fit`` is True.
            Q_col (str): Name of column in ``properties`` DataFrame with total quality factors to use if ``use_fit`` is True.
            mean_points (int): Number of points to average to determine max magnitude of resonator profile. Used if ``use_fit`` is False. Defaults to 10
        Returns:
            return (pl.DataFrame): Polars DataFrame with trimmed I & Q data
        """
        name_enums, prefix_enums = ccat_df.create_enums(
            [],
            prefix,
            ["trim_tail", "trim_resonance"],
            self.analysis_cfg,
        )

        num_prefix = len(name_enums)
        window = ccat_df.check_args(window, num_prefix, int)
        mean_points = ccat_df.check_args(mean_points, num_prefix, int)
        mag_prefix = ccat_df.check_args(mag_prefix, num_prefix, str)

        lower_bounds, upper_bounds = [[]] * num_prefix, [[]] * num_prefix
        for i, (mag_pre, mean_point, win) in enumerate(
            zip(mag_prefix, mean_points, window)
        ):
            HM_low, HM_mid, HM_high = (
                self.fwhm(
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

        return self.trim(col_name=col_name,
                         prefix=prefix,
                         lower_index=lower_bounds,
                         upper_index=upper_bounds,
                         name=[prefix_enum.TRIM_RESONANCE.value if invert else prefix_enum.TRIM_TAIL.value for prefix_enum in prefix_enums],
                         invert=invert,
                         include=include,
                         exclude=exclude,
                         recalc=recalc,
                        )

    def res_IQ_trim(self,
                prefix: str | list[str] = "",
                window: float | list[float] = 1.5,
                mean_points: int = 10,
                mag_prefix: str = "",
                invert: bool = False,
                include: int | list[int] | None = None,
                exclude: int | list[int] | None = None,
                recalc: bool = False,) -> pl.DataFrame:
        """
        Trim target sweep IQ data based on resonator linewidths. 

        Args:
            prefix: Prefix of the IQ data to trim.
            window (float | list[float], optional): How many linewidths of data to include around the magnitude minimum. Defaults to 1.5
            use_fit (bool): Whether to use resonant frequency and total quality factor from a fit to determine linewidth. Defaults to False
            f_0_col (str): Name of column in ``properties`` DataFrame with resonant frequencies to use if ``use_fit`` is True.
            Q_col (str): Name of column in ``properties`` DataFrame with total quality factors to use if ``use_fit`` is True.
            mean_points (int): Number of points to average to determine max magnitude of resonator profile. Used if ``use_fit`` is False. Defaults to 10
        Returns:
            return (pl.DataFrame): Polars DataFrame with trimmed I & Q data
        """

        name_dict = self.analysis_cfg['convention']['name']
        I_df = self.res_trim(name_dict['in_phase'],
                   prefix=prefix,
                   window=window,
                   mean_points=mean_points,
                   mag_prefix=mag_prefix,
                   invert=invert,
                   include=include,
                   exclude=exclude,
                   recalc=recalc)

        Q_df = self.res_trim(name_dict['quadrature'],
                   prefix=prefix,
                   window=window,
                   mean_points=mean_points,
                   mag_prefix=mag_prefix,
                   invert=invert,
                   include=include,
                   exclude=exclude,
                   recalc=recalc)

        IQ_df = pl.concat([I_df, Q_df], how='horizontal')
        return IQ_df

    def linewidth(self,
                        prefix: str | list[str] = "",
                        include: int | list[int] | None = None,
                        exclude: int | list[int] | None = None,
                        recalc: bool = False,):
        name_enums, prefix_enums = ccat_df.create_enums(
            ["in_phase", "quadrature", "linewidth"], prefix, [], self.analysis_cfg
        )

        self.transform(
            [Target._calc_linewidth] * len(name_enums),
            include=include,
            exclude=exclude,
            recalc=recalc,
            col_enum=name_enums,
            prefix_enum=prefix_enums,
        )
        return self.get_data(
            col_name=[name_enum.LINEWIDTH.value for name_enum in name_enums],
            include=include,
            exclude=exclude,
            strict=True,
        )
    
    def coupling_linewidth(self,
                            prefix: str | list[str] = "",
                            include: int | list[int] | None = None,
                            exclude: int | list[int] | None = None,
                            recalc: bool = False,):
        name_enums, prefix_enums = ccat_df.create_enums(
            ["linewidth", "IQ_circle_diameter", "coupling_linewidth"], prefix, [], self.analysis_cfg
        )

        self.transform(
            [Target._calc_coupling_linewidth] * len(name_enums),
            include=include,
            exclude=exclude,
            recalc=recalc,
            col_enum=name_enums,
            prefix_enum=prefix_enums,
        )
        return self.get_data(
            col_name=[name_enum.COUPLING_LINEWIDTH.value for name_enum in name_enums],
            include=include,
            exclude=exclude,
            strict=True,
        )
    
    def linewidth_shift(self,
                        prefix: str | list[str] = "",
                        mag_prefix = "",
                        coupling_linewidth = True,
                        m: float | list[float] | None = None,
                        b: float | list[float] | None = None,
                        res_trim_window = 2,
                        tail_trim_window = 3,
                        mean_points=10,
                        include=None,
                        exclude=None,
                        recalc=False, 
                        return_fit_params = False,
                        max_workers=1, 
                        ex=None):
        if isinstance(prefix, str): prefix = [prefix]
        name_enums, prefix_enums = ccat_df.create_enums(
            ["frequency", "linewidth", "coupling_linewidth", "linear_fit_slope", "linear_fit_intercept"], 
            prefix, 
            ["translate", "trim_tail", "trim_resonance", "trim", "linear", "linear_fit"], 
            self.analysis_cfg,
            no_prefix = ["frequency", "linewidth", "coupling_linewidth", "linear_fit_slope", "linear_fit_intercept"]
        )

        num_prefix = len(name_enums)
        mag_prefix = ccat_df.check_args(mag_prefix, num_prefix, str)
        coupling_linewidth = ccat_df.check_args(coupling_linewidth, num_prefix, bool)
        m = ccat_df.check_args(m, num_prefix, float)
        b = ccat_df.check_args(b, num_prefix, float)
        res_trim_window = ccat_df.check_args(res_trim_window, num_prefix, float)
        tail_trim_window = ccat_df.check_args(tail_trim_window, num_prefix, float)
        mean_points = ccat_df.check_args(mean_points, num_prefix, int)

        f_col = name_enums[0].FREQUENCY.value
        ms, bs = [[]]*num_prefix, [[]]*num_prefix
        for i, (pre, mag_pre, mm, bb, clw, res_win, tail_win, mean_point) in enumerate(zip(prefix, mag_prefix, m, b, coupling_linewidth, res_trim_window, tail_trim_window, mean_points)):
            lw_col = name_enums[0].COUPLING_LINEWIDTH.value if clw else name_enums[0].LINEWIDTH.value 
            if mm is None or bb is None:
                self.res_trim(col_name=lw_col, prefix=pre, mag_prefix=mag_pre, 
                              window=tail_win, mean_points=mean_point, 
                              invert = False,
                              include=include, exclude = exclude, recalc=recalc)
                tail_prefix = f"{prefix_enums[0].TRIM_TAIL.value}_{prefix_enums[0].TRIM.value}{'_' if pre else ''}{pre}"
                self.res_trim(col_name=lw_col, prefix=tail_prefix, mag_prefix=mag_pre, 
                              window=res_win, mean_points=mean_point, 
                              invert=True, 
                              include=include, exclude=exclude, recalc=recalc)
                res_prefix = "_".join([prefix_enums[0].TRIM_RESONANCE.value, prefix_enums[0].TRIM.value, tail_prefix])
                self.linear_fit(x_col_name=f_col, 
                                y_col_name=lw_col,
                                y_prefix=res_prefix,
                                include=include, exclude=exclude, recalc=recalc,
                                max_workers=max_workers, ex=ex)
                fit_prefix = "_".join([prefix_enums[0].LINEAR_FIT.value, res_prefix, lw_col, f_col])
                mm, bb = self.get_properties(col_name = ["_".join([fit_prefix, name_enums[0].LINEAR_FIT_SLOPE.value]),
                                                         "_".join([fit_prefix, name_enums[0].LINEAR_FIT_INTERCEPT.value])],
                                             include=include, exclude=exclude, strict=True).to_numpy().T[1:]
            ms[i], bs[i] = mm, bb
            y_0 = [-1*(mmm*f + bbb) for f, mmm, bbb in zip(self.get_data(f_col, include=include, exclude=exclude, strict=True).to_numpy().T, mm, bb)]
            self.shift(col_name=lw_col, prefix=pre, 
                       shift=y_0, name=prefix_enums[0].LINEAR.value, 
                       include=include, exclude=exclude, recalc=recalc)
        
        shift_df = self.get_data(
            col_name=[
                f"{prefix_enums[0].LINEAR.value}_{prefix_enums[0].TRANSLATE.value}_{pre}{'_' if pre else ''}{name_enums[0].COUPLING_LINEWIDTH.value if clw else name_enums[0].LINEWIDTH.value}"
                for pre, clw in zip(prefix, coupling_linewidth)
            ],
            include=include,
            exclude=exclude,
        )

        if return_fit_params:
            return shift_df, ms, bs
        else:
            return shift_df

    # ======= #
    # Fitting #
    # ======= #
  
    def linewidth_fit(
        self,
        lw_prefix: str = "",
        lw_shift_prefix: str = "",
        method: str = "least_squares",
        coupling_linewidth = True,
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
        if isinstance(lw_shift_prefix, str): lw_shift_prefix = [lw_shift_prefix]
        name_enums, prefix_enums = ccat_df.create_enums(
            [
                "linewidth",
                "coupling_linewidth",
                "fit_result",
                "ki_nonlinearity_parameter",
                "qp_nonlinearity_parameter",
                "linewidth_fit_offset",
                "linewidth_fit_y_ki",
                "linewidth_fit_y_qp",
                "linewidth_fit_exponent"
            ],
            lw_shift_prefix,
            ["linear", "translate", "linewidth_fit", "coupling_linewidth_fit"],
            self.analysis_cfg,
            no_prefix=["linewidth", "coupling_linewidth"],
        )

        num_prefix = len(name_enums)
        lw_prefix = ccat_df.check_args(lw_prefix, num_prefix, str)
        coupling_linewidth = ccat_df.check_args(coupling_linewidth, num_prefix, bool)
        params = ccat_df.check_args(params, num_prefix, lmfit.parameter.Parameters)
        method = ccat_df.check_args(method, num_prefix, str)

        args = [
            [
                self,
                lw_pre, 
                lw_shift_pre,
                clw, 
                meth,
                param,
                ccat_mp.check_max_workers(max_workers),
                ex,
            ]
            for lw_pre, lw_shift_pre, clw, param, meth in zip(lw_prefix, lw_shift_prefix, coupling_linewidth, params, method)
        ]
        self.transform(
            [Target._calc_linewidth_fit] * num_prefix,
            *args,
            include=include,
            exclude=exclude,
            recalc=recalc,
            col_enum=name_enums,
            prefix_enum=prefix_enums,
        )
        
        col_names = []
        for clw, prefix_enum, name_enum, lw_shift_pre in zip(coupling_linewidth, prefix_enums, name_enums, lw_shift_prefix):
            lw_fit_pre = prefix_enum.COUPLING_LINEWIDTH_FIT.value if clw else prefix_enum.LINEWIDTH_FIT.value
            col_names += [
                "_".join([lw_fit_pre, lw_shift_pre, name_enum.COUPLING_LINEWIDTH.value if clw else name_enum.LINEWIDTH.value])
            ]

            a_col, b_col, y_a_col, y_b_col, y_off_col, y_exp_col = ("_".join([lw_fit_pre, name_enum.KI_NONLINEARITY_PARAMETER.value]), 
                                                                    "_".join([lw_fit_pre, name_enum.QP_NONLINEARITY_PARAMETER.value]),
                                                                    "_".join([lw_fit_pre, name_enum.LINEWIDTH_FIT_Y_KI.value]),
                                                                    "_".join([lw_fit_pre, name_enum.LINEWIDTH_FIT_Y_QP.value]),
                                                                    "_".join([lw_fit_pre, name_enum.LINEWIDTH_FIT_OFFSET.value]),
                                                                    "_".join([lw_fit_pre, name_enum.LINEWIDTH_FIT_EXPONENT.value]))

            self._properties_df = (
                self.properties
                .with_columns((pl.col(y_off_col) + pl.col(a_col)/(1 + pl.col(y_a_col)**2) - pl.col(b_col)/(1 + (-1*pl.col(y_b_col))**pl.col(y_exp_col))).alias('a_eff'))
            )

        self.data = ccat_df.unnest(
            self, [f"struct_{name}" for name in col_names]
        )
        return self.get_data(
            col_name=col_names,
            include=include,
            exclude=exclude,
        )
 
    # ========== #
    # Properties #
    # ========== #

    def cable_delay(self, prefix="", window = 5, mean_points = 10, mag_prefix = "", include=None, exclude=None, recalc=False, stitch_workers=1, linear_fit_workers=1, ex=None):
        '''
        '''

        name_enums, prefix_enums = ccat_df.create_enums(
            ["frequency", "phase", "linear_fit_slope", "linear_fit_intercept", "cable_delay"],
            prefix,
            ["stitch", "trim", "trim_resonance", "linear_fit"],
            self.analysis_cfg,
            no_prefix=["frequency", "phase", "linear_fit_slope", "linear_fit_intercept"],
        )
        
        name_enum, prefix_enum = name_enums[0], prefix_enums[0]
        cable_col = "_".join(['targ', name_enum.CABLE_DELAY.value])

        include_subset = ccat_df.check_properties(
            self,
            cable_col,
            include=include,
            exclude=exclude,
            recalc=recalc,
        )
        if not len(include_subset) == 0:
            self.phase(prefix=prefix, include=include, exclude=exclude, recalc=recalc)
            with ccat_mp.optional_executor(max_workers=max(linear_fit_workers, stitch_workers), ex=ex) as ex:
                # Stitch discontinuities in phase
                self.stitch_phase(prefix=prefix, 
                                include=include, 
                                exclude=exclude, 
                                recalc=recalc, 
                                max_workers = stitch_workers, 
                                ex = ex)
                stitch_prefix = f"{prefix_enum.STITCH.value}{'_' if prefix else ''}{prefix}"

                # Trim on-resonance data
                self.res_trim(col_name=name_enum.PHASE.value,
                        prefix=stitch_prefix,
                        window=window,
                        mag_prefix=mag_prefix,
                        invert=True,
                        include=include,
                        exclude=exclude,
                        recalc=recalc,
                        )
                trim_prefix = "_".join([prefix_enum.TRIM_RESONANCE.value, prefix_enum.TRIM.value, stitch_prefix])

                # Fit line to off-resonance data
                self.linear_fit(x_col_name=name_enum.FREQUENCY.value,
                                y_col_name=name_enum.PHASE.value,
                                x_prefix="",
                                y_prefix=trim_prefix,
                                include=include,
                                exclude=exclude,
                                recalc=recalc,
                                max_workers=linear_fit_workers,
                                ex=ex)


            slope_col = "_".join([prefix_enum.LINEAR_FIT.value, trim_prefix, name_enum.PHASE.value, name_enum.FREQUENCY.value, name_enum.LINEAR_FIT_SLOPE.value])
            slope_df = self.get_properties(slope_col, include=include, exclude=exclude, strict=True)            
            shared_cols = [cable_col] if cable_col in self._properties_df.schema else []
            self._properties_df = ccat_df.coalesce_join(
                self._properties_df,
                slope_df.select(["det", (pl.col(slope_col) * pl.lit(1e9/(2*np.pi))).alias(cable_col)]),
                "det",
                shared_cols,
            )
        
        return self.get_properties(
            cable_col,
            include=include,
            exclude=exclude,
            strict=True,
        )

    def cable_gain(self, prefix= "", window = 5, mean_points = 10, mag_prefix="", err_fill = 1, include=None, exclude=None, recalc=False, linear_fit_workers=1, ex=None):
        name_dict, prefix_dict = (
            self.analysis_cfg["convention"]["name"],
            self.analysis_cfg["convention"]["prefix"],
        )
        
        self.mag(prefix=prefix, include=include, exclude=exclude, recalc=recalc)
        self.res_trim(col_name=name_dict['magnitude'],
                      prefix=prefix,
                      mag_prefix=mag_prefix,
                      window=window,
                      mean_points=mean_points,
                      invert=True,
                      include=include,
                      exclude=exclude,
                      recalc=recalc,
                      )
        trim_prefix = "_".join([prefix_dict['trim_resonance'], prefix_dict['trim']])
        trim_prefix = f"{trim_prefix}{'_' if prefix else ''}{prefix}"

        # Fit line to off-resonance data
        return self.linear_fit(x_col_name=name_dict['frequency'],
                               y_col_name=name_dict['magnitude'],
                               x_prefix="",
                               err_fill = err_fill,
                               y_prefix=trim_prefix,
                               include=include,
                               exclude=exclude,
                               recalc=recalc,
                               max_workers=linear_fit_workers,
                               ex=ex)

    def mismatch_angle(self, prefix="", mean_points=10, include=None, exclude=None, recalc=False):
        name_enums, prefix_enums = ccat_df.create_enums(
            ["in_phase", "quadrature", "phase"],
            prefix,
            ["remove_impedance_mismatch"],
            self.analysis_cfg,
            no_prefix=[],
        )

        name_enum, prefix_enum = name_enums[0], prefix_enums[0]
        mismatch_col_name = (
            f"{prefix_enum.REMOVE_IMPEDANCE_MISMATCH.value}_{name_enum.PHASE.value}"
        )
        include_subset = ccat_df.check_properties(
            self,
            mismatch_col_name,
            include=include,
            exclude=exclude,
            recalc=recalc,
        )
        if not len(include_subset) == 0:
            pi = pl.lit(np.pi)
            I_df = self.get_data(
                name_enum.IN_PHASE.value,
                include=include_subset,
                strict=True,
            )
            Q_df = self.get_data(
                name_enum.QUADRATURE.value,
                include=include_subset,
                strict=True,
            )
            I_cols, Q_cols = I_df.columns, Q_df.columns
            IQ_df = pl.concat([I_df, Q_df], how="horizontal")

            mismatch_df = (
                IQ_df.lazy()
                .select(
                    pl.all().head(mean_points).mean().name.prefix("first_"),
                    pl.all().tail(mean_points).mean().name.prefix("last_"),
                )
                .select(
                    [
                        (
                            pi
                            - (
                                pl.arctan2(
                                    pl.col(f"first_{Q_col}"),
                                    pl.col(f"first_{I_col}"),
                                )
                                % (2 * pi)
                                + pl.arctan2(
                                    pl.col(f"last_{Q_col}"),
                                    pl.col(f"last_{I_col}"),
                                )
                                % (2 * pi)
                            )
                            / 2
                        ).alias(I_col.split("_")[-1])
                        for I_col, Q_col in zip(I_cols, Q_cols)
                    ]
                )
                .collect()
            )
            ccat_df.add_data_to_properties(self, mismatch_df, mismatch_col_name)
        return self.get_properties(
            mismatch_col_name, include=include, exclude=exclude, strict=True
        )

    def mismatch_dist(self, prefix="", mean_points=10, include=None, exclude=None, recalc=False):
        name_enums, prefix_enums = ccat_df.create_enums(
            ["in_phase", "quadrature", "magnitude"],
            prefix,
            ["remove_impedance_mismatch"],
            self.analysis_cfg,
            no_prefix=[],
        )

        name_enum, prefix_enum = name_enums[0], prefix_enums[0]
        mismatch_col_name = (
            f"{prefix_enum.REMOVE_IMPEDANCE_MISMATCH.value}_{name_enum.MAGNITUDE.value}"
        )
        include_subset = ccat_df.check_properties(
            self,
            mismatch_col_name,
            include=include,
            exclude=exclude,
            recalc=recalc,
        )
        if not len(include_subset) == 0:
            I_df = self.get_data(
                name_enum.IN_PHASE.value,
                include=include_subset,
                strict=True,
            )
            Q_df = self.get_data(
                name_enum.QUADRATURE.value,
                include=include_subset,
                strict=True,
            )
            I_cols, Q_cols = I_df.columns, Q_df.columns
            IQ_df = pl.concat([I_df, Q_df], how="horizontal")

            mismatch_df = (
                IQ_df.lazy()
                .select(
                    pl.all().head(mean_points).mean().name.prefix("first_"),
                    pl.all().tail(mean_points).mean().name.prefix("last_"),
                )
                .select(
                    [
                        (
                            (
                                (pl.col(f"first_{Q_col}")**2
                                + pl.col(f"first_{I_col}")**2).sqrt()
                                
                                + (
                                    pl.col(f"last_{Q_col}")**2
                                    +pl.col(f"last_{I_col}")**2
                                ).sqrt()
                            )
                            / 2
                        ).alias(I_col.split("_")[-1])
                        for I_col, Q_col in zip(I_cols, Q_cols)
                    ]
                )
                .collect()
            )
            ccat_df.add_data_to_properties(self, mismatch_df, mismatch_col_name)
        return self.get_properties(
            mismatch_col_name, include=include, exclude=exclude, strict=True
        )
 
    def fwhm(self, mag_prefix="", mean_points=10, peak=False, include=None, exclude=None, recalc=False):
        '''
        '''

        "TODO: Use min/max values of phase instead potentially. Should be accurate as long as no large impedance mismatch"
        name_enums, prefix_enums = ccat_df.create_enums(
            [
                "sample",
                "frequency",
                "magnitude",
                "full_width_half_max",
            ],
            mag_prefix,
            ["low_frequency_side", "middle_frequency_point", "high_frequency_side"],
            self.analysis_cfg,
            no_prefix=["sample", 'frequency'],
        )

        name_enum, prefix_enum = name_enums[0], prefix_enums[0]
        low_sample, mid_sample, high_sample, low_f, mid_f, high_f = (
            f"{prefix_enum.LOW_FREQUENCY_SIDE.value}_{name_enum.FULL_WIDTH_HALF_MAX.value}_{name_enum.SAMPLE.value}",
            f"{prefix_enum.MIDDLE_FREQUENCY_POINT.value}_{name_enum.FULL_WIDTH_HALF_MAX.value}_{name_enum.SAMPLE.value}",
            f"{prefix_enum.HIGH_FREQUENCY_SIDE.value}_{name_enum.FULL_WIDTH_HALF_MAX.value}_{name_enum.SAMPLE.value}",
            f"{prefix_enum.LOW_FREQUENCY_SIDE.value}_{name_enum.FULL_WIDTH_HALF_MAX.value}_{name_enum.FREQUENCY.value}",
            f"{prefix_enum.MIDDLE_FREQUENCY_POINT.value}_{name_enum.FULL_WIDTH_HALF_MAX.value}_{name_enum.FREQUENCY.value}",
            f"{prefix_enum.HIGH_FREQUENCY_SIDE.value}_{name_enum.FULL_WIDTH_HALF_MAX.value}_{name_enum.FREQUENCY.value}",
        )

        include_subset = ccat_df.check_properties(
            self, mid_sample, include=include, exclude=exclude, recalc=recalc
        )
        if not len(include_subset) == 0:
            sample_col, mag_col, f_col = name_enum.SAMPLE.value, name_enum.MAGNITUDE.value, name_enum.FREQUENCY.value
            # Get detector magnitudes and sample numbers and unpivot DataFrame from wide to long format
            mag_df = (self.get_data(col_name=[sample_col, mag_col], strict=True, include=include_subset)
                        .unpivot(
                            index=sample_col,
                            variable_name="det",
                            value_name=mag_col,
                                )
                            .with_columns(pl.col("det").str.strip_prefix(f"{mag_col}_").cast(pl.Int32))
            )

            f_df = (self.get_data(col_name=f_col, strict=True, include=include_subset)
                        .unpivot(variable_name='tmp', value_name=f_col)
                        .drop('tmp'))

            mag_df = pl.concat([mag_df, f_df], how='horizontal').lazy().sort(mag_col, descending=not peak)

            # Get minimum magnitude values for each detector and corresponding sample numbers
            expr = pl.col(mag_col).max() if peak else pl.col(mag_col).min()
            min_df = (
                mag_df.filter((pl.col(mag_col) == expr).over("det"))
                .rename(
                    {
                        sample_col: f"min_{sample_col}",
                        f_col: f"min_{f_col}",
                        mag_col: f"min_{mag_col}",
                    }
                )
                .collect()
            )
            shared_cols = [mid_sample, mid_f] if mid_sample in self._properties_df.schema else []
            self._properties_df = ccat_df.coalesce_join(
                self._properties_df,
                min_df.select(["det", pl.col(f"min_{sample_col}").alias(mid_sample), pl.col(f"min_{f_col}").alias(mid_f)]),
                "det",
                shared_cols,
            )

            mag_min_df = (
                mag_df.collect()
                .join(min_df, on="det", how="left", coalesce=True)
                .lazy()
                .with_columns(
                    (pl.col(sample_col) < pl.col(f"min_{sample_col}")).alias("low")
                )
            )
            # Get mean maximum magnitude values for both the low and high frequency sides of each detector
            max_df = (
                mag_min_df.group_by(["low", "det"], maintain_order=True)
                .agg(pl.col(mag_col).head(mean_points).mean())
                .collect()
                .pivot(on="low", index="det", values=mag_col)
                .lazy()
                .sort("det")
                .rename(
                    {
                        "true": f"max_{mag_col}_low",
                        "false": f"max_{mag_col}_high",
                    }
                )
                .collect()
            )
            min_max_df = (
                mag_min_df.collect()
                .join(max_df, on="det", how="left", coalesce=True)
                .lazy()
                .with_columns(
                    [
                        (
                            pl.col(mag_col)
                            - (
                                (pl.col(f"min_{mag_col}") + pl.col(f"max_{mag_col}_{side}"))
                                / 2
                            )
                        )
                        .abs()
                        .alias(name)
                        for side, name in zip(["low", "high"], [low_sample, high_sample])
                    ]
                )
            )
            # Get the samples corresponding to the half max on the low and high frequency sides of each detector
            for side, sample_name, f_name in zip(["low", "high"], [low_sample, high_sample], [low_f, high_f]):
                HM_df = (
                    min_max_df.filter(pl.col("low") == ("low" == pl.lit(side)))
                    .sort(sample_name)
                    .select("det", pl.col(sample_col).first().over("det"), pl.col(f_col).first().over('det'))
                    .unique()
                    .sort("det")
                    .rename({sample_col: sample_name, f_col: f_name})
                    .collect()
                )
                shared_cols = [sample_name, f_name] if sample_name in self._properties_df.schema else []
                self._properties_df = ccat_df.coalesce_join(
                    self._properties_df, HM_df, "det", shared_cols
                )
        return self.get_properties(
            [low_sample, mid_sample, high_sample, low_f, mid_f, high_f],
            include=include,
            exclude=exclude,
            strict=True,
        )

    def IQ_circle_diameter(self,
                           prefix: str | list[str] = "",
                           include: int | list[int] | None = None,
                           exclude: int | list[int] | None = None,
                           recalc: bool = False,):
        name_enums, prefix_enums = ccat_df.create_enums(
            ["in_phase", "quadrature", "IQ_circle_diameter"], prefix, [], self.analysis_cfg
        )

        self.transform(
            [Target._calc_IQ_circle_diameter] * len(name_enums),
            include=include,
            exclude=exclude,
            recalc=recalc,
            col_enum=name_enums,
            prefix_enum=prefix_enums,
        )
        return self.get_data(
            col_name=[name_enum.IQ_CIRCLE_DIAMETER.value for name_enum in name_enums],
            include=include,
            exclude=exclude,
            strict=True,
        )
    
    # ================ #
    # Analysis Methods # 
    # ================ #

    @staticmethod
    def _calc_stitch_phase(
        schema,
        *args,
        tones: list[int],
        padding: int = 4,
        recalc: bool = False,
        col_enum=None,
        prefix_enum=None,):
        def _stitch_phase(df):
            data = ccat_mp.struct_batches(df, 1, batch_len, max_workers)

            results_dict = {}
            with ccat_mp.optional_executor(max_workers, ex=ex) as executor:
                future_to_batch = {
                    executor.submit(
                        ccat_mp.process_batches,
                        np.unwrap,
                        data[i][0],
                        period = threshold[inds],
                    ): all_tones[inds]
                    for i, inds in enumerate(calc_ind)
                }

                for future in concurrent.futures.as_completed(future_to_batch):
                    tones = future_to_batch[future]
                    stitch_cols = future.result()
                    for tone, stitch_col in zip(tones, stitch_cols):
                        if isinstance(stitch_col, Exception):
                            log.log(
                                "DEBUG",
                                "Phase stitching failed for tone %s with exception: %s",
                                tone,
                                stitch_col,
                            )
                            stitch_data = (
                                np.full(df.len(), np.nan),
                            )
                        else:
                            stitch_data = stitch_col

                        results_dict[ccat_df.add_tone(return_col[0], tone, padding)] = (
                            stitch_data
                        )
                        
            return ccat_mp.package_results(results_dict)

        if not len(args) == 3:
            log.log(
                "ERROR",
                "det, bounds, loss, f_scale, method, and max_workers are required arguments.",
            )
        threshold, max_workers, ex = np.array(args)
        max_workers, ex = (
            int(max_workers[0]),
            ex[0],
        )
        all_tones = np.array(tones)

        phase_col, stitch_name_col = (
            col_enum.PHASE.value,
            prefix_enum.STITCH.value,
        )

        return_col, return_type = (
            ["_".join([stitch_name_col, phase_col])],
            [pl.Float64],
        )
        expr, calc_ind, batch_len = ccat_mp.create_batches(
            _stitch_phase,
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
    def _calc_linewidth(
        schema: pl.Schema,
        *args,
        tones: list[int] | None = None,
        padding: int = 4,
        recalc: bool = False,
        col_enum: Enum | None = None,
        prefix_enum: Enum | None = None,
    ) -> list[pl.Expr]:
        """Generates pl.Expr for calculating the phase of a tone using I & Q data

        Args:
            schema (pl.Schema)
        """
        if tones is None:
            tones = [tones]

        I_col, Q_col, linewidth_col = (
            col_enum.IN_PHASE.value,
            col_enum.QUADRATURE.value,
            col_enum.LINEWIDTH.value,
        )
        exprs = []
        for tone in tones:
            if (
                linewidth_tone_col := ccat_df.add_tone(linewidth_col, tone, padding)
            ) not in schema or recalc:
                I_tone_col, Q_tone_col = (
                    ccat_df.add_tone(I_col, tone, padding),
                    ccat_df.add_tone(Q_col, tone, padding),
                )
                exprs.append(
                    (pl.col(Q_tone_col)/(-2*pl.col(I_tone_col))).alias(
                        linewidth_tone_col
                    )
                )
        return exprs

    @staticmethod
    def _calc_IQ_circle_diameter(
        schema: pl.Schema,
        *args,
        tones: list[int] | None = None,
        padding: int = 4,
        recalc: bool = False,
        col_enum: Enum | None = None,
        prefix_enum: Enum | None = None,
    ) -> list[pl.Expr]:
        """Generates pl.Expr for calculating the phase of a tone using I & Q data

        Args:
            schema (pl.Schema)
        """
        if tones is None:
            tones = [tones]

        I_col, Q_col, diameter_col = (
            col_enum.IN_PHASE.value,
            col_enum.QUADRATURE.value,
            col_enum.IQ_CIRCLE_DIAMETER.value,
        )
        exprs = []
        for tone in tones:
            if (
                diameter_tone_col := ccat_df.add_tone(diameter_col, tone, padding)
            ) not in schema or recalc:
                I_tone_col, Q_tone_col = (
                    ccat_df.add_tone(I_col, tone, padding),
                    ccat_df.add_tone(Q_col, tone, padding),
                )
                exprs.append(
                    ((1 + (pl.col(Q_tone_col)/pl.col(I_tone_col))**2)*pl.col(I_tone_col)).alias(
                        diameter_tone_col
                    )
                )
        return exprs

    @staticmethod
    def _calc_coupling_linewidth(
        schema: pl.Schema,
        *args,
        tones: list[int] | None = None,
        padding: int = 4,
        recalc: bool = False,
        col_enum: Enum | None = None,
        prefix_enum: Enum | None = None,
    ) -> list[pl.Expr]:
        """Generates pl.Expr for calculating the phase of a tone using I & Q data

        Args:
            schema (pl.Schema)
        """
        if tones is None:
            tones = [tones]

        linewidth_col, diameter_col, coupling_col = (
            col_enum.LINEWIDTH.value,
            col_enum.IQ_CIRCLE_DIAMETER.value,
            col_enum.COUPLING_LINEWIDTH.value,
        )
        exprs = []
        for tone in tones:
            if (
                coupling_tone_col := ccat_df.add_tone(coupling_col, tone, padding)
            ) not in schema or recalc:
                linewidth_tone_col, diameter_tone_col = (
                    ccat_df.add_tone(linewidth_col, tone, padding),
                    ccat_df.add_tone(diameter_col, tone, padding),
                )
                exprs.append(
                    (pl.col(linewidth_tone_col)/pl.col(diameter_tone_col)).alias(
                        coupling_tone_col
                    )
                )
        return exprs
      
    @staticmethod
    def _calc_linewidth_fit(
        schema,
        *args,
        tones: list[int],
        padding: int = 4,
        recalc: bool = False,
        col_enum=None,
        prefix_enum=None,
    ):
        def _linewidth_fit(df):
            data = ccat_mp.struct_batches(df, 2, batch_len, max_workers)

            results_dict = {}
            with ccat_mp.optional_executor(max_workers, ex=ex) as executor:
                future_to_batch = {
                    executor.submit(
                        ccat_mp.process_batches,
                        ccat_fit.linewidth_fit,
                        data[i][0],
                        data[i][1],
                        method=method[inds],
                        params=params[inds],
                    ): all_tones[inds]
                    for i, inds in enumerate(calc_ind)
                }

                fit_result = targ.fit_result.get(f"{fit_name_col}_{fit_result_col}", [None]*targ.num_tones)
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
                        targ._properties[ccat_df.add_tone("det", tone, padding)] = (
                            properties_dict
                        )
                        results_dict[ccat_df.add_tone(return_col[0], tone, padding)] = (
                            best_fit
                        )
                targ.fit_result[f"{fit_name_col}_{fit_result_col}"] = fit_result
            return ccat_mp.package_results(results_dict)

        if not len(args) == 8:
            log.log(
                "ERROR",
                "nonlinear, params, window, and max_workers are required arguments.",
            )

        (
            targ,
            lw_prefix,
            lw_shift_prefix,
            coupling_linewidth,
            method,
            params_list,
            max_workers,
            ex,
        ) = args
        params = np.empty(len(params_list), dtype=object)
        for i in range(len(params_list)):
            params[i] = params_list[i]
        method = np.array(method)
        targ, lw_prefix, lw_shift_prefix, coupling_linewidth, max_workers, ex = (
            targ[0],
            lw_prefix[0],
            lw_shift_prefix[0],
            coupling_linewidth[0],
            int(max_workers[0]),
            ex[0],
        )
        all_tones = np.array(tones)

        linewidth_name, fit_name_col, fit_result_col = (
            col_enum.COUPLING_LINEWIDTH.value if coupling_linewidth else col_enum.LINEWIDTH.value,
            prefix_enum.COUPLING_LINEWIDTH_FIT.value if coupling_linewidth else prefix_enum.LINEWIDTH_FIT.value,
            col_enum.FIT_RESULT.value
        )

        linewidth_col = "_".join([lw_prefix, linewidth_name])
        linewidth_shift_col = "_".join([lw_shift_prefix, linewidth_name])

        param_names = {
            "a": col_enum.KI_NONLINEARITY_PARAMETER.value,
            "b": col_enum.QP_NONLINEARITY_PARAMETER.value,
            "y_ki": col_enum.LINEWIDTH_FIT_Y_KI.value,
            "y_qp": col_enum.LINEWIDTH_FIT_Y_QP.value,
            "y_off": col_enum.LINEWIDTH_FIT_OFFSET.value,
            "y_exp": col_enum.LINEWIDTH_FIT_EXPONENT.value,
        }

        return_col, return_type = ["_".join([fit_name_col, linewidth_shift_col])], [pl.Float64]
        expr, calc_ind, batch_len = ccat_mp.create_batches(
            _linewidth_fit,
            tones,
            schema,
            input_col=[linewidth_col, linewidth_shift_col],
            return_col=return_col,
            return_type=return_type,
            padding=padding,
            max_workers=max_workers,
            recalc=recalc,
        )
        return expr

    # ============= #
    # Concatenation #
    # ============= #

    def join(self, other, in_place=False):
        new_data = super().join(other, in_place=in_place)
        left_prop, right_prop = self.properties, other.properties
        new_data._properties_df = pl.concat(
            [left_prop, right_prop], how="diagonal"
        ).with_columns(pl.Series("det", new_data.tones))
        return new_data

    # ================ #
    # Pickling Methods #
    # ================ #

    def __getstate__(self):
        state = super().__getstate__()
        if not self.analysis_cfg["io"]["pickle"]["pickle_dataframes"] and state.get(
            "_data", True
        ):
            properties = self.properties
            del state["_properties_df"]

            file_name = (
                "properties"
                if (pickle_count := state["pickle_count"]) is None
                else f"properties_{pickle_count}"
            )
            save_path = Path(self.pickle_dir) / "dataframe" / f"{file_name}.parquet"

            properties.write_parquet(save_path)
        return state

    def __setstate__(self, state):
        super().__setstate__(state)

        if (
            not self.analysis_cfg["io"]["pickle"]["pickle_dataframes"]
            and not isinstance(
                prop_df := getattr(self, "_properties_df", True), pl.DataFrame
            )
            and prop_df
        ):
            file_name = (
                "properties"
                if (pickle_count := self.pickle_count) is None
                else f"properties_{pickle_count}"
            )
            self.properties = pl.scan_parquet(
                Path(self.pickle_dir) / "dataframe" / f"{file_name}.parquet"
            )

def stitch_phase(phase, threshold: float = 1.9 * np.pi):
    '''
    Stitch discontinuities in phase from crossing the negative real axis. 
    '''

    curr_shift = 0
    for i in range(len(phase) - 1):
        diff = phase[i + 1] - phase[i]
        phase[i] -= curr_shift
        if diff > threshold:
            curr_shift += 2 * np.pi
        elif diff < -1 * threshold:
            curr_shift -= 2 * np.pi
    phase[-1] -= curr_shift
    return phase