"""Module for analyzing kinetic inductance detector (KID) time ordered (timestream) data

Authors:
    - Darshan Patel <dp649@cornell.edu>

TODO:
    - Multiprocess FFT & PSD calcultions

"""

import so3g  # noqa: F401
import numpy as np
import polars as pl
import concurrent.futures

from scipy.signal import welch
from functools import cached_property
from pathlib import Path
from collections.abc import Iterable
from typing import TypeAlias, Literal
from spt3g import core

import holoviews as hv
import datashader as ds
from holoviews import opts
from holoviews.operation.datashader import datashade, dynspread

# Local Imports
import ccatkidlib.io as log
import ccatkidlib.utils as utils
import ccatkidlib.analysis.viz.viz_utils as viz_utils
import ccatkidlib.analysis.utils.pair as pair
import ccatkidlib.analysis.utils.multiprocess as ccat_mp
import ccatkidlib.analysis.utils.dataframe as ccat_df

from ccatkidlib.analysis.core.data import Data

Format: TypeAlias = Literal["png", "jpeg", "pdf"]


class Timestream(Data):
    """Class representing a timestream taken with a radio frequency system on a chip (RFSoC)

    Attributes:
        tones (list[int]): List of tones with loaded timestreams
        start (float): Start time of loaded timestreams in seconds (0 seconds is beginning of timestream)
        end (float): End time of loaded timestreams in seconds (relative to start time)
        packet_counts (list[int]): List of packet numbers of loaded timestreams
        data (pl.DataFrame): Polars DataFrame with loaded (and transformed) timestream data
        properties (pl.DataFrame): Polars DataFrame with timestream properties (a 'property' has one value per tone)
        sampling_freq (float): Sampling frequency of timestream data in Hz
    """

    def __init__(
        self,
        com_to,
        tones: int | list[int] = -1,
        noise_tones: int | list[int] | None = None,
        start=0,
        end=-1,
        **kwargs,
    ):
        """
        Constructor for Timestream.

        Args:
            com_to (str): Which board and drone were used to take the timestreams (in form ``bid.drid``)
            tones (list): List of tones for which to load timestreams
            start (float): Start time in seconds (0 seconds is beginning of timestream)
            end (float): End time in seconds (relative to start time, -1 for no end time)
            analysis_cfg (str): File path of analysis configuration file. Defaults to analysis configuration file in *ccatkidlib/analysis* directory.
        """
        kwargs["data_type"] = "timestream"
        super().__init__(com_to, **kwargs)

        # Define list of tones
        # --------------------
        if isinstance(tones, int):
            if tones >= 0:
                self.tones = [tones]
            else:
                self.tones = list(range(self.num_tones))
        elif isinstance(tones, Iterable) and all(
            [isinstance(tone, int) for tone in tones]
        ):
            self.tones = tones
        else:
            error = f"Invalid type {type(tones)} for argument 'tones'. Should be int, list[int], or None."
            log.log("CRITICAL", error)
            raise ValueError(error)

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

        # Define timestream start and end times
        # -------------------------------------
        self.start = start
        self.end = np.inf if end < 0 else end

        # End time should be larger than start time
        if not (self.end - self.start > 0):
            error = "End time must be greater than start time!"
            log.log("ERROR", error)
            raise ValueError(error)

        # Create packet count attribute
        # -----------------------------
        self.packet_counts = []

        self._properties = {}
        self._properties_df = None

        self.spline_dict = {"None": None}

    # ==================#
    # Plotting Methods #
    # ==================#
    @staticmethod
    def _plot(df, plot_opts, *args, **kwargs):
        x_dim, y_dim = args
        datashade_plot = kwargs.pop("datashade")

        if datashade_plot:
            kwargs["s"] = kwargs.pop("ms")
            stream_scatter = df.hvplot.scatter(
                x=x_dim, y=y_dim, label="DataScatter", **kwargs
            ).relabel(group="Stream")
            plot = stream_scatter
            if kwargs.pop("linewidth") > 0:
                stream_line = df.hvplot.line(
                    x=x_dim, y=y_dim, label="DataLine", **kwargs
                ).relabel(group="Stream")
                plot = plot * stream_line
            plot = dynspread(datashade(plot))
        else:
            stream = df.hvplot.line(x=x_dim, y=y_dim, label="Data", **kwargs).relabel(
                group="Stream"
            )
            plot = stream

        plot.opts(*plot_opts)
        return plot

    def plot(
        self,
        x_dim,
        y_dim,
        x_prefix: str = "",
        y_prefix: str = "",
        xlabel: str = "",
        ylabel: str = "",
        grouping: str = "groupby",
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        plot_opts=None,
        filter_exprs=[],
        unpivot_x=True,
        return_df=False,
        save_fig: bool | None = None,
        figs_per_file: int | None = None,
        overwrite: bool | None = None,
        save_dir: str | Path | None = None,
        save_name: str = None,
        save_fmt: Format | None = None,
        return_fig=True,
        df: pl.DataFrame | None = None,
        by: str | list[str] | None = None,
        **kwargs,
    ):
        # Get DataFrame with data to plot
        # -------------------------------
        col_dict = {"sample": "sample", "x": x_dim, "y": y_dim}

        if df is None or by is None:
            df, by = self._get_plot_df(
                col_dict,
                x_prefix=x_prefix,
                y_prefix=y_prefix,
                include=include,
                exclude=exclude,
                unpivot_x=unpivot_x,
            )
            col_dict["x"], col_dict["y"] = df.select(
                pl.exclude("det", "sample")
            ).columns
            df = df.filter(
                (~pl.col(col_dict["x"]).is_nan()) & (~pl.col(col_dict["y"]).is_nan())
            )
            if filter_exprs:
                df = df.filter(filter_exprs)
        else:
            col_dict["x"], col_dict["y"] = df.select(pl.exclude(by, "sample")).columns
        if not return_fig:
            return df, by

        # Set default hvplot key word arguments
        # -------------------------------------
        if "aspect" not in kwargs:
            kwargs["aspect"] = self.viz_cfg["static_plot"]["stream"]["aspect"]
        if "marker" not in kwargs:
            kwargs["marker"] = self.viz_cfg["static_plot"]["stream"]["marker"]
        if "ms" not in kwargs:
            kwargs["ms"] = self.viz_cfg["static_plot"]["stream"]["marker_size"]
        if "linewidth" not in kwargs:
            kwargs["linewidth"] = self.viz_cfg["static_plot"]["stream"]["linewidth"]
        if "dynamic" not in kwargs:
            kwargs["dynamic"] = True
        fig_size = (
            kwargs.pop("fig_size")
            if "fig_size" in kwargs
            else self.viz_cfg["static_plot"]["stream"]["fig_size"]
        )

        if grouping == "by":
            kwargs["by"] = by
        elif grouping == "groupby":
            kwargs["groupby"] = by
        else:
            error = 'Invalid string specified for argument "grouping"! Must be either "by" or "groupby".'
            log.log("CRITICAL", error)
            raise ValueError(error)

        # Create opts for plots
        # ----------------------
        cfg = self.drone_cfg["det_config"]
        title = rf"${cfg['detector_type']}\ {cfg['network']}$"

        curve_opts = opts.Curve(
            show_legend=False,
            title=title,
            xlabel=xlabel if xlabel is not None else col_dict["x"],
            ylabel=ylabel if ylabel is not None else col_dict["y"],
            fig_size=fig_size,
        )
        RGB_opts = opts.RGB(
            show_legend=False,
            title=title,
            xlabel=xlabel if xlabel is not None else col_dict["x"],
            ylabel=ylabel if ylabel is not None else col_dict["y"],
            fig_size=fig_size,
            aspect=kwargs["aspect"],
        )
        all_opts = [curve_opts, RGB_opts]
        if plot_opts is not None:
            all_opts.append(plot_opts)

        # Create plot for immediate visualization
        # ---------------------------------------
        plot = Timestream._plot(df, all_opts, *(col_dict["x"], col_dict["y"]), **kwargs)

        # Save plot in background
        # -----------------------
        viz_utils.save_fig(
            self,
            Timestream._plot,
            df,
            all_opts,
            *(col_dict["x"], col_dict["y"]),
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

    def stream_plot(
        self,
        col_name: str,
        prefix: str = "",
        time_col="t",
        grouping="groupby",
        datashade=None,
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        return_df=False,
        save_fig: bool | None = None,
        overwrite: bool | None = None,
        **kwargs,
    ):
        """Plot the specified data column as a function of time

        Args:
            col_name (str): Name of data column (e.g., *I*, *Q*, *mag*, etc.)
            prefix (str, optional): Defaults to ""
            return_df (bool): Whether to return the Polars DataFrame that was used to create the plot. Defaults to *False*
            include (int | list[int] | None, optional): Defaults to *None*
            exclude (int | list[int] | None, optional): Defaults to *None*
        Returns:
            return (hv.NdOverlay | tuple[hv.NdOverlay, pl.DataFrame]):

        """
        xlabel, ylabel = r"$Time [s]$", f"{prefix}_{col_name}"
        save_name = f"timestream_{prefix}{'_' if prefix else ''}{col_name}_stream"

        kwargs["datashade"] = (
            datashade
            if datashade is not None
            else self.viz_cfg["static_plot"]["stream"]["datashade"]
        )
        rtn = self.plot(
            time_col,
            col_name,
            y_prefix=prefix,
            grouping=grouping,
            include=include,
            exclude=exclude,
            unpivot_x=False,
            xlabel=xlabel,
            ylabel=ylabel,
            return_df=return_df,
            save_fig=save_fig,
            save_name=save_name,
            overwrite=overwrite,
            **kwargs,
        )
        return rtn

    def mag_plot(
        self,
        x_prefix: str = "",
        y_prefix: str = "",
        grouping="groupby",
        datashade=None,
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        return_df=False,
        save_fig: bool | None = None,
        overwrite: bool | None = None,
        **kwargs,
    ):
        xlabel, ylabel = r"$Frequency\ [Hz]$", r"$|S_{21}|$"
        save_name = f"timestream_{y_prefix}{'_' if y_prefix else ''}mag"

        if "linewidth" not in kwargs:
            kwargs["linewidth"] = 0
        kwargs["datashade"] = (
            datashade
            if datashade is not None
            else self.viz_cfg["static_plot"]["stream"]["datashade"]
        )
        rtn = self.plot(
            "f",
            "mag",
            x_prefix=x_prefix,
            y_prefix=y_prefix,
            grouping=grouping,
            include=include,
            exclude=exclude,
            xlabel=xlabel,
            ylabel=ylabel,
            return_df=return_df,
            save_fig=save_fig,
            save_name=save_name,
            overwrite=overwrite,
            **kwargs,
        )
        return rtn

    def phase_plot(
        self,
        x_prefix: str = "",
        y_prefix: str = "",
        grouping="groupby",
        datashade=None,
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        return_df=False,
        save_fig: bool | None = None,
        overwrite: bool | None = None,
        **kwargs,
    ):
        xlabel, ylabel = r"$Frequency\ [Hz]$", r"$Phase\ [rad]$"
        save_name = f"timestream_{y_prefix}{'_' if y_prefix else ''}phase"

        if "linewidth" not in kwargs:
            kwargs["linewidth"] = 0
        kwargs["datashade"] = (
            datashade
            if datashade is not None
            else self.viz_cfg["static_plot"]["stream"]["datashade"]
        )
        rtn = self.plot(
            "f",
            "phase",
            x_prefix=x_prefix,
            y_prefix=y_prefix,
            grouping=grouping,
            include=include,
            exclude=exclude,
            xlabel=xlabel,
            ylabel=ylabel,
            return_df=return_df,
            save_fig=save_fig,
            save_name=save_name,
            overwrite=overwrite,
            **kwargs,
        )
        return rtn

    def IQ_plot(
        self,
        prefix: str = "",
        projection="IQ",
        grouping="groupby",
        datashade=True,
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        return_df=False,
        save_fig: bool | None = None,
        overwrite: bool | None = None,
        **kwargs,
    ):
        if projection == "IQ" or datashade:
            xlabel, ylabel = r"$I\ [arb]$", r"$Q\ [arb]$"
            x_dim, y_dim = "I", "Q"
            plot_opts = None
        elif projection == "polar":
            xlabel, ylabel = r"$Phase\ [deg]$", r"$|S_{21}|$"
            x_dim, y_dim = "phase", "mag"
            plot_opts = opts.Curve(projection="polar", show_grid=True)
        else:
            error = 'Invalid projection specified, must be either "IQ" or "polar".'
            log.log("CRITICAL", error)
            raise ValueError(error)
        save_name = f"timestream_{prefix}{'_' if prefix else ''}{projection}"

        if "linewidth" not in kwargs:
            kwargs["linewidth"] = 0
        kwargs["datashade"] = (
            datashade
            if datashade is not None
            else self.viz_cfg["static_plot"]["stream"]["datashade"]
        )
        rtn = self.plot(
            x_dim,
            y_dim,
            x_prefix=prefix,
            y_prefix=prefix,
            grouping=grouping,
            include=include,
            exclude=exclude,
            xlabel=xlabel,
            ylabel=ylabel,
            return_df=return_df,
            save_fig=save_fig,
            save_name=save_name,
            overwrite=overwrite,
            plot_opts=plot_opts,
            **kwargs,
        )
        return rtn

    def psd_plot(
        self,
        col_name,
        prefix: str = "",
        grouping="groupby",
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        return_df=False,
        save_fig: bool | None = None,
        overwrite: bool | None = None,
        **kwargs,
    ):
        xlabel, ylabel = r"$PSD\ Frequency\ [Hz]$", rf"psd_{prefix}_{col_name}"
        save_name = f"timestream_{prefix}{'_' if prefix else ''}{col_name}_psd"

        if "logx" not in kwargs:
            kwargs["logx"] = True
        if "logy" not in kwargs:
            kwargs["logy"] = True
        if "datashade" not in kwargs:
            kwargs["datashade"] = False

        rtn = self.plot(
            col_name,
            col_name,
            x_prefix=f"psd_f{'_' if prefix else ''}{prefix}",
            y_prefix=f"psd{'_' if prefix else ''}{prefix}",
            grouping=grouping,
            include=include,
            exclude=exclude,
            xlabel=xlabel,
            ylabel=ylabel,
            return_df=return_df,
            save_fig=save_fig,
            save_name=save_name,
            overwrite=overwrite,
            **kwargs,
        )
        return rtn

    def fft_plot(
        self,
        col_name,
        prefix: str = "",
        grouping="groupby",
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        return_df=False,
        save_fig: bool | None = None,
        overwrite: bool | None = None,
        **kwargs,
    ):
        xlabel, ylabel = r"$FFT\ Frequency\ [Hz]$", f"fft_{prefix}_{col_name}"
        save_name = f"timestream_{prefix}{'_' if prefix else ''}{col_name}_fft"

        if "logx" not in kwargs:
            kwargs["logx"] = True
        if "logy" not in kwargs:
            kwargs["logy"] = True
        rtn = self.plot(
            "f",
            col_name,
            x_prefix=f"fft_f{'_' if prefix else ''}{prefix}",
            y_prefix=f"fft{'_' if prefix else ''}{prefix}",
            grouping=grouping,
            include=include,
            exclude=exclude,
            xlabel=xlabel,
            ylabel=ylabel,
            return_df=return_df,
            save_fig=save_fig,
            save_name=save_name,
            overwrite=overwrite,
            **kwargs,
        )
        return rtn

    # ==========================#
    # Lazily Loaded Attributes #
    # ==========================#

    @property
    def data(self):
        """
        Load timestream I, Q data.
        """

        if self._data is None:
            ftype = Path(self.data_path[0]).suffix
            data = {}

            # Load g3 timestreams
            # -------------------
            if ftype == ".g3" or ftype == ".txt":
                ts, Is, Qs = self._load_g3_timestream(ftype)
                dt, unit = np.array(ts) * 1e9, "ns"

            # Load python timestreams (.npy or .npz)
            # --------------------------------------
            elif ftype == ".npz" or ftype == ".npy":
                ts, Is, Qs = self._load_npy_timestream(ftype)
                dt, unit = np.array(ts) * 1e5, "us"
            else:
                error = f"Invalid timestream file type: '{ftype}'!"
                log.log("ERROR", error)
                raise ValueError(error)

            ts, Is, Qs = (
                np.array(ts),
                np.array(Is, dtype=np.float64),
                np.array(Qs, dtype=np.float64),
            )

            name_mapping = self.analysis_cfg["convention"]["name"]
            data[name_mapping["sample"]] = range(len(ts))
            data[name_mapping["time"]], data[name_mapping["datetime"]] = ts, dt
            for t, I, Q in zip(self.tones, Is, Qs):
                data[ccat_df.add_tone(name_mapping["in_phase"], t, self.padding)] = I
                data[ccat_df.add_tone(name_mapping["quadrature"], t, self.padding)] = Q
            self._data = pl.DataFrame(data)
            self._data = self._data.with_columns(
                pl.col(name_mapping["datetime"]).cast(pl.Datetime(unit)),
                (
                    pl.col(name_mapping["time"]) - pl.col(name_mapping["time"]).first()
                ).alias(name_mapping["zerotime"]),
            )
        elif isinstance(self._data, pl.LazyFrame):
            self._data = self._data.collect()
        return self._data

    @data.setter
    def data(self, value: pl.lazyframe.frame.LazyFrame | None):
        if value is None or isinstance(value, (pl.DataFrame, pl.LazyFrame)):
            self._data = value
        else:
            log.log(
                "ERROR",
                "Cannot set data with type %s. Must be a Polars LazyFrame! Convert DataFrame to lazy frame with .lazy() before setting.",
                type(value),
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

    @cached_property
    def sampling_freq(self):
        return self.io_cfg["boards"][f"b{self.bid}"]["sampling_freq"]

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

    # =====================#
    # Data Getter Methods #
    # =====================#

    def psd(
        self,
        prefix: str | list[str] = "",
        col_name: str = "phase",
        window="hann",
        nperseg=None,
        detrend=False,
        average="mean",
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        recalc: bool = False,
        max_workers: int = 1,
        ex=None,
    ) -> pl.dataframe.frame.DataFrame:

        enum_key, mapping = None, self.analysis_cfg["convention"]["name"]
        for key, val in mapping.items():
            if val == col_name:
                enum_key = key
                break

        if enum_key is None:
            error = f"Could not find column {col_name}. Ensure that a mapping exists in the analysis configuration file."
            log.log("ERROR", error)
            raise KeyError(error)

        name_enums, prefix_enums = ccat_df.create_enums(
            [enum_key],
            prefix,
            ["power_spectral_density", "power_spectral_density_frequency"],
            self.analysis_cfg,
        )

        num_prefix = len(name_enums)
        sampling_freq, height = self.sampling_freq, self.data.height
        window = ccat_df.check_args(window, num_prefix, str)
        average = ccat_df.check_args(average, num_prefix, str)
        nperseg = ccat_df.check_args(nperseg, num_prefix, int)
        detrend = ccat_df.check_args(detrend, num_prefix, str)

        args = [
            [
                sampling_freq,
                height,
                win,
                npseg,
                dtrend,
                avg,
                ccat_mp.check_max_workers(max_workers),
                ex,
            ]
            for win, npseg, dtrend, avg in zip(window, nperseg, detrend, average)
        ]
        self.transform(
            [Timestream._calc_psd] * num_prefix,
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
                f"{prefix_enum.POWER_SPECTRAL_DENSITY.value}_{name_enum[enum_key.upper()].value}",
                f"{prefix_enum.POWER_SPECTRAL_DENSITY_FREQUENCY.value}_{name_enum[enum_key.upper()].value}",
            ]

        self.data = ccat_df.unnest(self, [f"struct_{name}" for name in col_names])

        return self.get_data(
            col_name=col_names,
            include=include,
            exclude=exclude,
        )

    def fft(
        self,
        prefix: str | list[str] = "",
        col_name: str = "phase",
        sampling_freq=None,
        include: int | list[int] | None = None,
        exclude: int | list[int] | None = None,
        recalc: bool = False,
        max_workers: int = 1,
        ex=None,
    ) -> pl.dataframe.frame.DataFrame:

        enum_key, mapping = None, self.analysis_cfg["convention"]["name"]
        for key, val in mapping.items():
            if val == col_name:
                enum_key = key
                break

        if enum_key is None:
            error = f"Could not find column {col_name}. Ensure that a mapping exists in the analysis configuration file."
            log.log("ERROR", error)
            raise KeyError(error)

        name_enums, prefix_enums = ccat_df.create_enums(
            [enum_key],
            prefix,
            ["fourier_transform", "fourier_transform_frequency"],
            self.analysis_cfg,
        )

        num_prefix = len(name_enums)
        if sampling_freq is None:
            sampling_freq = self.sampling_freq
        height = self.data.height
        args = [
            [sampling_freq, height, ccat_mp.check_max_workers(max_workers), ex]
        ] * num_prefix
        self.transform(
            [Timestream._calc_fft] * num_prefix,
            *args,
            include=include,
            exclude=exclude,
            recalc=recalc,
            col_enum=name_enums,
            prefix_enum=prefix_enums
        )

        col_names = []
        for prefix_enum, name_enum in zip(prefix_enums, name_enums):
            col_names += [
                f"{prefix_enum.FOURIER_TRANSFORM.value}_{name_enum[enum_key.upper()].value}",
                f"{prefix_enum.FOURIER_TRANSFORM_FREQUENCY.value}_{name_enum[enum_key.upper()].value}",
            ]

        self.data = ccat_df.unnest(self, [f"struct_{name}" for name in col_names])

        return self.get_data(
            col_name=col_names,
            include=include,
            exclude=exclude,
        )

    # ==================#
    # Analysis Methods #
    # ==================#

    @staticmethod
    def _calc_psd(
        schema,
        *args,
        tones: list[int] | None = None,
        padding: int = 4,
        recalc: bool = False,
        col_enum=None,
        prefix_enum=None
    ):
        def _mp_psd(df):
            data = ccat_mp.struct_batches(df, 1, batch_len, max_workers)

            results_dict = {}
            with ccat_mp.optional_executor(max_workers, ex=ex) as executor:
                future_to_batch = {
                    executor.submit(
                        ccat_mp.process_batches,
                        _psd,
                        data[i][0],
                        height[inds],
                        sampling_freq[inds],
                        window=window[inds],
                        nperseg=nperseg[inds],
                        detrend=detrend[inds],
                        average=average[inds],
                    ): all_tones[inds]
                    for i, inds in enumerate(calc_ind)
                }
                for future in concurrent.futures.as_completed(future_to_batch):
                    tones = future_to_batch[future]
                    psd_cols = future.result()
                    for tone, psd_col in zip(tones, psd_cols):
                        if isinstance(psd_col, Exception):
                            log.log(
                                "WARNING",
                                "PSD calculation for tone %s failed with exception: %s",
                                tone,
                                psd_col,
                            )
                            psd_f, psd = (
                                np.full(df.len(), np.nan),
                                np.full(df.len(), np.nan),
                            )
                        else:
                            psd_f, psd = psd_col
                        results_dict[ccat_df.add_tone(return_col[0], tone, padding)] = psd
                        results_dict[ccat_df.add_tone(return_col[1], tone, padding)] = psd_f
            return ccat_mp.package_results(results_dict)

        if not len(args) == 8:
            log.log(
                "ERROR",
                "sampling_freq, height, window, nperseg, detrend, average, max_workers, and ex are required arguments",
            )

        (
            sampling_freq,
            height,
            window,
            nperseg,
            detrend,
            average,
            max_workers,
            ex,
        ) = np.array(args, dtype=object)
        max_workers, ex = max_workers[0], ex[0]
        all_tones = np.array(tones)

        data_col, psd_prefix, psd_f_prefix = (
            [name for name in col_enum][0].value, # Enum should only ever have one member so can extract without name. Could pass name as arg but would create more overhead
            prefix_enum.POWER_SPECTRAL_DENSITY.value,
            prefix_enum.POWER_SPECTRAL_DENSITY_FREQUENCY.value
        )
            
        return_col, return_type = (
            [f"{psd_prefix}_{data_col}", f"{psd_f_prefix}_{data_col}"],
            [pl.Float64, pl.Float64],
        )
        expr, calc_ind, batch_len = ccat_mp.create_batches(
            _mp_psd,
            tones,
            schema,
            input_col=[data_col],
            return_col=return_col,
            return_type=return_type,
            padding=padding,
            max_workers=max_workers,
            recalc=recalc,
        )
        return expr

    @staticmethod
    def _calc_fft(
        schema,
        *args,
        tones: list[int] | None = None,
        padding: int = 4,
        recalc: bool = False,
        col_enum=None,
        prefix_enum=None
    ):
        def _mp_fft(df):
            data = ccat_mp.struct_batches(df, 1, batch_len, max_workers)

            results_dict = {}
            with ccat_mp.optional_executor(max_workers, ex=ex) as executor:
                future_to_batch = {
                    executor.submit(
                        ccat_mp.process_batches,
                        _fft,
                        data[i][0],
                        height[inds],
                        sampling_freq[inds],
                    ): all_tones[inds]
                    for i, inds in enumerate(calc_ind)
                }
                for future in concurrent.futures.as_completed(future_to_batch):
                    tones = future_to_batch[future]
                    fft_cols = future.result()
                    for tone, fft_col in zip(tones, fft_cols):
                        if isinstance(fft_col, Exception):
                            log.log(
                                "WARNING",
                                "FFT calculation for tone %s failed with exception: %s",
                                tone,
                                fft_col,
                            )
                            fft_f, fft = (
                                np.full(df.len(), np.nan),
                                np.full(df.len(), np.nan),
                            )
                        else:
                            fft_f, fft = fft_col
                        results_dict[ccat_df.add_tone(return_col[0], tone, padding)] = fft
                        results_dict[ccat_df.add_tone(return_col[1], tone, padding)] = fft_f
            return ccat_mp.package_results(results_dict)

        if not len(args) == 4:
            log.log(
                "ERROR",
                "sampling_freq, height, max_workers and ex are required arguments",
            )
        sampling_freq, height, max_workers, ex = np.array(args, dtype=object)
        max_workers, ex = max_workers[0], ex[0]
        all_tones = np.array(tones)

        data_col, fft_prefix, fft_f_prefix = (
            [name for name in col_enum][0].value, # Enum should only ever have one member so can extract without name. Could pass name as arg but would create more overhead
            prefix_enum.FOURIER_TRANSFORM.value,
            prefix_enum.FOURIER_TRANSFORM_FREQUENCY.value
        )
            
        return_col, return_type = (
            [f"{fft_prefix}_{data_col}", f"{fft_f_prefix}_{data_col}"],
            [pl.Float64, pl.Float64],
        )
        expr, calc_ind, batch_len = ccat_mp.create_batches(
            _mp_fft,
            tones,
            schema,
            input_col=[data_col],
            return_col=return_col,
            return_type=return_type,
            padding=padding,
            max_workers=max_workers,
            recalc=recalc,
        )
        return expr

    # ==========================#
    # Internal Loading Methods #
    # ==========================#

    def load_frame(
        self,
        frame,
        start_time: float,
        time_precision: int = 1e8,
        mask: list[bool] = None,
    ) -> tuple[list[float], list[list[float]], list[list[float]]]:
        if "packet_counts" in frame:
            self.packet_counts += list(frame["packet_counts"])

        ts = []
        g3_data = frame["data"]
        data = g3_data.data

        if mask is None:
            ts = np.array(g3_data.times) / time_precision

            t_diff = np.array(ts - start_time)
            mask = np.where((t_diff >= self.start) & (t_diff <= self.end), True, False)

        inds = 2 * np.array(self.tones)

        I = data[list(inds)][:, mask]
        Q = data[list(inds + 1)][:, mask]

        if not len(ts) == 0:
            return ts[mask], I, Q
        else:
            return ts, I, Q

    def _load_g3_timestream(self, ftype):
        """
        Load g3 timestream data.
        """

        time_precision = 1e8
        start_time = -1

        g3_files = []
        for path in sorted(self.data_path):
            if ftype == ".txt":
                g3_root = self.analysis_cfg["file_paths"]["g3_root_dir"]
                try:
                    original_g3_root = self.io_cfg["file_paths"]["g3_root_dir"]
                except KeyError:
                    original_g3_root = self.analysis_cfg["file_paths"][
                        "original_g3_root_dir"
                    ]

                if not g3_root[-1] == "/":
                    g3_root += "/"
                if not original_g3_root[-1] == "/":
                    original_g3_root += "/"

                with open(path, "r") as file:
                    g3_files += file.readlines()
                    g3_files = [
                        pair.replace_root(g3_file, original_g3_root, g3_root)
                        for g3_file in g3_files
                    ]
                break  # If user specifies multiple txt files only load the first one since there can only be one txt file per timestream
            else:
                g3_files += [path]

        # Do initial pass through of frames in G3 file without fully loading the data to:
        # 1. Aggregate frames from different G3 files into one list
        # 2. Filter out frames that do not contain timestream data
        # 3. Get the number of tones
        # 4. Filter out frames that are not within the specified time range and construct the array of times
        # --------------------------------------------------------------------------------------------------
        frames = []
        masks = []
        ts = []
        for g3_file in g3_files:
            g3_data = core.G3File(str(g3_file))

            for frame in g3_data:
                # Filter out frames that are not G3 Scan frames (those that contain the timestream data)
                if frame.type == core.G3FrameType.Scan:
                    # Determine the number of tones used for timestream
                    if self.tones is None:
                        channel_count = frame[
                            "channel_count"
                        ]  # Get number of tones directly from G3 frame

                        # Send warning if there is a mismatch in the number of tones (but continue execution using the number in the G3 frame)
                        if not self.num_tones == channel_count:
                            print(
                                "WARNING | There is a mismatch between the number of tones in the comb and the number of tones in the timestream packets!"
                            )
                            self.num_tones = channel_count
                        self.tones = range(channel_count)

                    # Filter out frames that are not within specified time range
                    times = np.array(frame["data"].times) / time_precision
                    if start_time == -1:
                        start_time = times[0]
                    if times[0] - start_time <= self.end:
                        if times[-1] - start_time >= self.start:
                            t_diff = np.array(times - start_time)
                            mask = np.where(
                                (t_diff >= self.start) & (t_diff <= self.end),
                                True,
                                False,
                            )
                            frames.append(frame)
                            masks.append(mask)
                            ts = np.append(ts, times[mask])
                    else:
                        break
            else:
                continue
            break

        shape = (len(self.tones), len(ts))

        Is, Qs = np.empty(shape, dtype=np.int32), np.empty(shape, dtype=np.int32)
        curr_ind = 0
        for i in range(len(frames)):
            mask = masks[i]
            t, I, Q = self.load_frame(frames[i], start_time, time_precision, mask=mask)  # noqa: E741

            num_samps = int(np.sum(mask))
            Is[:, curr_ind : num_samps + curr_ind] = I
            Qs[:, curr_ind : num_samps + curr_ind] = Q
            curr_ind += num_samps

        return ts, Is, Qs

    def _load_npy_timestream(self, ftype):
        """
        Load timestream data from .npz or .npy file
        """

        def _load_npy(data, start_time):
            """
            Load timestream I, Q data from npy file generated by python timestream
            """

            start_ind = 0

            # If data has two more rows than the number of tones than both packet counts and timestamps were recorded
            if len(data) % 2 == 0:
                self.packet_counts += list(data[0])
                start_ind += 1

            ts = data[start_ind].real / time_precision
            t_diff = np.array(ts - start_time)
            in_time = np.where(
                (t_diff >= self.start) & (t_diff <= self.end), True, False
            )

            inds = 2 * np.array(self.tones) + start_ind + 1

            I = data[list(inds)][:, in_time]
            Q = data[list(inds + 1)][:, in_time]

            return ts[in_time], I, Q

        ts, Is, Qs = [], None, None

        time_precision = 1e9
        tstamp_ind = 0
        start_time = -1

        # Do an initial iteration through the timestream data without fully loading into memory to determine which files lie within the specified time span
        # -------------------------------------------------------------------------------------------------------------------------------------------------

        data_arr = []
        for data_path in self.data_path:
            npy_file = (
                np.load(data_path, mmap_mode="r").values()
                if ftype == ".npz"
                else [np.load(data_path, mmap_mode="r")]
            )

            for npy_data in npy_file:
                if start_time == -1:
                    if len(npy_data) % 2 == 0:
                        tstamp_ind += 1
                    start_time = npy_data[tstamp_ind][0] / time_precision
                if npy_data[tstamp_ind][0] / time_precision - start_time <= self.end:
                    if (
                        npy_data[tstamp_ind][-1] / time_precision - start_time
                        >= self.start
                    ):
                        data_arr.append(npy_data)
                else:
                    break
            else:
                continue
            break

        # Load the data within the specified timespan
        # -------------------------------------------
        for npy_data in data_arr:
            t, I, Q = _load_npy(npy_data, start_time)  # noqa: E741
            ts = np.append(ts, t)
            Is = np.append(Is, I, axis=1) if Is is not None else I
            Qs = np.append(Qs, Q, axis=1) if Qs is not None else Q

        return ts, Is, Qs

    # =====================#
    # Data Getter Methods #
    # =====================#

    def t(self):
        return self.get_data(col_name=self.analysis_cfg["convention"]["name"]["time"])

    def dt(self):
        return self.get_data(
            col_name=self.analysis_cfg["convention"]["name"]["datetime"]
        )

    def zt(self):
        return self.get_data(
            col_name=self.analysis_cfg["convention"]["name"]["zerotime"]
        )

    def join(self, other, in_place=False):
        new_data = super().join(other, in_place=in_place)
        left_prop, right_prop = self.properties, other.properties
        new_data._properties_df = pl.concat(
            [left_prop, right_prop], how="diagonal"
        ).with_columns(pl.Series("det", new_data.tones))
        return new_data

    # =============================== #
    # Define Custom Pickling Behavior #
    # =============================== #

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


# ====================#
# Analysis Functions #
# ====================#


def _psd(data, height, fs, window="hann", nperseg=None, detrend=False, average="mean"):
    try:
        if nperseg is None:
            nperseg = 2 ** round(np.log2(height / 5))
        psd_f, psd = welch(
            data,
            fs=fs,
            window=window,
            nperseg=int(nperseg),
            detrend=detrend,
            average=average,
        )
        psd_f, psd = psd_f[1:-1], np.sqrt(psd[1:-1])

        pad_len = height - len(psd_f)
        psd_f, psd = (
            np.pad(psd_f, (0, pad_len), constant_values=None),
            np.pad(psd, (0, pad_len), constant_values=None),
        )
    except Exception as e:
        psd_f, psd = np.zeros(height), np.zeros(height)
    return psd_f, psd

def _fft(data, height, fs):
    try:
        fft_f, fft = (
            np.fft.fftshift(np.fft.fftfreq(height, d=1 / fs)),
            np.abs(np.fft.fftshift(np.fft.fft(data))),
        )

        pad_len = height - len(fft_f)
        fft_f, fft = (
            np.pad(fft_f, (0, pad_len), constant_values=None),
            np.pad(fft, (0, pad_len), constant_values=None),
        )
    except Exception as e:
        fft_f, fft = np.zeros(height), np.zeros(height)
    return fft_f, fft
