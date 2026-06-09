import gc
import sys
import numpy as np
import polars as pl
from numba import njit

from functools import cached_property
from collections.abc import Iterable
from pathlib import Path

# Local Imports
import ccatkidlib.log as log
import ccatkidlib.utils as utils
import ccatkidlib.analysis.utils.pair as pair
import ccatkidlib.analysis.utils.dataframe as ccat_df

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

    # ==========================#
    # Lazily Loaded Attributes #
    # ==========================#

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


    def join(self, other, in_place=False):
        new_data = super().join(other, in_place=in_place)
        left_prop, right_prop = self.properties, other.properties
        new_data._properties_df = pl.concat(
            [left_prop, right_prop], how="diagonal"
        ).with_columns(pl.Series("det", new_data.tones))
        return new_data

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
