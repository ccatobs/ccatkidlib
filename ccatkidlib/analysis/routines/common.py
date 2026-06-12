import polars as pl
import numpy as np

import ccatkidlib.analysis.utils.dataframe as ccat_df
import ccatkidlib.analysis.utils.multiprocess as ccat_mp


def IQ_circle_center(
    det,
    data="both",
    savgol_window=9,
    savgol_order=1,
    trim_window=10,
    trim_mean_points=10,
    mismatch_mean_points=10,
    include=None,
    exclude=None,
    recalc=False,
    savgol_workers=1,
    fit_workers=1,
    ex=None,
):
    name, prefix = (
        det.analysis_cfg["convention"]["name"],
        det.analysis_cfg["convention"]["prefix"],
    )

    # Remove Cable Delay
    # ------------------
    det.vna.phase(recalc=recalc)
    det.cable_delay
    det.IQ_unwind(
        delay_col=f"vna_{name['cable_delay']}",
        data=data,
        include=include,
        exclude=exclude,
        recalc=recalc,
    )
    cable_prefix = f"{prefix['remove_cable']}_{prefix['rotate']}"

    # Fit IQ Circle
    # -------------
    det.targ.mag(include=include, exclude=exclude, recalc=recalc)

    with ccat_mp.optional_executor(max_workers=max(savgol_workers, fit_workers), ex=ex) as ex:
        det.targ.savgol(
            col_name=name["magnitude"],
            deriv=0,
            window=savgol_window,
            k=savgol_order,
            include=include,
            exclude=exclude,
            recalc=recalc,
            max_workers=savgol_workers,
            ex=ex,
        )
        savgol_prefix = f"{prefix['savgol_filter']}0"

        det.IQ_trim(
            prefix=cable_prefix,
            window=trim_window,
            mean_points=trim_mean_points,
            use_fit=False,
            mag_prefix=savgol_prefix,
            include=include,
            exclude=exclude,
            recalc=recalc,
        )
        trim_prefix = f"{prefix['trim_tail']}_{prefix['trim']}_{cable_prefix}"

        det.IQ_circle_fit(
            prefix=trim_prefix,
            include=include,
            exclude=exclude,
            recalc=recalc,
            max_workers=fit_workers,
            ex=ex,
        )
        circle_prefix = f"{prefix['IQ_circle_fit']}_{trim_prefix}"

    # Center IQ Circle
    # ----------------
    det.IQ_circle_origin(
        prefix=cable_prefix,
        circle_fit_prefix=circle_prefix,
        data=data,
        include=include,
        exclude=exclude,
        recalc=recalc,
    )
    center_prefix = f"{prefix['center_origin']}_{prefix['translate']}_{prefix['center_origin']}_{prefix['rotate']}_{cable_prefix}"

    # Remove Impedance Mismatch
    # -------------------------
    centered_df = det.IQ_circle_rotate(
        prefix=center_prefix,
        data=data,
        rotation="mismatch",
        mean_points=mismatch_mean_points,
        include=include,
        exclude=exclude,
        recalc=recalc,
    )

    return centered_df

def phase_to_ff():
    return 

# def IQ_noise(
#     self,
#     prefix: str | list[str] = "mismatch_rotate_origin_shift_origin_rotate",
#     use_noise_tones: bool = True,
#     include: int | list[int] | None = None,
#     exclude: int | list[int] | None = None,
#     recalc: bool = False,
# ) -> pl.DataFrame:
#     """
#     Transform timestream data to isolate readout noise.
#     Will shift nearest frequency noise tone onto detector tone if ``use_noise_tones``, otherwise will rotate detector tone by ninety degrees around its center

#     Args:
#         prefix (str | list[str]):
#         use_noise_tones (bool): Whether to use noise tones. Defaults to True
#     """

#     def _get_medians(prefix):
#         med_col = ["stream_median"]

#         include_subset = ccat_df.check_properties(
#             self,
#             f"{med_col[0]}_{prefix}{'_' if prefix else ''}{col_name[0]}",
#             include=include,
#             exclude=exclude,
#             recalc=recalc,
#         )
#         if not len(include_subset) == 0:
#             median_I_df = self.stream.get_data(
#                 f"{prefix}{'_' if prefix else ''}{col_name[0]}",
#                 include=include_subset,
#                 strict=True,
#             ).select(pl.all().median().name.map(lambda s: s.split("_")[-1]))
#             median_Q_df = self.stream.get_data(
#                 f"{prefix}{'_' if prefix else ''}{col_name[1]}",
#                 include=include_subset,
#                 strict=True,
#             ).select(pl.all().median().name.map(lambda s: s.split("_")[-1]))

#             ccat_df.add_data_to_properties(
#                 self,
#                 median_I_df,
#                 f"{med_col[0]}_{prefix}{'_' if prefix else ''}{col_name[0]}",
#             )
#             ccat_df.add_data_to_properties(
#                 self,
#                 median_Q_df,
#                 f"{med_col[0]}_{prefix}{'_' if prefix else ''}{col_name[1]}",
#             )
#         median_I, median_Q = (
#             self.get_properties(
#                 [
#                     f"{med_col[0]}_{prefix}{'_' if prefix else ''}{col_name[0]}",
#                     f"{med_col[0]}_{prefix}{'_' if prefix else ''}{col_name[1]}",
#                 ],
#                 include=include,
#                 exclude=exclude,
#                 strict=True,
#             )
#             .to_numpy()
#             .T[1:3]
#         )
#         return median_I, median_Q

#     col_name = ["I", "Q"]

#     if isinstance(prefix, str):
#         prefix = [prefix]
#     num_prefix = len(prefix)
#     noise_tones = self.stream.noise_tones

#     if use_noise_tones and noise_tones is not None:  # Use noise tones
#         col_name += ["noise_shift"]

#         include_subset = ccat_df.check_properties(
#             self,
#             "closest_noise_tone",
#             include=include,
#             exclude=exclude,
#             recalc=recalc,
#         )
#         if not len(include_subset) == 0:
#             noise_freqs = (
#                 self.get_properties("tone_freqs", include=noise_tones, strict=True)
#                 .to_numpy()
#                 .T[1]
#             )
#             closest_tones = (
#                 self.get_properties("tone_freqs", include=include_subset, strict=True)
#                 .lazy()
#                 .select(
#                     ["det"]
#                     + [
#                         ((pl.col("tone_freqs") - freq).abs() / pl.lit(1e6)).alias(
#                             f"{tone:0{self.stream.padding}d}"
#                         )
#                         for tone, freq in zip(noise_tones, noise_freqs)
#                     ]
#                 )
#                 .collect()
#                 .unpivot(
#                     index="det",
#                     variable_name="closest_noise_tone",
#                     value_name="dist",
#                 )
#                 .lazy()
#                 .with_columns(pl.col("closest_noise_tone").cast(pl.Int32))
#                 .filter(pl.col("dist") == pl.col("dist").min().over("det"))
#                 .sort("det")
#                 .select(["det", "closest_noise_tone"])
#                 .collect()
#             )
#             shared_cols = (
#                 "closest_noise_tone"
#                 if "closest_noise_tone" in self._properties_df.schema
#                 else []
#             )
#             self._properties_df = ccat_df.coalesce_join(
#                 self._properties_df,
#                 closest_tones,
#                 on="det",
#                 shared_cols=shared_cols,
#             )
#         closest_tones = (
#             self.get_properties(
#                 "closest_noise_tone", include=include, exclude=exclude, strict=True
#             )
#             .to_numpy()
#             .T[1]
#         )
#         noise_median_I, noise_median_Q = _get_medians("")

#         col_names = [[]] * num_prefix
#         median_Is, median_Qs = [[]] * num_prefix, [[]] * num_prefix
#         for i, pre in enumerate(prefix):
#             col_names[i] = col_name[:-1] + [f"{col_name[-1]}{'_' if pre else ''}{pre}"]
#             median_Is[i], median_Qs[i] = _get_medians(pre)
#         args = [
#             [median_I, median_Q, noise_median_I, noise_median_Q, closest_tones]
#             for median_I, median_Q in zip(median_Is, median_Qs)
#         ]
#         self.stream.transform(
#             [Detector.calc_noise_shift] * num_prefix,
#             *args,
#             include=include,
#             exclude=exclude,
#             recalc=recalc,
#             col_name=col_names,
#         )
#         return self.stream.get_data(
#             col_name=[f"{col_name[-1]}_{col_name[0]}" for col_name in col_names]
#             + [f"{col_name[-1]}_{col_name[1]}" for col_name in col_names],
#             include=include,
#             exclude=exclude,
#         )
#     else:
#         col_name += ["noise_rotate"]

#         for pre in prefix:
#             median_I, median_Q = _get_medians(pre)

#             self.stream.IQ_shift(
#                 prefix=pre,
#                 shift_I=-1 * median_I,
#                 shift_Q=-1 * median_Q,
#                 name="",
#                 include=include,
#                 exclude=exclude,
#                 recalc=recalc,
#             )
#             self.stream.IQ_rotate(
#                 prefix=f"shift_{pre}",
#                 angle=np.pi / 2,
#                 name="noise",
#                 include=include,
#                 exclude=exclude,
#                 recalc=recalc,
#             )
#             self.stream.IQ_shift(
#                 prefix=f"noise_rotate_shift_{pre}",
#                 shift_I=median_I,
#                 shift_Q=median_Q,
#                 name="",
#                 include=include,
#                 exclude=exclude,
#                 recalc=recalc,
#             )

#             self.stream.data = self.stream.data.with_columns(
#                 [
#                     pl.col(col).alias(
#                         col.replace("shift_noise_rotate_shift", col_name[-1])
#                     )
#                     for col in self.stream.data.select(
#                         pl.col("^shift_noise_rotate_shift_.*$")
#                     ).columns
#                 ]
#             )
#         return self.stream.get_data(
#             col_name=[f"{col_name[-1]}{'_' if pre else ''}{pre}" for pre in prefix],
#             include=include,
#             exclude=exclude,
#         )
