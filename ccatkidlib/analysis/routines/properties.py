import polars as pl
import numpy as np

import ccatkidlib.analysis.utils.dataframe as ccat_df
import ccatkidlib.log as log

from ccatkidlib.analysis.core.timestream import Timestream


def is_bifurcated(
    targ,
    prefix="",
    bifurcation_threshold=60,
    qifurcation_threshold=50,
    trim_window: int = 2,
    trim_savgol_window: int = 9,
    trim_savgol_k: int = 1,
    include=None,
    exclude=None,
    recalc=False,
    max_workers=1,
    ex=None,
):
    """
    Determine if a detector is bifurcated or has a high quasiparticle nonlinearity using the angle of the maximally seperated points in IQ space


    """

    name_enums, prefix_enums = ccat_df.create_enums(
    [
        "in_phase",
        "quadrature",
        "magnitude",
    ],
    prefix,
    ["remove_impedance_mismatch"],
    targ.analysis_cfg,
    no_prefix=[],
    )

    name_enum, prefix_enum = name_enums[0], prefix_enums[0]

    property_routines.mismatch_dist(det.targ, prefix=_mismatch_prefix)
    I_shifts = det.targ.properties[f"{_prefix['remove_impedance_mismatch']}_{_mismatch_prefix}_{_name['magnitude']}"]

    det.targ.IQ_shift(prefix=_mismatch_prefix, shift_I = I_shifts, name='anchor')

    det.IQ_trim(prefix=f"anchor_shift_{_mismatch_prefix}", window=2, mag_prefix=f"{_prefix['savgol_filter']}0")
    det.targ.linewidth_shift(prefix=f"tail_trim_anchor_shift_{_mismatch_prefix}")
    _fwhm_q = det.targ.properties.select((pl.col('mid_savgol0_FWHM_f')/ (pl.col('high_savgol0_FWHM_f') -pl.col('low_savgol0_FWHM_f'))).alias('FWHM_Q')).to_numpy().T[0]
    _qs = det.targ.properties.select(pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q')).to_numpy().T[0]
    det.targ.scale(col_name='lw', prefix=f"tail_trim_anchor_shift_{_mismatch_prefix}", scale=10e5/_fwhm_q, name='FWHM')
    det.targ.scale(col_name='lw', prefix=f"tail_trim_anchor_shift_{_mismatch_prefix}", scale=10e5/_qs, name='Q')

    det.targ.diff(col_name='lw', prefix=f"Q_scale_tail_trim_anchor_shift_{_mismatch_prefix}")
    det.targ.diff(col_name='lw', prefix=f"FWHM_scale_tail_trim_anchor_shift_{_mismatch_prefix}")

    # Get frequency corresponding to the |S_21| minimum
    # -------------------------------------------------
    self.mag_min(include=include, exclude=exclude, recalc=recalc)

    include_subset = ccat_df.check_properties(
        self, "bifurcated", include=include, exclude=exclude, recalc=recalc
    )

    added_cols = [
        "bifurcated",
        "qifurcated",
        "sin_half_max_IQ_angle",
        "chord_length_ratio",
        "max_IQ_angle_rad",
        "max_IQ_angle_deg",
    ]
    if not len(include_subset) == 0:
        df = self.get_properties(
            [
                "max_IQ_dist",
                "max_IQ_dist_f",
                "max_IQ_dist_adj_f",
                "circle_fit_tail_trim_unwind_rotate_R",
                "min_mag_f",
            ],
            include=include_subset,
            strict=True,
        )

        bif_df = (
            df.lazy()
            .with_columns(
                (
                    0.5
                    * (
                        pl.col("max_IQ_dist")
                        / pl.col("circle_fit_tail_trim_unwind_rotate_R")
                    )
                ).alias("sin_half_max_IQ_angle"),
                (
                    0.5
                    * (
                        4
                        - (
                            pl.col("max_IQ_dist")
                            / pl.col("circle_fit_tail_trim_unwind_rotate_R")
                        )
                        ** 2
                    ).sqrt()
                ).alias("chord_length_ratio"),
            )
            .with_columns(
                (2 * pl.col("sin_half_max_IQ_angle").arcsin()).alias("max_IQ_angle_rad")
            )
            .with_columns(
                ((180 / np.pi) * pl.col("max_IQ_angle_rad")).alias("max_IQ_angle_deg")
            )
            .with_columns(
                (
                    (pl.col("max_IQ_angle_deg") >= bifurcation_threshold)
                    & (pl.col("max_IQ_dist_adj_f") < pl.col("min_mag_f"))
                ).alias("bifurcated"),
                (
                    (pl.col("max_IQ_angle_deg") >= qifurcation_threshold)
                    & (pl.col("max_IQ_dist_f") > pl.col("min_mag_f"))
                ).alias("qifurcated"),
            )
            .select(["det"] + added_cols)
            .collect()
        )
        shared_cols = added_cols if "bifurcated" in self._properties_df.schema else []
        self._properties_df = ccat_df.coalesce_join(
            self.properties, bif_df, "det", shared_cols
        )

    bif_df = self.get_properties(
        ["bifurcated", "qifurcated"], include=include, exclude=exclude, strict=True
    )
    return bif_df


# def is_bifurcated(
#     self,
#     bifurcation_threshold=60,
#     qifurcation_threshold=50,
#     trim_window: int = 2,
#     trim_savgol_window: int = 9,
#     trim_savgol_k: int = 1,
#     include=None,
#     exclude=None,
#     recalc=False,
#     max_workers=1,
#     ex=None,
# ):
#     """
#     Determine if a detector is bifurcated or has a high quasiparticle nonlinearity using the angle of the maximally seperated points in IQ space


#     """

#     # Get maximally distant points in IQ space
#     # ----------------------------------------
#     self.IQ_max_dist(
#         diff_savgol_window=1,
#         trim_window=trim_window,
#         trim_savgol_window=trim_savgol_window,
#         trim_savgol_k=trim_savgol_k,
#         include=include,
#         exclude=exclude,
#         recalc=recalc,
#         max_workers=max_workers,
#         ex=ex,
#     )

#     # Fit IQ circle to get radius
#     # ---------------------------
#     self.IQ_circle_fit(
#         prefix="tail_trim_unwind_rotate",
#         include=include,
#         exclude=exclude,
#         recalc=recalc,
#         max_workers=max_workers,
#         ex=ex,
#     )

#     # Get frequency corresponding to the |S_21| minimum
#     # -------------------------------------------------
#     self.mag_min(include=include, exclude=exclude, recalc=recalc)

#     include_subset = ccat_df.check_properties(
#         self, "bifurcated", include=include, exclude=exclude, recalc=recalc
#     )

#     added_cols = [
#         "bifurcated",
#         "qifurcated",
#         "sin_half_max_IQ_angle",
#         "chord_length_ratio",
#         "max_IQ_angle_rad",
#         "max_IQ_angle_deg",
#     ]
#     if not len(include_subset) == 0:
#         df = self.get_properties(
#             [
#                 "max_IQ_dist",
#                 "max_IQ_dist_f",
#                 "max_IQ_dist_adj_f",
#                 "circle_fit_tail_trim_unwind_rotate_R",
#                 "min_mag_f",
#             ],
#             include=include_subset,
#             strict=True,
#         )

#         bif_df = (
#             df.lazy()
#             .with_columns(
#                 (
#                     0.5
#                     * (
#                         pl.col("max_IQ_dist")
#                         / pl.col("circle_fit_tail_trim_unwind_rotate_R")
#                     )
#                 ).alias("sin_half_max_IQ_angle"),
#                 (
#                     0.5
#                     * (
#                         4
#                         - (
#                             pl.col("max_IQ_dist")
#                             / pl.col("circle_fit_tail_trim_unwind_rotate_R")
#                         )
#                         ** 2
#                     ).sqrt()
#                 ).alias("chord_length_ratio"),
#             )
#             .with_columns(
#                 (2 * pl.col("sin_half_max_IQ_angle").arcsin()).alias("max_IQ_angle_rad")
#             )
#             .with_columns(
#                 ((180 / np.pi) * pl.col("max_IQ_angle_rad")).alias("max_IQ_angle_deg")
#             )
#             .with_columns(
#                 (
#                     (pl.col("max_IQ_angle_deg") >= bifurcation_threshold)
#                     & (pl.col("max_IQ_dist_adj_f") < pl.col("min_mag_f"))
#                 ).alias("bifurcated"),
#                 (
#                     (pl.col("max_IQ_angle_deg") >= qifurcation_threshold)
#                     & (pl.col("max_IQ_dist_f") > pl.col("min_mag_f"))
#                 ).alias("qifurcated"),
#             )
#             .select(["det"] + added_cols)
#             .collect()
#         )
#         shared_cols = added_cols if "bifurcated" in self._properties_df.schema else []
#         self._properties_df = ccat_df.coalesce_join(
#             self.properties, bif_df, "det", shared_cols
#         )

#     bif_df = self.get_properties(
#         ["bifurcated", "qifurcated"], include=include, exclude=exclude, strict=True
#     )
#     return bif_df
