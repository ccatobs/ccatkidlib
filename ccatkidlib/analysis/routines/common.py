"""
Common high-level |KID| analysis routines that combine multiple ``Detector`` operations into
single convenience functions.

.. codeauthor:: Darshan Patel <dp649@cornell.edu>
"""

from __future__ import annotations

import polars as pl
import numpy as np

from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, Literal

import ccatkidlib.analysis.utils.dataframe as ccat_df
import ccatkidlib.analysis.utils.multiprocess as ccat_mp
import ccatkidlib.analysis.routines.properties as properties_routines

from ccatkidlib.analysis.utils.config import return_prefix

if TYPE_CHECKING:
    from ccatkidlib.analysis.core.detector import Detector


def IQ_circle_origin(
    det: Detector,
    data: Literal["targ", "timestream", "both"] = "both",
    cable_delay: Literal["vna", "targ"] = "vna",
    normalize: bool = False,
    normalization: Literal["constant", "linear"] = "constant",
    savgol_window: int = 9,
    savgol_order: int = 1,
    cable_trim_window: float = 5,
    fit_trim_window: float = 2,
    mean_points: int = 10,
    include: int | list[int] | None = None,
    exclude: int | list[int] | None = None,
    recalc: bool = False,
    savgol_workers: int = 1,
    circle_fit_workers: int = 1,
    stitch_workers: int = 1,
    linear_fit_workers: int = 1,
    ex: ProcessPoolExecutor | None = None,
) -> pl.DataFrame:
    """Remove cable delay and impedance mismatch from |KID| |I|/|Q| data, centering the |IQ| circle at the origin.

    Applies the following pipeline to the ``Detector`` ``det``:

    1. Compute magnitude (Savitzky-Golay smoothed if ``savgol_window > 1``)
    2. Estimate and remove cable delay
    3. Trim |IQ| data to the resonance window for circle fitting
    4. Fit the |IQ| circle
    5. Shift the circle centre to the origin
    6. Remove impedance mismatch rotation
    7. Optionally normalise by cable gain (``constant`` or ``linear`` method)

    Args:
        det: *ccatkidlib* ``Detector`` object to process
        data (Literal['targ', 'timestream', 'both'], optional): Which data streams to process.
            Defaults to ``'both'``.
        cable_delay (Literal['vna', 'targ'], optional): Source sweep to use for cable-delay
            estimation. Defaults to ``'vna'``.
        normalize (bool, optional): Whether to normalize |IQ| data by cable gain. Defaults to **False**.
        normalization (Literal['constant', 'linear'], optional): Normalisation method when
            ``normalize`` is **True**. Defaults to ``'constant'``.
        savgol_window (int, optional): Window length for Savitzky-Golay smoothing of the
            magnitude. Set to ``1`` to disable smoothing. Defaults to ``9``.
        savgol_order (int, optional): Polynomial order for Savitzky-Golay filter. Defaults to ``1``.
        cable_trim_window (float, optional): Half-window (in units of :math:`f_0 / Q_r`) used
            to trim data for cable-delay estimation. Defaults to ``5``.
        fit_trim_window (float, optional): Half-window (in units of :math:`f_0 / Q_r`) used
            to trim data for circle fitting. Defaults to ``2``.
        mean_points (int, optional): Number of edge points used for mean-based estimates.
            Defaults to ``10``.
        include (int | list[int] | None, optional): |tone| number(s) to include. Defaults to **None**.
        exclude (int | list[int] | None, optional): |tone| number(s) to exclude. Defaults to **None**.
        recalc (bool, optional): Force recalculation of all intermediate steps. Defaults to **False**.
        savgol_workers (int, optional): Worker processes for Savitzky-Golay step. Defaults to ``1``.
        circle_fit_workers (int, optional): Worker processes for circle-fit step. Defaults to ``1``.
        stitch_workers (int, optional): Worker processes for stitch step. Defaults to ``1``.
        linear_fit_workers (int, optional): Worker processes for linear-fit step. Defaults to ``1``.
        ex: Existing *concurrent.futures* ``ProcessPoolExecutor`` to reuse. Defaults to **None**.
    Returns:
        pl.DataFrame: DataFrame with the final processed |I| and |Q| columns (mismatch-corrected,
        and normalised if requested).
    """
    name_dict, prefix_dict = (
        det.analysis_cfg["convention"]["name"],
        det.analysis_cfg["convention"]["prefix"],
    )

    with return_prefix(det.analysis_cfg) as ret_prefix:
        det.targ.mag(include=include, exclude=exclude, recalc=recalc)
        with ccat_mp.optional_executor(
            max_workers=max(
                savgol_workers, circle_fit_workers, linear_fit_workers, stitch_workers
            ),
            ex=ex,
        ) as ex:
            savgol_prefix = ""
            if savgol_window > 1:
                _, savgol_prefix = det.targ.savgol(
                    col_name=name_dict["magnitude"],
                    deriv=0,
                    window=savgol_window,
                    k=savgol_order,
                    include=include,
                    exclude=exclude,
                    recalc=recalc,
                    max_workers=savgol_workers,
                    ex=ex,
                )

            # Remove Cable Delay
            # ------------------
            det.cable_delay(
                mag_prefix=savgol_prefix,
                window=cable_trim_window,
                mean_points=mean_points,
                include=include,
                exclude=exclude,
                recalc=recalc,
                stitch_workers=stitch_workers,
                linear_fit_workers=linear_fit_workers,
                ex=ex,
            )

            _, cable_prefix = det.IQ_unwind(
                delay_col="_".join([cable_delay, name_dict["cable_delay"]]),
                data=data,
                include=include,
                exclude=exclude,
                recalc=recalc,
            )

            _, trim_prefix = det.targ.res_IQ_trim(
                prefix=cable_prefix,
                window=fit_trim_window,
                mean_points=mean_points,
                mag_prefix=savgol_prefix,
                include=include,
                exclude=exclude,
                recalc=recalc,
            )

            _, circle_prefix = det.targ.IQ_circle_fit(
                prefix=trim_prefix,
                include=include,
                exclude=exclude,
                recalc=recalc,
                max_workers=circle_fit_workers,
                ex=ex,
            )

            # Center IQ Circle
            # ----------------
            _, center_prefix = det.IQ_circle_origin(
                prefix=cable_prefix,
                circle_fit_prefix=circle_prefix,
                data=data,
                include=include,
                exclude=exclude,
                recalc=recalc,
            )

            # Remove Impedance Mismatch
            # -------------------------
            output_df, mismatch_prefix = det.IQ_circle_rotate(
                prefix=center_prefix,
                data=data,
                rotation="mismatch",
                mean_points=mean_points,
                include=include,
                exclude=exclude,
                recalc=recalc,
            )
            output_prefix = mismatch_prefix

            # Normalize IQ Circle by Removing Cable Gain
            # ------------------------------------------
            if normalize:
                scale, _ = det.targ.mismatch_dist(
                    prefix=cable_prefix, include=include, exclude=exclude, recalc=recalc
                )
                scale = scale.to_numpy().T[1]

                if normalization == "constant":
                    data_objs = det._get_data_obj(data)
                    output_df = [None] * len(data_objs)
                    for i, data_obj in enumerate(data_objs):
                        output_df[i], output_prefix = data_obj.IQ_scale(
                            prefix=mismatch_prefix,
                            name=prefix_dict["normalize"],
                            scale=1 / scale,
                            include=include,
                            exclude=exclude,
                            recalc=recalc,
                        )
                elif normalization == "linear":
                    det.targ.cable_gain(
                        mag_prefix=savgol_prefix,
                        window=cable_trim_window,
                        mean_points=mean_points,
                        err_fill=scale,
                        include=include,
                        exclude=exclude,
                        recalc=recalc,
                        linear_fit_workers=linear_fit_workers,
                        ex=ex,
                    )

                    output_df, output_prefix = det.IQ_norm(
                        norm_prefix="_".join(
                            [
                                prefix_dict["linear_fit"],
                                prefix_dict["trim_resonance"],
                                prefix_dict["trim"],
                            ]
                        ),
                        prefix=mismatch_prefix,
                        data=data,
                        include=include,
                        exclude=exclude,
                        recalc=recalc,
                    )
                else:
                    error = "Invalid 'normalization' specified, not normalizing IQ data. Must be one of 'constant' or 'linear'."
        return output_df, (
            output_prefix,
            mismatch_prefix,
            center_prefix,
            circle_prefix,
            trim_prefix,
            cable_prefix,
            savgol_prefix,
        ) if ret_prefix else output_df


def phase_to_ff(
    det: Detector,
    prefix: str | list[str] = "",
    spline_prefix: str = "",
    spline_bounds: float = 1,
    spline_k: int = 2,
    spline_workers: int = 1,
    interp_workers: int = 1,
    ref_f: str = "",
    include: int | list[int] | None = None,
    exclude: int | list[int] | None = None,
    recalc: bool = False,
    ex: ProcessPoolExecutor | None = None,
) -> pl.DataFrame:
    """Convert |KID| phase timestream data to fractional frequency.

    Builds a cubic spline of ``phase`` vs ``frequency`` from the target sweep and
    timestream phase data, interpolates the timestream phase onto the spline to get
    instantaneous frequency, then computes fractional frequency
    :math:`\\delta f / f_0`.

    Args:
        det: *ccatkidlib* ``Detector`` object to process
        prefix (str | list[str], optional): Prefix of the timestream |IQ| data to use.
            Defaults to ``''``.
        spline_prefix (str, optional): Prefix of the phase data used to build the spline.
            Defaults to ``''``.
        spline_bounds (float, optional): Extension beyond the min/max observed phase when
            setting spline boundary conditions. Defaults to ``1``.
        spline_k (int, optional): Polynomial degree of the spline. Defaults to ``2``.
        spline_workers (int, optional): Worker processes for spline fitting. Defaults to ``1``.
        interp_workers (int, optional): Worker processes for spline interpolation. Defaults to ``1``.
        ref_f (str, optional): Reference frequency column name for fractional frequency
            calculation. Defaults to ``''``.
        include (int | list[int] | None, optional): |tone| number(s) to include. Defaults to **None**.
        exclude (int | list[int] | None, optional): |tone| number(s) to exclude. Defaults to **None**.
        recalc (bool, optional): Force recalculation. Defaults to **False**.
        ex: Existing *concurrent.futures* ``ProcessPoolExecutor`` to reuse. Defaults to **None**.
    Returns:
        pl.DataFrame: DataFrame with the fractional frequency column(s) added.
    """

    name_dict, prefix_dict = (
        det.analysis_cfg["convention"]["name"],
        det.analysis_cfg["convention"]["prefix"],
    )

    with return_prefix(det.analysis_cfg) as ret_prefix:
        det.targ.phase(
            prefix=spline_prefix, include=include, exclude=exclude, recalc=recalc
        )
        det.stream.phase(
            prefix=spline_prefix, include=include, exclude=exclude, recalc=recalc
        )
        det.stream.phase(prefix=prefix, include=include, exclude=exclude, recalc=recalc)

        min_phase, max_phase = (
            det.stream.agg(
                "min",
                name_dict["phase"],
                prefix=spline_prefix,
                include=include,
                exclude=exclude,
                recalc=recalc,
            )
            .to_numpy()
            .T[1],
            det.stream.agg(
                "max",
                name_dict["phase"],
                prefix=spline_prefix,
                include=include,
                exclude=exclude,
                recalc=recalc,
            )
            .to_numpy()
            .T[1],
        )

        with ccat_mp.optional_executor(
            max_workers=max(spline_workers, interp_workers), ex=ex
        ) as ex:
            _, phase_spline_prefix = det.phase_spline(
                prefix=spline_prefix,
                phase_low=min_phase - spline_bounds,
                phase_up=max_phase + spline_bounds,
                k=spline_k,
                include=include,
                exclude=exclude,
                recalc=recalc,
                max_workers=spline_workers,
                ex=ex,
            )

            _, f_prefix = det.phase_to_f(
                prefix=prefix,
                spline_prefix=spline_prefix,
                include=include,
                exclude=exclude,
                recalc=recalc,
                max_workers=interp_workers,
                ex=ex,
            )

        frac_f_df, ff_prefix = det.frac_f(
            prefix=prefix,
            ref_f=ref_f,
            include=include,
            exclude=exclude,
            recalc=recalc,
        )

        return frac_f_df, (
            ff_prefix,
            f_prefix,
            phase_spline_prefix,
        ) if ret_prefix else frac_f_df


def IQ_noise(
    det: Detector,
    prefix: str | list[str] = "mismatch_rotate_origin_shift_origin_rotate",
    use_noise_tones: bool = False,
    include: int | list[int] | None = None,
    exclude: int | list[int] | None = None,
    recalc: bool = False,
) -> pl.DataFrame:
    """Transform timestream data to isolate the |KID| readout noise component.

    Shifts the timestream |IQ| data so the circle centre sits at the origin,
    rotates by ninety degrees around that centre to place the noise direction along |I|,
    then shifts back — isolating the dissipation-noise axis from the frequency-noise axis.
    If ``use_noise_tones`` is **True** and noise tones are available, the nearest noise |tone|
    is shifted onto the detector |tone| instead (not yet implemented).

    Args:
        det: *ccatkidlib* ``Detector`` object to process
        prefix (str | list[str], optional): Prefix(es) of the timestream |IQ| data columns to
            use. Defaults to ``'mismatch_rotate_origin_shift_origin_rotate'``.
        use_noise_tones (bool, optional): Whether to use noise tones. Defaults to **False**.
        include (int | list[int] | None, optional): |tone| number(s) to include. Defaults to **None**.
        exclude (int | list[int] | None, optional): |tone| number(s) to exclude. Defaults to **None**.
        recalc (bool, optional): Force recalculation. Defaults to **False**.
    Returns:
        pl.DataFrame: DataFrame with the noise-isolated |I| and |Q| columns added.
    """
    name_dict, prefix_dict = (
        det.analysis_cfg["convention"]["name"],
        det.analysis_cfg["convention"]["prefix"],
    )

    if isinstance(prefix, str):
        prefix = [prefix]
    num_prefix = len(prefix)
    use_noise_tones = ccat_df.check_args(use_noise_tones, num_prefix, bool)

    with return_prefix(det.analysis_cfg) as ret_prefix:
        noise_tones = det.stream.noise_tones

        noise_names = [""] * num_prefix
        for i, (pre, use_noise_tone) in enumerate(zip(prefix, use_noise_tones)):
            if use_noise_tone and noise_tones is not None:  # Use noise tones
                print("Not yet implemented")
                # col_name += ["noise_shift"]

                # include_subset = ccat_df.check_properties(
                #     self,
                #     "closest_noise_tone",
                #     include=include,
                #     exclude=exclude,
                #     recalc=recalc,
                # )
                # if not len(include_subset) == 0:
                #     noise_freqs = (
                #         self.get_properties("tone_freqs", include=noise_tones, strict=True)
                #         .to_numpy()
                #         .T[1]
                #     )
                #     closest_tones = (
                #         self.get_properties("tone_freqs", include=include_subset, strict=True)
                #         .lazy()
                #         .select(
                #             ["det"]
                #             + [
                #                 ((pl.col("tone_freqs") - freq).abs() / pl.lit(1e6)).alias(
                #                     f"{tone:0{self.stream.padding}d}"
                #                 )
                #                 for tone, freq in zip(noise_tones, noise_freqs)
                #             ]
                #         )
                #         .collect()
                #         .unpivot(
                #             index="det",
                #             variable_name="closest_noise_tone",
                #             value_name="dist",
                #         )
                #         .lazy()
                #         .with_columns(pl.col("closest_noise_tone").cast(pl.Int32))
                #         .filter(pl.col("dist") == pl.col("dist").min().over("det"))
                #         .sort("det")
                #         .select(["det", "closest_noise_tone"])
                #         .collect()
                #     )
                #     shared_cols = (
                #         "closest_noise_tone"
                #         if "closest_noise_tone" in self._properties_df.schema
                #         else []
                #     )
                #     self._properties_df = ccat_df.coalesce_join(
                #         self._properties_df,
                #         closest_tones,
                #         on="det",
                #         shared_cols=shared_cols,
                #     )
                # closest_tones = (
                #     self.get_properties(
                #         "closest_noise_tone", include=include, exclude=exclude, strict=True
                #     )
                #     .to_numpy()
                #     .T[1]
                # )
                # noise_median_I, noise_median_Q = _get_medians("")

                # col_names = [[]] * num_prefix
                # median_Is, median_Qs = [[]] * num_prefix, [[]] * num_prefix
                # for i, pre in enumerate(prefix):
                #     col_names[i] = col_name[:-1] + [f"{col_name[-1]}{'_' if pre else ''}{pre}"]
                #     median_Is[i], median_Qs[i] = _get_medians(pre)
                # args = [
                #     [median_I, median_Q, noise_median_I, noise_median_Q, closest_tones]
                #     for median_I, median_Q in zip(median_Is, median_Qs)
                # ]
                # self.stream.transform(
                #     [Detector.calc_noise_shift] * num_prefix,
                #     *args,
                #     include=include,
                #     exclude=exclude,
                #     recalc=recalc,
                #     col_name=col_names,
                # )
                # return self.stream.get_data(
                #     col_name=[f"{col_name[-1]}_{col_name[0]}" for col_name in col_names]
                #     + [f"{col_name[-1]}_{col_name[1]}" for col_name in col_names],
                #     include=include,
                #     exclude=exclude,
                # )
            else:
                noise_prefix = f"{prefix_dict['isolate_dissipation_noise']}_{prefix_dict['rotate']}{'_' if pre else ''}{pre}"
                noise_names[i] = noise_prefix
                median_I, median_Q = (
                    det.stream.agg(
                        "median",
                        name_dict["in_phase"],
                        prefix=pre,
                        include=include,
                        exclude=exclude,
                        recalc=recalc,
                    )
                    .to_numpy()
                    .T[1],
                    det.stream.agg(
                        "median",
                        name_dict["quadrature"],
                        prefix=pre,
                        include=include,
                        exclude=exclude,
                        recalc=recalc,
                    )
                    .to_numpy()
                    .T[1],
                )

                _, shift_prefix = det.stream.IQ_shift(
                    prefix=pre,
                    shift_I=-1 * median_I,
                    shift_Q=-1 * median_Q,
                    name="tmp",
                    include=include,
                    exclude=exclude,
                    recalc=recalc,
                )
                _, rotate_prefix = det.stream.IQ_rotate(
                    prefix=shift_prefix,
                    angle=np.pi / 2,
                    name="tmp",
                    include=include,
                    exclude=exclude,
                    recalc=recalc,
                )
                _, final_prefix = det.stream.IQ_shift(
                    prefix=rotate_prefix,
                    shift_I=median_I,
                    shift_Q=median_Q,
                    name="tmp",
                    include=include,
                    exclude=exclude,
                    recalc=recalc,
                )

                det.stream.data = det.stream.data.with_columns(
                    [
                        pl.col(col).alias(col.replace(final_prefix, noise_prefix))
                        for col in det.stream.data.select(
                            pl.col(f"^{final_prefix}.*$")
                        ).columns
                    ]
                ).drop(pl.col(f"^tmp.*$"))

        col_names = []
        for noise_name in noise_names:
            col_names += [
                "_".join([noise_name, name_dict["in_phase"]]),
                "_".join([noise_name, name_dict["quadrature"]]),
            ]

        ret_df = det.stream.get_data(
            col_name=col_names,
            include=include,
            exclude=exclude,
        )

        ret_pre = noise_names
        if len(ret_pre) == 1:
            ret_pre = ret_pre[0]

        return ret_df, ret_pre if ret_prefix else ret_df
