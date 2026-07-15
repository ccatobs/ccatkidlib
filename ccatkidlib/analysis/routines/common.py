import polars as pl
import numpy as np


import ccatkidlib.analysis.utils.dataframe as ccat_df
import ccatkidlib.analysis.utils.multiprocess as ccat_mp
import ccatkidlib.analysis.routines.properties as properties_routines


def IQ_circle_center(
    det,
    data="both",
    normalize=False,
    savgol_window=9,
    savgol_order=1,
    trim_window=2,
    trim_mean_points=10,
    mismatch_mean_points=10,
    include=None,
    exclude=None,
    recalc=False,
    savgol_workers=1,
    circle_fit_workers=1,
    cable_fit_workers=1,
    ex=None,
):
    name_dict, prefix_dict = (
        det.analysis_cfg["convention"]["name"],
        det.analysis_cfg["convention"]["prefix"],
    )

    # Remove Cable Delay
    # ------------------
    det.vna.phase(recalc=recalc)
    det.cable_delay
    det.IQ_unwind(
        delay_col=f"vna_{name_dict['cable_delay']}",
        data=data,
        include=include,
        exclude=exclude,
        recalc=recalc,
    )
    cable_prefix = f"{prefix_dict['remove_cable']}_{prefix_dict['rotate']}"

    # Fit IQ Circle
    # -------------
    det.targ.mag(include=include, exclude=exclude, recalc=recalc)

    with ccat_mp.optional_executor(max_workers=max(savgol_workers, circle_fit_workers, cable_fit_workers), ex=ex) as ex:
        det.targ.savgol(
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
        savgol_prefix = f"{prefix_dict['savgol_filter']}0"

        if normalize:
            det.complex_fit(include=include,
                            exclude=exclude,
                            recalc=recalc,
                            max_workers=cable_fit_workers,
                            ex=ex)
            complex_fit_cable = f"{prefix_dict['complex_fit_cable']}_{prefix_dict['complex_fit']}"

            det.targ.mag(prefix=complex_fit_cable, dB=False, include=include, exclude=exclude, recalc=recalc)
            det.IQ_norm(norm_prefix=complex_fit_cable, prefix=cable_prefix, data='both', include=include, exclude=exclude, recalc=recalc)
            cable_prefix = f"{prefix_dict['normalize']}_{prefix_dict['scale']}_{cable_prefix}"


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
        trim_prefix = f"{prefix_dict['trim_tail']}_{prefix_dict['trim']}_{cable_prefix}"

        det.IQ_circle_fit(
            prefix=trim_prefix,
            include=include,
            exclude=exclude,
            recalc=recalc,
            max_workers=circle_fit_workers,
            ex=ex,
        )
        circle_prefix = f"{prefix_dict['IQ_circle_fit']}_{trim_prefix}"

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
    center_prefix = f"{prefix_dict['center_origin']}_{prefix_dict['translate']}_{prefix_dict['center_origin']}_{prefix_dict['rotate']}_{cable_prefix}"

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

def phase_to_ff(det,
                prefix: str | list[str] = '',
                spline_prefix='',
                spline_bounds = 1,
                spline_k = 2,
                spline_workers = 1, 
                interp_workers = 1,
                ref_f = '',
                include=None,
                exclude=None,
                recalc=False,
                ex=None):

        name_dict, prefix_dict = (
            det.analysis_cfg["convention"]["name"],
            det.analysis_cfg["convention"]["prefix"],
        )

        det.targ.phase(prefix=spline_prefix, include=include, exclude=exclude, recalc=recalc)
        det.stream.phase(prefix=spline_prefix, include=include, exclude=exclude, recalc=recalc)
        det.stream.phase(prefix=prefix, include=include, exclude=exclude, recalc=recalc)

        min_phase, max_phase = (
            properties_routines.agg(det.stream, "min", name_dict['phase'], prefix=spline_prefix, include=include, exclude=exclude, recalc=recalc)
            .to_numpy()
            .T[1],
            properties_routines.agg(det.stream, "max", name_dict['phase'], prefix=spline_prefix, include=include, exclude=exclude, recalc=recalc)
            .to_numpy()
            .T[1],
        )

        with ccat_mp.optional_executor(max_workers=max(spline_workers, interp_workers), ex=ex) as ex:
            det.phase_spline(
                prefix=spline_prefix,
                phase_low=min_phase - spline_bounds,
                phase_up=max_phase + spline_bounds,
                k=spline_k,
                include=include,
                exclude=exclude,
                recalc=recalc,
                max_workers=spline_workers,
                ex=ex
            )

            det.phase_to_f(
                prefix=prefix,
                spline_prefix=spline_prefix,
                include=include,
                exclude=exclude,
                recalc=recalc,
                max_workers=interp_workers,
                ex=ex
            )

        frac_f_df = det.frac_f(
            prefix=prefix,
            ref_f=ref_f,
            include=include,
            exclude=exclude,
            recalc=recalc,
        )

        return frac_f_df

def IQ_noise(
    det,
    prefix: str | list[str] = "mismatch_rotate_origin_shift_origin_rotate",
    use_noise_tones: bool = False,
    include: int | list[int] | None = None,
    exclude: int | list[int] | None = None,
    recalc: bool = False,
) -> pl.DataFrame:
    """
    Transform timestream data to isolate readout noise.
    Will shift nearest frequency noise tone onto detector tone if ``use_noise_tones``, otherwise will rotate detector tone by ninety degrees around its center

    Args:
        prefix (str | list[str]):
        use_noise_tones (bool): Whether to use noise tones. Defaults to True
    """
    name_dict, prefix_dict = (
        det.analysis_cfg["convention"]["name"],
        det.analysis_cfg["convention"]["prefix"],
    )

    if isinstance(prefix, str):
        prefix = [prefix]
    num_prefix = len(prefix)
    use_noise_tones = ccat_df.check_args(use_noise_tones, num_prefix, bool)

    noise_tones = det.stream.noise_tones
    noise_names = ['']*num_prefix
    for i, (pre, use_noise_tone) in enumerate(zip(prefix, use_noise_tones)):
        if use_noise_tone and noise_tones is not None:  # Use noise tones
            print('Not yet implemented')
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
            #)
        else:
            noise_prefix = f'{prefix_dict['isolate_readout_noise']}_{prefix_dict['rotate']}_{pre}'
            noise_names[i] = noise_prefix
            median_I, median_Q = (
                properties_routines.agg(det.stream, "median", name_dict['in_phase'], prefix=pre, include=include, exclude=exclude, recalc=recalc)
                .to_numpy()
                .T[1],
                properties_routines.agg(det.stream, "median", name_dict['quadrature'], prefix=pre, include=include, exclude=exclude, recalc=recalc)
                .to_numpy()
                .T[1],
            )

            det.stream.IQ_shift(
                prefix=pre,
                shift_I=-1 * median_I,
                shift_Q=-1 * median_Q,
                name="tmp",
                include=include,
                exclude=exclude,
                recalc=recalc,
            )
            shift_prefix = f"tmp_{prefix_dict['translate']}_{pre}"
            det.stream.IQ_rotate(
                prefix=shift_prefix,
                angle=np.pi / 2,
                name="tmp",
                include=include,
                exclude=exclude,
                recalc=recalc,
            )
            rotate_prefix = f'tmp_{prefix_dict['rotate']}_{shift_prefix}'
            det.stream.IQ_shift(
                prefix=rotate_prefix,
                shift_I=median_I,
                shift_Q=median_Q,
                name="tmp",
                include=include,
                exclude=exclude,
                recalc=recalc,
            )
            final_prefix = f'tmp_{prefix_dict['translate']}_{rotate_prefix}'

            det.stream.data = det.stream.data.with_columns(
                [
                    pl.col(col).alias(
                        col.replace(final_prefix, noise_prefix)
                    )
                    for col in det.stream.data.select(
                        pl.col(f"^{final_prefix}.*$")
                    ).columns
                ]
            )
        
        col_names = []
        for noise_name in noise_names:
            col_names += [
                f"{noise_name}_{name_dict['in_phase']}",
                f"{noise_name}_{name_dict['quadrature']}",
            ]

        return det.stream.get_data(
            col_name=col_names,
            include=include,
            exclude=exclude,
        )
