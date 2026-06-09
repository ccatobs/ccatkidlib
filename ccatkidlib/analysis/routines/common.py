import polars as pl
import numpy as np 

import ccatkidlib.analysis.utils.dataframe as ccat_df

def IQ_noise(
    self,
    prefix: str | list[str] = "mismatch_rotate_origin_shift_origin_rotate",
    use_noise_tones: bool = True,
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

    def _get_medians(prefix):
        med_col = ["stream_median"]

        include_subset = ccat_df.check_properties(
            self,
            f"{med_col[0]}_{prefix}{'_' if prefix else ''}{col_name[0]}",
            include=include,
            exclude=exclude,
            recalc=recalc,
        )
        if not len(include_subset) == 0:
            median_I_df = self.stream.get_data(
                f"{prefix}{'_' if prefix else ''}{col_name[0]}",
                include=include_subset,
                strict=True,
            ).select(pl.all().median().name.map(lambda s: s.split("_")[-1]))
            median_Q_df = self.stream.get_data(
                f"{prefix}{'_' if prefix else ''}{col_name[1]}",
                include=include_subset,
                strict=True,
            ).select(pl.all().median().name.map(lambda s: s.split("_")[-1]))

            ccat_df.add_data_to_properties(
                self,
                median_I_df,
                f"{med_col[0]}_{prefix}{'_' if prefix else ''}{col_name[0]}",
            )
            ccat_df.add_data_to_properties(
                self,
                median_Q_df,
                f"{med_col[0]}_{prefix}{'_' if prefix else ''}{col_name[1]}",
            )
        median_I, median_Q = (
            self.get_properties(
                [
                    f"{med_col[0]}_{prefix}{'_' if prefix else ''}{col_name[0]}",
                    f"{med_col[0]}_{prefix}{'_' if prefix else ''}{col_name[1]}",
                ],
                include=include,
                exclude=exclude,
                strict=True,
            )
            .to_numpy()
            .T[1:3]
        )
        return median_I, median_Q

    col_name = ["I", "Q"]

    if isinstance(prefix, str):
        prefix = [prefix]
    num_prefix = len(prefix)
    noise_tones = self.stream.noise_tones

    if use_noise_tones and noise_tones is not None:  # Use noise tones
        col_name += ["noise_shift"]

        include_subset = ccat_df.check_properties(
            self,
            "closest_noise_tone",
            include=include,
            exclude=exclude,
            recalc=recalc,
        )
        if not len(include_subset) == 0:
            noise_freqs = (
                self.get_properties("tone_freqs", include=noise_tones, strict=True)
                .to_numpy()
                .T[1]
            )
            closest_tones = (
                self.get_properties(
                    "tone_freqs", include=include_subset, strict=True
                )
                .lazy()
                .select(
                    ["det"]
                    + [
                        ((pl.col("tone_freqs") - freq).abs() / pl.lit(1e6)).alias(
                            f"{tone:0{self.stream.padding}d}"
                        )
                        for tone, freq in zip(noise_tones, noise_freqs)
                    ]
                )
                .collect()
                .unpivot(
                    index="det",
                    variable_name="closest_noise_tone",
                    value_name="dist",
                )
                .lazy()
                .with_columns(pl.col("closest_noise_tone").cast(pl.Int32))
                .filter(pl.col("dist") == pl.col("dist").min().over("det"))
                .sort("det")
                .select(["det", "closest_noise_tone"])
                .collect()
            )
            shared_cols = (
                "closest_noise_tone"
                if "closest_noise_tone" in self._properties_df.schema
                else []
            )
            self._properties_df = ccat_df.coalesce_join(
                self._properties_df,
                closest_tones,
                on="det",
                shared_cols=shared_cols,
            )
        closest_tones = (
            self.get_properties(
                "closest_noise_tone", include=include, exclude=exclude, strict=True
            )
            .to_numpy()
            .T[1]
        )
        noise_median_I, noise_median_Q = _get_medians("")

        col_names = [[]] * num_prefix
        median_Is, median_Qs = [[]] * num_prefix, [[]] * num_prefix
        for i, pre in enumerate(prefix):
            col_names[i] = col_name[:-1] + [
                f"{col_name[-1]}{'_' if pre else ''}{pre}"
            ]
            median_Is[i], median_Qs[i] = _get_medians(pre)
        args = [
            [median_I, median_Q, noise_median_I, noise_median_Q, closest_tones]
            for median_I, median_Q in zip(median_Is, median_Qs)
        ]
        self.stream.transform(
            [Detector.calc_noise_shift] * num_prefix,
            *args,
            include=include,
            exclude=exclude,
            recalc=recalc,
            col_name=col_names,
        )
        return self.stream.get_data(
            col_name=[f"{col_name[-1]}_{col_name[0]}" for col_name in col_names]
            + [f"{col_name[-1]}_{col_name[1]}" for col_name in col_names],
            include=include,
            exclude=exclude,
        )
    else:
        col_name += ["noise_rotate"]

        for pre in prefix:
            median_I, median_Q = _get_medians(pre)

            self.stream.IQ_shift(
                prefix=pre,
                shift_I=-1 * median_I,
                shift_Q=-1 * median_Q,
                name="",
                include=include,
                exclude=exclude,
                recalc=recalc,
            )
            self.stream.IQ_rotate(
                prefix=f"shift_{pre}",
                angle=np.pi / 2,
                name="noise",
                include=include,
                exclude=exclude,
                recalc=recalc,
            )
            self.stream.IQ_shift(
                prefix=f"noise_rotate_shift_{pre}",
                shift_I=median_I,
                shift_Q=median_Q,
                name="",
                include=include,
                exclude=exclude,
                recalc=recalc,
            )

            self.stream.data = self.stream.data.with_columns(
                [
                    pl.col(col).alias(
                        col.replace("shift_noise_rotate_shift", col_name[-1])
                    )
                    for col in self.stream.data.select(
                        pl.col("^shift_noise_rotate_shift_.*$")
                    ).columns
                ]
            )
        return self.stream.get_data(
            col_name=[f"{col_name[-1]}{'_' if pre else ''}{pre}" for pre in prefix],
            include=include,
            exclude=exclude,
        )

def IQ_max_dist(
    det,
    trim_window: int = 2,
    trim_savgol_window: int = 9,
    diff_savgol_window: int = 21,
    trim_savgol_k: int = 1,
    diff_savgol_k: int = 1,
    include=None,
    exclude=None,
    recalc=False,
    max_workers: int = 1,
    ex=None,
):
    """
    Get the frequency corresponding to the max geometric distance between adjacent points in IQ space
    """
    trim_savgol, diff_savgol = trim_savgol_window > 1, diff_savgol_window > 1
    # Calculate the geometric distance between adjacent points in IQ space
    det.cable_delay
    det.targ.mag(include=include, exclude=exclude, recalc=recalc)

    if trim_savgol:
        det.targ.savgol(
            col_name="mag",
            prefix="",
            window=trim_savgol_window,
            k=trim_savgol_k,
            deriv=0,
            max_workers=max_workers,
            ex=ex,
            include=include,
            exclude=exclude,
            recalc=recalc,
        )
    det.IQ_unwind(
        prefix="",
        data="targ",
        delay_col="network_cable_delay",
        include=include,
        exclude=exclude,
        recalc=recalc,
    )
    det.IQ_trim(
        prefix="unwind_rotate",
        window=trim_window,
        use_fit=False,
        mag_prefix=f"{'savgol0' if trim_savgol else ''}",
        include=include,
        exclude=exclude,
        recalc=recalc,
    )

    if diff_savgol:
        det.targ.savgol(
            col_name="I",
            prefix="tail_trim_unwind_rotate",
            window=diff_savgol_window,
            k=diff_savgol_k,
            deriv=1,
            include=include,
            exclude=exclude,
            recalc=recalc,
            max_workers=max_workers,
            ex=ex,
        )
        det.targ.savgol(
            col_name="Q",
            prefix="tail_trim_unwind_rotate",
            window=diff_savgol_window,
            k=diff_savgol_k,
            deriv=1,
            include=include,
            exclude=exclude,
            recalc=recalc,
            max_workers=max_workers,
            ex=ex,
        )
    else:
        det.targ.diff(
            col_name="I",
            prefix="tail_trim_unwind_rotate",
            include=include,
            exclude=exclude,
            recalc=recalc,
        )
        det.targ.diff(
            col_name="Q",
            prefix="tail_trim_unwind_rotate",
            include=include,
            exclude=exclude,
            recalc=recalc,
        )
    det.targ.mag(
        prefix=f"{'savgol1' if diff_savgol else 'diff'}_tail_trim_unwind_rotate",
        include=include,
        exclude=exclude,
        recalc=recalc,
    )

    # Get the frequencies corresponding to the max distance in IQ space (same as frequency with steepest phase gradient)
    include_subset = ccat_df.check_properties(
        det, "max_IQ_dist_f", include=include, exclude=exclude, recalc=recalc
    )
    if not len(include_subset) == 0:
        diff_IQ = (
            det.targ.get_data(
                [
                    "sample",
                    f"{'savgol1' if diff_savgol else 'diff'}_tail_trim_unwind_rotate_mag",
                ],
                strict=True,
                include=include_subset,
            )
            .rechunk()
            .lazy()
            .unpivot(index="sample", value_name="IQ", variable_name="temp")
            .drop("temp")
            .collect()
        )
        f = (
            det.targ.get_data(["f"], strict=True, include=include_subset)
            .lazy()
            .unpivot(value_name="f", variable_name="det")
            .with_columns((pl.col("det").str.strip_prefix("f_")).cast(pl.Int32))
            .collect()
        )
        full_df = pl.concat([f, diff_IQ], how="horizontal")
        max_sample = (
            full_df.lazy()
            .filter(~pl.col("IQ").is_nan())
            .filter((pl.col("IQ") == pl.col("IQ").max()).over("det"))
            .select("det", pl.col("sample").alias("max_sample"))
            .group_by("det")
            .agg(pl.col("max_sample").first())
            .collect()
        )

        full_df = full_df.join(max_sample, on="det", how="left")
        max_IQ = (
            full_df.lazy()
            .filter(pl.col("sample") == pl.col("max_sample"))
            .select(
                "det",
                pl.col("sample").alias("max_IQ_dist_sample"),
                pl.col("f").alias("max_IQ_dist_f"),
                pl.col("IQ").alias("max_IQ_dist"),
            )
            .collect()
        )
        adj_IQ = (
            full_df.lazy()
            .filter(pl.col("sample") == pl.col("max_sample") - 1)
            .select("det", pl.col("f").alias("max_IQ_dist_adj_f"))
            .collect()
        )

        max_IQ = max_IQ.join(adj_IQ, on="det", how="left")

        # Use tone frequencies for detectors where finding the max distance frequency failed
        tone_freq_df = det.get_properties(
            "tone_freqs", strict=True, include=include_subset
        )
        max_IQ = (
            max_IQ.join(tone_freq_df, on="det", how="right", coalesce=True)
            .lazy()
            .with_columns(
                pl.when(pl.col("max_IQ_dist_f").is_null())
                .then(pl.col("tone_freqs"))
                .otherwise(pl.col("max_IQ_dist_f"))
                .alias("max_IQ_dist_f")
            )
            .drop("tone_freqs")
            .collect()
        )
        shared_cols = (
            [
                "max_IQ_dist_f",
                "max_IQ_dist",
                "max_IQ_dist_sample",
                "max_IQ_dist_adj_f",
            ]
            if "max_IQ_dist_f" in det._properties_df.schema
            else []
        )
        det._properties_df = ccat_df.coalesce_join(
            det.properties, max_IQ, "det", shared_cols
        )
        det.targ._properties_df = ccat_df.coalesce_join(
            det.targ.properties, max_IQ, "det", shared_cols
        )

    max_IQ_f = det.get_properties(
        "max_IQ_dist_f", include=include, exclude=exclude, strict=True
    )
    return max_IQ_f