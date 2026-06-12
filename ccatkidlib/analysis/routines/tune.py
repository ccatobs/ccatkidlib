import polars as pl
import numpy as np

import ccatkidlib.analysis.utils.multiprocess as ccat_mp
import ccatkidlib.analysis.utils.dataframe as ccat_df

from ccatkidlib.analysis.core.network import Network
# Tone Power Tuning
# -----------------


class TonePowerNetwork(Network):
    def __init__(self, com_to, **kwargs):
        super.__init__(com_to, **kwargs)

    def drive_to_power():
        """
        Convert drive attenuations to RFSoC tone powers
        """

        return


# Tone Frequency Tuning
# ---------------------

def max_IQ_dist_f(
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

    det.vna.phase()
    det.cable_delay

    name, prefix = (
        det.analysis_cfg["convention"]["name"],
        det.analysis_cfg["convention"]["prefix"],
    )

    # Apply Savgol filter
    # -------------------
    with ccat_mp.optional_executor(max_workers=max_workers, ex=ex) as ex:
        det.targ.mag(include=include, exclude=exclude, recalc=recalc)
        if trim_savgol:
            det.targ.savgol(
                col_name=name["magnitude"],
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

        # Remove cable delay and trim tails
        det.IQ_unwind(
            prefix="",
            data="targ",
            delay_col=f"vna_{name['cable_delay']}",
            include=include,
            exclude=exclude,
            recalc=recalc,
        )
        cable_prefix = f"{prefix['remove_cable']}_{prefix['rotate']}"
        savgol_prefix = f"{prefix['savgol_filter']}0"

        det.IQ_trim(
            prefix=cable_prefix,
            window=trim_window,
            use_fit=False,
            mag_prefix=savgol_prefix if trim_savgol else "",
            include=include,
            exclude=exclude,
            recalc=recalc,
        )
        trim_prefix = f"{prefix['trim_tail']}_{prefix['trim']}_{cable_prefix}"

        if diff_savgol:
            det.targ.savgol(
                col_name=name["in_phase"],
                prefix=trim_prefix,
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
                col_name=name["quadrature"],
                prefix=trim_prefix,
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
                col_name=name["in_phase"],
                prefix=trim_prefix,
                include=include,
                exclude=exclude,
                recalc=recalc,
            )
            det.targ.diff(
                col_name=name["quadrature"],
                prefix=trim_prefix,
                include=include,
                exclude=exclude,
                recalc=recalc,
            )

    deriv_savgol_prefix, diff_prefix = (
        f"{prefix['savgol_filter']}1",
        prefix["difference"],
    )
    det.targ.mag(
        prefix=f"{deriv_savgol_prefix if diff_savgol else diff_prefix}_{trim_prefix}",
        include=include,
        exclude=exclude,
        recalc=recalc,
    )

    # Get the frequencies corresponding to the max distance in IQ space (same as frequency with steepest phase gradient)
    # ------------------------------------------------------------------------------------------------------------------
    sample_col, freq_col, dist_col = name["sample"], name["frequency"], name["distance"]
    include_subset = ccat_df.check_properties(
        det,
        f"high_max_{dist_col}_{freq_col}",
        include=include,
        exclude=exclude,
        recalc=recalc,
    )
    if not len(include_subset) == 0:
        # Get samples, frequencies, and IQ distances in long format DataFrame
        IQ_dist_df = (
            det.targ.get_data(
                [
                    sample_col,
                    f"{deriv_savgol_prefix if diff_savgol else diff_prefix}_{trim_prefix}_{name['magnitude']}",
                ],
                strict=True,
                include=include_subset,
            )
            .rechunk()
            .lazy()
            .unpivot(index=sample_col, value_name="IQ_dist", variable_name="temp")
            .drop("temp")
            .collect()
        )
        f_df = (
            det.targ.get_data([freq_col], strict=True, include=include_subset)
            .lazy()
            .unpivot(value_name=freq_col, variable_name="det")
            .with_columns(
                (pl.col("det").str.strip_prefix(f"{freq_col}_")).cast(pl.Int32)
            )  # Extract detector IDs from column names
            .collect()
        )
        full_df = pl.concat([f_df, IQ_dist_df], how="horizontal")

        # Calculate samples & frequencies corresponding to max IQ distance
        # ----------------------------------------------------------------
        max_sample = (
            full_df.lazy()
            .filter(~pl.col("IQ_dist").is_nan())
            .filter((pl.col("IQ_dist") == pl.col("IQ_dist").max()).over("det"))
            .select("det", pl.col(sample_col).name.prefix("max_"))
            .group_by("det")
            .agg(pl.col(f"max_{sample_col}").first())
            .collect()
        )

        full_df = full_df.join(max_sample, on="det", how="left")

        # Get both points used to calculate distance
        low_max_IQ = (
            full_df.lazy()
            .filter(pl.col(sample_col) == pl.col(f"max_{sample_col}"))
            .select(
                "det",
                pl.col(sample_col).alias(f"high_max_{dist_col}_{sample_col}"),
                pl.col(freq_col).alias(f"high_max_{dist_col}_{freq_col}"),
                pl.col("IQ_dist").alias(f"max_{dist_col}"),
            )
            .collect()
        )
        high_max_IQ = (
            full_df.lazy()
            .filter(pl.col(sample_col) == pl.col(f"max_{sample_col}") - 1)
            .select(
                "det",
                pl.col(sample_col).alias(f"low_max_{dist_col}_{sample_col}"),
                pl.col(freq_col).alias(f"low_max_{dist_col}_{freq_col}"),
            )
            .collect()
        )

        max_IQ = low_max_IQ.join(high_max_IQ, on="det", how="left")

        # Use tone frequencies for detectors where finding the max distance frequency failed
        # ----------------------------------------------------------------------------------
        tone_freq_col = name["tone_frequency"]
        tone_freq_df = det.get_properties(
            tone_freq_col, strict=True, include=include_subset
        )

        is_null = pl.when(pl.col(f"high_max_{dist_col}_{freq_col}").is_null())
        max_IQ = (
            max_IQ.join(tone_freq_df, on="det", how="right", coalesce=True)
            .lazy()
            .with_columns(
                is_null.then(pl.col(tone_freq_col))
                .otherwise(pl.col(f"high_max_{dist_col}_{freq_col}"))
                .alias(f"high_max_{dist_col}_{freq_col}"),
                is_null.then(pl.col(tone_freq_col))
                .otherwise(pl.col(f"low_max_{dist_col}_{freq_col}"))
                .alias(f"low_max_{dist_col}_{freq_col}"),
            )
            .drop(tone_freq_col)
            .collect()
        )
        shared_cols = (
            [
                f"low_max_{dist_col}_{freq_col}",
                f"low_max_{dist_col}_{sample_col}",
                f"high_max_{dist_col}_{freq_col}",
                f"high_max_{dist_col}_{sample_col}",
                f"max_{dist_col}",
            ]
            if f"high_max_{dist_col}_{freq_col}" in det._properties_df.schema
            else []
        )

        det.targ._properties_df = ccat_df.coalesce_join(
            det.targ.properties, max_IQ, "det", shared_cols
        )

    max_IQ_f = det.get_properties(
        f"high_max_{dist_col}_{freq_col}", include=include, exclude=exclude, strict=True
    )
    return max_IQ_f
