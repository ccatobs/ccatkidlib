import polars as pl
import numpy as np

import ccatkidlib.analysis.utils.dataframe as ccat_df
import ccatkidlib.log as log

from ccatkidlib.analysis.core.timestream import Timestream


def mismatch_angle(
    targ, prefix="", mean_points=10, include=None, exclude=None, recalc=False
):
    name_enums, prefix_enums = ccat_df.create_enums(
        ["in_phase", "quadrature", "phase"],
        prefix,
        ["remove_impedance_mismatch"],
        targ.analysis_cfg,
        no_prefix=[],
    )

    name_enum, prefix_enum = name_enums[0], prefix_enums[0]
    mismatch_col_name = (
        f"{prefix_enum.REMOVE_IMPEDANCE_MISMATCH.value}_{name_enum.PHASE.value}"
    )
    include_subset = ccat_df.check_properties(
        targ,
        mismatch_col_name,
        include=include,
        exclude=exclude,
        recalc=recalc,
    )
    if not len(include_subset) == 0:
        pi = pl.lit(np.pi)
        I_df = targ.get_data(
            name_enum.IN_PHASE.value,
            include=include_subset,
            strict=True,
        )
        Q_df = targ.get_data(
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
        ccat_df.add_data_to_properties(targ, mismatch_df, mismatch_col_name)
    return targ.get_properties(
        mismatch_col_name, include=include, exclude=exclude, strict=True
    )


def timestream_angle(stream, prefix="", include=None, exclude=None, recalc=False):
    name_enums, prefix_enums = ccat_df.create_enums(
        ["in_phase", "quadrature", "phase"],
        prefix,
        ["center_timestream"],
        stream.analysis_cfg,
        no_prefix=[],
    )

    name_enum, prefix_enum = name_enums[0], prefix_enums[0]
    timestream_col_name = (
        f"{prefix_enum.CENTER_TIMESTREAM.value}_{name_enum.PHASE.value}"
    )

    # Calculate angle of center of timestream (center determined using I and Q medians)
    include_subset = ccat_df.check_properties(
        stream,
        timestream_col_name,
        include=include,
        exclude=exclude,
        recalc=recalc,
    )
    if not len(include_subset) == 0:
        pi = pl.lit(np.pi)
        I_df = stream.get_data(
            name_enum.IN_PHASE.value,
            include=include_subset,
            strict=True,
        )
        Q_df = stream.get_data(
            name_enum.QUADRATURE.value,
            include=include_subset,
            strict=True,
        )
        I_cols, Q_cols = I_df.columns, Q_df.columns
        IQ_df = pl.concat([I_df, Q_df], how="horizontal")
        timestream_df = (
            IQ_df.lazy()
            .select(pl.all().median())
            .select(
                [
                    (pl.arctan2(pl.col(Q_col), pl.col(I_col)) % (2 * pi)).alias(
                        I_col.split("_")[-1]
                    )
                    for I_col, Q_col in zip(I_cols, Q_cols)
                ]
            )
            .collect()
        )
        ccat_df.add_data_to_properties(stream, timestream_df, timestream_col_name)
    return stream.get_properties(
        timestream_col_name,
        include=include,
        exclude=exclude,
        strict=True,
    )


def fwhm(targ, mag_prefix="", mean_points=10, include=None, exclude=None, recalc=False):
    name_enums, prefix_enums = ccat_df.create_enums(
        [
            "sample",
            "magnitude",
            "full_width_half_max",
        ],
        mag_prefix,
        ["low_frequency_side", "middle_frequency_point", "high_frequency_side"],
        targ.analysis_cfg,
        no_prefix=["sample"],
    )

    name_enum, prefix_enum = name_enums[0], prefix_enums[0]
    low_sample, mid_sample, high_sample = (
        f"{prefix_enum.LOW_FREQUENCY_SIDE.value}_{name_enum.FULL_WIDTH_HALF_MAX.value}_{name_enum.SAMPLE.value}",
        f"{prefix_enum.MIDDLE_FREQUENCY_POINT.value}_{name_enum.FULL_WIDTH_HALF_MAX.value}_{name_enum.SAMPLE.value}",
        f"{prefix_enum.HIGH_FREQUENCY_SIDE.value}_{name_enum.FULL_WIDTH_HALF_MAX.value}_{name_enum.SAMPLE.value}",
    )

    include_subset = ccat_df.check_properties(
        targ, mid_sample, include=include, exclude=exclude, recalc=recalc
    )
    if not len(include_subset) == 0:
        sample_col, mag_col = name_enum.SAMPLE.value, name_enum.MAGNITUDE.value
        # Get detector magnitudes and sample numbers and unpivot DataFrame from wide to long format
        mag_df = targ.get_data(
            col_name=[sample_col, mag_col], strict=True, include=include_subset
        )
        mag_df = (
            mag_df.unpivot(
                index=sample_col,
                variable_name="det",
                value_name=mag_col,
            )
            .lazy()
            .with_columns(pl.col("det").str.strip_prefix(f"{mag_col}_").cast(pl.Int32))
            .sort(mag_col, descending=True)
        )

        # Get minimum magnitude values for each detector and corresponding sample numbers
        min_df = (
            mag_df.filter((pl.col(mag_col) == pl.col(mag_col).min()).over("det"))
            .rename(
                {
                    sample_col: f"min_{sample_col}",
                    mag_col: f"min_{mag_col}",
                }
            )
            .collect()
        )
        shared_cols = mid_sample if mid_sample in targ._properties_df.schema else []
        targ._properties_df = ccat_df.coalesce_join(
            targ._properties_df,
            min_df.select(["det", pl.col(f"min_{sample_col}").alias(mid_sample)]),
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
        for side, name in zip(["low", "high"], [low_sample, high_sample]):
            HM_df = (
                min_max_df.filter(pl.col("low") == ("low" == pl.lit(side)))
                .sort(name)
                .select("det", pl.col(sample_col).first().over("det"))
                .unique()
                .sort("det")
                .rename({sample_col: name})
                .collect()
            )
            shared_cols = name if name in targ._properties_df.schema else []
            targ._properties_df = ccat_df.coalesce_join(
                targ._properties_df, HM_df, "det", shared_cols
            )
    return targ.get_properties(
        [low_sample, mid_sample, high_sample],
        include=include,
        exclude=exclude,
        strict=True,
    )


def agg(obj, operation, col_name, prefix="", include=None, exclude=None, recalc=False):
    AGGS = {
        "mean": pl.all().mean(),
        "median": pl.all().median(),
        "max": pl.all().max(),
        "min": pl.all().min(),
        'std': pl.all().std(),
    }
    agg = AGGS.get(operation, None)
    if agg is None:
        error = f"Invalid operation specified, must be one of {', '.join(AGGS.keys())}"
        log.log("ERROR", error)
        raise ValueError(error)

    enum_key, mapping = None, obj.analysis_cfg["convention"]["name"]
    for key, val in mapping.items():
        if val == col_name:
            enum_key = key
            break

    if enum_key is None:
        error = f"Could not find column '{col_name}'. Ensure that a mapping exists in the analysis configuration file."
        log.log("ERROR", error)
        raise KeyError(error)

    name_enums, prefix_enums = ccat_df.create_enums(
        [enum_key],
        prefix,
        [],
        obj.analysis_cfg,
    )

    name_enum, prefix_enum = name_enums[0], prefix_enums[0]
    agg_name = f"{operation}_{'stream' if isinstance(obj, Timestream) else 'targ'}_{name_enum[enum_key.upper()].value}"

    include_subset = ccat_df.check_properties(
        obj,
        agg_name,
        include=include,
        exclude=exclude,
        recalc=recalc,
    )
    if not len(include_subset) == 0:
        df = obj.get_data(
            name_enum[enum_key.upper()].value, include=include_subset, strict=True
        )
        agg_df = df.select([agg.name.map(lambda s: s.split("_")[-1])])
        ccat_df.add_data_to_properties(obj, agg_df, agg_name)

    return obj.get_properties(
        agg_name,
        include=include,
        exclude=exclude,
        strict=True,
    )