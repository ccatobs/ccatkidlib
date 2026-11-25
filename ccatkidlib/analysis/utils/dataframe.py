"""
Helper functions for constructing and manipulating *Polars* DataFrames used throughout the *ccatkidlib* analysis pipeline.

.. codeauthor:: Darshan Patel <dp649@cornell.edu>
"""

import polars as pl
import ccatkidlib.log as log

from enum import Enum
from collections.abc import Iterable
from typing import Callable, TypeAlias, Any

ExprFunction: TypeAlias = Callable[[list[int], Any], list[pl.Expr]]


def check_args(args: Any, num_prefix: int, data_type: type) -> list:
    """Normalize ``args`` into a list of length ``num_prefix``.

    If ``args`` is an instance of ``data_type``, is not iterable, or does not have
    length ``num_prefix``, it is broadcast to a list of ``num_prefix`` copies.

    Args:
        args (Any): Argument value(s) to normalize
        num_prefix (int): Expected length of the resulting list
        data_type (type): Type that, if matched, triggers broadcasting
    Returns:
        list: Normalized list of length ``num_prefix``
    """
    if (
        isinstance(args, data_type)
        or not isinstance(args, Iterable)
        or not len(args) == num_prefix
    ):
        args = [args] * num_prefix
    return args


def create_enums(
    name: list[str],
    input_prefix: str | list[str],
    output_prefix: list[str],
    mapping: dict,
    no_prefix: list[str] = [],
) -> tuple[list[Enum], list[Enum]]:
    """Create *Polars* column-name and prefix ``Enum`` objects from the analysis config ``mapping``.

    Builds one ``Name`` enum per input prefix and a single ``Prefix`` enum for the output prefixes,
    using the ``convention`` section of the analysis configuration dictionary.

    Args:
        name (list[str]): Short data-column names to look up in the name convention mapping
        input_prefix (str | list[str]): Input prefix or list of input prefixes
        output_prefix (list[str]): Output prefix names to look up in the prefix convention mapping
        mapping (dict): Analysis configuration dictionary containing a ``'convention'`` key
        no_prefix (list[str], optional): Column names that should not be prefixed. Defaults to ``[]``
    Returns:
        tuple[list[Enum], list[Enum]]: ``(name_enums, prefix_enums)`` — one ``Name`` enum per
        input prefix and one ``Prefix`` enum (broadcast across all input prefixes).
    """
    if isinstance(input_prefix, str):
        input_prefix = [input_prefix]
    name_mapping, prefix_mapping = (
        mapping["convention"]["name"],
        mapping["convention"]["prefix"],
    )

    name_enums = [
        Enum(
            "Name",
            [
                (
                    n.upper(),
                    f"{pre}_{name_mapping[n]}"
                    if pre and n not in no_prefix
                    else name_mapping[n],
                )
                for n in name
                if n in name_mapping
            ],
        )
        for pre in input_prefix
    ]
    prefix_enums = [
        Enum(
            "Prefix",
            [
                (p.upper(), prefix_mapping[p])
                for p in output_prefix
                if p in prefix_mapping
            ],
        )
    ] * len(input_prefix)
    return name_enums, prefix_enums


def add_tone(col_name: str, tone: int | None, padding: int) -> str:
    """Append a zero-padded |tone| suffix to a column name.

    Args:
        col_name (str): Base column name (e.g., ``'I'``)
        tone (int | None): |tone| number to append. If **None**, ``col_name`` is returned unchanged.
        padding (int): Total width of the zero-padded tone suffix (e.g., ``4`` → ``'_0007'``)
    Returns:
        str: Column name with appended tone suffix, or the original ``col_name`` if ``tone`` is **None**.
    """
    return col_name if tone is None else f"{col_name}_{tone:0{padding}d}"


def parse_tones(
    func_include: ExprFunction,
    func_exclude: ExprFunction,
    func_all: Callable[[Any], list[pl.Expr]],
    include: int | list[int] | None = None,
    exclude: int | list[int] | None = None,
    *args,
) -> list[pl.Expr]:
    """Dispatch to one of three tone-selection functions based on ``include``/``exclude`` arguments.

    Exactly one of ``include`` or ``exclude`` may be specified. If both are given an error is logged.
    If neither is given, ``func_all`` is called.

    Args:
        func_include (ExprFunction): Function to call when ``include`` is specified. Receives the
            list of included tone numbers plus any extra ``*args``.
        func_exclude (ExprFunction): Function to call when ``exclude`` is specified. Receives the
            list of excluded tone numbers plus any extra ``*args``.
        func_all (Callable[[Any], list[pl.Expr]]): Function to call when neither is specified.
            Receives any extra ``*args``.
        include (int | list[int] | None, optional): Tone number(s) to include. Defaults to **None**.
        exclude (int | list[int] | None, optional): Tone number(s) to exclude. Defaults to **None**.
        *args: Additional positional arguments forwarded to the dispatched function.
    Returns:
        list[pl.Expr]: *Polars* expression list returned by the dispatched function.
    """

    if include is not None and exclude is not None:
        log.log(
            "ERROR", "Can't include and exclude tones. Must specify one or the other."
        )
    elif include is not None:
        if isinstance(include, int):
            include = [include]
        return func_include(include, *args)
    elif exclude is not None:
        if isinstance(exclude, int):
            exclude = [exclude]
        return func_exclude(exclude, *args)
    else:
        return func_all(*args)


def unnest(data_obj: Any, col_names: str | list[str]) -> pl.DataFrame:
    """Unnest struct columns matching ``col_names`` from a data object's ``data`` DataFrame.

    Struct columns whose names contain any of the specified ``col_names`` are unnested in-place
    on ``data_obj.data``. Existing columns that duplicate keys in the struct are dropped first.

    Args:
        data_obj (Any): *ccatkidlib* data object with a ``data`` attribute that is a *Polars* DataFrame
        col_names (str | list[str]): Column name substring(s) to match against struct columns
    Returns:
        pl.DataFrame: The unnested *Polars* DataFrame (also updates ``data_obj.data`` in place).
    """
    struct_cols = []
    if isinstance(col_names, str):
        col_names = [col_names]

    for col_name in col_names:
        schema = data_obj.data.schema
        for name, data in schema.items():
            if isinstance(data, pl.Struct) and col_name in name:
                data_obj.data = data_obj.data.drop(
                    [col for col in dict(data).keys() if col in schema]
                )
                struct_cols.append(name)
    return data_obj.data.unnest(struct_cols)


def coalesce_join(
    left_df: pl.DataFrame, right_df: pl.DataFrame, on: str, shared_cols: str | list[str]
) -> pl.DataFrame:
    """Join two Polars DataFrames, replacing shared columns with non null values from right DataFrame ``right_df``

    Args:
        left_df (pl.DataFrame): Left (*old*) DataFrame
        right_df (pl.DataFrame): Right (*new*) DataFrame
        on (str | list[str]): Columns to join two DataFrames on
        shared_columns (str | list[str]): Shared columns between both DataFrames
    Returns:
        pl.DataFrame: Joined DataFrame
    """

    if isinstance(shared_cols, str):
        shared_cols = [shared_cols]
    df = (
        left_df.join(right_df, on=on, how="left", coalesce=True)
        .lazy()
        .with_columns(
            [
                pl.coalesce([f"{column}_right", column]).alias(column)
                for column in shared_cols
            ]
        )
        .drop([f"{column}_right" for column in shared_cols])
        .collect()
    )
    return df


# ------------------------------ #
# Properties DataFrame Functions #
# ------------------------------ #


def get_properties(
    obj: Any,
    col_name: str | list[str] = ".*",
    include: int | list[int] | None = None,
    exclude: int | list[int] | None = None,
    strict: bool = False,
) -> pl.DataFrame:
    """Get the specified data columns and rows from the ``properties`` *Polars* DataFrame.

    Args:
        obj (Any): *ccatkidlib* data object with ``properties`` and ``tones`` attributes
        col_name (str | list[str], optional): Regex pattern(s) to select columns. Defaults to ``'.*'`` (all columns).
        include (int | list[int] | None, optional): Tone number(s) to include. Defaults to **None**.
        exclude (int | list[int] | None, optional): Tone number(s) to exclude. Defaults to **None**.
        strict (bool, optional): When **True**, column names are matched exactly; when **False** they are
            matched as substrings. Defaults to **False**.
    Returns:
        pl.DataFrame: Filtered ``properties`` *Polars* DataFrame with the ``'det'`` column plus the
        requested data columns.
    """

    def _get_expr(tones):
        expr = [pl.col("det").is_in(tones)]
        return expr

    def _include(include: list[int]):
        """Internal function for getting data rows when ``include`` is specified

        Args:
            include (list[int]): List of tones to get data for
        """
        return _get_expr(include)

    def _exclude(exclude: list[int]):
        """Internal function for getting data rows when ``exclude`` is specified

        Args:
            exclude (list[int]): List of tones to **not** get data for
        """
        tones = set(obj.tones) - set(exclude)
        return _get_expr(tones)

    def _all():
        """Internal function for getting all data rows (neither ``include`` or ``exclude`` specified)"""

        return _get_expr(obj.tones)

    if isinstance(col_name, str):
        col_name = [col_name]

    exprs = parse_tones(_include, _exclude, _all, include, exclude)
    return (
        obj.properties.lazy()
        .select(
            ["det"]
            + [
                f"^{'' if strict else '.*'}{name}{'' if strict else '.*'}$"
                for name in col_name
            ]
        )
        .filter(*exprs)
        .collect()
    )


def check_properties(
    obj: Any,
    col_name: str,
    include: int | list[int] | None = None,
    exclude: int | list[int] | None = None,
    recalc: bool = False,
) -> list[int]:
    """Check which subset of detectors do not have a value for the specified column.

    Args:
        obj (Any): *ccatkidlib* data object with a ``properties`` DataFrame
        col_name (str): Name of data column to check
        include (int | list[int] | None, optional): Tone number(s) to include. Defaults to **None**.
        exclude (int | list[int] | None, optional): Tone number(s) to exclude. Defaults to **None**.
        recalc (bool, optional): When **True**, returns all tones regardless of existing values.
            Defaults to **False**.
    Returns:
        list[int]: List of tones without a value for the specified column.
    """

    property_df = get_properties(
        obj, col_name, include=include, exclude=exclude, strict=True
    )
    if not recalc and not property_df.width == 1:
        property_df = property_df.filter(pl.col(col_name).is_null())
    tones = property_df["det"].to_numpy().T
    return tones


def add_data_to_properties(obj: Any, df: pl.DataFrame, col_name: str) -> pl.DataFrame:
    """Add a quantity to a data object's ``properties`` DataFrame.

    Pivots ``df`` from wide to long format, then coalesce-joins the result into
    ``obj._properties_df``, replacing any existing values for ``col_name``.

    Note:
        ``df`` must be in wide format with column names matching zero-padded tone numbers
        (e.g., ``'0000'``, ``'0001'``, …).

    Args:
        obj (Any): *ccatkidlib* data object with a ``_properties_df`` attribute
        df (pl.DataFrame): Wide-format *Polars* DataFrame whose columns are tone numbers and
            whose single value row holds the per-|tone| quantity to add
        col_name (str): Name of the column to add to the ``properties`` DataFrame
    Returns:
        pl.DataFrame: Updated ``properties`` *Polars* DataFrame.
    """

    df = (
        df.unpivot(variable_name="det", value_name=col_name)
        .with_columns(pl.col("det").cast(int))
        .unique()
    )
    shared_cols = col_name if col_name in obj.properties.schema else []
    obj._properties_df = coalesce_join(obj._properties_df, df, "det", shared_cols)
    return obj._properties_df
