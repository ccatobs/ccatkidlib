"""
Library of helper functions for multiprocessing data analysis code

.. codeauthor:: Darshan Patel <dp649@cornell.edu>

"""

from __future__ import annotations

import os
import math
import polars as pl
from concurrent.futures import ProcessPoolExecutor

from contextlib import contextmanager
from typing import Callable, Iterator, Any, TYPE_CHECKING

import ccatkidlib.analysis.utils.dataframe as ccat_df

if TYPE_CHECKING:
    import numpy as np


@contextmanager
def optional_executor(
    max_workers: int = 1, ex: ProcessPoolExecutor | None = None
) -> Iterator[ProcessPoolExecutor]:
    """
    Context manager that yields the *concurrent.futures* **ProcessPoolExecutor** provided or creates a new one if **None** provided

    Args:
        max_workers: Maximum number of worker processes to use for multiprocessing. Only used if ``ex`` is **None**
        ex: A *concurrent.futures* **ProcessPoolExecutor**
    Yields:
        The *concurrent.futures* **ProcessPoolExecutor** provided or a newly created one if **None** provided
    """

    if ex is None:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            yield executor
    else:
        yield ex


def check_max_workers(max_workers: int) -> int:
    """
    Ensure that the maximum number of worker processes specified is less than or equal to the number of available CPU cores

    Args:
        max_workers: Maximum number of workers to use for multiprocessing
    Returns:
        ``max_workers`` if it less than or equal to the number of CPUs, otherwise returns the number of available CPU cores
    """
    return min(int(max_workers), os.cpu_count())


def create_batches(
    func: Callable[[pl.DataFrame], pl.Series],
    tones: list[int],
    schema: pl.Schema,
    input_col: list[str],
    return_col: list[str],
    return_type: list[pl.DataType],
    padding: int = 4,
    max_workers: int = 1,
    recalc: bool = False,
) -> tuple[
    pl.Expr, list[list[int]], list[list[int]], str | list[str], list[list[str]], int
]:
    """Build *Polars* batch-processing expressions and tone index batches for parallel analysis.

    Determines which tones need to be (re-)calculated, constructs a ``pl.struct.map_batches``
    expression that applies ``func`` over all required tones at once, and splits the work into
    ``max_workers`` batches for parallel execution.

    Args:
        func (Callable[[pl.DataFrame], pl.Series]): Analysis function applied per |tone|.
            Must accept a *Polars* DataFrame and return a *Polars* Series.
        tones (list[int]): List of |tone| numbers to consider
        schema (pl.Schema): Current DataFrame schema used to skip already-computed columns
        input_col (list[str]): Base input column names (without tone suffix)
        return_col (list[str]): Base output column names (without tone suffix)
        return_type (list[pl.DataType]): *Polars* data types for each output column
        padding (int, optional): Zero-padding width for tone column name suffixes. Defaults to ``4``.
        max_workers (int, optional): Number of worker processes for batching. Defaults to ``1``.
        recalc (bool, optional): When **True**, recalculate even if the output column already exists.
            Defaults to **False**.
    Returns:
        tuple[pl.Expr, list[list[int]], list[list[int]], str | list[str], list[list[str]], int]:
        ``(expr, calc_ind, batch_len)`` — the batch *Polars* expression, per-worker tone index
        batches, and batch length. ``expr`` is an empty list if no tones need calculating.
    """
    # Determine which tones need calculating
    to_calc, calc_ind = [], []
    for i, tone in enumerate(tones):
        if recalc or ccat_df.add_tone(return_col[0], tone, padding) not in schema:
            to_calc.append(tone), calc_ind.append(i)

    expr, batches, batch_len = [], [], 0
    if (num_calc := (len(to_calc))) != 0:
        batches, returns = num_calc * [None], num_calc * [None]
        for i, tone in enumerate(to_calc):
            batches[i] = [
                ccat_df.add_tone(name, tone, padding) for name in input_col
            ]  # Get all input columns needed for calculations
            returns[i] = [
                pl.Field(ccat_df.add_tone(name, tone, padding), dtype)
                for name, dtype in zip(return_col, return_type)
            ]  # Map batches requires explicit return data types specified

        struct_col = f"struct_{return_col[0]}"
        batches_flat, returns_flat = (
            [col for batch in batches for col in batch],
            [col for ret_col in returns for col in ret_col],
        )
        expr = (
            pl.struct(batches_flat)
            .map_batches(func, return_dtype=pl.Struct(returns_flat))
            .alias(struct_col)
        )

        # Split data into batches based on number of workers
        batch_len = math.ceil(num_calc / max_workers)
        new_calc_ind = max_workers * [None]
        for i in range(max_workers):
            low_ind, up_ind = i * batch_len, (i + 1) * batch_len
            new_calc_ind[i] = calc_ind[low_ind:up_ind]
        calc_ind = new_calc_ind
    return expr, calc_ind, batch_len


def struct_batches(
    struct: pl.Struct, num_data_cols: int, batch_len: int, max_workers: int
) -> list[list[np.ndarray]]:
    """Unpack a *Polars* struct column into per-worker batches of NumPy arrays.

    Unnests the struct, converts to a NumPy array, reshapes into ``(num_data_cols, n_tones, n_samples)``,
    then slices along the tone dimension into ``max_workers`` batches of length ``batch_len``.

    Args:
        struct (pl.Struct): *Polars* struct column to unpack
        num_data_cols (int): Number of data columns packed into the struct per |tone|
        batch_len (int): Number of tones per worker batch
        max_workers (int): Number of worker batches to produce
    Returns:
        list[list[np.ndarray]]: Nested list of shape ``[max_workers][num_data_cols]`` where each
        element is a NumPy array of shape ``(batch_len, n_samples)``.
    """

    data = (
        struct.struct.unnest().to_numpy().T
    )  # Unnest Struct and convert to numpy array
    data = list(
        zip(*data.reshape(len(data) // num_data_cols, num_data_cols, -1))
    )  # Zip data by tone
    data = [
        [data_col[i * batch_len : (i + 1) * batch_len] for data_col in data]
        for i in range(max_workers)
    ]  # Batch data
    return data


def process_batches(func: Callable, *args, **kwargs) -> list[Any | Exception]:
    """Execute ``func`` over batches of positional and keyword arguments, capturing exceptions.

    Zips each positional ``args`` iterable and each keyword ``kwargs`` iterable element-wise,
    then calls ``func`` with each combined set of arguments. Any exception raised by ``func``
    for a given batch is caught and stored in the results list rather than propagated.

    Args:
        func (Callable): Function to apply to each batch
        *args: Iterables of positional arguments; zipped element-wise across batches
        **kwargs: Iterables of keyword argument values; zipped element-wise across batches
    Returns:
        list[Any | Exception]: Results (or caught exceptions) for each batch, in order.
    """
    func_args = list(zip(*args))
    func_kwargs = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]
    if not func_kwargs:
        func_kwargs = [{}] * len(func_args)

    results = [None] * len(func_args)
    for i, (func_arg, func_kwarg) in enumerate(zip(func_args, func_kwargs)):
        try:
            results[i] = func(*func_arg, **func_kwarg)
        except Exception as e:
            results[i] = e
    return results


def package_results(results_dict: dict) -> pl.Series:
    """Package a dictionary of per-|tone| results into a *Polars* struct Series.

    Sorts the dictionary by key, constructs a *Polars* DataFrame, then returns the rows as a
    struct Series — the format expected by ``pl.struct.map_batches`` return values.

    Args:
        results_dict (dict): Mapping of column names to result values, keyed by column name
    Returns:
        pl.Series: Single-element *Polars* Series of struct type containing the results.
    """

    df = pl.DataFrame(dict(sorted(results_dict.items())))
    return pl.Series(df.select(pl.struct(df.columns)))
