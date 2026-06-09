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
    """

    Args:
        func (Callable[[pl.DataFrame], pl.Series]): Analysis function to apply to tones. Must take a Polars DataFrame as the input and return a Polars Series
        tones [list[int]]:

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
            batches[i] = [ccat_df.add_tone(name, tone, padding) for name in input_col] # Get all input columns needed for calculations
            returns[i] = [
                pl.Field(ccat_df.add_tone(name, tone, padding), dtype)
                for name, dtype in zip(return_col, return_type)
            ] # Map batches requires explicit return data types specified

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
    """ """

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
    """ """
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
    """ """

    df = pl.DataFrame(dict(sorted(results_dict.items())))
    return pl.Series(df.select(pl.struct(df.columns)))
