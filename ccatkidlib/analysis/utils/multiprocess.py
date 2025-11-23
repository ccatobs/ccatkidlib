''' Library of helper functions related to multiprocessing

Authors:
    - Darshan Patel <dp649@cornell.edu>

'''

import os
import math
import polars as pl
import concurrent.futures
import numpy as np

from contextlib import contextmanager
from typing import Callable

@contextmanager
def optional_executor(max_workers, ex=None):
    if ex is None:
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            yield executor
    else:
        yield ex

def check_max_workers(max_workers: int):
    ''' Check that the number of max workers specified for multiprocessing is less than or equal to the numebr of available CPUs

    Args:
        max_workers (int): Number of workers to use for multiprocessing
    Returns:
        return (int): Returns ``max_workers`` if it less than or equal to the number of CPUs. Otherwise, returns the number of available CPUs.
    '''
    return min(int(max_workers), os.cpu_count())

def create_batches(func: Callable[[pl.DataFrame], pl.Series],
                   tones: list[int],
                   col_name: list[str], 
                   schema: pl.Schema, 
                   return_col: list[str], 
                   return_type: list[pl.DataType], 
                   padding: int = 4,
                   calc_col: list[str] | None = None,
                   max_workers=1,
                   recalc: bool = False):
    '''
    
    Args:
        func (Callable[[pl.DataFrame], pl.Series]): Analysis function to apply to tones. Must take a Polars DataFrame as the input and return a Polars Series
        tones [list[int]]: 

    '''
    if calc_col is None: calc_col = [f'{col_name[-1]}_{col_name[-2]}_{tone:0{padding}d}' for tone in tones]
    to_calc = tones if recalc else [tone for tone, col in zip(tones, calc_col) if col not in schema]
    calc_ind = [tones.index(tone) for tone in to_calc]
    if not len(to_calc) == 0:
        batches = [[f'{name}_{tone:0{padding}d}' for name in col_name[:-1]] for tone in to_calc]
        returns = [[pl.Field(f'{name}_{tone:0{padding}d}', dtype) for name, dtype in zip(return_col, return_type)] for tone in to_calc]
        calc_col = f'struct_{col_name[-1]}_{to_calc[0]:0{padding}d}'
        
        batches_flat = [col for batch in batches for col in batch]
        returns_flat = [col for ret_col in returns for col in ret_col]
        expr = pl.struct(batches_flat).map_batches(func, return_dtype=pl.Struct(returns_flat)).alias(calc_col)

        batch_len = math.ceil(len(to_calc) / max_workers)
        batches = [batches[i*batch_len:(i+1)*batch_len] for i in range(max_workers)]
        to_calc = [to_calc[i*batch_len:(i+1)*batch_len] for i in range(max_workers)]
        calc_ind = [calc_ind[i*batch_len:(i+1)*batch_len] for i in range(max_workers)]
    else:
        batches = []
        batch_len = 0
        expr = pl.col(calc_col)
    return expr, to_calc, calc_ind, calc_col, batches, batch_len

def struct_batches(df, num_data_cols, batch_len, max_workers):
    data = df.struct.unnest().to_numpy().T # Unnest Struct and convert to numpy array
    data = list(zip(*data.reshape(len(data)//num_data_cols, num_data_cols, -1))) # Zip data by tone
    data = [[data_col[i*batch_len:(i+1)*batch_len] for data_col in data] for i in range(max_workers)] # Batch data
    return data

def process_batches(func, *args, **kwargs):
    func_args = list(zip(*args))
    func_kwargs = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())] 

    results = [None]*len(func_args)
    for i, (func_arg, func_kwarg) in enumerate(zip(func_args, func_kwargs)):
        try:
            results[i] = func(*func_arg, **func_kwarg)
        except Exception as e:
            results[i] = e
    return results

def package_results(results_dict):
    df = pl.DataFrame(dict(sorted(results_dict.items())))
    return pl.Series(df.select(pl.struct(df.columns)))