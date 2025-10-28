''' Library of helper functions related to multiprocessing

Authors:
    - Darshan Patel <dp649@cornell.edu>

'''

import os
import polars as pl
 
from typing import Callable

def check_max_workers(max_workers: int):
    ''' Check that the number of max workers specified for multiprocessing is less than or equal to the numebr of available CPUs

    Args:
        max_workers (int): Number of workers to use for multiprocessing
    Returns:
        return (int): Returns ``max_workers`` if it less than or equal to the number of CPUs. Otherwise, returns the number of available CPUs.
    '''
    return min(int(max_workers), os.cpu_count())

def batch_calc(func: Callable[[pl.DataFrame], pl.Series],
               tones: list[int],
               col_name: list[str], 
               schema: pl.Schema, 
               return_col: list[str], 
               return_type: list[pl.DataType], 
               calc_col: list[str] | None = None,
               recalc: bool = False):
    '''
    
    Args:
        func (Callable[[pl.DataFrame], pl.Series]): Analysis function to apply to tones. Must take a Polars DataFrame as the input and return a Polars Series
        tones [list[int]]: 

    '''
    if calc_col is None: calc_col = [f'{col_name[-1]}_{col_name[-2]}_{tone:04d}' for tone in tones]
    to_calc = tones if recalc else [tone for tone, col in zip(tones, calc_col) if col not in schema]
    calc_ind = [tones.index(tone) for tone in to_calc]
    if not len(to_calc) == 0:
        batches = [[f'{name}_{tone:04d}' for name in col_name[:-1]] for tone in to_calc]
        returns = [[pl.Field(f'{name}_{tone:04d}', dtype) for name, dtype in zip(return_col, return_type)] for tone in to_calc]
        calc_col = f'struct_{col_name[-1]}_{to_calc[0]:04d}'
        
        batches_flat = [col for batch in batches for col in batch]
        returns_flat = [col for ret_col in returns for col in ret_col]
        expr = pl.struct(batches_flat).map_batches(func, return_dtype=pl.Struct(returns_flat)).alias(calc_col)
    else:
        batches = []
        expr = pl.col(calc_col)
    return expr, to_calc, calc_ind, calc_col, batches