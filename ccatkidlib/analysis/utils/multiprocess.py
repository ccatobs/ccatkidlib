import os
import polars as pl

def check_max_workers(max_workers):
    return min(int(max_workers), os.cpu_count())

def batch_calc(func, tones, col_name, schema, return_col, return_type, recalc=False, calc_col = None):
    '''
    
    '''
    if calc_col is None: calc_col = [f'{col_name[-1]}_{col_name[-2]}_{tone:04d}' for tone in tones]
    to_calc = tones if recalc else [tone for tone, col in zip(tones, calc_col) if col not in schema]
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
    return expr, to_calc, calc_col, batches