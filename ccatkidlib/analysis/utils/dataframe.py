import polars as pl

import ccatkidlib.rfsoc_io as rfsoc_io

from typing import Callable, TypeAlias, Any

ExprFunction: TypeAlias = Callable[[list[int], Any], list[pl.Expr]]

def parse_tones(func_include: ExprFunction, func_exclude: ExprFunction, func_all: Callable[[Any], list[pl.Expr]], include: int | list[int] | None = None, exclude: int | list[int] | None = None, *args) -> any:
        '''
        
        
        '''
        
        if include is not None and exclude is not None:
            rfsoc_io.send_msg('ERROR', "Can't include and exclude tones. Must specify one or the other.")
        elif include is not None:
            if isinstance(include, int): include = [include]
            return func_include(include, *args)
        elif exclude is not None:
            if isinstance(exclude, int): exclude = [exclude]
            return func_exclude(exclude, *args)
        else:
            return func_all(*args)
        
def coalesce_join(left_df: pl.DataFrame, right_df: pl.DataFrame, on: str, shared_cols: str | list[str]) -> pl.DataFrame:
    ''' Join two Polars DataFrames, replacing shared columns with non null values from right DataFrame ``right_df``

    Args:
        left_df (pl.DataFrame): Left (*old*) DataFrame
        right_df (pl.DataFrame): Right (*new*) DataFrame
        on (str | list[str]): Columns to join two DataFrames on
        shared_columns (str | list[str]): Shared columns between both DataFrames
    Returns:
        return (pl.DataFrame): Joined DataFrame
    '''

    if isinstance(shared_cols, str): shared_cols = [shared_cols]
    df = (left_df.join(right_df, on=on, how='left', coalesce=True)
                 .lazy()
                 .with_columns([pl.coalesce([f'{column}_right', column]).alias(column) for column in shared_cols])
                 .drop([f'{column}_right' for column in shared_cols])
                 .collect())
    return df