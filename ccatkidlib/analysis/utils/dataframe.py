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

# ------------------------------ #
# Properties DataFrame Functions #
# ------------------------------ # 

def get_properties(obj,
                   col_name: str | list[str] = '.*',
                   include: int | list[int] | None = None,
                   exclude: int | list[int] | None = None, 
                   strict: bool = False):
        ''' Get the specified data columns and rows from the ``properties`` Polars DataFrame

        Args:
            col_name (str | list[str], optional): Defaults to all columns
            include (int | list[int] | None, optional): Defaults to *None*
            exclude (int | list[int] | None, optional): Defaults to *None*
            strict (bool, optional): Defaults to *False*
        
        '''

        def _get_expr(tones):
            expr = [pl.col('det').is_in(tones)]
            return expr
        
        def _include(include: list[int]):
            ''' Internal function for getting data rows when ``include`` is specified

            Args:
                include (list[int]): List of tones to get data for
            '''
            return _get_expr(include)

        def _exclude(exclude: list[int]):
            ''' Internal function for getting data rows when ``exclude`` is specified

            Args:
                exclude (list[int]): List of tones to **not** get data for
            '''
            tones = set(obj.tones) - set(exclude)
            return _get_expr(tones)

        def _all():
            ''' Internal function for getting all data rows (neither ``include`` or ``exclude`` specified)

            '''
            
            return _get_expr(obj.tones)

        if isinstance(col_name, str): col_name = [col_name] 
    
        exprs = parse_tones(_include, _exclude, _all, include, exclude)
        return (obj.properties.lazy()
                              .select(['det'] + [f"^{'' if strict else '.*'}{name}{'' if strict else '.*'}$" for name in col_name])
                              .filter(*exprs)
                              .collect())

def check_properties(obj, col_name: str, include: int | list[int] | None = None, exclude: int | list[int] | None = None, recalc: bool = False) -> list[int]:
    ''' Check which subset of detectors do not have a value for the specified column

    Args:
        col_name (str): Name of data column
        include ():
        exclude ():
        recalc (bool):
    Returns:
        return (list[int]): List of tones without a value for the specified column
    '''

    property_df = get_properties(obj, col_name, include=include, exclude=exclude, strict=True)
    if not recalc and not property_df.width == 1: property_df = property_df.filter(pl.col(col_name).is_null())
    tones = property_df['det'].to_numpy().T
    return tones

def add_data_to_properties(obj, df, col_name) -> pl.DataFrame:
    '''
    Add a quantity calculated with a data object's ``data`` DataFrame to the ``properties`` DataFrame
    
    Note:
        - The ``df`` DataFrame does not necessarily need to derive from a data object's ``data`` DataFrame, but the structure of this method is designed specifically for that use case

    Example:
        -
    
    Args:
        df (pl.DataFrame): Polars DataFrame with the data to be added to the ``properties`` DataFrame. The DataFrame must be in wide format with the column names being tone numbers (e.g., '0000', '0001', etc.)
        col_name (str): Name of column to add to ``properties`` DataFrame 
    '''

    df = df.unpivot(variable_name='det', value_name=col_name).with_columns(pl.col('det').cast(int)).unique()
    shared_cols = col_name if col_name in obj.properties.schema else []
    obj._properties_df = coalesce_join(obj._properties_df, df, 'det', shared_cols)
    return obj._properties_df