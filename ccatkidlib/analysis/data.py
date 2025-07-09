import gc
import sys
import pathlib
import numpy as np
import polars as pl

from pathlib import Path
from numba import njit
from functools import cached_property

# Local Imports

import ccatkidlib.rfsoc_io as rfsoc_io
import ccatkidlib.utils as utils
import ccatkidlib.analysis.pair as pair
import ccatkidlib.analysis.plot_utils as putils

from ccatkidlib.utils import method_timer



class Data:
    '''Class representing a raw RFSoC output data product (VNA sweep, target sweep, or timestream)

    Attributes:
        bid  (str): RFSoC board that took the data 
        drid (str): RFSoC drone that took the data

        tone (int | list[int], optional): Which tones are loaded. For target sweep and timestream data.
        data_path (pathlib.PosixPath): Path to data file
        data (polars.dataframe.frame.DataFrame): Polars DataFrame with loaded data
        timestamp (int): Timestamp of data file

        io_cfg    (dict): Loaded IO config
        ext_cfg   (dict): Loaded external config
        drone_cfg (dict): Loaded drone config

        comb (polars.dataframe.frame.DataFrame): Polars DataFrame with tone frequencies, powers, and phis used to take data
        num_tones (int): Number of tones used to take data
    '''

    def __init__(self, com_to: str,
                 analysis_cfg: str = str(Path(__file__).parent / 'analysis_config.yaml'),
                 data_type: str | None = None,
                 timestamp: int | str | None = None,
                 data_path: str | pathlib.PosixPath | None = None, 
                 **kwargs):
        ''' Initialize Data object by finding data file and associated config files
        
        Note:
            Either the full data path must be provided via the ``data_path`` key word argument or
            both the timestamp and data type must be provided via the ``data_type`` and ``timestamp`` key word arguments respectively

        Args:
            com_to (str): Drone that took the data. In form 'Board.Drone'
            analysis_cfg (str, optional): Path to analysis config. Defaults to analysis config in ccatkidlib/ccatkidlib/analysis directory.
            
            data_type (str, optional): Type of data file. Should be one of 'vna', 'targ', 'timestream'. 
            timestamp (int | str, optional): Timestamp of data file
            data_path (str | pathlib.PosixPath, optional): Full path to data file 
            
            **kwargs: Key word arguments for finding data file. See below:
            root_data_dir (str, optional): Root directory where data is stored. Defaults to that specified in analysis config
            data_dir (str, optional): Directory where data is stored
            date (str, optional): Date data was taken
            sess_id (str, optional): ccatkidlib session ID of data
        '''
        # Define data attributes
        # -----------------------
        self.bid, self.drid = com_to.split('.') # Baard and drone data was taken with

        self.tone = None
        self._data = None
        self.analysis_cfg, self.plot_cfg = rfsoc_io.load_config(analysis_cfg) # File path of analysis config

        # If full data path is not provided, find data file based on timestamp and (optional) file path parts
        # ---------------------------------------------------------------------------------------------------
        if data_path is None:
            # Find sweep data file using 
            if data_type is None or timestamp is None:
                error = ("Both the data type ('vna', 'targ', or 'timestream') and the timestamp must be provided" 
                         "or the full data path needs to be specified.")
                rfsoc_io.send_msg('CRITICAL', error)
                raise ValueError(error)

            # Parse data file part key word arguments
            # ---------------------------------------------
            root_data_dir = self.analysis_cfg['data_load']['root_data_dir']
            data_dir = '**'
            date = '**'
            sess_id = '**'

            for key, value in kwargs.items():
                if key == 'root_data_dir':
                    root_data_dir = value
                elif key == 'data_dir':
                    data_dir = value
                elif key == 'date':
                    date = value
                elif key == 'sess_id':
                    sess_id = value

            # Try to find data file using given information
            self.data_path = pair.get_data_file(com_to, timestamp, data_type, data_dir = data_dir, date = date, sess_id = sess_id, root_data_dir=root_data_dir)
            self.timestamp = timestamp
        else:
            self.timestamp = pair.get_timestamp(self.data_path)
        
        self.data_path = Path(self.data_path)

        # Check that data path exists
        # ---------------------------
        if not self.data_path.exists():
            error = f'Could not find {data_type} file for board {self.bid}, drone {self.drid} with timestamp {timestamp}! Check that all optional file path segments are correct!'
            rfsoc_io.send_msg('CRITICAL', error)
            raise FileNotFoundError(error)

        # Get io, ext, and drone configs associated with the sweep data file
        # ------------------------------------------------------------------
        self._configs = pair.get_config(self.data_path, all_cfg=False)

    #=====================#
    # Data Getter Methods # 
    #=====================#

    def get_data(self, col_name: str | list[str] = '.*', include: int | list[int] | None = None, exclude: int | list[int] | None = None):
        '''Get the specified data columns from the self.data Polars DataFrame for the specified tones
    
        Note:
            Only one of ``include`` and ``exclude`` arguments can be specified. If neither are specified, returns data for all tones.

        Args:
            col_name (str | list[str]): List of data column names without tone number suffix (e.g., 'I', 'mag', 'phase')
            include (int | list[int], optional): List of tones to include. Defaults to None
            exclude (int | list[int], optional): List of tones to exclude. Defaults to None
        Returns:
            polars.dataframe.frame.DataFrame: Polars DataFrame with specified data columns

        '''
        def _include(include, *args):
            col_name = args[0]
            expr = []
            for tone in include:
                expr.append(pl.col([f'^{name}_{tone:04d}$' for name in col_name]))
            return expr
        
        def _exclude(exclude, *args):
            col_name = args[0]
            expr = []
            for tone in exclude:
                expr += [f'^{name}_{tone:04d}$' for name in col_name]
            return [pl.col([f'^{name}_.*$' for name in col_name]).exclude(expr)]

        def _all(*args):
            col_name = args[0]
            return [pl.col([f'^{name}_.*$' for name in col_name])]

        if self.tone is not None:
            if isinstance(col_name, str): col_name = [col_name]
            args = [col_name]
            return self.data.select(*self._parse_tone(_include, _exclude, _all, include, exclude, *args))
        else:
            return self.data.select(pl.col(col_name))
    
    def tone(self, include: int | list[int] | None = None, exclude: int | list[int] | None = None):
        '''Get all of the data columns in the self.data Polars DataFrame for the specified tones

        Note:
            Only one of ``include`` and ``exclude`` arguments can be specified. If neither are specified, returns data for all tones.

        Args:
            include (int | list[int], optional): List of tones to include. Defaults to None
            exclude (int | list[int], optional): List of tones to exclude. Defaults to None
        Returns:
            polars.dataframe.frame.DataFrame: Polars DataFrame with specified data columns
        '''
        if self.tone is not None:
            return self.get_data(col_name = '.*', include=include, exclude=exclude)
        else:
            rfsoc_io.send_msg('ERROR', "Cannot load individual tone data when self.tone is None.")

    def I(self, include: int | list[int] | None = None, exclude: int | list[int] | None = None):
        '''Get the in-phase ``I`` data for the specified tones

        Note:
            Only one of ``include`` and ``exclude`` arguments can be specified. If neither are specified, returns data for all tones.

        Args:
            include (int | list[int], optional): List of tones to include. Defaults to None
            exclude (int | list[int], optional): List of tones to exclude. Defaults to None
        Returns:
            polars.dataframe.frame.DataFrame: Polars DataFrame with specified data columns

        '''

        return self.get_data(col_name='I', include=include, exclude=exclude)
    
    def Q(self, include: int | list[int] | None = None, exclude: int | list[int] | None = None):
        '''Get the quadrature ``Q`` data for the specifed tones

        Note:
            Only one of ``include`` and ``exclude`` arguments can be specified. If neither are specified, returns data for all tones.

        Args:
            include (int | list[int], optional): List of tones to include. Defaults to None
            exclude (int | list[int], optional): List of tones to exclude. Defaults to None
        Returns:
            polars.dataframe.frame.DataFrame: Polars DataFrame with specified data columns
        '''
        return self.get_data(col_name='Q', include=include, exclude=exclude)

    def phase(self, include: int | list[int] | None = None, exclude: int | list[int] | None = None):
        '''Calculate and get the phase ``arctan(Q/I)`` data for the specified tones. 

        Note:
            Only one of ``include`` and ``exclude`` arguments can be specified. If neither are specified, returns data for all tones.

        Args:
            include (int | list[int], optional): List of tones to include. Defaults to None
            exclude (int | list[int], optional): List of tones to exclude. Defaults to None
        Returns:
            polars.dataframe.frame.DataFrame: Polars DataFrame with specified data columns

        '''
        self.transform(Data.calc_phase, include=include, exclude=exclude).collect().lazy()
        return self.get_data(col_name='phase', include=include, exclude=exclude)

    def mag(self, include: int | list[int] | None = None, exclude: int | list[int] | None = None):
        '''Calculate and get the magnitude ``I^2 + Q^2`` data of the specified tones

        Note:
            Only one of ``include`` and ``exclude`` arguments can be specified. If neither are specified, returns data for all tones.

        Args:
            include (int | list[int], optional): List of tones to include. Defaults to None
            exclude (int | list[int], optional): List of tones to exclude. Defaults to None
        Returns:
            polars.dataframe.frame.DataFrame: Polars DataFrame with specified data columns
        '''
        self.transform(Data.calc_mag, include=include, exclude=exclude).collect().lazy()
        return self.get_data(col_name='mag', include=include, exclude=exclude)
    
    #==================#
    # Analysis Methods #
    #==================#

    def transform(self, funcs, include: int | list[int] | None = None, exclude: int | list[int] | None = None):
        '''Apply transformations specified by ``funcs`` argument to the specified tones

        Note:
            Only one of ``include`` and ``exclude`` arguments can be specified. If neither are specified, returns data for all tones.

        Args:
            funcs (list): List of functions to apply to data. Functions should take tone number and DataFrame schema as arguments and should return a polars.expr.expr.Expr object.
            include (int | list[int], optional): List of tones to include. Defaults to None
            exclude (int | list[int], optional): List of tones to exclude. Defaults to None
        Returns:
            polars.dataframe.frame.DataFrame: Full self.data Polars DataFrame including columns with the transformed data

        '''
        def _include(include, *args):
            schema = args[0]
            expr = []
            for tone in include:
                for func in funcs:
                    expr.append(func(tone, schema))
            return expr
        
        def _exclude(exclude, *args):
            schema = args[0]
            expr = []
            for tone in self.tone:
                if not tone in exclude:
                    for func in funcs:
                        expr.append(func(tone, schema))
            return expr

        def _all(*args):
            schema = args[0]
            expr = []
            for tone in self.tone:
                for func in funcs:
                    expr.append(func(tone, schema))
            return expr

        if callable(funcs): funcs = [funcs]

        if self.tone is not None:
            args = [self.data.collect_schema()]
            self.data = self.data.with_columns(*self._parse_tone(_include, _exclude, _all, include, exclude, *args))
        else:
            self.data = self.data.with_columns(*[func(None, self.data.schema) for func in funcs])
        return self.data
    
    @staticmethod
    def calc_phase(tone, schema):
        if tone is not None:
            phase_col = f'phase_{tone:04d}'
            I_col = f'I_{tone:04d}'
            Q_col = f'Q_{tone:04d}'
        else:
            phase_col = 'phase'
            I_col = 'I'
            Q_col = 'Q'

        if not phase_col in schema:
            return np.arctan2(pl.col(Q_col), pl.col(I_col)).alias(phase_col)
        else:
            return pl.col(phase_col)
    
    @staticmethod
    def calc_mag(tone, schema):
        if tone is not None:
            mag_col = f'mag_{tone:04d}'
            I_col = f'I_{tone:04d}'
            Q_col = f'Q_{tone:04d}'
        else:
            mag_col = 'mag'
            I_col = 'I'
            Q_col = 'Q'

        if not mag_col in schema:
            return (pl.col(I_col)**2 + pl.col(Q_col)**2).sqrt().alias(mag_col)
        else:
            return pl.col(mag_col)

    #==========================#
    # Lazily Loaded Attributes #
    #==========================#
    
    @cached_property
    def io_cfg(self) -> dict:
        return self._load_cfg('_io_')
    
    @cached_property 
    def ext_cfg(self) -> dict:
        return self._load_cfg('_ext_')

    @cached_property
    def drone_cfg(self) -> dict:
        return self._load_cfg('_drone_')

    @cached_property
    def num_tones(self) -> int:
        return self.drone_cfg['tones']['num_tones']

    @cached_property
    def comb(self) -> pl.dataframe.frame.DataFrame:
        '''
        Load comb frequencies, powers, and phases
        '''
        comb = {'tone_freqs': [], 'tone_powers': [], 'tone_phis': []}
        for key in comb.keys():
            value = self.drone_cfg['tones'][key]
            if isinstance(value, list):
                value = value.real
            else:
                try:
                    value = np.load(value).real
                except:
                    value = None
            comb[key] = value
        comb = pl.DataFrame(comb)
        return comb

    #================#
    # Helper Methods #
    #================#

    def _parse_tone(self, func_include, func_exclude, func_all, include: int | list[int] | None = None, exclude: int | list[int] | None = None, *args):
        if include is not None and exclude is not None:
            rfsoc_io.send_msg('ERROR', "Can't include and exclude tones. Must specify one or the other.")
        elif include:
            if isinstance(include, int): include = [include]
            return func_include(include, *args)
        elif exclude:
            if isinstance(exclude, int): exclude = [exclude]
            return func_exclude(exclude, *args)
        else:
            return func_all(*args)

    def _load_cfg(self, id: str) -> dict:
        '''Internal method for loading io, ext, or drone cfg file corresponding to data file.

        Args:
            id (str): Which config to load. Should be one of '_io_', '_ext_', '_drone_'.
        Returns:
            dict: Loaded config file. Returns empty dictionary if loading fails.
        '''
        cfg = {}
        for i, config_path in enumerate(self._configs):
            if id in str(config_path): 
                cfg = rfsoc_io.load_config(config_path)
                self._configs.pop(i)
                break
        return cfg