import pathlib
import numpy as np
import polars as pl
import concurrent.futures
import time
import pathlib
import re
import copy

from pathlib import Path
from scipy.signal import savgol_filter
from functools import cached_property
from collections.abc import Iterable
from typing import Callable, TypeAlias, Any
from math import ceil

# Local Imports
import ccatkidlib.io as io
import ccatkidlib.log as log
import ccatkidlib.utils as utils
import ccatkidlib.analysis.utils.pair as pair
import ccatkidlib.analysis.utils.dataframe as ccat_df
import ccatkidlib.analysis.utils.multiprocess as ccat_mp

from ccatkidlib.log import header
from ccatkidlib.utils import method_timer

CalcFunction: TypeAlias = Callable[[pl.Schema, Any], pl.Expr]

class Data:
    '''Class representing a raw RFSoC output data product (VNA sweep, target sweep, or timestream)

    Attributes:
        bid  (str): RFSoC board that took the data
        drid (str): RFSoC drone that took the data

        tones (int | list[int], optional): Which tones are loaded for target sweep and timestream data. Defaults to None
        data_path (pathlib.PosixPath): Path to data file
        data (polars.DataFrame): Polars DataFrame with loaded data
        timestamp (int): Timestamp of data file

        io_cfg    (dict): Loaded IO config
        ext_cfg   (dict): Loaded external config
        drone_cfg (dict): Loaded drone config

        comb (pl.DataFrame): Polars DataFrame with tone frequencies, powers, and phases
        num_tones (int): Number of tones used to take data
    '''

    def __init__(self, 
                 com_to: str,
                 cfg_path: str = str(Path(__file__).parents[1] / 'analysis_config.yaml'),
                 data_type: str | None = None,
                 timestamp: int | str | None = None,
                 data_path: str | pathlib.PosixPath | list[str] | list[pathlib.PosixPath] | None = None,
                 analysis_cfg: dict | None = None,
                 viz_cfg: dict | None = None,
                 **kwargs):
        ''' Initialize Data object by finding data file and associated config files

        Note:
            Either the full data path must be provided via the ``data_path`` key word argument or
            both the timestamp and data type must be provided via the ``data_type`` and ``timestamp`` key word arguments respectively

        Args:
            com_to (str): Drone that took the data. In form *'\<board>.\<drone>'*
            analysis_cfg (str, optional): Path to analysis config. Defaults to analysis config in *ccatkidlib/ccatkidlib/analysis* directory.

            data_type (str, optional): Type of data file. Should be one of 'vna', 'targ', 'timestream'.
            timestamp (int | str, optional): Timestamp of data file
            data_path (str | pathlib.PosixPath, optional): Full path to data file

            **kwargs: Key word arguments for finding data file. See below:
            root_data_dir (str, optional): Root directory where data is stored. Defaults to that specified in analysis config
            data_dir (str, optional): Directory where data is stored
            date (str, optional): Date data was taken
            sess_id (str, optional): ccatkidlib session ID of data
        
        Raises:
            ValueError: If multiple data files are specified with differing file types or timestamps
            FileNotFoundError: If any data files cannot be found 
        '''
        # Define data attributes
        # -----------------------
        self.bid, self.drid = com_to.split('.') # Baard and drone data was taken with

        self.tones = None
        self._data = None

        # Load configs and initialize logger
        # ----------------------------------
        self.analysis_cfg, self.viz_cfg = io.load_config(cfg_path)
        if analysis_cfg is not None: self.analysis_cfg = analysis_cfg
        if viz_cfg is not None: self.viz_cfg = viz_cfg
        
        self.root_dir = self.analysis_cfg['file_paths']['root_data_dir']
        self.padding = len(str(self.analysis_cfg['tones']['max_tones']*100))
        if not self.root_dir[-1] == '/': self.root_dir += '/'

        # If full data path is None, [], or "", find data file based on timestamp and (optional) file path parts
        # ------------------------------------------------------------------------------------------------------
        if not data_path:
            if data_type is None or timestamp is None:
                error = ("Both the data type ('vna', 'targ', or 'timestream') and the timestamp must be provided "
                         "or the full data path needs to be specified.")
                log.log('CRITICAL', error)
                raise ValueError(error)

            data_dir = '**'
            date = '**'
            sess_id = '**'

            for key, value in kwargs.items():
                if key == 'root_data_dir':
                    self.root_dir = value
                elif key == 'data_dir':
                    data_dir = value
                elif key == 'date':
                    date = value
                elif key == 'sess_id':
                    sess_id = value

            # Try to find data file using given information
            self.data_path = pair.get_data_file(com_to, timestamp, data_type, data_dir = data_dir, date = date, sess_id = sess_id, root_data_dir=self.root_dir)
            self.timestamp = timestamp
        else:
            self.data_path = data_path if isinstance(data_path, Iterable) and not isinstance(data_path, str) else [data_path]
            if not all((isinstance(path, str) or isinstance(path, pathlib.PosixPath)) for path in self.data_path):
                error = 'All data paths must be of type str or pathlib.PosixPath!'
                log.log('CRITICAL', error)
                raise ValueError(error)

            self.timestamp = io.get_timestamp(self.data_path[0])
            if not all(io.get_timestamp(path) == self.timestamp for path in self.data_path):
                error = 'All data paths must have the same timestamp!'
                log.log('CRITICAL', error)
                raise ValueError(error)              

        # Check that data path(s) exist
        # -----------------------------
        ftype = Path(self.data_path[0]).suffix
        for path in self.data_path:
            path = Path(path)
            if not path.exists():
                error = f'Could not find {data_type} file for board {self.bid}, drone {self.drid} with timestamp {timestamp}! Check that all optional file path segments are correct!'
                log.log('CRITICAL', error)
                raise FileNotFoundError(error)
            elif not path.suffix == ftype:
                error = f'All data files must have the same file type!'
                log.log('CRITICAL', error)
                raise ValueError(error)

        # Get io, ext, and drone configs associated with the sweep data file
        # ------------------------------------------------------------------
        self._configs = pair.get_config(self.data_path[0], all_cfg=False)

        log_dir = io.add_dir('log', 
                             str(self.data_path[0]), 
                             save_root = self.analysis_cfg['io']['file_logging']['logging_root_dir'],
                             data_root = self.root_dir,
                             sub_dirs=[""])
        log.setup_logging(Path(log_dir) / self.analysis_cfg['io']['file_logging']['logging_fname'], 
                          self.analysis_cfg['io']['file_logging']['data_level'], 
                          self.analysis_cfg['io']['terminal_logging']['data_level'],
                          name='analysis.data')

    #=====================#
    # Data Getter Methods #
    #=====================#

    def get_data(self,
                 col_name: str | list[str] = '.*',
                 include: int | list[int] | None = None, 
                 exclude: int | list[int] | None = None, 
                 strict=False) -> pl.DataFrame:
        '''Get the specified data columns from the ``data`` Polars DataFrame for the specified tones

        Note:
            Only one of ``include`` and ``exclude`` arguments can be specified. If neither are specified, returns data for all tones.

        Args:
            col_name (str | list[str], optional): List of data column names without tone number suffix (e.g., *'I'*, *'mag'*, *'phase'*). Defaults to all columns
            include  (int | list[int], optional): List of tones to include. Defaults to *None*
            exclude  (int | list[int], optional): List of tones to exclude. Defaults to *None*
            strict (bool, optional): Whether the column names must match ``col_name`` exactly. If *False*, the column names can have additional prefixes **(not suffixes)**. Defaults to *False* 
        Returns:
            return (pl.DataFrame): Polars DataFrame with specified data columns

        '''
        def _get_exclude(exclude):
            expr = []
            for tones in exclude:
                expr += [f"^{'' if strict else '.*'}{name}_{tones:0{self.padding}d}$" for name in col_name]
            return [pl.col([rf"^{'' if strict else '.*'}{name}_\d+.*$" for name in col_name]).exclude(expr)]

        def _include(include: list[int]):
            ''' Internal function for getting data columns when ``include`` is specified

            Args:
                include (list[int]): List of tones to get data for
                *args: Name of data column
            '''
            exclude = set(self.tones) - set(include)
            return _get_exclude(exclude)

        def _exclude(exclude: list[int]):
            ''' Internal function for getting data columns when ``exclude`` is specified

            Args:
                exclude (list[int]): List of tones to **not** get data for
                *args: Name of data column
            '''
            
            return _get_exclude(exclude)

        def _all():
            ''' Internal function for getting all data columns (neither ``include`` or ``exclude`` specified)

            Args:
                *args: Name of data column
            '''
            return _get_exclude([])

        col_name = [col_name] if isinstance(col_name, str) else col_name.copy() # Copy col_name list since it may be modified
        
        if self.tones is not None: 
            exprs=[]

            # Timestreams never have self.tones = None but **do** have columns (the time columns in particular) without tones so need to handle those seperately
            no_tone_name = r'^(sample|t|dt|fft_f)(?:_\d+)?$'
            pattern = re.compile(no_tone_name) # Create regex pattern

            # If a specified data column is in the no_tone_name list, add it to the list of Polars Exprs without additional processing
            for name in col_name[::-1]:
                if pattern.match(name):
                    exprs.append(pl.col(name))
                    col_name.remove(name)
                    
            # Parse data columns that have tones
            exprs += ccat_df.parse_tones(_include, _exclude, _all, include, exclude)
            return (self.data.lazy()
                             .select(*exprs)
                             .collect())
        else:
            return (self.data.lazy()
                             .select([pl.col(f"^{'' if strict else '.*'}{name}$") for name in col_name])
                             .collect())

    def tone(self,
             include: int | list[int] | None = None, 
             exclude: int | list[int] | None = None) -> pl.DataFrame:
        '''Get all of the data columns in the ``data`` Polars DataFrame for the specified tones

        Note:
            Only one of ``include`` and ``exclude`` arguments can be specified. If neither are specified, returns data for all tones.

        Args:
            include (int | list[int], optional): List of tones to include. Defaults to *None*
            exclude (int | list[int], optional): List of tones to exclude. Defaults to *None*
        Returns:
            return (pl.DataFrame | None): Polars DataFrame with specified data columns or *None* if ``tones`` is *None*
        '''
        if self.tones is not None:
            return self.get_data(col_name = '.*', include=include, exclude=exclude)
        else:
            log.log('ERROR', "Cannot load individual tone data when self.tones is None.")
            return None

    def I(self, 
          prefix: str | list[str] = '', 
          include: int | list[int] | None = None, 
          exclude: int | list[int] | None = None) -> pl.DataFrame:
        '''Get the in-phase ``I`` data for the specified tones

        Note:
            Only one of ``include`` and ``exclude`` arguments can be specified. If neither are specified, returns data for all tones.

        Args:
            include (int | list[int], optional): List of tones to include. Defaults to None
            exclude (int | list[int], optional): List of tones to exclude. Defaults to None
        Returns:
            return (pl.DataFrame): Polars DataFrame with specified data columns

        '''
        col_names = ['I']

        if isinstance(prefix, str): prefix = [prefix]
        col_names = [f"{pre}{'_' if pre else ''}{col_names[0]}" for pre in prefix]
        return self.get_data(col_name=col_names, include=include, exclude=exclude)

    def Q(self, 
          prefix: str | list[str] = '', 
          include: int | list[int] | None = None, 
          exclude: int | list[int] | None = None) -> pl.DataFrame:
        '''Get the quadrature ``Q`` data for the specifed tones

        Note:
            Only one of ``include`` and ``exclude`` arguments can be specified. If neither are specified, returns data for all tones.

        Args:
            include (int | list[int], optional): List of tones to include. Defaults to None
            exclude (int | list[int], optional): List of tones to exclude. Defaults to None
        Returns:
            return (pl.DataFrame): Polars DataFrame with specified data columns
        '''
        col_names = ['Q']

        if isinstance(prefix, str): prefix = [prefix]
        col_names = [f"{pre}{'_' if pre else ''}{col_names[0]}" for pre in prefix]
        return self.get_data(col_name='Q', include=include, exclude=exclude)

    def phase(self, 
              prefix: str | list[str] = '', 
              include: int | list[int] | None = None, 
              exclude: int | list[int] | None = None, 
              recalc: bool = False) -> pl.DataFrame:
        '''Calculate and get the phase *arctan(Q/I)* data for the specified tones.

        Note:
            Only one of ``include`` and ``exclude`` arguments can be specified. If neither are specified, returns data for all tones.

        Args:
            include (int | list[int], optional): List of tones to include. Defaults to *None*
            exclude (int | list[int], optional): List of tones to exclude. Defaults to *None*
        Returns:
            return (pl.DataFrame): Polars DataFrame with specified data columns

        '''

        col_name = ['I', 'Q', 'phase']
        if isinstance(prefix, str): prefix = [prefix]
        num_prefix = len(prefix)

        col_names = [[]]*num_prefix
        for i, pre in enumerate(prefix):
            col_names[i] = [f"{pre}{'_' if pre else ''}{name}" for name in col_name]
        self.transform([Data.calc_phase]*num_prefix, include=include, exclude=exclude, recalc=recalc, col_name = col_names)
        return self.get_data(col_name=[col_name[-1] for col_name in col_names], include=include, exclude=exclude)

    def mag(self, 
            prefix: str | list[str] = '', 
            include: int | list[int] | None = None, 
            exclude: int | list[int] | None = None, 
            dB: bool = False, 
            recalc: bool = False) -> pl.DataFrame:
        '''Calculate and get the magnitude *sqrt(I^2 + Q^2)* data of the specified tones

        Note:
            Only one of ``include`` and ``exclude`` arguments can be specified. If neither are specified, returns data for all tones.

        Args:
            include (int | list[int], optional): List of tones to include. Defaults to *None*
            exclude (int | list[int], optional): List of tones to exclude. Defaults to *None*
        Returns:
            return (pl.DataFrame): Polars DataFrame with specified data columns
        '''
        col_name = ['I', 'Q', f"{'dB_' if dB else ''}mag"]
        if isinstance(prefix, str): prefix = [prefix]
        num_prefix = len(prefix)

        col_names = [[]]*num_prefix
        for i, pre in enumerate(prefix):
            col_names[i] = [f"{pre}{'_' if pre else ''}{name}" for name in col_name]

        args = [[dB]]*num_prefix
        self.transform([Data.calc_mag]*num_prefix, *args, include=include, exclude=exclude, recalc=recalc, col_name = col_names)
        return self.get_data(col_name=[col_name[-1] for col_name in col_names], include=include, exclude=exclude)

    #============================#
    # General IQ Transformations #
    #============================#

    def IQ_rotate(self, prefix: str | list[str] = '', angle: float | list[float] | list[list[float]] = 0, name: str = '', include: int | list[int] | None = None, exclude: int | list[int] | None = None, recalc: bool = False):
        ''' Rotate the IQ data around the origin by the specified angle
        
        Args:
            prefix (str | list[str], optional): 
            angle (float | list[float] | list[list[float]], optional): Angle by which to rotate IQ data.  Defaults to 0
        '''
        
        col_name = ['I', 'Q', f"{name}{'_' if name else ''}rotate"]
        if isinstance(prefix, str): prefix = [prefix]
        num_prefix = len(prefix)

        if not isinstance(angle, Iterable) or not len(angle) == num_prefix: angle = [angle]

        col_names = [[]]*num_prefix
        for i, pre in enumerate(prefix):
            col_names[i] = [f"{pre}{'_' if pre else ''}{name}" for name in col_name[:-1]] + [col_name[-1]]

        args = [[a] for a in angle]
        self.transform([Data.calc_IQ_rotate]*num_prefix, *args, include=include, exclude=exclude, recalc=recalc, col_name = col_names)
        return self.get_data(col_name=[f'{col_name[-1]}_{col_name[0]}' for col_name in col_names] + 
                                      [f'{col_name[-1]}_{col_name[1]}' for col_name in col_names], include=include, exclude=exclude)
    
    def IQ_scale(self, prefix: str | list[str] = '', scale: float | list[float] | list[list[float]] = 1, name: str = '', include: int | list[int] | None = None, exclude: int | list[int] | None = None, recalc: bool = False):
        '''
        Divide IQ data by the specified scale factor

        Args:
            prefix ():
            scale (float | list[float] | list[list[float]]): Scale factor by which to divde IQ data
            name (str): Custom name of scaling. Will be added to the column name as a prefix. Defaults to ''
        
        '''
        
        col_name = ['I', 'Q', f"{name}{'_' if name else ''}scale"]
        if isinstance(prefix, str): prefix = [prefix]
        num_prefix = len(prefix)

        if not isinstance(scale, Iterable) or not len(scale) == num_prefix: scale = [scale]
        col_names = [[]]*num_prefix
        for i, pre in enumerate(prefix):
            col_names[i] = [f"{pre}{'_' if pre else ''}{name}" for name in col_name[:-1]] + [col_name[-1]]

        args = [[s] for s in scale]
        self.transform([Data.calc_IQ_scale]*num_prefix, *args, include=include, exclude=exclude, recalc=recalc, col_name = col_names)
        return self.get_data(col_name=[f'{col_name[-1]}_{col_name[0]}' for col_name in col_names] + 
                                      [f'{col_name[-1]}_{col_name[1]}' for col_name in col_names], include=include, exclude=exclude)

    def IQ_shift(self, prefix: str | list[str] = '', shift_I: float = 0, shift_Q: float = 0, name = '', include: int | list[int] | None = None, exclude: int | list[int] | None = None, recalc: bool = False):
        col_name = ['I', 'Q', f"{name}{'_' if name else ''}shift"]
        if isinstance(prefix, str): prefix = [prefix]
        num_prefix = len(prefix)

        if not isinstance(shift_I, Iterable) or not len(shift_I) == num_prefix: shift_I = [shift_I]
        if not isinstance(shift_Q, Iterable) or not len(shift_Q) == num_prefix: shift_Q = [shift_Q]
    
        col_names = [[]]*num_prefix
        for i, pre in enumerate(prefix):
            col_names[i] = [f"{pre}{'_' if pre else ''}{name}" for name in col_name[:-1]] + [col_name[-1]]

        args = [[I, Q] for I, Q in zip(shift_I, shift_Q)]
        self.transform([Data.calc_IQ_shift]*num_prefix, *args, include=include, exclude=exclude, recalc=recalc, col_name = col_names)
        return self.get_data(col_name=[f'{col_name[-1]}_{col_name[0]}' for col_name in col_names] + 
                                      [f'{col_name[-1]}_{col_name[1]}' for col_name in col_names], include=include, exclude=exclude)
    
    def IQ_trim(self, prefix: str | list[str] = '', lower_bound = 0, upper_bound = -1, name='', include: int | list[int] | None = None, exclude: int | list[int] | None = None, recalc: bool = False):
        def _neg_indexing(bounds, df_height):
            for i, bound in enumerate(bounds): 
                try:
                    bound = np.array(bound)
                    neg_inds = np.where(bound < 0)
                    bound[neg_inds] += df_height
                    bounds[i] = bound
                except:
                    if bound < 0: bounds[i] = bound + df_height
            return bounds
        
        col_name = ['sample', 'I', 'Q', f"{name}{'_' if name else ''}trim"]
        if isinstance(prefix, str): prefix = [prefix]
        num_prefix = len(prefix)

        if not isinstance(lower_bound, Iterable) or not len(lower_bound) == num_prefix: lower_bound = [lower_bound]
        if not isinstance(upper_bound, Iterable) or not len(upper_bound) == num_prefix: upper_bound = [upper_bound]

        # Handle negative indicing
        df_height = int(self.data.height)
        lower_bound = _neg_indexing(lower_bound, df_height)
        upper_bound = _neg_indexing(upper_bound, df_height)
        
        col_names = [[]]*num_prefix
        for i, pre in enumerate(prefix):
            col_names[i] = [col_name[0]] + [f"{pre}{'_' if pre else ''}{name}" for name in col_name[1:-1]] + [col_name[-1]]
        args = [[low, up] for low, up in zip(lower_bound, upper_bound)]
        self.transform([Data.calc_IQ_trim]*num_prefix, *args, include=include, exclude=exclude, recalc=recalc, col_name = col_names)
        return self.get_data(col_name=[f'{col_name[-1]}_{col_name[0]}' for col_name in col_names] + 
                                      [f'{col_name[-1]}_{col_name[1]}' for col_name in col_names], include=include, exclude=exclude)
    
    def diff(self, col_name: str, prefix: str | list[str] = '', include: int | list[int] | None = None, exclude: int | list[int] | None = None, recalc: bool = False):
        ''' Calculate the difference between adjacent elements of a specified data column
        
        Args:
            col_name (str): Name of data column
        '''
        col_name = [col_name, 'diff']
        if isinstance(prefix, str): prefix = [prefix]
        num_prefix = len(prefix)

        col_names = [[]]*num_prefix
        for i, pre in enumerate(prefix):
            col_names[i] = [f"{pre}{'_' if pre else ''}{col_name[0]}", col_name[-1]]
        self.transform([Data.calc_diff]*num_prefix, include=include, exclude=exclude, recalc=recalc, col_name = col_names)
        return self.get_data(col_name=[f'{col_name[-1]}_{col_name[0]}' for col_name in col_names], include=include, exclude=exclude)

    def savgol(self, 
               col_name: str, 
               prefix: str | list[str] = '', 
               window: int = 3, 
               k: int = 1, 
               deriv: int = 0, 
               include: int | list[int] | None = None, 
               exclude: int | list[int] | None = None, 
               recalc: bool = False, 
               max_workers=1, 
               ex=None):
        ''' Calculate the difference between adjacent elements of a specified data column
        
        Args:
            col_name (str): Name of data column
        '''
        col_name = [col_name, 'savgol']
        if isinstance(prefix, str): prefix = [prefix]
        num_prefix = len(prefix)

        if not isinstance(window, Iterable) or not len(window) == num_prefix: window = [window]*num_prefix
        if not isinstance(k, Iterable) or not len(k) == num_prefix: k = [k]*num_prefix
        if not isinstance(deriv, Iterable) or not len(deriv) == num_prefix: deriv = [deriv]*num_prefix

        col_names = [[]]*num_prefix
        for i, pre in enumerate(prefix):
            col_names[i] = [f"{pre}{'_' if pre else ''}{col_name[0]}", col_name[-1]]
        args = [[win, order, der, ccat_mp.check_max_workers(max_workers), ex] for win, order, der in zip(window, k, deriv)]
        self.transform([Data.calc_savgol]*num_prefix, *args, include=include, exclude=exclude, recalc=recalc, col_name = col_names, batch_size=len(self.tones))
        self.data = self._unnest([f'struct_{col_name[-1]}{der:01d}' for col_name, der in zip(col_names, deriv)])
        return self.get_data(col_name=[f'{col_name[-1]}{der}_{col_name[0]}' for col_name, der in zip(col_names, deriv)], include=include, exclude=exclude)

    #==================#
    # Analysis Methods #
    #==================#

    def transform(self, funcs: list[CalcFunction], *funcs_args, include: int | list[int] | None = None, exclude: int | list[int] | None = None, col_name: str | list[str] = [], batch_size: int | list[int] = [], recalc: bool = False) -> pl.DataFrame:
        '''Apply transformations specified by ``funcs`` argument to the specified tones

        Mostly for internal use, but can be used externally if wanting to perform multiple transformations in parallel to increase efficiency.

        Note:
            Only one of ``include`` and ``exclude`` arguments can be specified. If neither are specified, transforms data for all tones.

        Examples:
            To calculate the phase and magnitude of tones 1, 5, and 7, one would run::

            >>> data.transform(['calc_phase', 'calc_mag'], include=[1,5,7], col_name=[['I', 'Q'], ['I', 'Q']])

        Args:
            funcs (list[CalcFunction]): List of transformation functions to apply to data.
            *func_args (list[list], optional): Additional arguments to pass to funcs as a nested list. For funcs that don't take additional arguments, can specify [None].
            include (int | list[int], optional): List of tones to include. Defaults to None
            exclude (int | list[int], optional): List of tones to exclude. Defaults to None
        Returns:
            return (pl.DataFrame): Full ``data`` Polars DataFrame including columns with the transformed data
        '''
        
        def _get_expr(tones):
            expr = []

            for func, f_arg, name, size in zip(funcs, funcs_args, col_name, batch_size):
                num_batches = ceil(len(tones) / size)
                batches = [[]]*num_batches
                args = [[]]*num_batches
                for i in range(num_batches):
                    slice_left, slice_right = i*size, (i+1)*size
                    batches[i] = tones[slice_left:slice_right]
                    try:
                        args[i] = [a[slice_left:slice_right] if isinstance(a, Iterable) and not isinstance(a, str) else [a]*(slice_right-slice_left) for a in f_arg]
                    except IndexError: 
                        args[i] = [[None]*(slice_right-slice_left) for _ in range(len(f_arg))]
                        log.log('WARNING', 'More tones than arguments specified. Ensure that each tone has a corresponding argument.')

                for batch, arg  in zip(batches, args):
                    exp = func(schema, *arg, tones=batch, padding=self.padding, recalc=recalc, col_name = name)

                    # Handle funcs that return a list of expressions and funcs that return single expressions
                    if isinstance(exp, Iterable):
                        expr += exp
                    else:
                        expr.append(exp)
            return expr

        def _include(include: list[int]) -> list[pl.Expr]:
            '''Internal method for generating polars query expressions for the tones to include

            Args:
                include (list[int]): List of tones to include
                *args: Additional args to pass to query generating function
            Returns:
                list[pl.expr.expr.Expr]: List of query expressions
            '''

            return _get_expr(include)

        def _exclude(exclude: list[int]) -> list[pl.Expr]:
            '''Internal method for generating polars query expressions for the tones to exclude

            Args:
                exclude (list[int]): List of tones to exclude
                *args: Additional args to pass to query generating function
            Returns:
                list[pl.expr.expr.Expr]: List of query expressions
            '''
            
            tones = set(self.tones) - set(exclude)
            return _get_expr(tones)

        def _all() -> list[pl.Expr]:
            '''Internal method for generating polars query expressions for all tones
            Args:
                *args: Additional args to pass to query generating function
            Returns:
                list[pl.expr.expr.Expr]: List of query expressions
            '''
            return _get_expr(self.tones)

        def _check_len(arg_list, num_funcs, error: str = ''):
            if not len(arg_list) == num_funcs:
                if num_funcs == 1:
                    arg_list = [arg_list]
                else:
                    log.log('ERROR', error)
            return arg_list

        if callable(funcs): funcs = [funcs]

        # Parse funcs_args arg
        # --------------------        
        num_funcs = len(funcs)
        if len(funcs_args) == 0:
            funcs_args = num_funcs*[[None]]
        else:
            funcs_args = _check_len(funcs_args, num_funcs, error = ('When using multiple transformation functions with at least one requiring additional args,'
                                                                    'args must be provided for all functions. Use ``[None]`` for functions that do not take args.'))
        # Parse col_name arg
        # ------------------
        if isinstance(col_name, str): col_name = [col_name]
        col_name = _check_len(col_name, num_funcs, error = 'All column names used must be specified for each transformation.')

        # Parse batch_size arg
        # --------------------
        if isinstance(batch_size, int): batch_size = [batch_size]
        if len(batch_size) == 0:
            batch_size = num_funcs*[1]
        else:
            batch_size = _check_len(batch_size, num_funcs, error =  ('When using multiple transformation functions with at least unique batch size,'
                                                                     'batch_size must be provided for all functions.'))
        
        data = self.data.lazy()
        if self.tones is not None:
            schema = data.collect_schema()
            self.data = data.with_columns(*ccat_df.parse_tones(_include, _exclude, _all, include, exclude)).collect()
        else:
            self.data = data.with_columns(*[func(data.collect_schema(), *f_arg, tones = None, recalc = recalc, col_name = name) for func, f_arg, name in zip(funcs, funcs_args, col_name)]).collect()
        return self.data

    @staticmethod
    def calc_phase(schema: pl.Schema, *args, tones: list[int] | None = None, padding: int = 4, recalc: bool = False, col_name = ['I', 'Q', 'phase']) -> pl.Expr:
        ''' Generates pl.Expr for calculating the phase of a tone using I & Q data

        Args:
            schema (pl.Schema)
        '''
        
        
        if tones is not None:
            tone = tones[0]
            col_name = [f'{name}_{tone:0{padding}d}' for name in col_name]

        I_col, Q_col, phase_col = col_name

        if recalc or not (phase_col in schema):
            return pl.arctan2(pl.col(Q_col), pl.col(I_col)).alias(phase_col)
        else:
            return pl.col(phase_col)

    @staticmethod
    def calc_mag(schema: pl.Schema, *args, tones: list[int] | None = None, padding: int = 4, recalc: bool = False, col_name = ['I', 'Q', 'mag']) -> pl.Expr:
        ''' Generates pl.Expr for calculating the magnitude of a tone using I & Q data
        '''
    
        if tones is not None:
            tone = tones[0]
            col_name = [f'{name}_{tone:0{padding}d}' for name in col_name]
        
        I_col, Q_col, mag_col = col_name

        if len(args) == 1:
            dB = args[0]
            if tones is not None: dB = dB[0]
        else:
            log.log('ERROR', 'dB is a required argument.')

        if recalc or not (mag_col in schema):
            mag_expr = (pl.col(I_col)**2 + pl.col(Q_col)**2).sqrt()
            if dB: mag_expr = pl.lit(20)*mag_expr.log10()
            return mag_expr.alias(mag_col)            
        else:
            return pl.col(mag_col)

    @staticmethod
    def calc_IQ_rotate(schema: pl.Schema, *args, tones: list[int] | None = None, padding: int = 4, recalc: bool = False, col_name = ['I', 'Q', 'rotate']) -> pl.Expr:        
        if tones is not None:
            tone = tones[0]
            col_name = [f'{name}_{tone:0{padding}d}' for name in col_name[:-1]] + [col_name[-1]]

        I_col, Q_col, rotate_col = col_name

        if len(args) == 1:
            angle = args[0]
            if tones is not None: angle = angle[0]
        else:
            log.log('ERROR', 'angle is a required argument.')
        if recalc or not (f'{rotate_col}_{I_col}' in schema):
            return [(pl.col(I_col)*pl.lit(angle).cos() - pl.col(Q_col)*pl.lit(angle).sin()).alias(f'{rotate_col}_{I_col}'),
                    (pl.col(I_col)*pl.lit(angle).sin() + pl.col(Q_col)*pl.lit(angle).cos()).alias(f'{rotate_col}_{Q_col}')]
        else:
            return pl.col(f'{rotate_col}_{I_col}')    

    @staticmethod
    def calc_IQ_shift(schema: pl.Schema, *args, tones: list[int] | None = None, padding: int = 4, recalc: bool = False, col_name = ['I', 'Q', 'shift']) -> pl.Expr:
        if tones is not None:
            tone = tones[0]
            col_name = [f'{name}_{tone:0{padding}d}' for name in col_name[:-1]] + [col_name[-1]]

        I_col, Q_col, shift_col = col_name

        if len(args) == 2:
            I_shift, Q_shift = args
            if tones is not None: I_shift, Q_shift = I_shift[0], Q_shift[0]
        else:
            log.log('ERROR', 'I_shift and Q_shift are required arguments.')

        if recalc or not (f'{shift_col}_{I_col}' in schema):
            return [(pl.col(I_col) + I_shift).name.prefix(shift_col + '_'),
                    (pl.col(Q_col) + Q_shift).name.prefix(shift_col + '_')]
        else:
            return pl.col(f'{shift_col}_{I_col}')

    @staticmethod
    def calc_IQ_scale(schema: pl.Schema, *args, tones: list[int] | None = None, padding: int = 4, recalc: bool = False, col_name = ['I', 'Q', 'scale']) -> pl.Expr:
        if tones is not None:
            tone = tones[0]
            col_name = [f'{name}_{tone:0{padding}d}' for name in col_name[:-1]] + [col_name[-1]]

        I_col, Q_col, scale_col = col_name

        if len(args) == 1:
            scale = args[0]
            if tones is not None: scale = scale[0]
        else:
            log.log('ERROR', 'scale is a required argument.')

        if recalc or not (f'{scale_col}_{I_col}' in schema):
            return (pl.col([I_col, Q_col]) * scale).name.prefix(f'{scale_col}_')
        else:
            return pl.col(f'{scale_col}_{I_col}')

    @staticmethod
    def calc_IQ_trim(schema: pl.Schema, *args, tones: list[int] | None = None, padding: int = 4, recalc: bool = False, col_name = ['sample', 'I', 'Q', 'trim']) -> pl.Expr:
        if tones is not None:
            tone = tones[0]
            col_name = [col_name[0]] + [f'{name}_{tone:0{padding}d}' for name in col_name[1:-1]] + [col_name[-1]]

        sample_col, I_col, Q_col, trim_col = col_name
        if len(args) == 2:
            lower_bound, upper_bound = args
            if tones is not None: lower_bound, upper_bound = lower_bound[0], upper_bound[0]
        else:
            log.log('ERROR', 'lower_bound, upper_bound, and tone_list are required arguments.')

        if recalc or not (f'{trim_col}_{I_col}' in schema):
            return [pl.when(pl.col(sample_col).is_between(pl.lit(lower_bound), pl.lit(upper_bound), closed='none'))
                      .then(pl.col(col))
                      .otherwise(pl.lit(None))
                      .alias(f'{trim_col}_{col}') for col in [I_col, Q_col]]  
        else:
            return pl.col(f'{trim_col}_{I_col}')

    @staticmethod
    def calc_diff(schema: pl.Schema, *args, tones: list[int] | None = None, padding: int = 4, recalc: bool = False, col_name = ['', 'diff']) -> pl.Expr:
        ''' Generates pl.Expr for calculating difference between adjacent data points for the specified column
        Args:
            schema (pl.Schema)
        '''
        if tones is not None:
            tone = tones[0]
            col_name = [f'{name}_{tone:0{padding}d}' for name in col_name[:-1]] + [col_name[-1]]

        data_col, diff_col = col_name
        if recalc or not (f'{diff_col}_{data_col}' in schema):
            return pl.col(data_col).diff().name.prefix(f'{diff_col}_')
        else:
            return pl.col(f'{diff_col}_{data_col}')
        
    @staticmethod
    def calc_savgol(schema: pl.Schema, *args, tones: list[int] | None = None, padding: int = 4, recalc: bool = False, col_name = ['', 'savgol']) -> pl.Expr:
        def _calc_savgol(df):
            data = ccat_mp.struct_batches(df, 1, batch_len, max_workers)
            results_dict = {}
            with ccat_mp.optional_executor(max_workers, ex=ex) as executor:
                future_to_batch = {executor.submit(ccat_mp.process_batches,
                                                   savgol_filter, 
                                                   data[i][0], 
                                                   window[inds], 
                                                   k[inds], 
                                                   deriv = deriv[inds]):  (tones, cols) for i, (tones, inds, (cols)) in enumerate(zip(to_calc, calc_ind, batches))}

                for future in concurrent.futures.as_completed(future_to_batch):
                    tones, cols = future_to_batch[future]
                    filtered_cols = future.result()
                    for tone, col, filtered_col in zip(tones, cols, filtered_cols):
                        if isinstance(filtered_col, Exception):
                            log.log('WARNING', 'Savgol filter for tone %s failed with exception: %s', tone, filtered_col)
                            filtered_col = np.full(df.len(), np.nan)
                        results_dict[f'{col_name[-1]}_{col[0]}'] = filtered_col
            return ccat_mp.package_results(results_dict)

        col_name = col_name.copy()
        if len(args) == 5:
            window, k, deriv, max_workers, ex = np.array(args)
            if tones is not None: max_workers, ex = int(max_workers[0]), ex[0]
            col_name[-1] = f'{col_name[-1]}{deriv[0]}'
        else:
            log.log('ERROR', 'window, k, deriv, and max_workers are required arguments.')

        return_col, return_type = [f'{col_name[-1]}_{col_name[-2]}'], [pl.Float64]
        expr, to_calc, calc_ind, calc_col, batches, batch_len = ccat_mp.create_batches(_calc_savgol, tones, col_name, schema, padding=padding, return_col=return_col, return_type=return_type, max_workers=max_workers, recalc=recalc)
        return expr

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
        return self.drone_cfg.get('tones', {'num_tones': -1})['num_tones']
        
    @cached_property
    def original_root(self) -> str:
        try:
            original_root = self.io_cfg['file_paths']['root_dir']
        except:
            # Fall back to using original root data directory specified in analysis config for old data files
            original_root = self.analysis_cfg['file_paths']['original_root_data_dir']
        if not original_root[-1] == '/': original_root += '/'
        return original_root

    @cached_property
    def fig_dir(self) -> str:
        '''
        Directory where figures should be saved. Create if it does not already exist.
        '''

        return io.add_dir('fig', 
                          str(self.data_path[0]), 
                          save_root = self.viz_cfg['save']['fig_root_dir'],
                          data_root = self.root_dir,
                          timestamp = str(self.timestamp))
    
    @cached_property
    def pickle_dir(self) -> str:
        '''
        Directory where pickle files should be saved. Create if it does not already exist.
        '''

        pickle_dir = io.add_dir('pickle', 
                                str(self.data_path[0]), 
                                save_root = self.analysis_cfg['io']['pickle']['pickle_root_dir'],
                                data_root = self.root_dir,
                                timestamp = str(self.timestamp))
        io.create_dir(Path(pickle_dir) / 'dataframe')
        return pickle_dir

    @cached_property
    def comb(self) -> pl.DataFrame:
        '''
        Load comb frequencies, powers, and phases
        '''
        comb = {'tone_freqs': [], 'tone_powers': [], 'tone_phis': []}
        for key in comb.keys():
            value = self.drone_cfg.get('tones', {f'{key}': []})[key]
            if isinstance(value, list):
                value = np.array(value).real
            elif isinstance(value, str):
                comb_path = pair.replace_root(value, self.original_root, self.root_dir)
                value = np.load(comb_path).real if Path(comb_path).exists() else np.zeros(self.num_tones)
            else:
                value = np.zeros(self.num_tones)
            comb[key] = value if self.tones is None else value[self.tones]
        comb['det'] = range(len(comb['tone_freqs'])) if self.tones is None else self.tones
        comb = pl.DataFrame(comb)
        return comb

    #==========#
    # Plotting #
    #==========#

    def _get_plot_df(self, col_dict: dict, x_prefix: str = '', y_prefix: str = '', unpivot_x: bool = True, include: int | list[int] | None = None, exclude: int | list[int] | None = None) -> tuple[pl.DataFrame, str | None]:
        '''
        
        '''
        
        # Get frequency and magnitude data
        x_df = self.get_data([col_dict['sample'], f"{x_prefix}{'_' if x_prefix else ''}{col_dict['x']}"], include=include, exclude=exclude, strict=True)
        y_df = self.get_data([col_dict['sample'], f"{y_prefix}{'_' if y_prefix else ''}{col_dict['y']}"], include=include, exclude=exclude, strict=True)
        on, by = [col_dict['sample']], None

        # Convert frequency and magnitude DataFrames from wide to long
        if self.tones is None:
            y_df = y_df.rename({f"{y_prefix}{'_' if y_prefix else ''}{col_dict['y']}": col_dict['y']})
        elif unpivot_x:
            x_df = self.unpivot(x_df, col_dict['x'], index_cols=[col_dict['sample']])
            y_df = self.unpivot(y_df, col_dict['y'], index_cols=[col_dict['sample']])
            on += ['det']
            if include is not None or exclude is not None: by = 'det'

        # Combine into a single DataFrame
        df = x_df.join(y_df, on=on, how='left')

        if not unpivot_x: 
            df = df.rename({f"{x_prefix}{'_' if x_prefix else ''}{col_dict['x']}": col_dict['x']})
            df = self.unpivot(df, col_dict['y'], index_cols=[col_dict['sample'], col_dict['x']])
            on += ['det']
            if include is not None or exclude is not None: by = 'det'

        return df, by

    #================#
    # Helper Methods #
    #================#

    def unpivot(self, df: pl.DataFrame, data_name: str, index_cols: list[str] = ['sample']) -> pl.DataFrame:
        '''
        
        '''
        
        unpivot_cols = df.columns
        for col in index_cols: unpivot_cols.remove(col)
        prefix = self.get_prefix(unpivot_cols[0], data_name)
        value_name = data_name if not data_name in df.columns else data_name*2

        df = (df.unpivot(index=index_cols,
                         on=unpivot_cols,
                         variable_name='det',
                         value_name=value_name))
        df = df.with_columns(pl.col('det').str.strip_prefix(f"{prefix}{'_' if prefix else ''}{data_name}_").cast(int))
        return df

    def get_prefix(self, col_name: str, data_name: str) -> str:
        ''' Get the prefix of the given column name

        Assumes that the column name is of the form {prefix}_{data_name}_{tone}

        Args:
            col_name (str): Full name of the column
            data_name(str): Data that column corresponds to (e.g., 'phase' or 'f')
        Returns:
            return (str): Prefix of the column name
        '''

        segments = 0
        if self.tones is not None: segments -= 1
        segments -= len(data_name.split('_'))
        return '_'.join(col_name.split('_')[:segments])

    def _unnest(self, col_names: str | list[str]) -> pl.DataFrame:
        '''
        
        '''
        struct_cols = []
        if isinstance(col_names, str): col_names = [col_names]
        for col_name in col_names:
            schema = self.data.schema
            for name, data in schema.items():
                if isinstance(data, pl.Struct) and col_name in name:
                    self.data = self.data.drop([col for col in dict(data).keys() if col in schema])
                    struct_cols.append(name)
        return self.data.unnest(struct_cols)

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
                cfg = io.load_config(config_path)
                self._configs.pop(i)
                break
        return cfg

    def join(self, other, in_place=False):
        ''' 
        Join two Data objects together.

        Args:
            other (Data): Data object to join with
        Returns:
            return (Data): New Data object created from the two joined Data objects
        '''
        def _join_consts(left_const: Any | list[Any], right_const: Any | list[Any]) -> list[Any]:
            if not isinstance(left_const, list): left_const = [left_const]
            if not isinstance(right_const, list): right_const = [right_const]
            return left_const + right_const

        def _join_cfg(left_cfg: dict, right_cfg: dict) -> dict:
            if len(left_mapping) == 1: 
                left_cfg['index'] = 0
                left_cfg = {f'{self.bid}.{self.drid}': {str(self.timestamp): {**left_cfg}}}
            
            left_ind = left_mapping[-1] + 1
            if len(right_mapping) == 1:
                right_cfg['index'] = int(left_ind)
                right_cfg = {f'{other.bid}.{other.drid}': {str(other.timestamp): {**right_cfg}}}
            else:
                for i, (bid, drid, timestamp) in enumerate(zip(other.bid, other.drid, other.timestamp)):
                    utils.dict_set(right_cfg, [f'{bid}.{drid}', str(timestamp), 'index'], int(i + left_ind))

            return left_cfg | right_cfg

        if not isinstance(other, Data):
            error = f'Cannot join with object of type {type(other)}. Must be of type Data.'
            log.log('ERROR', error)
            raise ValueError(error)
        elif self.tones is None or other.tones is None:
            error = f'Both Data objects must have tones not equal to None.'
            log.log('ERROR', error)
            raise NotImplementedError(error)

        # Create a copy of the Data object
        new_data = self if in_place else copy.deepcopy(self) 

        # Load data before joining to prevent in-place errors
        # ---------------------------------------------------
        left_df, right_df = self.data, other.data 
        left_comb, right_comb = self.comb, other.comb

        # Join tones arrays
        # -----------------
        max_tones = self.analysis_cfg['tones']['max_tones']
        left_tones, right_tones = np.array(self.tones), np.array(other.tones)
        left_mapping, right_mapping = np.unique(left_tones // max_tones), np.unique(right_tones // max_tones)
        shift = (left_mapping[-1] - right_mapping[0] + 1)        
        new_data.tones = list(np.append(left_tones, right_tones + shift * max_tones))
        
        # Join data DataFrames
        # --------------------
        if len(str(new_data.tones[-1])) == new_data.padding: 
            new_data.padding += 1
            left_df.rename({col: re.sub(r"(0\d+)$", lambda tone: f'{int(tone.group(1)):0{new_data.padding}d}', col) for col in left_df.columns})
        right_df = right_df.rename({col: re.sub(r"(0\d+)$", lambda tone: f'{int(tone.group(1)) + shift*max_tones:0{new_data.padding}d}', col) for col in right_df.columns})
        new_data.data = left_df.join(right_df, on='sample', how='full', coalesce=True)

        # Join comb DataFrames
        # --------------------
        new_data.comb = pl.concat([left_comb, right_comb], how='diagonal').with_columns(pl.Series('det', new_data.tones))

        # Join configs
        # ------------
        new_data.ext_cfg = _join_cfg(self.ext_cfg, other.ext_cfg)
        new_data.io_cfg = _join_cfg(self.io_cfg, other.io_cfg)
        new_data.drone_cfg = _join_cfg(self.drone_cfg, other.drone_cfg)

        # Join constant attributes
        # ------------------------
        consts = ['bid', 'drid', 'timestamp', 'num_tones', 'data_path']
        for const in consts:
            left_const, right_const = getattr(self, const), getattr(other, const)
            setattr(new_data, const, _join_consts(left_const, right_const))

        return new_data

    # ------------------------------- #
    # Define Custom Pickling Behavior #
    # ------------------------------- #

    def __getstate__(self):
        state = self.__dict__.copy()
        if not self.analysis_cfg['io']['pickle']['pickle_dataframes'] and (data := self._data) is not None:
            del state['_data']
            save_path, file_count = io.increment_file(Path(self.pickle_dir) / 'dataframe',
                                                      f'data_', 
                                                      '.parquet',
                                                      overwrite=self.analysis_cfg['io']['pickle']['overwrite'])
            state['pickle_count'] = file_count
            data.write_parquet(save_path)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not self.analysis_cfg['io']['pickle']['pickle_dataframes'] and getattr(self, '_data', True):
            file_name = 'data' if (pickle_count := self.pickle_count) is None else f'data_{pickle_count}'

            analysis_cfg, _ = io.load_config(str(Path(__file__).parents[1] / 'analysis_config.yaml'))
            if (curr_dir := analysis_cfg['io']['pickle']['curr_pickle_root_dir']): self.analysis_cfg['io']['pickle']['pickle_root_dir'] = curr_dir

            self.data = pl.scan_parquet(Path(self.pickle_dir) / 'dataframe' / f'{file_name}.parquet')





