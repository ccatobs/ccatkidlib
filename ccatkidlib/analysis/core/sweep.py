import sys
import numpy as np
import gc
import polars as pl

from pathlib import Path
from functools import cached_property

# Bokeh Imports
from bokeh.models import CheckboxButtonGroup, CustomJS, ColumnDataSource
from bokeh.layouts import layout, column
from bokeh.io import show
from bokeh.plotting import curdoc

# Local Imports

import ccatkidlib.rfsoc_io as rfsoc_io
import ccatkidlib.utils as utils
import ccatkidlib.analysis.pair as pair

from ccatkidlib.analysis.core.data import Data


class Sweep(Data):
    '''Class representing a sweep (VNA or target) taken using a Radio Frequency System on a Chip (RFSoC).

    Subclasses the general ccatkidlib Data class.
    '''

    def __init__(self, com_to: str, analysis_cfg: str = str(Path(__file__).parents[1] / 'analysis_config.yaml'), **kwargs):
        super().__init__(com_to, analysis_cfg, **kwargs)
        
    #==================#
    # Plotting Methods #
    #==================#
    
    #==========================#
    # Lazily Loaded Attributes #
    #==========================#

    @property
    def data(self) -> pl.lazyframe.frame.LazyFrame:
        if self._data is None:
            data = {'sample': [], 'f': [], 'I': [], 'Q': []}
            fs, s21z = np.load(self.data_path[0], mmap_mode='r')
            I, Q = s21z.real, s21z.imag

            data['sample'], data['f'], data['I'], data['Q'] = range(len(fs)), fs.real, I, Q
            self._data = pl.DataFrame(data)
        return self._data

    @data.setter
    def data(self, value: pl.lazyframe.frame.LazyFrame | None): 
        if value is None or isinstance(value, pl.dataframe.frame.DataFrame): 
            self._data = value
        else:
            rfsoc_io.send_msg('ERROR', 'Cannot set data with type %s. Must be a Polars LazyFrame! Convert DataFrame to lazy frame with .lazy() before setting.', type(value))

    @cached_property
    def det_f(self) -> np.ndarray:
        '''Found detector frequencies by find_resonators or find_resonators_fine

        Note:
            The found detector frequencies are ``NOT`` necessarily the same as the tone frequencies of the sweep!
        
        Returns:
            np.ndarray: Array of found detector frequencies

        Raises:
            FileNotFoundError: Unable to load file with found detector frequencies
        '''

        det_f = self.drone_cfg['det_config']['found_detector_freqs']
        if isinstance(det_f, list):
            det_f = np.real(det_f)
        else:
            try:
                f_path = pair.replace_root(det_f, self.original_root, self.root_dir)
                det_f = np.real(np.load(det_f))
            except:
                error = f'Failed to load detector frequencies file {det_f}.'
                rfsoc_io.send_msg('ERROR', error)
                raise FileNotFoundError(error)
        return det_f
    
    #=====================#
    # Data Getter Methods #
    #=====================#

    def f(self, include = None, exclude = None):
        return self.get_data(col_name='f', include=include, exclude=exclude)
