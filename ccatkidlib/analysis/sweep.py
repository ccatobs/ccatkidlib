import sys
import numpy as np
import gc
import polars as pl

from typing import override
from pathlib import Path

# Bokeh Imports
from bokeh.models import CheckboxButtonGroup, CustomJS, ColumnDataSource
from bokeh.layouts import layout, column
from bokeh.io import show
from bokeh.plotting import curdoc

# Local Imports

import ccatkidlib.rfsoc_io as rfsoc_io
import ccatkidlib.utils as utils
import ccatkidlib.analysis.pair as pair
import ccatkidlib.analysis.plot_utils as putils

from ccatkidlib.analysis.data import Data


class Sweep(Data):
    '''Class representing a sweep (VNA or target) taken using a Radio Frequency System on a Chip (RFSoC).

    Subclasses the general ccatkidlib Data class.
    '''

    @override
    def __init__(self, com_to: str, analysis_cfg: str = str(Path(__file__).parent / 'analysis_config.yaml'), **kwargs):
        super().__init__(com_to, analysis_cfg, **kwargs)

        self.res_freqs = None
        self.res_s21z  = None
        
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
            fs, s21z = np.load(self.data_path, mmap_mode='r')
            I, Q = s21z.real, s21z.imag

            data['sample'], data['f'], data['I'], data['Q'] = range(len(fs)), fs.real, I, Q
            self._data = pl.DataFrame(data).lazy()
        return self._data

    @data.setter
    def data(self, value: pl.lazyframe.frame.LazyFrame | None): 
        if value is None or isinstance(value, pl.lazyframe.frame.LazyFrame): 
            self._data = value
        else:
            rfsoc_io.send_msg('ERROR', 'Cannot set data with type %s. Must be a Polars LazyFrame! Convert DataFrame to lazy frame with .lazy() before setting.', type(value))

    #===============================#
    # Internal Data Loading Methods #
    #===============================#
    def _load_res_freqs(self):
        '''
        '''
        res_freqs = self.drone_cfg['det_config']['found_detector_freqs']
        if isinstance(res_freqs, list):
            res_freqs = np.real(res_freqs)
        else:
            try:
                res_freqs = np.real(np.load(res_freqs))
            except:
                res_freqs = None
        return res_freqs
    
    def _get_res_s21z(self):
        res_s21z = None
        res_freqs = self.res_freqs
        if res_freqs is not None and len(res_freqs) > 0:
            res_s21z = [self.s21z[np.argmin(np.abs(self.freqs - freq))] for freq in res_freqs]
        return res_s21z
    #=====================#
    # Data Getter Methods #
    #=====================#

    def f(self, include = None, exclude = None):
        return self.get_data(col_name='f', include=include, exclude=exclude)
