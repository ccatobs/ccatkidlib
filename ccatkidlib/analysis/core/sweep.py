import sys
import numpy as np
import gc
import polars as pl

from pathlib import Path
from functools import cached_property

import holoviews as hv
from holoviews import opts

# Local Imports

import ccatkidlib.rfsoc_io as rfsoc_io
import ccatkidlib.utils as utils
import ccatkidlib.analysis.utils.pair as pair

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

    def plot(self, x_dim, y_dim, x_prefix: str = '', y_prefix: str = '', include: int | list[int] | None = None, exclude: int | list[int] | None = None):
        col_dict = {'sample': 'sample',
                    'x': x_dim,
                    'y': y_dim}

        df, by = self._get_plot_df(col_dict, x_prefix = x_prefix, y_prefix = y_prefix, include = include, exclude = exclude)
        col_dict['x'], col_dict['y'] = df.select(pl.exclude('det', 'sample')).columns    
        df = df.filter((~pl.col(col_dict['x']).is_nan()) & (~pl.col(col_dict['y']).is_nan()))

        tone_sample = int((self.drone_cfg['tones']['sweep_steps']-1)/2)
        df = (df.with_columns(pl.when(pl.col(col_dict['sample']) == tone_sample)
                               .then(pl.lit('diamond_dot'))
                               .otherwise(pl.lit('circle'))
                               .alias('markers'))
                .with_columns(pl.when(pl.col('markers') == 'diamond_dot')
                               .then(pl.lit(400))
                               .otherwise(pl.lit(20))
                               .alias('size')))

        # Create HoloViews plot objects
        line = df.hvplot.line(x=col_dict['x'],
                              y=col_dict['y'],
                              by=by,
                              label='Curve',
                              width=self.viz_cfg['plot']['width'],
                              height=self.viz_cfg['plot']['height'])

        scatter = df.hvplot.scatter(x=col_dict['x'],
                                    y=col_dict['y'],
                                    by=by,
                                    s='size',
                                    scale=1,
                                    label='Scatter',
                                    width=self.viz_cfg['plot']['width'],
                                    height=self.viz_cfg['plot']['height'])
        
        overlay = hv.Overlay([line, scatter])

        cfg = self.drone_cfg['det_config']
        title = rf"${cfg['detector_type']}\ {cfg['network']}$"

        if not (include is None and exclude is None):
            overlay.NdOverlay.Curve.opts(opts.Curve(title=title))
            overlay.NdOverlay.Scatter.opts(opts.Scatter(title=title))
        else:
            overlay.Curve.Curve.opts(opts.Curve(title=title))
            overlay.Scatter.Scatter.opts(opts.Scatter(title=title))

        return overlay, df

    def mag_plot(self, prefix: str = '', include: int | list[int] | None = None, exclude: int | list[int] | None = None, return_df = False):
        overlay, df = self.plot('f', 'mag', y_prefix=prefix, include=include, exclude=exclude)
        xlabel = r'$Frequency\ [Hz]$'
        ylabel = r'$|S_{21}|$'

        curve_opts = opts.Curve(xlabel=xlabel,
                                ylabel=ylabel)
        scatter_opts = opts.Scatter(xlabel=xlabel,
                                    ylabel=ylabel)

        if not (include is None and exclude is None):
            overlay.NdOverlay.Curve.opts(curve_opts)
            overlay.NdOverlay.Scatter.opts(scatter_opts)
        else:
            overlay.Curve.Curve.opts(curve_opts)
            overlay.Scatter.Scatter.opts(scatter_opts)

        if return_df:
            return overlay, df
        else:
            return overlay
    
    def phase_plot(self, prefix: str = '', include: int | list[int] | None = None, exclude: int | list[int] | None = None, return_df = False):
        overlay, df = self.plot('f', 'phase', y_prefix=prefix, include=include, exclude=exclude)
        xlabel = r'$Frequency\ [Hz]$'
        ylabel = r'$Phase\ [rad]$'

        curve_opts = opts.Curve(xlabel=xlabel,
                                ylabel=ylabel)
        scatter_opts = opts.Scatter(xlabel=xlabel,
                                    ylabel=ylabel)

        if not (include is None and exclude is None):
            overlay.NdOverlay.Curve.opts(curve_opts)
            overlay.NdOverlay.Scatter.opts(scatter_opts)
        else:
            overlay.Curve.Curve.opts(curve_opts)
            overlay.Scatter.Scatter.opts(scatter_opts)

        if return_df:
            return overlay, df
        else:
            return overlay
    
    def IQ_plot(self, prefix: str = '', include: int | list[int] | None = None, exclude: int | list[int] | None = None, return_df = False):
        overlay, df = self.plot('I', 'Q', x_prefix=prefix, y_prefix=prefix, include=include, exclude=exclude)
        xlabel = r'$I\ [arb]$'
        ylabel = r'$Q\ [arb]$'

    
        I_min, I_max = df.select(pl.col('I').min().alias('min'), pl.col('I').max().alias('max'))[0].to_numpy()[0]
        Q_min, Q_max = df.select(pl.col('Q').min().alias('min'), pl.col('Q').max().alias('max'))[0].to_numpy()[0]

        I_diff = I_max - I_min
        Q_diff = Q_max - Q_min
        I_avg = (I_min + I_max)/2
        Q_avg = (Q_min + Q_max)/2

        curve_opts = opts.Curve(xlabel=xlabel,
                                ylabel=ylabel,
                                aspect=1)
        scatter_opts = opts.Scatter(xlabel=xlabel,
                                    ylabel=ylabel,
                                    aspect=1)

        if not (include is None and exclude is None):
            overlay.NdOverlay.Curve.opts(curve_opts)
            overlay.NdOverlay.Scatter.opts(scatter_opts)
        else:
            overlay.Curve.Curve.opts(curve_opts)
            overlay.Scatter.Scatter.opts(scatter_opts)

        if return_df:
            return overlay, df
        else:
            return overlay

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
