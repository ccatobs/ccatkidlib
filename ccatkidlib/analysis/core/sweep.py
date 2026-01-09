import gc
import sys
import numpy as np
import polars as pl

from pathlib import Path
from functools import cached_property

import holoviews as hv
from holoviews import opts

# Local Imports

import ccatkidlib.rfsoc_io as rfsoc_io
import ccatkidlib.analysis.utils.pair as pair
import ccatkidlib.utils as utils
import ccatkidlib.analysis.viz.viz_utils as viz_utils

from ccatkidlib.analysis.core.data import Data


class Sweep(Data):
    '''Class representing a sweep (VNA or target) taken using a Radio Frequency System on a Chip (RFSoC).

    Subclasses the general ccatkidlib Data class.
    '''

    def __init__(self, com_to: str, analysis_cfg: str = str(Path(__file__).parents[1] / 'analysis_config.yaml'), **kwargs):
        super().__init__(com_to, analysis_cfg, **kwargs)
        self.save_dir = Path(self.save_dir) / f'sweep_{self.timestamp}'
        rfsoc_io.create_dir(self.save_dir)
        
    #==================#
    # Plotting Methods #
    #==================#
        
    @staticmethod
    def _plot(df, x_dim, y_dim, plot_opts, **kwargs):
        tone_marker, tone_ms = kwargs.pop('tone_marker'), kwargs.pop('tone_ms')

        sweep = df.hvplot.line(x=x_dim,
                               y=y_dim,
                               label='Data',
                               **kwargs).relabel(group='Sweep')

        kwargs['marker'], kwargs['ms'], kwargs['linewidth'] = tone_marker, tone_ms, 0

        tone = (df.filter(pl.col('tone'))
                  .hvplot.line(x=x_dim,
                               y=y_dim,
                               label='Tone',
                               **kwargs)).relabel(group='Sweep')
        plot = sweep*tone
        plot.opts(*plot_opts)

        return plot

    def plot(self, 
             x_dim: str,
             y_dim: str, 
             x_prefix: str = '', 
             y_prefix: str = '',
             xlabel: str | None = None,
             ylabel: str | None = None,
             grouping: str = 'groupby',
             include: int | list[int] | None = None, 
             exclude: int | list[int] | None = None, 
             return_df = False,
             plot_opts = None,
             save_fig: bool | None = None,
             overwrite: bool | None = None,
             save_name: str = None,
             **kwargs):
        '''
        
        '''
        
        # Get DataFrame with data to plot
        # -------------------------------
        col_dict = {'sample': 'sample',
                    'x': x_dim,
                    'y': y_dim}

        df, by = self._get_plot_df(col_dict, x_prefix = x_prefix, y_prefix = y_prefix, include = include, exclude = exclude)
        col_dict['x'], col_dict['y'] = df.select(pl.exclude('det', 'sample')).columns    
        df = df.filter((~pl.col(col_dict['x']).is_nan()) & (~pl.col(col_dict['y']).is_nan()))

        tone_sample = int((self.drone_cfg['tones']['sweep_steps']-1)/2)
        df = (df.with_columns(pl.when(pl.col(col_dict['sample']) == tone_sample)
                                .then(pl.lit(True))
                                .otherwise(pl.lit(False))
                                .alias('tone')))

        # Set default hvplot key word arguments
        # -------------------------------------
        if not 'aspect' in kwargs: kwargs['aspect'] = self.viz_cfg['static_plot']['sweep']['aspect']
        if not 'marker' in kwargs: kwargs['marker'] = self.viz_cfg['static_plot']['sweep']['marker']
        if not 'ms' in kwargs: kwargs['ms'] = self.viz_cfg['static_plot']['sweep']['marker_size']
        if not 'linewidth' in kwargs: kwargs['linewidth'] = self.viz_cfg['static_plot']['sweep']['linewidth']
        if not 'tone_marker' in kwargs: kwargs['tone_marker'] = self.viz_cfg['static_plot']['sweep']['tone_marker']
        if not 'tone_ms' in kwargs: kwargs['tone_ms'] = self.viz_cfg['static_plot']['sweep']['tone_marker_size']
        if not 'dynamic' in kwargs: kwargs['dynamic'] = True

        if grouping == 'by': 
            kwargs['by'] = by
        elif grouping == 'groupby':
            kwargs['groupby'] = by
        else:
            error = 'Invalid string specified for argument "grouping"! Must be either "by" or "groupby".'
            rfsoc_io.send_msg('CRITICAL', error)
            raise ValueError(error)

        # Create opts for plots
        # ---------------------
        cfg = self.drone_cfg['det_config']
        title = rf"${cfg['detector_type']}\ {cfg['network']}$"
        
        overlay_opts = opts.Overlay(title=title,
                                    xlabel = xlabel if xlabel is not None else col_dict['x'],
                                    ylabel = ylabel if ylabel is not None else col_dict['y'])
        curve_opts = opts.Curve(show_legend=False, 
                                fig_size = kwargs.pop('fig_size') if 'fig_size' in kwargs else self.viz_cfg['static_plot']['sweep']['fig_size'])
        data_opts = opts.Curve('Sweep.Data')
        tone_opts = opts.Curve('Sweep.Tone')
        all_opts = [overlay_opts,
                    curve_opts,
                    data_opts,
                    tone_opts]
        if plot_opts is not None: all_opts.append(plot_opts)
        
        # Create plot for immediate visualization
        # ---------------------------------------
        plot = Sweep._plot(df, col_dict['x'], col_dict['y'], all_opts, **kwargs)

        # Save plot in background
        # -----------------------
        viz_utils.save_fig(self, Sweep._plot, df, col_dict['x'], col_dict['y'], all_opts, save_fig = save_fig, overwrite=overwrite, save_name=save_name, **kwargs)

        if return_df:
            return plot, df
        else:
            return plot

    def mag_plot(self, prefix: str = '', grouping = 'groupby', include: int | list[int] | None = None, exclude: int | list[int] | None = None, return_df = False, save_fig: bool | None = None, overwrite: bool | None = None, **kwargs):
        '''
        Plot the magnitude of the complex transmission |S_21| = \sqrt{I^2 + Q^2} of a frequency sweep
        '''
        xlabel, ylabel = r'$Frequency\ [Hz]$', r'$|S_{21}|$'
        save_name = f'sweep_{prefix}{'_' if prefix else ''}mag'

        rtn = self.plot('f', 'mag', 
                        y_prefix=prefix, 
                        grouping=grouping, 
                        include=include, 
                        exclude=exclude, 
                        xlabel=xlabel, 
                        ylabel=ylabel, 
                        return_df = return_df,
                        save_fig = save_fig,
                        save_name = save_name,
                        overwrite = overwrite,
                        **kwargs)
        return rtn
    
    def phase_plot(self, prefix: str = '', grouping = 'groupby', include: int | list[int] | None = None, exclude: int | list[int] | None = None, return_df = False, save_fig: bool | None = None, overwrite: bool | None = None, **kwargs):
        '''
        Plot the phase of the complex transmission \phi = \arctan{Q/I} of a frequency sweep
        '''
        xlabel, ylabel = r'$Frequency\ [Hz]$', r'$Phase\ [rad]$'
        save_name = f'sweep_{prefix}{'_' if prefix else ''}phase'
        rtn = self.plot('f', 'phase', 
                             y_prefix=prefix, 
                             grouping=grouping, 
                             include=include, 
                             exclude=exclude, 
                             xlabel=xlabel, 
                             ylabel=ylabel, 
                             return_df = return_df,
                             save_fig = save_fig,
                             save_name = save_name,
                             overwrite = overwrite,
                             **kwargs)

        return rtn

    def IQ_plot(self, prefix: str = '', projection='IQ', grouping = 'groupby', include: int | list[int] | None = None, exclude: int | list[int] | None = None, return_df = False, save_fig: bool | None = None, overwrite: bool | None = None, **kwargs):
        if projection == 'IQ':
            xlabel, ylabel = r'$I\ [arb]$', r'$Q\ [arb]$'
            x_dim, y_dim = 'I', 'Q'
            plot_opts=None
        elif projection == 'polar':
            xlabel, ylabel = r'$Phase\ [deg]$', r'$|S_{21}|$'
            x_dim, y_dim = 'phase', 'mag'
            plot_opts = opts.Curve(projection='polar', show_grid=True)
        else:
            error = 'Invalid projection specified, must be either "IQ" or "polar".'
            rfsoc_io.send_msg('CRITICAL', error)
            raise ValueError(error)
        save_name = f'sweep_{prefix}{'_' if prefix else ''}{projection}'

        if not 'linewidth' in kwargs: kwargs['linewidth'] = 0
        #I_min, I_max = df.select(pl.col('I').min().alias('min'), pl.col('I').max().alias('max'))[0].to_numpy()[0]
        #Q_min, Q_max = df.select(pl.col('Q').min().alias('min'), pl.col('Q').max().alias('max'))[0].to_numpy()[0]

        #max_diff = 1.1*max(I_max - I_min, Q_max - Q_min) / 2
        #I_avg, Q_avg = (I_min + I_max)/2, (Q_min + Q_max)/2

        #x_min, x_max = I_avg - max_diff, I_avg + max_diff
        #y_min, y_max = Q_avg - max_diff, Q_avg + max_diff
        #plot_opts = opts.Curve(xlim = (x_min, x_max), ylim = (y_min, y_max), linewidth=0)            
    
        rtn = self.plot(x_dim=x_dim,
                        y_dim=y_dim,
                        x_prefix=prefix, 
                        y_prefix=prefix, 
                        grouping=grouping, 
                        include=include, 
                        exclude=exclude, 
                        xlabel=xlabel, 
                        ylabel=ylabel, 
                        return_df = return_df,
                        save_fig = save_fig,
                        save_name = save_name,
                        overwrite = overwrite,
                        plot_opts=plot_opts,
                        **kwargs)
        return rtn

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
                det_f = np.real(np.load(f_path))
            except:
                error = f'Failed to load detector frequencies file {det_f}.'
                rfsoc_io.send_msg('ERROR', error)
                raise FileNotFoundError(error)
        return det_f
    
    #=====================#
    # Data Getter Methods #
    #=====================#

    def f(self, include: int | list[int] | None = None, exclude: int | list[int] | None = None):
        '''
        Get the frequency data of the sweep in Hz
        '''
        return self.get_data(col_name='f', include=include, exclude=exclude)
