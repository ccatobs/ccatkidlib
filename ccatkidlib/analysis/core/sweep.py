'''

'''

import gc
import sys
import numpy as np
import polars as pl

from pathlib import Path
from functools import cached_property
from collections.abc import Iterable
from typing import TypeAlias, Literal
import ccatkidlib.analysis.utils.dataframe as ccat_df

import holoviews as hv
from holoviews import opts

# Local Imports

import ccatkidlib.log as log
import ccatkidlib.analysis.utils.pair as pair
import ccatkidlib.utils as utils
import ccatkidlib.analysis.viz.viz_utils as viz_utils

from ccatkidlib.analysis.core.data import Data

Format: TypeAlias = Literal['png', 'jpeg', 'pdf']
SweepFigure: TypeAlias = hv.Overlay | hv.HoloMap | hv.DynamicMap


class Sweep(Data):
    '''Class representing a |RFSoC| frequency sweep taken using the *ccatkidlib* data acquisition. 
    Sub-classes the **Data** base class to implement data loading and plotting. 
    '''

    def __init__(self, com_to: str, **kwargs):
        '''
        Args:
            com_to: |RFSoC| drone that took the frequency sweep
            **kwargs**: Keyword arguments of the base **Data** class
        '''
        super().__init__(com_to, cfg_path, **kwargs)
        
    #==================#
    # Plotting Methods #
    #==================#
        
    @staticmethod
    def _plot(df, plot_opts, *args, **kwargs):
        x_dim, y_dim = args

        tone_marker = kwargs.pop('tone_marker')
        tone_ms = kwargs.pop('tone_ms')

        sweep = df.hvplot.line(x=x_dim,
                               y=y_dim,
                               label='Data',
                               **kwargs).relabel(group='Sweep')

        kwargs.pop('ms'), kwargs.pop('linewidth')
        kwargs['marker'], kwargs['s'] = tone_marker, tone_ms**2

        tone = (df.filter(pl.col('tone'))
                  .hvplot.scatter(x=x_dim,
                                  y=y_dim,
                                  label='Tone',
                                  **kwargs)).relabel(group='Sweep')
        plot = sweep*tone
        if 'area_sample' in df.schema:
            if df.schema['area_sample'] == pl.List: df = df.explode('area_sample')
            area = df.filter(pl.col('sample') == pl.col('area_sample')).hvplot.area(x=x_dim,
                                                                                    y=y_dim,
                                                                                    label='Area Data',
                                                                                    **kwargs).relabel(group='Sweep')
            plot *= area

        plot.opts(*plot_opts)

        return plot

    def plot(self, 
             x_dim: str,
             y_dim: str, 
             x_prefix: str = '', 
             y_prefix: str = '',
             xlabel: str | None = None,
             ylabel: str | None = None,
             grouping: Literal['by', 'groupby'] = 'groupby',
             include: int | list[int] | None = None, 
             exclude: int | list[int] | None = None, 
             plot_opts: hv.Options | list[hv.Options] | None = None,
             filter_exprs: list[pl.Expr] = [],
             return_df: bool = False,
             save_fig: bool | None = None,
             figs_per_file: int | None = None,
             overwrite: bool | None = None,
             save_dir: str | Path | None = None,
             save_name: str | None = None,
             save_fmt: Format | None = None,
             return_fig: bool = True,
             df: pl.DataFrame | None = None,
             by: str | list[str] | None = None,
             area_df: pl.DataFrame | None = None,
             **kwargs) -> SweepFigure | tuple[SweepFigure, pl.DataFrame] | tuple[pl.DataFrame, str | None]:
        r'''
        Plot sweep data and tones

        Args:
            x_dim: Name of data to plot on x-axis without prefixes or |tone| suffix (e.g., **'f'**)
            y_dim: Name of data to plot on y-axis without prefixes or |tone| suffix (e.g., **'mag'**)
            x_prefix: Prefix of data to plot on x-axis
            y_prefix: Prefix of data to plot on y-axis
            xlabel: Label of x-axis
            ylabel: Label of y-axis
            grouping: How to handle plotting sweeps for multiple tones. 
                      **'by'** will overlay all plots whereas **'groupby'** will allow scrubbing through sweep plots of individual tones
            include: List of tones to plot
            exclude: List of tones to not plot
            plot_opts: *Holoviews* **Options** to apply to figure(s)
            filter_exprs: List of *Polars* **Expr** to use for filtering *Polars* **DataFrame** used for plotting
            return_df: Whether to return the *Polars* **DataFrame** that was used to create figure(s)
            save_fig: Whether to save figure. Defaults to that specified in viz configuration file
            figs_per_file: Number of figures to save in a single file. Will make a :math:`\sqrt{\text{figs_per_file}} \times \sqrt{\text{figs_per_file}}` grid of figures. Defaults to that specified in viz configuration file
            overwrite: Whether to overwrite figure files that already exist. Defaults to that specified in viz configuration file
            save_dir: Directory where figure should be saved. Defaults to ``fig_dir`` attribute
            save_name: Save name of file. Will always append sweep ``timestamp`` to the end of file name. Defaults to that specified in viz configuration file
            save_fmt: Format to save figure as. Defaults to that specified in viz configuration file
            return_fig: Whether to return *Holoviews* figure
            df: *Polars* **DataFrame** to use for creating figure(s). Must have the following columns: **'sample'**, **'det'**, ``x_dim``, ``y_dim``
            by: List of column names to group data by
            area_df: *Polars* **DataFrame** specifying which samples should be used to create *Holoviews* **Area** plot. 
                     Must have a **'det'** column and a **'area_sample'** column with lists of samples to use for area plots for each tone
            **kwargs**: Keyword arguments to supply to the *Holoviews* ``hvplot`` call(s) that create the figure(s)
        Returns:
            Will return *Holoviews* figure of sweep data if ``return_fig`` is **True**. Will also return *Polars* **DataFrame** used to create figure
            if ``return_df`` is **True**. If ``return_fig`` is **False**, will return DataFrame and column names to group data by
        '''
        
        # Get DataFrame with data to plot
        # -------------------------------
        col_dict = {'sample': 'sample',
                    'x': x_dim,
                    'y': y_dim}

        if df is None or by is None:
            df, by = self._get_plot_df(col_dict, x_prefix = x_prefix, y_prefix = y_prefix, include = include, exclude = exclude)
            col_dict['x'], col_dict['y'] = df.select(pl.exclude('det', 'sample')).columns    
            df = df.filter((~pl.col(col_dict['x']).is_nan()) & (~pl.col(col_dict['y']).is_nan()))
            if filter_exprs: df = df.filter(filter_exprs)

            tone_sample = int((self.drone_cfg['tones']['sweep_steps']-1)/2)
            df = (df.with_columns(pl.when(pl.col(col_dict['sample']) == tone_sample)
                                    .then(pl.lit(True))
                                    .otherwise(pl.lit(False))
                                    .alias('tone')))
            if area_df is not None: df = df.join(area_df, on='det', how='left')

        if not return_fig: return df, by

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
            log.log('CRITICAL', error)
            raise ValueError(error)

        # Create opts for plots
        # ---------------------
        cfg = self.drone_cfg['det_config']
        title = rf"${cfg['detector_type']}\ {cfg['network']}$"
        
        overlay_opts = opts.Overlay(title=title,
                                    xlabel = xlabel if xlabel is not None else col_dict['x'],
                                    ylabel = ylabel if ylabel is not None else col_dict['y'], 
                                    fig_size = kwargs.pop('fig_size') if 'fig_size' in kwargs else self.viz_cfg['static_plot']['sweep']['fig_size'])
        curve_opts = opts.Curve(show_legend=False)
        area_opts = opts.Area(alpha = kwargs.pop('area_alpha') if 'area_alpha' in kwargs else self.viz_cfg['static_plot']['sweep']['area_alpha'])
        all_opts = [overlay_opts,
                    curve_opts,
                    area_opts]
        if plot_opts is not None: all_opts += plot_opts if isinstance(plot_opts, Iterable) else [plot_opts]
        
        # Create plot for immediate visualization
        # ---------------------------------------
        plot = Sweep._plot(df, all_opts, *(col_dict['x'], col_dict['y']), **kwargs)

        # Save plot in background
        # -----------------------
        viz_utils.save_fig(self, Sweep._plot, df, all_opts, *(col_dict['x'], col_dict['y']), 
                           save_fig = save_fig, figs_per_file=figs_per_file, overwrite=overwrite, save_dir = save_dir,  save_name=save_name, save_fmt = save_fmt,
                           **kwargs)

        if return_df:
            return plot, df
        else:
            return plot

    def mag_plot(self, 
                 prefix: str = '', 
                 grouping: Literal['by', 'groupby'] = 'groupby', 
                 include: int | list[int] | None = None, 
                 exclude: int | list[int] | None = None, 
                 filter_exprs: list[pl.Expr] = [],
                 return_df: bool = False, 
                 save_fig: bool | None = None, 
                 **kwargs) -> SweepFigure | tuple[SweepFigure, pl.DataFrame] | tuple[pl.DataFrame, str | None]:
        r'''
        Plot magnitude :math:`|S_{21}| = \sqrt{I^2 + Q^2}` versus frequency data of the sweep

        Args:
            prefix: Prefix of the magnitude data 
            grouping: How to handle plotting sweeps for multiple tones. 
                      **'by'** will overlay all plots whereas **'groupby'** will allow scrubbing through sweep plots of individual tones
            include: List of tones to plot
            exclude: List of tones to not plot
            filter_exprs: List of *Polars* **Expr** to use for filtering *Polars* **DataFrame** used for plotting
            return_df: Whether to return the *Polars* **DataFrame** that was used to create figure(s)
            save_fig: Whether to save figure. Defaults to that specified in viz configuration file
            **kwargs**:
        Returns:
            Will return *Holoviews* figure of sweep data if ``return_fig`` is **True**. Will also return *Polars* **DataFrame** used to create figure
            if ``return_df`` is **True**. If ``return_fig`` is **False**, will return DataFrame and column names to group data by
        '''
        xlabel = r'$Frequency\ [Hz]$' if not 'xlabel' in kwargs else kwargs.pop('xlabel')
        ylabel = r'$|S_{21}|$' if not 'ylabel' in kwargs else kwargs.pop('ylabel')
        save_name = f"sweep_{prefix}{'_' if prefix else ''}mag" if not 'save_name' in kwargs else kwargs.pop('save_name')

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
    
    def phase_plot(self, 
                   prefix: str = '', 
                   grouping: Literal['by', 'groupby'] = 'groupby', 
                   include: int | list[int] | None = None, 
                   exclude: int | list[int] | None = None, 
                   filter_exprs: list[pl.Expr] = [],
                   return_df: bool = False, 
                   save_fig: bool | None = None, 
                   **kwargs) -> SweepFigure | tuple[SweepFigure, pl.DataFrame] | tuple[pl.DataFrame, str | None]:
        r'''
        Plot the phase :math:`\arctan{Q/I}` versus frequency data of the sweep

        Args:
            prefix: Prefix of the phase data
            grouping: How to handle plotting sweeps for multiple tones. 
                      **'by'** will overlay all plots whereas **'groupby'** will allow scrubbing through sweep plots of individual tones
            include: List of tones to plot
            exclude: List of tones to not plot
            filter_exprs: List of *Polars* **Expr** to use for filtering *Polars* **DataFrame** used for plotting
            return_df: Whether to return the *Polars* **DataFrame** that was used to create figure(s)
            save_fig: Whether to save figure. Defaults to that specified in viz configuration file
            **kwargs**:

        Returns:
            Will return *Holoviews* figure of sweep data if ``return_fig`` is **True**. Will also return *Polars* **DataFrame** used to create figure
            if ``return_df`` is **True**. If ``return_fig`` is **False**, will return DataFrame and column names to group data by
        '''
        xlabel = r'$Frequency\ [Hz]$' if not 'xlabel' in kwargs else kwargs.pop('xlabel')
        ylabel = r'$Phase\ [rad]$' if  not 'ylabel' in kwargs else kwargs.pop('ylabel')
        save_name = f"sweep_{prefix}{'_' if prefix else ''}phase" if not 'save_name' in kwargs else kwargs.pop('save_name')
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

    def IQ_plot(self, 
                prefix: str = '', 
                y_prefix: str | None = None, 
                projection: Literal['IQ', 'polar'] = 'IQ', 
                max_IQ_sliver: bool = False, 
                grouping: Literal['by', 'groupby'] = 'groupby', 
                include: int | list[int] | None = None, 
                exclude: int | list[int] | None = None, 
                filter_exprs: list[pl.Expr] = [],
                return_df: bool = False,
                save_fig: bool = True,
                **kwargs) -> SweepFigure | tuple[SweepFigure, pl.DataFrame] | tuple[pl.DataFrame, str | None]:
        r'''
        Plot the quadrature |Q| versus the in-phase |I| data of the sweep if `projection` is **'IQ'**. 
        If `projection` is **'polar'**, plot the magnitude :math:`|S_{21}| = \sqrt{I^2 + Q^2}` versus phase :math:`\arctan{Q/I}` data of the sweep.

        Args:
            prefix: Prefix of data to plot on the x-axis (either |I| or the phase data). Will also use as the prefix of the y-axis data if `y_prefix` not specified.
            y_prefix: Prefix of data to plot on the y-axis (either |Q| or the magnitude data).
            projection: Whether to plot data in polar (`projection` = **'polar'**) or cartesian (`projection` = **'IQ'**) coordinates
            max_IQ_sliver:
            grouping: How to handle plotting sweeps for multiple tones. 
                      **'by'** will overlay all plots whereas **'groupby'** will allow scrubbing through sweep plots of individual tones
            include: List of tones to plot
            exclude: List of tones to not plot
            filter_exprs: List of *Polars* **Expr** to use for filtering *Polars* **DataFrame** used for plotting
            return_df: Whether to return the *Polars* **DataFrame** that was used to create figure(s)
            save_fig: Whether to save figure. Defaults to that specified in viz configuration file
            **kwargs**: 
        Returns:
            Will return *Holoviews* figure of sweep data if ``return_fig`` is **True**. Will also return *Polars* **DataFrame** used to create figure
            if ``return_df`` is **True**. If ``return_fig`` is **False**, will return DataFrame and column names to group data by
        '''
        shared_opts = {'padding': 0.1}
        area_df = None
        if projection == 'IQ':
            xlabel, ylabel = r'$I\ [arb]$', r'$Q\ [arb]$'
            x_dim, y_dim = 'I', 'Q'

            plot_opts=opts.Curve(data_aspect=1, **shared_opts)
        elif projection == 'polar':
            xlabel, ylabel = r'$Phase\ [deg]$', r'$|S_{21}|$'
            x_dim, y_dim = 'phase', 'mag'

            # Define custom angle tick marks that do not block the radial label or title
            # --------------------------------------------------------------------------
            x_ticks = np.arange(0, 2*np.pi, np.pi/6)
            mask = [np.all(filt) for filt in zip(x_ticks != np.pi, x_ticks != np.pi/2)]
            x_ticks = x_ticks[mask]

            # Get DataFrame of samples with the maximum seperation in IQ space 
            # ----------------------------------------------------------------
            if max_IQ_sliver:
                area_df = ccat_df.get_properties(self, col_name='max_IQ_dist_sample', include=include, exclude=exclude, strict=True)
                if 'max_IQ_dist_sample' in area_df.schema:
                    area_df = area_df.select(pl.col('det'),
                                             pl.concat_list([pl.col('max_IQ_dist_sample'), pl.col('max_IQ_dist_sample') - 1]).alias('area_sample'))
                
            # Create plot options
            # -------------------
            plot_opts = opts.Curve(projection='polar',
                                   show_grid=True, 
                                   ylim=(0, None), 
                                   xticks=x_ticks,
                                   **shared_opts)
        else:
            error = 'Invalid projection specified, must be either "IQ" or "polar".'
            log.log('CRITICAL', error)
            raise ValueError(error)

        if not 'linewidth' in kwargs: kwargs['linewidth'] = 0
        if 'plot_opts' in kwargs: plot_opts += extra_opts if isinstance(extra_opts := kwargs.pop('plot_opts'), Iterable) else [extra_opts]


        save_name = f"sweep_{prefix}{'_' if prefix else ''}{projection}" if not 'save_name' in kwargs else kwargs.pop('save_name')
        if 'xlabel' in kwargs: xlabel = kwargs.pop('xlabel')
        if 'ylabel' in kwargs: ylabel = kwargs.pop('ylabel')
        
        rtn = self.plot(x_dim=x_dim,
                        y_dim=y_dim,
                        x_prefix=prefix, 
                        y_prefix=prefix if y_prefix is None else y_prefix, 
                        grouping=grouping, 
                        include=include, 
                        exclude=exclude, 
                        xlabel=xlabel, 
                        ylabel=ylabel, 
                        return_df = return_df,
                        area_df = area_df,
                        save_fig = save_fig,
                        save_name = save_name,
                        plot_opts=plot_opts,
                        **kwargs)
        return rtn

    #==========================#
    # Lazily Loaded Attributes #
    #==========================#

    @property
    def data(self) -> pl.DataFrame:
        '''
        *Polars* **DataFrame** with sweep data. Load data if it is not already loaded.
        Can only be set with *Polars* **DataFrame** or **LazyFrame** objects (or **None**)

        '''
        if self._data is None:
            data = {'sample': [], 'f': [], 'I': [], 'Q': []}
            fs, s21z = np.load(self.data_path[0], mmap_mode='r')
            I, Q = s21z.real, s21z.imag

            data['sample'], data['f'], data['I'], data['Q'] = range(len(fs)), fs.real, I, Q
            self._data = pl.DataFrame(data)
        elif isinstance(self._data, pl.LazyFrame): 
            self._data = self._data.collect()
        return self._data

    @data.setter
    def data(self, value: pl.DataFrame | pl.LazyFrame | None) -> None: 
        if value is None or isinstance(value, (pl.DataFrame, pl.LazyFrame)): 
            self._data = value
        else:
            log.log('ERROR', 'Cannot set data with type %s. Must be a Polars LazyFrame! Convert DataFrame to lazy frame with .lazy() before setting.', type(value))

    @cached_property
    def det_f(self) -> pl.Series:
        '''
        |KID| frequencies identified by *ccatkidlib* ``find_detectors`` or ``tune_tone_placement`` data acquisition methods

        .. important::
            The identified detector frequencies are **NOT** necessarily the same as the tone frequencies of the sweep!
        
        Returns:
            *Polars* **Series** of identified detector frequencies

        Raises:
            FileNotFoundError: If unable to load *NumPy* file of detector frequencies
        '''

        det_f = self.drone_cfg['det_config']['found_detector_freqs']
        if isinstance(det_f, list):
            det_f = np.real(det_f)
        else:
            try:
                f_path = pair.replace_root(det_f, self._original_root, self._root_dir)
                det_f = np.real(np.load(f_path))
            except:
                error = f'Failed to load detector frequencies file {det_f}.'
                log.log('ERROR', error)
                raise FileNotFoundError(error)
        return det_f
    
    #=====================#
    # Data Getter Methods #
    #=====================#

    def f(self, include: int | list[int] | None = None, exclude: int | list[int] | None = None) -> pl.DataFrame:
        '''
        Frequency data of the sweep

        Args:
            include: List of tones to include
            exclude: List of tones to exclude
        Returns:
            *Polars **DataFrame** with frequency data

        '''
        return self.get_data(col_name='f', include=include, exclude=exclude)