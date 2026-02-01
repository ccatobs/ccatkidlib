from __future__ import annotations

import numpy as np
import holoviews as hv

from multiprocessing import Process
from functools import wraps
from pathlib import Path
from collections.abc import Iterable
from typing import Any, Callable, TYPE_CHECKING

import ccatkidlib.io as io

if TYPE_CHECKING:
    from ccatkidlib.analysis.core.data import Data
    from ccatkidlib.analysis.core.detector import Detector
    from ccatkidlib.analysis.core.network import Network
    from holoviews import opts


'''
Library of helper functions for plotting KID data.
'''

def save_fig(obj: Data | Detector | Network, 
             plot_func: Callable, 
             data: Any, 
             plot_opts: list[opts], 
             *args, 
             save_fig: bool | None = None, 
             overwrite: bool | None = None, 
             save_name: str | None = None, 
             **kwargs):
    '''
    Create and save Holoviews figure

    Args:
        obj (Data | Detector | Network): ccatkidlib Data, Detector, or Network analysis object
        plot_func (Callable[]): Function that creates Holoviews figure
        data (Any): Data to be passed to ``plot_func`` for creating figure
        plot_opts (list[Opts]): List of Holoviews Opts to be passed to ``plot_func`` for styling figure
        save_fig (bool | None): Whether to save figure. Defaults to that specified in analysis config
        overwrite (bool | None): Whether to overwrite figure files that already exist. Defaults to that specified in analysis config
        save_name (str | None): Save name of file. Will always add timestamp . Defaults to that specified in analysis config
    ''' 
    viz_cfg = obj.viz_cfg

    if save_fig is None: save_fig = viz_cfg['save']['save_fig']
    if overwrite is None: overwrite = viz_cfg['save']['overwrite']
    if save_name is None: save_name = 'tmp'

    save_name = save_name.replace('/', '-') # Backslashes cannot be used in filenames

    figs_per_file = kwargs.pop('figs_per_file') if 'figs_per_file' in kwargs else viz_cfg['save']['figs_per_file']

    if save_fig:
        save_dir, save_fmt = Path(obj.fig_dir), viz_cfg['save']['fig_fmt']
        timestamp = obj.timestamp

        pickle_dataframes = obj.analysis_cfg['io']['pickle']['pickle_dataframes']
        obj.analysis_cfg['io']['pickle']['pickle_dataframes'] = True

        save_process = Process(target=_save_fig, args=(plot_func, data, plot_opts, overwrite, save_dir, save_name, save_fmt, timestamp, figs_per_file, args), kwargs = kwargs)
        save_process.start()

        obj.analysis_cfg['io']['pickle']['pickle_dataframes'] = pickle_dataframes

def _save_fig(plot_func, data, plot_opts, overwrite, save_dir, save_name, save_fmt, timestamp, figs_per_file, args, **kwargs):
    '''
    Worker function for saving Holoviews figures in a background process
    '''
    import holoviews as hv
    import hvplot.polars    
    hv.extension('matplotlib', enable_mathjax=True)

    kwargs['dynamic'] = False
    plot = plot_func(data, plot_opts, *args, **kwargs)

    if isinstance(plot, hv.HoloMap):
        kdims = [name for kdim in plot.kdims if (name := kdim.name) != 'Default']
        num_files = (len(plot) + figs_per_file - 1) // figs_per_file
        plots = num_files*[[]]
        for i, (key, fig) in enumerate(plot.items()):
            if kdims: # Edit subfigure titles to include key dimensions
                if not isinstance(key, Iterable) or isinstance(key, str): key = [key]
                fig.opts(title=', '.join([f'{name}={value}' for name, value in zip(kdims, key)]))
            plots[i // figs_per_file] += [fig]
        for i in range(len(plots)): plots[i] = hv.Layout(plots[i]).opts(sublabel_format='', shared_axes=False).cols(int(np.sqrt(figs_per_file)))
    else:
        plots = [plot]

    save_path = io.increment_file(save_dir, f'{save_name}_{timestamp}_', f'_0.{save_fmt}', overwrite=overwrite)[0]
    save_path = '_'.join(str(save_path.with_suffix('')).split('_')[:-1])
    
    for i, plot in enumerate(plots): hv.save(plot, f'{save_path}_{i}', fmt=save_fmt, backend='matplotlib', toolbar=False)