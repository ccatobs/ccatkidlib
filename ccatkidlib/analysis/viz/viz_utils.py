'''
Library of helper functions for plotting :term:`KID` data.

.. codeauthor:: Darshan Patel <dp649@cornell.edu>
'''

from __future__ import annotations

import numpy as np
import holoviews as hv
import matplotlib as mpl

from multiprocessing import Process
from functools import wraps
from pathlib import Path
from collections.abc import Iterable
from typing import Any, Callable, TypeAlias, Literal, TYPE_CHECKING

import ccatkidlib.io as io

if TYPE_CHECKING:
    from ccatkidlib.analysis.core.data import Data
    from ccatkidlib.analysis.core.detector import Detector
    from ccatkidlib.analysis.core.network import Network
    from holoviews import Options

Format: TypeAlias = Literal['png', 'jpeg', 'pdf']

def cycle_cmap(cmap: str, num_colors: int, cmap_range: tuple[float, float] = (0, 1)) -> hv.Cycle | None:
    ''' Create *Holoviews* **Cycle** object of specified *matplotlib* color map discretized into ``num_colors`` colors

    Args:
        cmap: *Matplotlib* `color map <https://matplotlib.org/stable/users/explain/colors/colormaps.html#id10>`_
        num_colors: Number of colors to include
        cmap_range: Range of colors to use from color map. Defaults to a range of **(0, 1)** for use with continuous color maps.
    Returns:
        *Holoviews* **Cycle** object with ``num_colors`` colors from specified color map or *None* if invalid color map is specified
    '''
    colors = None
    if cmap in list(mpl.colormaps):
        cmap = mpl.colormaps[cmap]

        # Need to convert RGBL colors from matplotlib color map into hexidecimal to use with hv.Cycle
        colors = [mpl.colors.rgb2hex(c) for c in cmap(np.linspace(cmap_range[0], cmap_range[1], num_colors))]
        colors = hv.Cycle(colors)
    return colors

def save_fig(obj: Data | Detector | Network, 
             plot_func: Callable, 
             data: Any, 
             plot_opts: list[Options], 
             *args, 
             save_fig: bool | None = None, 
             figs_per_file: int | None = None,
             overwrite: bool | None = None, 
             save_dir: str | Path | None = None,
             save_name: str | None = None, 
             save_fmt: Format | None = None,
             **kwargs) -> None:
    r''' Create a *Holoviews* figure using the specified plotting function and save figure to disk

    Args:
        obj: *ccatkidlib* **Data**, **Detector**, or **Network** object
        plot_func: Function that generates *Holoviews* figure
        data: Data to be passed to ``plot_func`` for creating figure
        plot_opts: List of *Holoviews* **Options** to be passed to ``plot_func`` for styling figure
        args: Positional arguments to pass to ``plot_func``
        save_fig: Whether to save figure. Defaults to that specified in viz configuration file
        figs_per_file: Number of figures to save in a single file. Will make a :math:`\sqrt{\text{figs_per_file}} \times \sqrt{\text{figs_per_file}}` grid of figures. Defaults to that specified in viz configuration file
        overwrite: Whether to overwrite figure files that already exist. Defaults to that specified in viz configuration file
        save_dir: Directory where figure should be saved. Defaults to the ``fig_dir`` of ``obj``
        save_name: Save name of file. Will always append ``obj.timestamp`` to the end of file name. Defaults to that specified in viz configuration file
        save_fmt: Format to save figure as. Defaults to that specified in viz configuration file
        kwargs: Key word arguments to pass to ``plot_func``
    ''' 
    viz_cfg = obj.viz_cfg

    if save_fig is None: save_fig = viz_cfg['save']['save_fig']
    if figs_per_file is None: figs_per_file = viz_cfg['save']['figs_per_file']
    if overwrite is None: overwrite = viz_cfg['save']['overwrite']
    if save_dir is None: save_dir = Path(obj.fig_dir)
    if save_name is None: save_name = 'tmp'
    if save_fmt is None: save_fmt = viz_cfg['save']['fig_fmt']

    save_name = save_name.replace('/', '-') # Backslashes cannot be used in filenames

    if save_fig:
        timestamp = obj.timestamp

        pickle_dataframes = obj.analysis_cfg['io']['pickle']['pickle_dataframes']
        obj.analysis_cfg['io']['pickle']['pickle_dataframes'] = True

        worker_args = (plot_func, data, timestamp, plot_opts, 
                       overwrite, save_dir, save_name, save_fmt, figs_per_file, 
                       args)
        # Start worker function in a new process to save figure(s) in the background
        save_process = Process(target=_save_fig, args=worker_args, kwargs = kwargs)
        save_process.start()

        # Run worker function in main process for easier debugging
        #_save_fig(*worker_args, **kwargs)

        obj.analysis_cfg['io']['pickle']['pickle_dataframes'] = pickle_dataframes

def _save_fig(plot_func, data, timestamp, plot_opts, overwrite, save_dir, save_name, save_fmt, figs_per_file, args, **kwargs):
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
        plots = [[] for _ in range(num_files)]
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