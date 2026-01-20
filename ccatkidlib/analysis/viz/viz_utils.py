import numpy as np
import holoviews as hv

from multiprocessing import Process
from functools import wraps
from pathlib import Path
from collections.abc import Iterable

'''
Library of helper functions for plotting KID data.
'''

def save_fig(obj, plot_func, data, plot_opts, *args, save_fig = None, overwrite=None, save_name = None, **kwargs):
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
    if save_fig is None: save_fig = obj.save_fig
    if overwrite is None: overwrite = obj.overwrite
    if save_name is None: save_name = 'tmp'
    figs_per_file = kwargs.pop('figs_per_file') if 'figs_per_file' in kwargs else obj.figs_per_file

    if save_fig:
        save_dir, save_fmt = Path(obj.save_dir), obj.save_fmt
        timestamp = obj.timestamp

        print(args)
        save_process = Process(target=_save_fig, args=(plot_func, data, plot_opts, overwrite, save_dir, save_name, save_fmt, timestamp, figs_per_file, args), kwargs = kwargs)
        save_process.start()
    return

def _save_fig(plot_func, data, plot_opts, overwrite, save_dir, save_name, save_fmt, timestamp, figs_per_file, args, **kwargs):
    '''
    Worker function for saving Holoviews figure

    '''
    import holoviews as hv
    import hvplot.polars    
    hv.extension('matplotlib', enable_mathjax=True)

    kwargs['dynamic'] = False
    plot = plot_func(data, plot_opts, *args, **kwargs)

    if isinstance(plot, hv.HoloMap):
        kdims = [kdim.name for kdim in plot.kdims]

        for key, fig in plot.items():
            if not isinstance(key, Iterable) or isinstance(key, str): key = [key]
            fig.opts(title=', '.join([f'{name}={value}' for name, value in zip(kdims, key)]))

        num_files = (len(plot) + figs_per_file - 1) // figs_per_file
        plots = num_files*[None]
        for i in range(num_files): 
            range_min, range_max = i*figs_per_file, (i+1)*figs_per_file 
            plots[i] = hv.Layout(plot[range_min:range_max]).opts(sublabel_format='', shared_axes=False).cols(int(np.sqrt(figs_per_file)))
    else:
        plots = [plot]

    file_count = 0
    save_path = save_dir / f'{save_name}_{timestamp}_{file_count}_0.{save_fmt}'
    while not overwrite and save_path.exists(): 
        file_count += 1
        save_path = save_dir / f'{save_name}_{timestamp}_{file_count}_0.{save_fmt}'
    save_path = '_'.join(str(save_path.with_suffix('')).split('_')[:-1])

    for i, plot in enumerate(plots):  hv.save(plot, f'{save_path}_{i}', fmt=save_fmt, backend='matplotlib', toolbar=False)