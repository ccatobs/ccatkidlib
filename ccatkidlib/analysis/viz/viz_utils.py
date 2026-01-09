import numpy as np
import holoviews as hv

from multiprocessing import Process
from functools import wraps
from pathlib import Path

'''
Library of helper functions for plotting KID data.
'''

def save_fig(self, plot_func, df, x_dim, y_dim, plot_opts, save_fig = None, overwrite=None, save_name = None, **kwargs):
    if save_fig is None: save_fig = self.save_fig
    if overwrite is None: overwrite = self.overwrite
    if save_name is None: save_name =  f'{x_dim}_{y_dim}'
    figs_per_file = kwargs.pop('figs_per_file') if 'figs_per_file' in kwargs else self.figs_per_file

    if save_fig:
        save_dir, save_fmt = Path(self.save_dir), self.save_fmt
        timestamp = self.timestamp

        save_process = Process(target=_save_fig, args=(plot_func, df, x_dim, y_dim, plot_opts, overwrite, save_dir, save_name, save_fmt, timestamp, figs_per_file), kwargs = kwargs)
        save_process.start()
    return

def _save_fig(plot_func, df, x_dim, y_dim, plot_opts, overwrite, save_dir, save_name, save_fmt, timestamp, figs_per_file, **kwargs):
    import holoviews as hv
    import hvplot.polars    
    hv.extension('matplotlib', enable_mathjax=True)

    kwargs['dynamic'] = False
    plot = plot_func(df, x_dim, y_dim, plot_opts, **kwargs)

    if isinstance(plot, hv.HoloMap):
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