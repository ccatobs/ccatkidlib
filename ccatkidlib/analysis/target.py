import gc
import sys
import numpy as np
import polars as pl

from collections.abc import Iterable
from pathlib import Path

from bokeh.layouts import layout
from bokeh.io import show
from bokeh.plotting import curdoc


# Local Imports
import ccatkidlib.rfsoc_io as rfsoc_io
import ccatkidlib.utils as utils
import ccatkidlib.analysis.pair as pair

from ccatkidlib.analysis.sweep import Sweep


class Target(Sweep):
    '''Class representing a target sweep taken with a Radio Frequency System on a Chip (RFSoC)
    
    Subclasses Sweep.
    '''

    def __init__(self, com_to: str, tones: int | list[int] | None = None, analysis_cfg: str = str(Path(__file__).parent / 'analysis_config.yaml'), **kwargs):
        '''Subclass of Sweep with additional arguments

        Args:
            tones (int | list[int] | None, optional): Which tones to load. None for loading all data without splitting into individual tones. -1 for all data split into individual tones. Defaults to None
        '''
        kwargs['data_type'] = 'targ'
        super().__init__(com_to, analysis_cfg, **kwargs)
        
        # Parse 'tone' argument specifying which tones should be loaded
        # -------------------------------------------------------------
        if isinstance(tones, int): 
            if tones >= 0: 
                tones = [tones]
            else:
                tones = list(range(self.num_tones))

        if tones is None or isinstance(tones, Iterable):
            self.tones = tones
        else:
            error = f"Invalid type {type(tones)} for argument 'tones'. Should be int, list[int], or None."
            rfsoc_io.send_msg('CRITICAL', error)
            raise ValueError(error)

    #==========================#
    # Lazily Loaded Attributes #
    #==========================#
    
    @property
    def data(self):
        if self._data is None:
            data = super().data # Get DataFrame of sweep data 
            tones = self.tones # Define local variable since it will be used often

            if tones is not None: # Run if a specific tone(s) is specified
                data = data.to_numpy().T
                try:
                    sweep_steps = self.drone_cfg['tones']['sweep_steps']
                except KeyError:
                    sweep_steps = self.drone_cfg['tones']['N_step']    

                data = data[1:].reshape((3, -1, sweep_steps))
                num_tones = len(tones)

                try:
                    fs, Is, Qs = [None]*num_tones, [None]*num_tones, [None]*num_tones
                    for i, t in enumerate(tones):
                        f, I, Q = data[:, t, :]
                        fs[i] = f
                        Is[i] = I
                        Qs[i] = Q
                    fs, Is, Qs = np.array(fs), np.array(Is), np.array(Qs)
                except Exception as e:
                    fs, Is, Qs = [None]*num_tones, [None]*num_tones, [None]*num_tones
                    rfsoc_io.send_msg('ERROR', 'Failed to reshape data array with error %s.', e)

                data_dict = {'sample': range(sweep_steps)}
                for t, f, I, Q in zip(tones, fs, Is, Qs):
                    data_dict[(f'f_{t:04d}')] = f
                    data_dict[(f'I_{t:04d}')] = I
                    data_dict[(f'Q_{t:04d}')] = Q
                df = pl.DataFrame(data_dict)
            else:
                df = data
            self._data = df 
        return self._data

    @data.setter
    def data(self, value: pl.lazyframe.frame.LazyFrame | None): 
        if value is None or isinstance(value, pl.dataframe.frame.DataFrame): 
            self._data = value
        else:
            rfsoc_io.send_msg('ERROR', 'Cannot set data with type %s. Must be a Polars DataFrame!')
    
    #==================#
    # Plotting Methods #
    #==================#

    #==========================#
    # Internal Loading Methods #
    #==========================#

    def _load_res_freqs(self):
        res_freqs = super()._load_res_freqs()
        if res_freqs is not None: res_freqs = np.array(res_freqs[self.res_num]) if self.res_num is not None else res_freqs
        return res_freqs

    def _get_res_s21z(self):
        res_s21z = None
        res_freqs = self.res_freqs
        if res_freqs is not None and len(res_freqs) > 0:
            try:
                sweep_steps = self.drone_cfg['tones']['sweep_steps']
            except KeyError:
                sweep_steps = self.drone_cfg['tones']['N_step']
            freq_bins = np.array(self.freqs).reshape((-1, sweep_steps))
            data_bins = np.array(self.s21z).reshape((-1, sweep_steps))
            res_s21z = [data[np.argmin(np.abs(freqs - res))] for res, freqs, data in zip(self.res_freqs, freq_bins, data_bins)]
        return res_s21z

