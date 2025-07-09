import gc
import sys
import numpy as np
import polars as pl

from typing import override
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

    @override
    def __init__(self, com_to: str, tone: int | list[int] | None = None, analysis_cfg: str = str(Path(__file__).parent / 'analysis_config.yaml'), **kwargs):
        '''Subclass of Sweep with additional arguments

        Args:
            tone (int | list[int] | None, optional): Which tones to load. None for loading all data without splitting into individual tones. -1 for all data split into individual tones. Defaults to None
        '''
        kwargs['data_type'] = 'targ'
        super().__init__(com_to, analysis_cfg, **kwargs)
        
        # Parse 'tone' argument specifying which tones should be loaded
        # -------------------------------------------------------------
        if isinstance(tone, int): 
            if tone >= 0: 
                tone = [tone]
            else:
                tone = list(range(self.num_tones))

        if tone is None or isinstance(tone, Iterable):
            self.tone = tone
        else:
            error = f"Invalid type {type(tone)} for argument 'tone'. Should be int, list[int], or None."
            rfsoc_io.send_msg('CRITICAL', error)
            raise ValueError(error)

    #==========================#
    # Lazily Loaded Attributes #
    #==========================#
    
    @property
    def data(self):
        if self._data is None:
            df = super().data # Get DataFrame of sweep data 
            tone = self.tone # Define local variable since it will be used often

            if tone is not None: # Run if a specific tone(s) is specified
                data = df.collect().to_numpy().T

                try:
                    sweep_steps = self.drone_cfg['tones']['sweep_steps']
                except KeyError:
                    sweep_steps = self.drone_cfg['tones']['N_step']    

                data = data[1:].reshape((3, -1, sweep_steps))
                num_tone = len(tone)

                try:
                    fs, Is, Qs = [None]*num_tone, [None]*num_tone, [None]*num_tone
                    for i, t in enumerate(tone):
                        f, I, Q = data[:, t, :]
                        fs[i] = f
                        Is[i] = I
                        Qs[i] = Q
                    fs, Is, Qs = np.array(fs), np.array(Is), np.array(Qs)
                except Exception as e:
                    fs, Is, Qs = [None]*num_tone, [None]*num_tone, [None]*num_tone
                    print(e)
                    rfsoc_io.send_msg('ERROR', 'Failed to reshape data array with error %s.', e)

                data_dict = {'sample': range(sweep_steps)}
                for t, f, I, Q in zip(tone, fs, Is, Qs):
                    data_dict[(f'f_{t:04d}')] = f
                    data_dict[(f'I_{t:04d}')] = I
                    data_dict[(f'Q_{t:04d}')] = Q
                df = pl.DataFrame(data_dict).lazy()
            self._data = df 
        return self._data

    @data.setter
    def data(self, value: pl.lazyframe.frame.LazyFrame | None): 
        if value is None or isinstance(value, pl.lazyframe.frame.LazyFrame): 
            self._data = value
        else:
            rfsoc_io.send_msg('ERROR', 'Cannot set data with type %s. Must be a Polars LazyFrame! Convert DataFrame to lazy frame with .lazy() before setting.', type(value))
    
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

