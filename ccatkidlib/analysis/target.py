from .sweep import Sweep
from pathlib import Path
import gc
import sys
import numpy as np
import pandas as pd

from bokeh.layouts import layout
from bokeh.io import show
from bokeh.plotting import curdoc

# Local Imports
import ccatkidlib.rfsoc_io as rfsoc_io
import ccatkidlib.utils as utils
import ccatkidlib.analysis.pair as pair

class Target(Sweep):
    '''
    Class representing a target sweep 
    Subclass of Sweep class.  
    '''

    def __init__(self, com_to, res_num = -1, analysis_cfg=str(Path(__file__).parent / 'analysis_config.yaml'), **kwargs):
        kwargs['data_type'] = 'targ'
        super().__init__(com_to, analysis_cfg, **kwargs)
        if isinstance(res_num, int) and res_num > 0: res_num = [res_num]
        self.res_num = res_num
    
    #@method_timer
    def transform(self, name, func, res_num = None):
        '''
        Method for adding data transformed by func to self.data pandas DataFrame.
        '''

        if res_num is None and self.res_num == -1:
            data = super().transform_sweep(name, func, res_num)
        else:
            data = super().transform(name, func, res_num)
        return data

    def resonator(self, res_num = None):
        if not self.res_num == -1:
            return super().resonator(res_num = res_num)
        else:
            return None
    ####################
    # Analysis Methods #
    ####################

    ####################
    # Plotting Methods #
    ####################

    def dashboard(self, dB = False, show_plot = True, **kwargs):
        plot_dash = show_plot if self.res_num is None else False
        lyot = super().dashboard(dB = dB, show_plot = plot_dash, **kwargs)
        
        if self.res_num is not None:
            # Get all column objects from current layout
            children = lyot.children
            columns = []
            for child in children:
                columns += child.children
            
            print(columns)
            # Restructure layout to have only one row
            lyot = layout(columns, sizing_mode='scale_width')
            if show_plot: 
                curdoc().add_root(lyot)
                show(lyot)
        return lyot

    def plot_mag(self, fig = None, freqs = None, s21m = None, source = None, dB = False, show_plot = True, **kwargs):
        res_num = self.res_num
        kwargs.setdefault('title', f"Target Sweep of {f'Resonator {res_num} in ' if res_num is not None else ''}{self.drone_cfg['det_config']['detector_type']} Network {self.drone_cfg['det_config']['network']} Taken on {utils.convert_timestamp(self.timestamp)} EST")
        if res_num is None: 
            kwargs['plot_scatter'] = False
        else:
            tools = self.plot_cfg['plot_defaults']['figure']['tools']
            if not 'pan' in tools: 
                tools += ',pan'
                kwargs['tools'] = tools
            show_bins = False 
        return super().plot_mag(fig = fig, freqs = freqs, s21m = s21m, dB = dB, source = source, show_plot = show_plot, **kwargs)
    
    def plot_phase(self, fig = None, freqs = None, phase = None, source = None, show_plot = True, **kwargs):
        res_num = self.res_num
        kwargs.setdefault('title', f"Target Sweep of {f'Resonator {res_num} in ' if res_num is not None else ''}{self.drone_cfg['det_config']['detector_type']} Network {self.drone_cfg['det_config']['network']} Taken on {utils.convert_timestamp(self.timestamp)} EST")
        if res_num is None: 
            kwargs['plot_scatter'] = False
        else:
            tools = self.plot_cfg['plot_defaults']['figure']['tools']
            if not 'pan' in tools: 
                tools += ',pan'
                kwargs['tools'] = tools
            show_bins = False 
        return super().plot_phase(fig = fig, freqs = freqs, phase = phase, source = source, show_plot = show_plot, **kwargs)

    def plot_IQ(self, fig = None, I = None, Q = None, source = None, show_plot = True, **kwargs):
        res_num = self.res_num
        kwargs.setdefault('title', f"Target Sweep of {f'Resonator {res_num} in ' if res_num is not None else ''}{self.drone_cfg['det_config']['detector_type']} Network {self.drone_cfg['det_config']['network']} Taken on {utils.convert_timestamp(self.timestamp)} EST")
        if res_num is None: 
            kwargs['plot_line'] = False
        else:
            tools = self.plot_cfg['plot_defaults']['figure']['tools']
            if not 'pan' in tools: 
                tools += ',pan'
                kwargs['tools'] = tools
        return super().plot_IQ(fig = fig, I = I, Q = Q, source = source, show_plot = show_plot, **kwargs)

    ############################
    # Internal Loading Methods #
    ############################

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

    def _load_sweep(self):
        data_dict = super()._load_sweep() # Call the Sweep _load_sweep method

        if self.res_num is None: self.res_num = range(self.drone_cfg['tones']['num_tones'])
        res_num = self.res_num

        if not res_num == -1: # Run if a specific resonator(s) is specified
            num_res = len(res_num)
            data = [data_dict['fs']] + [data_dict['I']] + [data_dict['Q']]
            try:
                sweep_steps = self.drone_cfg['tones']['sweep_steps']
            except KeyError:
                sweep_steps = self.drone_cfg['tones']['N_step']
            data = np.array(data).reshape((3, -1, sweep_steps))
            try:
                fs, Is, Qs = [], [], []
                for res in res_num:
                    f, I, Q = data[:, res, :]
                    fs += [f]
                    Is += [I]
                    Qs += [Q]
                fs, Is, Qs = np.array(fs), np.array(Is), np.array(Qs)
            except Exception as e:
                fs, Is, Qs = num_res*[None], num_res*[None], num_res*[None]

            data_dict = {}
            for res, f, I, Q in zip(res_num, fs, Is, Qs):
                data_dict[('fs', f'R_{res:04d}')] = f
                data_dict[('I',  f'R_{res:04d}')]  = I
                data_dict[('Q',  f'R_{res:04d}')]  = Q
        return data_dict

    #################
    # Magic Methods #
    #################
