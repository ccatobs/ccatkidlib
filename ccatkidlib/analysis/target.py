from .sweep import Sweep
from pathlib import Path
import sys
import numpy as np

from bokeh.layouts import layout
from bokeh.io import show

# Local Imports
import ccatkidlib.rfsoc_io as rfsoc_io
import ccatkidlib.utils as utils
import ccatkidlib.analysis.pair as pair

class Target(Sweep):
    '''
    Class representing a target sweep 
    Subclass of Sweep class.  
    '''

    def __init__(self, com_to, res_num = None, analysis_cfg=str(Path(__file__).parent / 'analysis_config.yaml'), **kwargs):
        kwargs['data_type'] = 'targ'
        super().__init__(com_to, analysis_cfg, **kwargs)
        self.res_num = res_num
    
    ####################
    # Plotting Methods #
    ####################

    def plot_mag(self, fig = None, freqs = None, s21m = None, source = None, dB = False, show_plot = True, **kwargs):
        res_num = self.res_num
        kwargs.setdefault('title', f"Target Sweep of {f'Resonator {res_num} in ' if res_num is not None else ''}{self.drone_cfg['det_config']['detector_type']} Network {self.drone_cfg['det_config']['network']} Taken on {utils.convert_timestamp(self.timestamp)} EST")
        if res_num is None: 
            kwargs['plot_scatter'] = False
        else:
            show_bins = False 
        fig = super().plot_mag(fig = fig, freqs = freqs, s21m = s21m, dB = dB, source = source, show_plot = show_plot, **kwargs)
        return fig
    
    def plot_phase(self, fig = None, freqs = None, phase = None, source = None, show_plot = True, **kwargs):
        res_num = self.res_num
        kwargs.setdefault('title', f"Target Sweep of {f'Resonator {res_num} in ' if res_num is not None else ''}{self.drone_cfg['det_config']['detector_type']} Network {self.drone_cfg['det_config']['network']} Taken on {utils.convert_timestamp(self.timestamp)} EST")
        if res_num is None: 
            kwargs['plot_scatter'] = False
        else:
            show_bins = False 
        fig = super().plot_phase(fig = fig, freqs = freqs, phase = phase, source = source, show_plot = show_plot, **kwargs)
        return fig

    def plot_IQ(self, fig = None, I = None, Q = None, source = None, show_plot = True, **kwargs):
        res_num = self.res_num
        kwargs.setdefault('title', f"Target Sweep of {f'Resonator {res_num} in ' if res_num is not None else ''}{self.drone_cfg['det_config']['detector_type']} Network {self.drone_cfg['det_config']['network']} Taken on {utils.convert_timestamp(self.timestamp)} EST")
        if res_num is None: kwargs['plot_line'] = False
        fig = super().plot_IQ(fig = fig, I = I, Q = Q, source = source, show_plot = show_plot, **kwargs)
        return fig

    def dashboard(self, dB = False, show_plot = True, **kwargs):
        plot_cfg = self.plot_cfg

        if self.res_num is not None:
            kwargs['aspect_ratio'] = 1
            mag_fig, source = self.plot_mag(dB = dB, show_plot = False, **kwargs)
            phase_fig, source = self.plot_phase(show_plot = False, source = source, **kwargs)
            IQ_fig, source =  self.plot_IQ(show_plot = False, source = source, **kwargs)

            lyot = layout([[mag_fig, phase_fig, IQ_fig]], sizing_mode='scale_width')
            if show_plot: show(lyot)
        else:
            lyot = super().dashboard(dB = dB, show_plot = show_plot, **kwargs)
        return lyot


    ############################
    # Internal Loading Methods #
    ############################


    def _load_res_freqs(self):
        res_freqs =  super()._load_res_freqs()
        res_freqs = [res_freqs[self.res_num]] if self.res_num is not None else res_freqs
        return np.array(res_freqs)

    def _get_res_s21z(self):
        res_s21z = None
        if self.res_freqs is not None:
            N_step = self.drone_cfg['tones']['N_step']
            freq_bins = np.array(self.freqs).reshape((-1, N_step))
            data_bins = np.array(self.s21z).reshape((-1, N_step))
            res_s21z = [data[np.argmin(np.abs(freqs - res))] for res, freqs, data in zip(self.res_freqs, freq_bins, data_bins)]
        return res_s21z

    def _load_sweep(self):
        data = list(super()._load_sweep()) # Call the Sweep _load_sweep method
        if self.res_num is not None: # Run if a specific resonator is specified
            N_step = self.drone_cfg['tones']['N_step']
            data = np.array(data).reshape((2, -1, N_step))
            try:
                freqs, s21z = data[:, self.res_num, :]
            except:
                freqs, s21z = None, None
        else:
            freqs, s21z = data
        return freqs.real, s21z

    #################
    # Magic Methods #
    #################
