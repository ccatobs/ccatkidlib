from .sweep import Sweep
from pathlib import Path
from scipy.stats import linregress
import numpy as np
import sys

# Local Imports
import ccatkidlib.rfsoc_io as rfsoc_io
import ccatkidlib.analysis.pair as pair



class VNA(Sweep):
    '''
    Class representing a vector network analyzer (VNA) esque sweep taken with a Radio Frequency System on a Chip (RFSoC). 
    Subclass of the Sweep class.
    '''

    def __init__(self, com_to, analysis_cfg=str(Path(__file__).parent / 'analysis_config.yaml'), **kwargs):
        kwargs['data_type'] = 'vna'
        super().__init__(com_to, analysis_cfg, **kwargs)

    def _get_res_s21z(self):
        res_s21z = None
        s21z = self.s21z
        freqs = list(self.freqs)
        res_freqs = self.res_freqs
        if res_freqs is not None:
            try:
                res_s21z = [s21z[freqs.index(freq)] for freq in res_freqs]
            except ValueError:
                res_s21z = super()._get_res_s21z()
        return res_s21z

    def get_cable_delay(self, freqs = None, phases = None, stitch_phase = True):
        freqs = freqs if freqs is not None else self.freqs
        if phases is not None:
            pass
        else:
            phases = np.arctan2(np.imag(self.s21z), np.real(self.s21z))
            if stitch_phase: phases = self.stitch_phase(freqs, phases)

        cable_delay, intercept, r, p, se = linregress(freqs, phases)

        return cable_delay*1e9/(2*np.pi), se*1e9/(2*np.pi)
    
    def stitch_mag():
        pass

    def stitch_phase(self, freqs = None, phases = None, threshold = 1.9*np.pi, stitch_ends_factor = 10):
        # Get frequency and phase data
        # ----------------------------
        freqs = freqs if freqs is not None else self.freqs
        phases = phases if phases is not None else np.arctan2(np.imag(self.s21z), np.real(self.s21z))

        # Stitch phase discontinuities caused by wrapping from -pi to pi
        # --------------------------------------------------------------
        curr_shift = 0
        for i in range(len(phases) - 1):
            diff = phases[i+1] - phases[i]
            phases[i] -= curr_shift
            if diff > threshold:
                curr_shift += 2*np.pi 
            elif diff < -1*threshold:
                curr_shift -= 2*np.pi
        
        # Reshape data into bins of size N_step
        # -------------------------------------
        try:
            sweep_steps = self.drone_cfg['tones']['sweep_steps']
        except KeyError:
            sweep_steps = self.drone_cfg['tones']['N_step']

        phase_bins = np.array(phases).reshape(-1, sweep_steps)
        freq_bins = np.array(freqs).reshape(-1, sweep_steps)

        # Stitch phase discontiuities at bin edges
        # ----------------------------------------
        curr_shift = 0
        stitch_ends = int(sweep_steps / stitch_ends_factor)

        slope, intercept, _, _, _ = linregress(freq_bins[0, -1*stitch_ends:], phase_bins[0, -1*stitch_ends:])

        prev = 0
        next = intercept + slope*freq_bins[0, -1]
        for i in range(len(phase_bins) - 1):
            slope_prev, intercept_prev, _, _ , _ = linregress(freq_bins[i + 1, :stitch_ends], phase_bins[i + 1, :stitch_ends])
            prev = intercept_prev + slope_prev*freq_bins[i+1, 0]
            diff = prev - next
            slope_next, intercept_next, _, _, _ = linregress(freq_bins[i + 1, -1*stitch_ends:], phase_bins[i + 1, -1*stitch_ends:])
            next = intercept_next + slope_next*freq_bins[i + 1, -1]
            
            #diff = phase_bins[i+1][0] - phase_bins[i][-1] # Calculate simple difference between points at bin edges instead of performing linear fits on edges of bins
            phase_bins[i] -= curr_shift
            curr_shift += diff
        phase_bins[-1] -= curr_shift

        return phase_bins.flatten()

    #################
    # Magic Methods #
    #################


                
                

