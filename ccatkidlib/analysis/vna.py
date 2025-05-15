from .sweep import Sweep
from pathlib import Path
from scipy.stats import linregress
import numpy as np
import sys
import gc

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

    ####################
    # Analysis Methods #    
    ####################

    def filter_freqs(self, fs = None, phase = None, res_freqs = None, w = 1, stitch = True):
        """
        classifies the peaks found by found_resonators into ones that are likely good resonators based on the slope of the complex phase at identified minima. 

        Parameters:
            res_freqs (list): The output list of resonators found using analysis._findResonators_alt in primecam_readout.
            fs (list): the list of frequencies swept over in the VNA sweep
            phase (list): the list of complex phase values returned by the VNA sweep. the function is structured this way in case you want to play
            around with the phase values you input to the fitting (i.e. with bin stitching or not with bin stitching). 
            w (int): the window size with which to fit the complex phase around identified minima. in testing, w = 1 or w = 2 works well when dealing
            with data that may have phase bin discontinuities. Could be increased if phase bin stitching improved. 
        Returns:
            good_resonators (list): The frequencies of found resonators that are flagged as real resonators (positive phase slope). 
            bad_resonators (list): The frequencies of found resonators that are flagged as bad (negative or zero phase slope). 
            data (list): the windowed data used to make determination of each resonator. This is mostly included for troubleshooting and can be
            discarded if desired. 
            idx_res (list): the indexes of all flagged real resonators as taken from the total list of f_res.
            idx_bad (list): the indexes of all bad or not real resonators as taken from the total list of f_res.
        """
        
        if fs is None: fs = self.fs()
        if res_freqs is None: res_freqs = self.res_freqs
        if phase is None: self.phase()
        
        if stitch: phase = stitch_phase(fs = fs, phase=phase)

        slopes = []
        data = []
        #variances = []
        for i in range(len(res_freqs)):
            peak_idx = np.where(fs == res_freqs[i])[0][0]
            X = fs[peak_idx-w: peak_idx+w+1] #defines window of data around peak_idx to fit
            Y = phase[peak_idx-w: peak_idx+w+1]
            
            X = np.c_[np.ones(X.shape[0]), X]  # Adds a column of ones needed for y-int
            
            fit_params = np.linalg.inv(X.T @ X) @ X.T @ Y
            slopes.append(fit_params[1])
            
            idx_res = np.where(np.array(slopes) > 0)[0]
            idx_bad = np.where(np.array(slopes) <= 0)[0]
            good_resonators = res_freqs[idx_res]
            bad_resonators = res_freqs[idx_bad]
            data.append(Y)

            #below calculated the variances of eat fitted window, and flags those with high variance. not clear this offers substantial improvement
            
            #window_var = np.std(Y)
            #variances.append(window_var)
            #flagged_high_var = np.intersect1d(np.where(variances > np.median(variances) + 1*np.std(variances))[0],idx_res)
            #var_flagged_good_res = f_res[flagged_high_var]
            
        return good_resonators, bad_resonators, data, idx_res, idx_bad

    def get_cable_delay(self, fs = None, phase = None, stitch_phase = True):
        if fs is None: fs = self.fs()

        if phase is None:
            phase = self.phase() 
            if stitch_phase: phase = self.stitch_phase(fs, phase)

        cable_delay, intercept, r, p, se = linregress(fs, phase)

        return cable_delay*1e9/(2*np.pi), se*1e9/(2*np.pi)
    
    def stitch_mag():
        pass

    def stitch_phase(self, fs = None, phase = None, threshold = 1.9*np.pi, stitch_ends_factor = 10):
        # Get frequency and phase data
        # ----------------------------
        if fs is None: fs = self.fs()
        if phase is None: phase = self.phase()

        # Stitch phase discontinuities caused by wrapping from -pi to pi
        # --------------------------------------------------------------
        curr_shift = 0
        for i in range(len(phase) - 1):
            diff = phase[i+1] - phase[i]
            phase[i] -= curr_shift
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

        phase_bins = np.array(phase).reshape(-1, sweep_steps)
        freq_bins = np.array(fs).reshape(-1, sweep_steps)

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

    ############################
    # Internal Loading Methods #
    ############################

    def _get_res_s21z(self):
        res_s21z = None
        s21z = self.s21z
        fs = list(self.fs())
        res_freqs = self.res_freqs
        if res_freqs is not None:
            try:
                res_s21z = [s21z[fs.index(freq)] for freq in res_freqs]
            except ValueError:
                res_s21z = super()._get_res_s21z()
        return res_s21z

    #################
    # Magic Methods #
    #################


                
                

