import sys
import numpy as np
import matplotlib.pyplot as plt
import lmfit
from scipy import stats

# Local imports
sys.path.append('/home/rfsoc/ccatkidlib/rfsoc/') 
import rfsoc_io
import plot_utils
import resonator_model_v3 as rm

class Sweep:
    def __init__(self, sweep_file, tone_num, cfg_file, cfg_io_file = None, **kwargs):
        '''
        Class representing a single target sweep taken with a radio frequency system on a chip (RFSoC). 
        Includes analysis functions for target sweeps of single microwave kinetic inductance detectors (MKIDs).
        '''
        self.output = True
        for key, value in kwargs.items():
            if key  == 'output':
                self.output = value

        # Load sweep data for all tones
        # -----------------------------
        fs, s21z = np.load(sweep_file, allow_pickle = True)

        # Try to load cfg file(s)
        # -----------------------
        self.cfg = rfsoc_io.load_config(cfg_file)

        if cfg_io_file:
            self.cfg_io = rfsoc_io.load_config(cfg_io_file)
        else:
            self.cfg_io = None

        # Get sweep data of specified tone
        # --------------------------------

        # Get number of steps used per tone
        N_steps = self.cfg['rfsoc_tones']['N_steps']      

        # Slice frequency and s21z data to get correct tone
        self.fs = np.real(fs[N_steps*tone_num:N_steps*(tone_num + 1)])
        self.s21z = s21z[N_steps*tone_num:N_steps*(tone_num + 1)]

        # Define other sweep parameters
        self.fitparams = None # Best fit parameters of resonator
        self.cablefine = None
        self.s21fine = None
        self.finefs = None

        # Fit S21 right away if needed
        if 'normalize' in kwargs.keys() and kwargs['normalize'] == True:
            self.normalize_sweep()


    ######################
    # Analysis Functions #
    ######################

    def fit_sweep(self, nonlinear = True, asymm = False, **kwargs):
        '''
        Fit resonator using either a nonliner or asymetric model (not both!). 
        Parameters:
            nonlinear (bool): Whether to use nonlinear model
            asymm (bool): Whether to use aysmmetric model.
        Returns:
            output (complex array): Complex S21 fit data
        '''
        
        refit = False
        # Evaluate kwargs
        for key, value in kwargs.items():
            if key == 'refit':
                refit = value

        I = np.real(self.s21z)
        Q = np.imag(self.s21z)

        # Attempt to fit resonator and get fit parameters
        if not (nonlinear and asymm):
            if self.fitparams is None or refit:
                try:
                    self.fitparams = rm.full_fit(self.fs, I, Q, nonlinear = nonlinear, asymm = asymm)
                except:
                    rfsoc_io.send_msg('WARNING', "Failed to fit resonator!", self.output)
                    return None
        else:
            rfsoc_io.send_msg('WARNING', "Fit must be either non-linear or asymmetric, not both!", self.output)
            return None

        # Use fit parameters to get fit s21z
        s21z_fit = rm.fine_s21_model(self.fs, self.fitparams.params, asymm = True)
        return s21z_fit

    def normalize_sweep(self, nonlinear = True, asymm = False, **kwargs):
        '''
        Normalize sweep data by dividing out cable delay and gain profile obtained from fitting resonator. 
        Parameters:
            nonlinear (bool): Whether use nonlinear model for fitting
            asymm (bool): Whether to use asymmetric model for fitting
        Returns:
            output (complex array): Normalized complex S21 data, normalized fit complex S21 data, and cable data
        
        '''
        # Fit resonator and get fit parameters
        s21z_fit = self.fit_sweep(nonlinear, asymm, **kwargs)
        fitparams = self.fitparams
        # Use fit parameters to fit cable profile
        cable_fit = rm.fine_s21_model(self.fs, fitparams.params, cable = True)

        # Normalize complex S21 data by dividing out cable profile
        s21z_norm = self.s21z/cable_fit
        s21z_fit_norm = s21z_fit/cable_fit

        return s21z_norm, s21z_fit_norm, cable_fit


    ######################
    # Plotting Functions #
    ######################
    
    def plot_S21():
        pass

    def plot_I():
        pass

    def plot_Q():
        pass

    def plot_IQ():
        pass

    def plot_phase():
        pass

    ###################
    # Getters/Setters #
    ###################

    def get_fit_params(self, nonlinear = True, asymm = False, **kwargs):
        self.fit_sweep(nonlinear, asymm, **kwargs)
        return self.fitparams

    def get_config(self):
        return self.cfg
