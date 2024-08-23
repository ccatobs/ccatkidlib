# Imports
import sys
import numpy as np
import matplotlib.pyplot as plt

# Local imports
sys.path.append('./../rfsoc/') 
from Sweep import Sweep
from Timestream import Timestream
import rfsoc_io
import plot_utils

class Resonator:
    def __init__(self, array_pos, **kwargs):
        '''
        Class representing a single microwave kinetic inductance detector (MKID) driven with a set tone power.
        '''
        self.output = True
        for key, value in kwargs.items():
            if key  == 'output':
                self.output = value

        # Resonator Quality
        self.good_quality = True # Flag for discarding poor quality data

        # Unique identifiers
        self.array_pos = array_pos # Numerical position in MKID array (sorted by smallest to largest resonant frequency)
        
        # Associated target sweep(s)
        self.sweeps = list([])

        # Associated timestream(s)
        self.streams = list([])

        # Define resonator frequencies
        self.nominal_res_freq = None
        self.res_freq = None # Resonant frequency
        self.min_freq = None # Frequency where |S21| is minimized
        self.sens_freq = None # Frequency where resonator is most sensitive frequency shifts

        # Define resonator quality factors
        self.Qr = None # Resonator quality factor
        self.Qe = None # Complex external quality factor

        # Bifurcation parameter
        self.a = None
        self.bifurcated = False

        # External parameters
        self.tone_power = None # Driving tone power

    ######################
    # Analysis Functions #
    ######################

    def characterize_resonator(self, nonlinear, asymm, **kwargs):
        '''
        Calculate key resonator properties and set their values.
        '''

        # Fit resonator and get parameters of fit. If multiple sweeps, use first successful fit
        # -------------------------------------------------------------------------------------
        for sweep in self.sweeps:
            fitparams = sweep.get_fit_params(nonlinear, asymm, **kwargs)
            if fitparams is not None:
                break
        
        if fitparams is None:
            rfsoc_io.send_msg('WARNING', 'Unable to characterize resonator, there were no succesful fits!', self.output)
            return False
        else:
            # Set resonator parameters equal to fit best fit parameters
            best_values = fitparams.summary()['best_values']
            if nonlinear:
                self.a = best_values['a']
                if self.a > 0.77:
                    self.bifurcated = True
            self.Qe = best_values['Q_e_real'] + 1j*best_values['Q_e_imag']
            self.Qr = best_values['Q']
            self.res_freq = best_values['f_0']
    
    ######################
    # Plotting Functions #
    ######################

    
    ###################
    # Getters/Setters #
    ###################

    def add_sweep(self, sweep_file, tone_num, cfg_file, cfg_io_file = None, **kwargs):
        sweep = Sweep(sweep_file, tone_num, cfg_file, cfg_io_file, **kwargs)
        self.sweeps.append(sweep)

    def add_timestream(self, stream_file, tone_num, cfg_file, cfg_io_file = None, **kwargs):
        stream = Timestream(stream_file, tone_num, cfg_file, cfg_io_file, **kwargs)
        self.streams.append(stream)
        
