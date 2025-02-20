# Imports
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

# Local imports
sys.path.append('./../rfsoc/') 
from Sweep import Sweep
from Timestream import Timestream
import rfsoc_io
import plot_utils
import resonator_model_v3 as rm

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

        # High Resolution Models
        self.s21 = None
        self.fs = None
        self.phis = None

    ######################
    # Analysis Functions #
    ######################

    def process_timestreams(self, **kwargs):
        '''
        Processes all the timestreams by first characterizing the resonantor,
        then divides out cable delay and centers
        finally converts to frequencies
        '''
        #t1 = time.time()
        self.characterize_resonator(nonlinear=True, asymm=False)
        self.create_models()
        self.correct_timestream_cables()
        self.center_rotate_timestreams()
        self.convert_phase()
        #print(time.time() - t1)

        #t1 = time.time()
        self.convert_frequencies()
        #print(time.time() - t1)

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
    
    def center_data(self, data):
        ''' 
        Shifts MKIDS to origin and rotates such that infinite far freqs lie on neg Re axis
        Given a set of points, outputs their values normalized on circle
        Input data must have already done phase and gain correction
        '''

        s = self.Qr/np.abs(self.Qe)

        # rotation needed to offset impedance mismatches
        r = np.arctan2(self.Qe.imag, self.Qe.real)

        # apply rotation and add s/2 to center at origin
        ret = np.exp(1j*r)*(data - 1) + s/2.0
        
        # Rotate by another pi such that frequencies infinitely far are at origin
        ret = np.exp(1j*np.pi)*ret
        
        return ret
    

    def create_models(self, resolution=10, deg=50):
        '''
        fits resonator and returns a high resolution fit for s21, cable based on a gaussian like 
        distribution of frequencies. Only need to fit once. Uses the most recent sweep ie. last in array
        Also populates self.phi_to_freq with a function
        Parameters:
            resolution (float): how many hundreds of points for the model
        Returns:
            fs (array): array of frequencies
            s21 (array): array of high resolution normalized and centered s21 
            phis (array): array of high resolution phase plots

        '''
        #finefs = np.linspace(self.fs.min(), self.fs.max(), resolution*len(self.fs))

        if self.fs is None:
            self.sweeps[-1].normalize_sweep()
            fitparams = self.sweeps[-1].fitparams.params
            

            self.fs = np.linspace(self.sweeps[-1].fs.min(), self.sweeps[-1].fs.max(), resolution*100)
            #self.fs = np.random.normal(loc=self.sweeps[0].fs.mean(), scale=self.sweeps[0].fs.std(), size=resolution*1000)

            s21z = rm.fine_s21_model(self.fs, fitparams)
            cable = rm.fine_s21_model(self.fs, fitparams, cable=True)
            s21z = s21z / cable
            self.s21 = self.center_data(s21z)

            self.phis = np.arctan2(self.s21.imag, self.s21.real)
        
        X = self.phis
        Y = self.fs

        p = np.polyfit(X, Y, deg)
        f = np.poly1d(p)
        self.phi_to_freq = f

        return self.fs, self.s21, self.phis
    
    
    def correct_timestream_cables(self, n=-1, res=5):
        '''
        Corrects all timestreams for cable delay in self.streams using the nth sweep in self.sweeps
        Parameters:
            n (int): the nth sweep to use to subtract cable. default to -1 so most recent sweep
            res (float): how many thousands of points to use for cable model
        '''
        s = self.sweeps[n]
        s.normalize_sweep()

        # create high resolution cable fit
        fs = np.linspace(s.fs.min(), s.fs.max(), int(res*1e3))
        cable = rm.fine_s21_model(fs, s.fitparams.params, cable=True)
        s21z = rm.fine_s21_model(fs, s.fitparams.params)

        # get the cable term for all streams using the first stream
        index = np.argmin(np.abs(self.streams[0].s21z.mean() - s21z))
        cable_term = cable[index]

        # iterate through timestreams to correct cable
        for stream in self.streams:
            if not stream.removed_cable:
                stream.remove_cable(cable_term)
    
    def center_rotate_timestreams(self):
        '''
        Uses parameters from self to rotate timestreams onto the centered IQ circle
        '''
        for stream in self.streams:
            if not stream.centered:
                stream.s21z = self.center_data(stream.s21z)
    
    def phi_to_freq(self, phis):
        '''
        Given an array of phases, turns into frequencies based on modeled phase plot
        waiting to be populated by self.create_models()
        '''
        pass

    def convert_phase(self):
        for stream in self.streams:
            if stream.phis is None:
                stream.phis = np.arctan2(stream.s21z.imag, stream.s21z.real)
    
    def convert_frequencies(self):
        '''
        Uses modeled phase plots to turn into timestreams into where the resonant frequency is 
        '''
        for stream in self.streams:
            if stream.fs is None:
                stream.fs = self.phi_to_freq(stream.phis)
        
    
    ######################
    # Plotting Functions #
    ######################

    
    ###################
    # Getters/Setters #
    ###################

    def get_all_data(self,dtype='freq'):
        '''
        returns all the timestream data into a 2D array
        first index in the ith timestream
        returns calculated frequency data on default
        '''
        if dtype == 'freq':
            data = np.zeros(shape=(len(self.streams), self.streams[0].fs.shape[0]), dtype='complex128')
            
            for i in range(len(self.streams)):
                data[i] = self.streams[i].fs
        
        if dtype == 'phi':
            data = np.zeros(shape=(len(self.streams), self.streams[0].fs.shape[0]), dtype='complex128')
            
            for i in range(len(self.streams)):
                data[i] = self.streams[i].phis
        
        if dtype == 'cfg':
            data = []

            for stream in self.streams:
                data.append(stream.get_config())

        return data

    def add_sweep(self, sweep_file, tone_num, cfg_file, cfg_io_file = None, **kwargs):
        sweep = Sweep(sweep_file, tone_num, cfg_file, cfg_io_file, **kwargs)
        self.sweeps.append(sweep)

    def add_timestream(self, stream_file, tone_num, cfg_file, cfg_io_file = None, **kwargs):
        stream = Timestream(stream_file, tone_num, cfg_file, cfg_io_file, **kwargs)
        self.streams.append(stream)
        
