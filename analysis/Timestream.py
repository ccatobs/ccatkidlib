# Imports
import sys
import numpy as np
import matplotlib.pyplot as plt

# Local imports
sys.path.append('./../rfsoc/') 
import rfsoc_io
import plot_utils

class Timestream:
    def __init__(self, stream_file, tone_num, cfg_file, cfg_io_file = None, **kwargs):
        '''
        '''

        self.output = True
        for key, value in kwargs.items():
            if key  == 'output':
                self.output = value

        # Load timestream file
        # --------------------
        data = np.load(stream_file, allow_pickle=True) 
        self.s21z = data[tone_num] # Get the stream data of specified tone

        # Try to load cfg file(s)
        # -----------------------
        self.cfg = rfsoc_io.load_config(cfg_file)

        if cfg_io_file:
            self.cfg_io = rfsoc_io.load_config(cfg_io_file)
        else:
            self.cfg_io = None

        # Define commonly used parameters
        # ----------------------------
        try:
            self.output = self.cfg_io['io']['terminal_output']
        except:
            self.output = True

        # Get tone information
        # ---------------------------
        self.tone_freq = self.cfg['rfsoc_tones']['tone_freqs'][tone_num]
        self.tone_power = self.cfg['rfsoc_tones']['tone_powers'][tone_num]
        self.tone_phi = self.cfg['rfsoc_tones']['tone_phis'][tone_num]

        # Determine stream times based on sampling freq
        # ---------------------------------------------
        try:
            sampling_freq = self.cfg_io['rfsoc_io']['sampling_freq']
        except:
            sampling_freq = 512e6/(2**20)

        # Final time is total number of samples/sampling rate
        final_t = len(self.s21z)/sampling_freq
        self.ts = np.linspace(0, final_t, len(self.s21z))

        self.remove_outliers()
        
    ##################
    # Analysis Funcs #
    ##################

    def remove_outliers(self):
        self.ts = self.ts[50:]
        self.s21z = self.s21z[50:]

    def remove_cable_delay():
        pass

    def fft_stream(self):
        from scipy.fft import fft, fftfreq, fftshift

        fft_freqs = fftshift(fftfreq(len(self.ts), d = self.ts[1] - self.ts[0]))
        stream_fft = fftshift(fft(np.abs(self.s21z)))

        return fft_freqs, stream_fft

    ##################
    # Plotting Funcs #
    ##################

    def plot_abs(self, figax = None, **kwargs):
        # Plot time vs. |S21|
        fig, ax = plot_utils.init_fig(figax, **kwargs)
        ax.plot(self.ts, 20*np.log10(np.abs(self.s21z)))

        # Set labels
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('|S21|')
        ax.set_title(f'Tone Freq: {self.tone_freq:.4E} Hz, Tone Power: {self.tone_power} dB')

        return (fig, ax)
    
    def plot_I(self, figax = None, **kwargs):
        fig, ax = plot_utils.init_fig(figax, **kwargs)
        ax.plot(self.ts, np.real(self.s21z))

        # Set labels
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('I')
        ax.set_title(f'Tone Freq: {self.tone_freq:.4E} Hz, Tone Power: {self.tone_power} dB')

        return (fig, ax)
    
    def plot_Q(self, figax = None, **kwargs):
        fig, ax = plot_utils.init_fig(figax, **kwargs)
        ax.plot(self.ts, np.imag(self.s21z))

        # Set labels
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Q')
        ax.set_title(f'Tone Freq: {self.tone_freq:.4E} Hz, Tone Power: {self.tone_power} dB')

        return (fig, ax)
    
    def plot_IQ(self, figax = None, **kwargs):
        '''
        Plot the real part of timestream S21 (I) vs. the imaginary part of timestream S21 (Q).
        
        Parameters:
            subtract_mean: Whether to subtract off average I and Q of timestream
        '''

        subtract_mean = False
        for key, value in kwargs.items():
            if key == 'subtract_mean':
                subtract_mean = value

        # Initialize figure
        fig, ax = plot_utils.init_fig(figax, **kwargs)
        
        # Subtract average I and Q if desired to center timestream at (I = 0, Q = 0)
        mean_I = 0
        mean_Q = 0

        if subtract_mean:
            mean_I = np.mean(np.real(self.s21z))
            mean_Q = np.mean(np.imag(self.s21z))

        # Plot I vs. Q
        ax.scatter(np.real(self.s21z) - mean_I, np.imag(self.s21z) - mean_Q, s = 1, alpha = 0.5)

        # Set labels
        ax.set_xlabel('I')
        ax.set_ylabel('Q')
        ax.set_title(f'Tone Freq: {self.tone_freq:.4E} Hz, Tone Power: {self.tone_power} dB')

        return (fig, ax)
    
    def plot_FFT(self, figax = None, **kwargs):
        fig, ax = plot_utils.init_fig(figax, **kwargs)
        fft_freqs, stream_fft = self.fft_stream()

        ax.plot(fft_freqs, np.abs(stream_fft))

        # Set labels
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('FFT of |S21|')
        ax.set_title(f'Tone Freq: {self.tone_freq:.4E} Hz, Tone Power: {self.tone_power} dB')

        return (fig, ax)
    
    ################
    # Helper Funcs #
    ################

    def get_config(self):
        return self.cfg