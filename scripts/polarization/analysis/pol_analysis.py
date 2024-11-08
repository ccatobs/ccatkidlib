"""
10/08/2024
Andy Yang

Pipline for analyzing polarization measurements for MKIDS as part of CCAT project

Reads out target sweeps -> models resonators -> normalize timestreams
analyze oscillations in Phi -> turn into fractional frequency shift

"""
import numpy as np
import time
import os
import random
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import sys

import yaml
from pathlib import Path

sys.path.append('/home/rfsoc/ccatkidlib/analysis') 
from Sweep import Sweep
from Resonator import Resonator
from Timestream import Timestream
import resonator_model_v3 as rm

##############
# Variables
#############

data_path = Path('/home/rfsoc/rfsoc_result/polarization/data/')
date = '20241007'
stamp = '1728322158'
board = 'B1D1'

io_path = Path('/home/rfsoc/Andy/pol_results')/date/board/stamp

PLOT = True

DEBUG = True

##############
# Pipeline
#############


##############
# Main
#############

def main():

    resonators = collect_polarization_data()
    
    for det in resonators:
        denoised = denoise_timestreams(det)
        cfgs  = det.get_all_data(dtype='cfg')

        dfs, dferr = analyze_oscillations(denoised, det)

        data = []
        # Use Oscillations in Phase to get frequency Shift
        for d, u, cfg in zip(dfs, dferr, cfgs):
            data.append([cfg['pol_config']['angle'], d, u])

        data = np.array(data)
        
        angles = np.unique(data[:,0])
        intens = []
        uncs = []
        for a in angles:
            condition = np.apply_along_axis(lambda x: x[0]==a, axis=1, arr=data)
            means = data[np.argwhere(condition)][:,0,1]

            uncers = data[np.argwhere(condition)][:,0,2]
            
            inten = means.mean()
            uncer = np.sqrt(np.sum(uncers * uncers))/np.sqrt(len(uncers))
            
            intens.append(inten)
            uncs.append(uncer)
        intens = np.array(intens)
        uncs = np.array(uncs)

        uncs = uncs/intens.max()
        intens = intens/intens.max()


###############
# Pipline Funcs
###############

def analyze_oscillations(denoised, det):
    # analyzes oscillations in a timestream to produce an array of
    # the amplitude of oscillations and their uncertainties
    dfs = np.zeros(shape=denoised.shape[0])
    dferr = np.zeros(shape=denoised.shape[0])
    for ts, i in zip(denoised, range(len(denoised))):
        highs, lows = find_local_extrema(ts, ts.std()/3)
        dfs[i] = highs.mean() - lows.mean()
        dferr[i] = (highs.std() + lows.std()) / np.sqrt(len(highs))
    
    return dfs, dferr


def denoise_timestreams(resonator, freqs=[0,10,30,50], s=[0.1,1,1,1], ):
    # denoises all the timestreams belonging to a resonator and returns 2D array
    data = resonator.get_all_data()
    denoised = np.zeros(shape=data.shape, dtype='complex128')
    cfgs  = resonator.get_all_data(dtype='cfg')

    for ts, cfg, n in zip(data, cfgs, range(len(data))):
        fft = np.fft.fft(ts)
        fs = np.fft.fftfreq(len(ts), 1/(512e6/(2**20)))

        window = np.zeros(fs.shape)

        if freqs is None:
            denoised[n] = ts
            continue
        
        for i in range(len(freqs)):
            window += gaussian_window(fs, s[i], freqs[i])

            if freqs[i] != 0:
                window += gaussian_window(fs, s[i], -freqs[i])
        
        # apply window
        fft = fft * window
        denoised[n] = np.fft.ifft(fft).real

    return denoised


def collect_polarization_data():
    # Collects all the data in directory and fits resonators using functions in ccatkidlib
    # Returns list of resonator classes which stores all the streams, sweeps, cfgs

    targs, targ_cfgs = get_targ_list(data_path, date, stamp, board)
    timestreams, ts_cfgs = get_timestream_list(data_path, date, stamp, board)

    with open(targ_cfgs[0]) as config:
        cfg = yaml.safe_load(config)

    resonators = [Resonator(i) for i in range(cfg['rfsoc_tones']['num_tones'])]

    #t1 = time.time()
    for i in range(len(resonators)):
        resonators[i].add_sweep(targs[-1], i, targ_cfgs[-1])

    for f, cfg in zip(timestreams, ts_cfgs):
        for i in range(len(resonators)):
            resonators[i].add_timestream(f, i,  cfg)

    #print(time.time() - t1)

    for res in resonators:
        res.process_timestreams()
    
    return resonators
       

def get_timestream_list(data_path, date, stamp, board):
    # Returns a list of Paths to timestreams and a list
    # of Paths to the corresponding cfgs

    timestreams = sorted((data_path/'timestream'/date/board/stamp).glob('*.npy'), key=lambda a: get_id(a.name))
    cfgs = sorted((data_path/'rfsoc'/date/board/stamp).glob('*_stream_config.yaml'), key=lambda a: get_id(a.name))

    return timestreams, cfgs


def get_targ_list(data_path, date, stamp, board):
    # Returns a list of Paths to target sweeps and a list
    # of Paths to the corresponding cfgs

    targs = sorted((data_path/'targ'/date/board/stamp).glob('*.npy'), key=lambda a: get_id(a.name))
    cfgs = sorted((data_path/'rfsoc'/date/board/stamp).glob('*_targ_config.yaml'), key=lambda a: get_id(a.name))

    return targs, cfgs

###############
# Fit Funcs
###############
def sinfunc(x, a, w, p, d):
    return a*np.sin(w*x-p) + d

def fit_sin(x_dat, y_dat, unc=None):
    guess = [(y_dat.max() - y_dat.min()), 2*np.pi/180, 0, y_dat.mean()]
    bounds = [(0, 0, -100, -np.inf) ,(1.6*(y_dat.max() - y_dat.min()), 100, 100, np.inf)]
    params, covariance = curve_fit(sinfunc, x_dat, y_dat, p0=guess, bounds=bounds, sigma=unc)

    a, w, p, d = params    
    return a, w, p, d

def nist_fit_func(x, A, C, W, p1, p2):
    # C is crosspol amplitude
    # W is wobble amplitude
    return A*(1-W*np.cos(np.pi*x/360 -p1)**2)*(np.cos(np.pi*x/180 - p2)**2 + C*np.sin(np.pi*x/180 -p2)**2)

def chi_squared_reduced(data, model, unc=None, n_params=0):
    if unc is None:
        chi2 = np.sum((data - model) ** 2)
    else:
        chi2 = np.sum(((data - model) / unc) ** 2)

    # Degrees of freedom: number of data points - number of model parameters
    dof = len(data) - n_params
    return chi2 / dof if dof > 0 else np.inf  # Avoid division by zero

###############
# Helper Funcs
###############

def find_local_extrema(arr, prom):
    # Find indices of local maxima
    peaks, _ = find_peaks(arr, prominence=prom)
    
    # Find indices of local minima by finding peaks of the inverted array
    inverted_peaks, _ = find_peaks(-arr, prominence=prom)
    
    # Extract the maxima and minima values from the array
    local_maxima = arr[peaks]
    local_minima = arr[inverted_peaks]
    
    return local_maxima, local_minima

def get_id(s):
    ret = ''
    flag = 0
    for i in s:
        if i in str([1,2,3,4,5,6,7,8,9,0]):
            ret += i
            flag = 1
        elif flag == 1:
            return ret
    return ret

def gaussian_window(x_values, sigma=1.0, mu=None):
    """
    Generates a Gaussian window based on the input array of x-values.

    Parameters:
    - x_values (array-like): The input array of x-values.
    - sigma (float): The standard deviation of the Gaussian window. Default is 1.0.
    - mu (float or None): The mean (center) of the Gaussian window. If None, the mean will be the center of the x_values.

    Returns:
    - gaussian (numpy array): The Gaussian window corresponding to the input x-values.
    """
    x_values = np.array(x_values)
    
    # Set the mean to the center of the x_values if not provided
    if mu is None:
        mu = np.mean(x_values)
    
    # Compute the Gaussian window
    gaussian = np.exp(- (x_values - mu)**2 / (2 * sigma**2))
    
    return gaussian

def sort_data(data): 
    # sorts list of tuples with (angle, value)
    # returns a list of angles, value means, and uncertainties
    # written by chatGPT

    # Sort data by angle
    sorted_data = sorted(data, key=lambda x: x[0])

    # Initialize lists for the final angles, average intensities, and uncertainties
    angles = []
    avg_intensities = []
    uncertainties = []

    # Variables to accumulate data for the current angle
    current_angle = sorted_data[0][0]
    intensity_values = []

    # Loop through sorted data
    for angle, intensity in sorted_data:
        if angle == current_angle:
            # Accumulate intensities for the same angle
            intensity_values.append(intensity)
        else:
            # Calculate average intensity and uncertainty (std dev) for the previous angle
            avg_intensity = np.mean(intensity_values)
            uncertainty = np.std(intensity_values)/np.sqrt(len(intensity_values)) if len(intensity_values) > 1 else 0
            
            # Store the results
            angles.append(current_angle)
            avg_intensities.append(avg_intensity)
            uncertainties.append(uncertainty)

            # Start accumulating for the new angle
            current_angle = angle
            intensity_values = [intensity]

    # Handle the last group
    avg_intensity = np.mean(intensity_values)
    uncertainty = np.std(intensity_values, ddof=1)/np.sqrt(len(intensity_values)) if len(intensity_values) > 1 else 0
    angles.append(current_angle)
    avg_intensities.append(avg_intensity)
    uncertainties.append(uncertainty)

    # Convert lists to numpy arrays
    angles_array = np.array(angles)
    intensities_array = np.array(avg_intensities)
    uncertainties_array = np.array(uncertainties)

    return angles_array, intensities_array, uncertainties_array

if __name__ == '__main__':
    main()
