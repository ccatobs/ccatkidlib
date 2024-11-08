"""
11/05/2024
Andy Yang

Pipline for analyzing beam maps for MKIDS as part of CCAT project

Reads out target sweeps -> models resonators -> normalize timestreams
analyze oscillations in Phi -> turn into fractional frequency shift -> create map

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
import file_utils
import resonator_model_v3 as rm

##############
# Variables
#############

data_path = Path('/home/rfsoc/rfsoc_result/polarization/data/')
date = '20241007'
stamp = '1728322158'
board = 'B1D1'

io_path = Path('/home/rfsoc/Andy/beammap_results/')

def main():
    resonators = file_utils.load_folder_data(data_path, date, stamp, board)
    for res in resonators: res.process_timestreams()

    # get all related configs for creating a map
    beam_cfg = resonators[0].streams[0].cfg['beam_config']
    center = beam_cfg['center']
    beam_type = beam_cfg['beam_type']
    size = beam_cfg['resolution']

    x_meshes = []
    y_meshes = []
    maps = []

    for det in resonators:
        # interate through each detector to create a map for each
        denoised = denoise_timestreams(det)
        cfgs  = det.get_all_data(dtype='cfg')

        # Use Oscillations in Phase to get frequency Shift
        dfs, dferr = analyze_oscillations(denoised, det)

        data = np.zeros(shape=(len(cfgs), 3))
        for cfg, df, i in zip(cfgs, dfs, range(len(data))):
            data[i] = np.array(*cfg['beam_config']['position'], df)
        
        # Extract x and y values to determine grid dimensions
        x_vals = np.array([x for x, y, intensity in data])
        y_vals = np.array([y for x, y, intensity in data])

        # Calculate grid boundaries
        x_min, x_max = x_vals.min(), x_vals.max()
        y_min, y_max = y_vals.min(), y_vals.max()

        # Determine number of grid cells based on range and pixel size
        x_steps = int(np.ceil((x_max - x_min) / size)) + 1
        y_steps = int(np.ceil((y_max - y_min) / size)) + 1

        # Calculate starting coordinates based on the center
        x_start = center[0] - size * (x_steps - 1) / 2
        y_start = center[1] - size * (y_steps - 1) / 2

        # Generate 1D arrays of x and y positions
        x_positions = x_start + np.arange(x_steps) * size
        y_positions = y_start + np.arange(y_steps) * size

        # Generate coordinate matrices using meshgrid
        x_coords, y_coords = np.meshgrid(x_positions, y_positions)

        # Initialize an intensity matrix with NaNs (or zeros, depending on preference)
        intensity_matrix = np.full((y_steps, x_steps), np.nan)

        # Fill the intensity matrix based on the data
        for x, y, intensity in data:
            # Calculate the closest grid indices for x and y
            x_index = int(np.round((x - x_start) / size))
            y_index = int(np.round((y - y_start) / size))
            # Assign intensity to the appropriate cell
            intensity_matrix[y_index, x_index] = intensity
        
        x_meshes.append(x_coords)
        y_meshes.append(y_coords)
        maps.append(intensity_matrix)
    
    dat = np.array(x_meshes, y_meshes, maps)

    np.save(io_path / f'date_stamp_board_maps.npy')




def analyze_oscillations(denoised, det):
    # analyzes oscillations in a timestream to produce the values df
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

def find_local_extrema(arr, prom):
    # Find indices of local maxima
    peaks, _ = find_peaks(arr, prominence=prom)
    
    # Find indices of local minima by finding peaks of the inverted array
    inverted_peaks, _ = find_peaks(-arr, prominence=prom)
    
    # Extract the maxima and minima values from the array
    local_maxima = arr[peaks]
    local_minima = arr[inverted_peaks]
    
    return local_maxima, local_minima

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

if __name__ == '__main__':
    main()
