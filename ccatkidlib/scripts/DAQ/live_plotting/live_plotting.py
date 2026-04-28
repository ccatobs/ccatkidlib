# Imports
from ocs.ocs_client import OCSClient
from pathlib import Path
from tqdm import tqdm
from collections import deque
import argparse
import sys
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from functools import partial

matplotlib.use('qt5agg')

# Local imports
sys.path.append('./../../rfsoc/') # Append path with RFSoC_DAQ.py

import rfsoc_io

def main():
    '''
    Main method run when 'live_plotting.py' is called directly
    '''


    # Import RFSoC control module
    from rfsoc_daq import R

    # Parse command line arguments
    args = eval_args()
   
    # Initialize RFSoC data acquisition object
    # ----------------------------------------
    R = R(args.cfg)

    sys.path.append(R.io_cfg['file_paths']['analysis_dir'])

    # Store common variables
    output = R.io_cfg['io']['terminal_output']

    # Tune detectors before taking data
    if args.tune: 
        rfsoc_io.send_msg('INFO', 'Tuning Detectors', output)
        tune_detectors(R, args)
        rfsoc_io.send_msg('INFO', 'Finished tuning detectors', output)

    # Live plot data
    rfsoc_io.send_msg('INFO', 'Starting Live Plot', output)
    live_plot(R, args)

###########################
# General Live Plotting Functions #
###########################

def live_plot(R, args):
    save = R.io_cfg['live_plot_config']['save_data']
    save_interval = R.io_cfg['live_plot_config']['save_interval']
    save_params = (save, save_interval)

    if save:
        global save_data

        # save_data = np.array(cfg['rfsoc_tones']['num_tones']*[[]])
    
    # Get autoscale from config
    autoscale = R.io_cfg['live_plot_config']['autoscale']

    # Get sampling frequency of RFSoC
    #sampling_freq = eval(R.io_cfg['rfsoc_io']['sampling_freq'])

    sampling_freq = 488.28

    # Number of points to plot to achieve the specified plot time
    num_points = int(args.time*sampling_freq)

    # Total number of iterations (refreshes) to perform
    max_its = int(args.uptime/args.time_per_it)

    # Create a deque for each resonator
    global data
    data = list()
    for i in range(len(args.resonators)):
        data.append(deque(np.full(num_points, np.nan)))

    # Create figure object
    fig, axs = plt.subplots(figsize = (8, 8), tight_layout = True)
    figax = (fig, axs)

    artists = list()
    for i in range(len(data)):
        if args.style == 'line' or args.style == 'both':
            artists.append(plt.plot([], [], animated=True)[0])
        if args.style == 'scatter' or args.style == 'both':
            artists.append(plt.scatter([],[], s = 2, alpha = 0.5))

    # Define kwargs to pass to animation function
    kwargs = {'num_points':num_points}
    # Define S21 plotting functions
    if args.format == 'S21':
        init_func = setup_S21
        func = S21
    # Define FFT plotting functions
    elif args.format == 'FFT':
        init_func =  setup_FFT
        func = FFT
        kwargs['sampling_freq'] = sampling_freq
    elif args.format == 'IQ':
        init_func = setup_IQ
        func = IQ

    init = partial(init_func, figax = figax, artists = artists, args = args)
    update = partial(update_artists, artists=artists, figax = figax, R = R, func = func, save_params = save_params, autoscale = autoscale, args = args, **kwargs)
    
    global last_frame

    last_frame = 0
    
    # Turn on timestreaming
    R.take_timestream(1, write_tones = False, save_data = False, reset=True, turn_off = False)
    anim = FuncAnimation(fig, func = update, init_func=init, frames = max_its, interval = 0, repeat = False, blit = True)
    plt.show()

def update_artists(frames, artists, figax, R, func, save_params, autoscale, args, **kwargs):
    global data #Have 'data' point to the global list storing past data
    global last_frame

    fig, ax = figax
    for key, value in kwargs.items():
        if key == 'num_points':
            num_points = value
 
    # Take a timestream
    s21z = R.take_timestream(args.time_per_it, write_tones = False, save_data = False, reset=False, turn_off = False)
    print(s21z)
    # Run if data is to be saved
    if save_params[0]:
        # Define global variables
        global save_data 

        # Add most recently taken data to list of saved data
        save_data = np.append(save_data, s21z, axis = 1)

        # If it has been 'interval' time since last save, save the data to disk
        if (frames - last_frame)*args.time_per_it > save_params[1]:
            R._save_timestream(save_data) # Save data to disk
            save_data = np.array(len(save_data)*[[]]) # Reset list of saved data (to save memory)
            last_frame = frames # Reset time since last save        

    # Update data list and plot artist corresponding to data
    rescale = False

    sf = 0.001 # Scale factor
    xmin = np.inf
    xmax = -np.inf
    ymin = np.inf
    ymax = -np.inf

    # Loop over each resonator (parallelize?)
    for i in range(len(data)):
        data[i].extend(s21z[args.resonators[i]])
        while len(data[i]) > num_points:
            data[i].popleft()
        xdata, ydata = func(data[i], args, **kwargs)
        base = int(len(artists)/len(data))

        # Loop over each artist associated with resonator
        for j in range(base):
            try:
                artists[int(j + i*base)].set_data(xdata, ydata)
            except:
                artists[int(j + i*base)].set_offsets(np.column_stack([xdata, ydata]))

        if autoscale and frames % 100 == 0:
            currxmin = np.nanmin(xdata)
            currxmax = np.nanmax(xdata)
            currymin = np.nanmin(ydata)
            currymax = np.nanmax(ydata)

            if currxmin < xmin: xmin = currxmin
            if currxmax > xmax: xmax = currxmax
            if currymin < ymin: ymin = currymin
            if currymax > ymax: ymax = currymax
        
    if autoscale and frames % 100 == 0: 
        rescale = autoscale_plot(figax, (xmin, xmax), (ymin, ymax), sf)

    if rescale: fig.canvas.draw_idle()

    return artists

def autoscale_plot(figax, xparams, yparams, sf):
    fig, ax = figax
    xmin, xmax = xparams
    ymin, ymax = yparams

    sxmin = xmin - abs(xmin*sf)
    sxmax = xmax + abs(xmax*sf)

    symin = ymin - abs(ymin*sf)
    symax = ymax + abs(ymax*sf)

    rescale = False

    xlims = list(ax.get_xlim())
    ylims = list(ax.get_ylim()) 

    if xlims[0] > sxmin or xlims[1] < sxmax:
        ax.set_xlim(sxmin, sxmax)
        rescale = True
    if ylims[0] > sxmin or ylims[1] < sxmax:
        ax.set_ylim(symin, symax)
        rescale = True

    return rescale

###############################
# Specific Plotting Functions #
###############################

def setup_S21(figax, artists, args, **kwargs):
    fig, axs = figax
    
    # Try to loop over subplots and set axis titles and limits
    try:
        for ax in axs:
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("S21 (dB)")
            plt.xlim(-args.time, 0)
            plt.ylim(0, 1)

    # If error thrown, assume only one subplot
    except:
        ax = axs
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("S21 (dB)")
        plt.xlim(-args.time, 0)
        plt.ylim(0, 1)

    return artists

def S21(s21z, args, **kwargs):
    return s21z[0], np.abs(np.array(s21z)[1:])

def setup_FFT(figax, artists, args, **kwargs):
    fig, axs = figax

    # Try to loop over subplots and set axis titles and limits
    try:
        for ax in axs:
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("FFT of S21")
            plt.xlim(0, 1)
            plt.ylim(0, 1)

    # If error thrown, assume only one subplot
    except:
        ax = axs
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("FFT of S21")
        plt.xlim(0, 1)
        plt.ylim(0, 1)

    return artists

def FFT(s21z, args, **kwargs):
    for key, value in kwargs.items():
        if key == 'num_points': 
            num_points = value
        elif key == 'sampling_freq':
            sampling_freq = value
    freqs  = np.fft.rfftfreq(num_points, d = 1/sampling_freq)
    return freqs[1:-2], np.abs(np.fft.rfft(S21(s21z, args, **kwargs)[1]))[1:-2]

def setup_IQ(figax, artists, args, **kwargs):
    fig, axs = figax
    
    # Try to loop over subplots and set axis titles and limits
    try:
        for ax in axs:
            ax.set_xlabel("I")
            ax.set_ylabel("Q")
            plt.xlim(-1,1)
            plt.ylim(-1, 1)

    # If error thrown, assume only one subplot
    except:
        ax = axs
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)

    return artists    

def IQ(s21z, args, **kwargs):
    return np.real(s21z[1:]), np.imag(s21z[1:])

###################
# Other Functions #
###################


def tune_detectors(R, args):
    # Find detectors and set tones
    # ----------------------------
    R.find_detectors(new_sweep = True)
    det_freqs = R.find_detectors_fine(new_sweep = True)
    R.take_target_sweep(write_tones = False)
      
def eval_args():
    # Initialize arg parser
    parser = argparse.ArgumentParser(prog='live_plotting',
                                     description='''Live plotting of detector timestream data in various formats.''')
    parser.add_argument('-r', '--resonators', type = int, nargs = "+", help = 'Which resonator(s) to plot.')    
    parser.add_argument('-f', '--format', default = "S21", choices=['S21', 'FFT', 'IQ'], help = "Format of the plot.")
    parser.add_argument('-s', '--style', default = 'line', choices=['line', 'scatter', 'both'], help = 'How to plot the data.')
    parser.add_argument('-t', '--time', type = float, default = 60, help = "Length of time series data to use for the plot.")
    parser.add_argument('-u', '--uptime', type = float, default = 2*60, help = 'Length of time to keep the live plotter running.')
    parser.add_argument('-p', '--time_per_it', type = float, default = 1, help = 'Amount of time per refresh (1/refresh rate).')
    parser.add_argument('--tune', action = 'store_true', help = 'Flag indicating detectors should be tuned.')
    parser.add_argument('-c', "--cfg", type = str, default='./live_plotting_system_config.yaml', help='Path of system config file.')


    return parser.parse_args()

if __name__ == '__main__':
    main()
