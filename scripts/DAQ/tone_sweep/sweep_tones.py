# Imports
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
import numpy as np
import time

# Local imports
sys.path.append('./../../rfsoc/') # Append path with RFSoC_DAQ.py
sys.path.append('./../../analysis')  # Append path with Sweep Timestream Resonator

import rfsoc_io
from Sweep import Sweep

def main():
    '''
    Main method run when 'sweep_tones_pol.py' is called directly
    '''
    
    # Import RFSoC control module
    from rfsoc_daq import R

    # Parse command line arguments
    args = eval_args()

    # Read configuration file
    cfg = rfsoc_io.load_config(args.cfg)

    try:
        cfg, cfg_io = cfg
    except:
        cfg_io = None

    # Store common variables
    output = cfg_io['io']['terminal_output']
   
    # Initialize RFSoC data acquisition object
    # ----------------------------------------
    R = R(args.cfg)

    # Tune detectors before taking data
    rfsoc_io.send_msg('INFO', '=======================Finding Detectors=========================', output)
    tune_detectors(R, cfg, cfg_io, args)
    rfsoc_io.send_msg('INFO', 'Finished tuning detectors', output)
    rfsoc_io.send_msg('INFO', f'Number of Detectors Found: {len(R.cfg['rfsoc_tones']['tone_freqs'])}', output)


    # Rotate grid and take timestream data
    rfsoc_io.send_msg('INFO', '========================Starting Tone Sweep==============================', output)
    acquire_data(R, args)
    
def tune_detectors(R, cfg, cfg_io, args):
    # Find detectors and set tones
    # ----------------------------
    det_freqs = R.find_detectors(new_sweep = True, N_steps = 400, peak_prom_db = 0.07, peak_dis = 8800, peak_width_min = 25, peak_width_max = 400)

    # We wont need find detectors fine
    #det_freqs = R.find_detectors_fine(N_steps = 300, bandwidth = 4)
    # Take a test sweep
    
    # Makes sure that the correct number of resonators are found
    #assert len(det_freqs) == cfg['rfsoc_tones']['num_tones'], 'Number of found detectors do not match nominal number of detectors'
    
    # Write a tone at all det_freqs
    # --------------------------------------
    tone_freqs = np.array(det_freqs)
    tone_powers = np.array(cfg['rfsoc_tones']['tone_powers'])
    tone_phis = np.array(cfg['rfsoc_tones']['tone_phis'])
    
    data_dir = Path(cfg_io['file_paths']['base_data_dir'])

    # Use tone powers and phis found from VNA sweep custom parameters not provided 
    if len(tone_powers) == 0 or len(tone_powers) != len(det_freqs):
        tone_powers = np.load(data_dir / "tmp" / "a_tones_comb_cust.npy")
    if len(tone_phis) == 0 or len(tone_phis) != len(det_freqs):
        tone_phis = np.load(data_dir / "tmp" / "p_tones_comb_cust.npy")

    # Edit tones in RFSoC main config file
    R.edit_main_config('tone_freqs', tone_freqs.tolist())
    R.edit_main_config('tone_powers', tone_powers.tolist())
    R.edit_main_config('tone_phis', tone_phis.tolist())

def acquire_data(R, args):
    '''
    Sweeps through different tone powers and collect target sweeps and each tone pwoer

    Parameters:
        R: RFSoC data acquisition object
    '''

    # Create list of angles to iterate over
    # -------------------------------------
    tones = None
    start, end, num = args.tones
    if args.num:
        tones = np.linspace(start, end, num)
    else:
        tones = np.arange(start, end + 2*num, num)

    # Iterate over angles and take timestream at each angle
    # -----------------------------------------------------
    with tqdm(range(len(tones) - 1), desc = f'DAQ:') as pbar:
        for i in pbar:
            # Update/add angle to config file
            tone_powers = np.full(shape=(len(R.get_main_config()['rfsoc_tones']['tone_freqs'])), fill_value=tones[i])
            R.edit_main_config('tone_powers', tone_powers.tolist(), append = True)
            pbar.set_postfix_str(f'Current Tone: {tones[i]}')

            # Take target sweep data (also saves config file)
            R.take_target_sweep()

def eval_args():
    # Initialize arg parser
    parser = argparse.ArgumentParser(prog='sweep_tones',
                                     description='''Performs target sweeps at different tone powers for Kinetic Inductance Detectors.
                                     the tone powers for all the detectors will be the same''')
    parser.add_argument('t', type = float, help='Length of timestreams in seconds')
    parser.add_argument('-a', "--tones", nargs = 3, type = float,
                         default = (0, 100, 1), metavar=('Start', 'End', 'Spacing/Num'), 
                         help='tone powers for which to collect data.')
    parser.add_argument('-n', "--num", action = 'store_true', 
                        help='Last value in -a argument is number of angles')
    parser.add_argument('-c', "--cfg", type = str, default='./tone_config.yaml', help='Path of a custom config file.')
    #parser.add_argument('-t', "--trials", type = int, default=1, help='Number of sweeps to take per angle.')
    return parser.parse_args()


if __name__ == '__main__':
    main()
