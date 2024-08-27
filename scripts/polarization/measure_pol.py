# Imports
from ocs.ocs_client import OCSClient
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
import numpy as np
import time

# Local imports
sys.path.append('./../../rfsoc/') # Append path with RFSoC_DAQ.py
import rfsoc_io

def main():
    '''
    Main method run when 'measure_pol.py' is called directly
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

    # Initialize PCS agent to control polarizing grid
    P = OCSClient(cfg_io['pcs_agents']['pol_agent'], args=[])
    rfsoc_io.send_msg('INFO', 'Initialized polarized grid PCS agent', output)
   
    # Initialize RFSoC data acquisition object
    # ----------------------------------------
    R = R(args.cfg)

    # Tune detectors before taking data
    rfsoc_io.send_msg('INFO', '=======================Tuning Detectors=========================', output)
    tune_detectors(R, cfg, cfg_io, args)
    rfsoc_io.send_msg('INFO', 'Finished tuning detectors', output)

    # Wait for light source to be plugged in
    rfsoc_io.send_msg('INFO', 'Waiting for light source to be plugged in...')
    time.sleep(60)

    # Rotate grid and take timestream data
    rfsoc_io.send_msg('INFO', '========================Starting Data Acquisition==============================', output)
    #acquire_data(P, R, args)

def tune_detectors(R, cfg, cfg_io, args):
    # Find detectors and set tones
    # ----------------------------
    R.find_detectors(new_sweep = True, N_steps = 200, peak_dis = 6800, peak_width_min = 40, peak_width_max = 300)
    det_freqs = R.find_detectors_fine(N_steps = 200, bandwidth = 0.5)
    shifts = np.array(cfg['pol_config']['shifts']) * 1e6

    # Write a tone at all det_freqs + shifts 
    # --------------------------------------
    tone_freqs = det_freqs + shifts
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

def acquire_data(P, R, args):
    '''
    Rotates polarizing grid to specified angles and collects timestream data.

    Parameters:
        R: RFSoC data acquisition object
        t: Length of timestreams in seconds
    '''

    # Create list of angles to iterate over
    # -------------------------------------
    angles = None
    start, end, num = args.angles
    if args.num:
        angles = np.linspace(start, end, num)
    else:
        angles = np.arange(start, end + 2*num, num)

    # Iterate over angles and take timestream at each angle
    # -----------------------------------------------------
    with tqdm(range(len(angles) - 1), desc = f'DAQ:') as pbar:
        for i in pbar:
            # Update/add angle to config file
            R.edit_main_config('angle', float(angles[i]), append = True)
            pbar.set_postfix_str(f'Current Angle: {angles[i]}')

            # Take timestream data (also saves config file)
            for i in tqdm(range(args.trials), desc = "Trial"):
                R.take_timestream(args.t, write_tones = False)

            # Rotate polarizing grid to next angle
            dtheta = angles[i+1] - angles[i]
            P.moveStepper(deg=dtheta)

            # Wait before taking next timestream
            time.sleep(args.wait)

def eval_args():
    # Initialize arg parser
    parser = argparse.ArgumentParser(prog='measure_pol',
                                     description='''Perform polarization measurements on 
                                     microwave kinetic inductance detectors (MKID) using a rotating
                                     polarizing grid.''')
    parser.add_argument('t', type = float, help='Length of timestreams in seconds')
    parser.add_argument('-a', "--angles", nargs = 3, type = float,
                         default = (0, 360, 1), metavar=('Start', 'End', 'Spacing/Num'), 
                         help='Angles for which to collect data.')
    parser.add_argument('-n', "--num", action = 'store_true', 
                        help='Last value in -a argument is number of angles')
    parser.add_argument('-w', "--wait", type= float, default = 1,
                         help = 'Time to wait after moving grid before taking timestream.')
    parser.add_argument('-c', "--cfg", type = str, default='./pol_config.yaml', help='Path of a custom config file.')
    parser.add_argument('-t', "--trials", type = int, default=1, help='Number of timestreams to take per angle.')
    return parser.parse_args()

if __name__ == '__main__':
    main()
