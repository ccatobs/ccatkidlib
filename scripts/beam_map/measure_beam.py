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
sys.path.append('./../../analysis')  # Append path with Sweep Timestream Resonator

import rfsoc_io
from Sweep import Sweep

def main():
    '''
    Main method run when 'measure_beam.py' is called directly
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
    #tune_detectors(R, cfg, cfg_io, args)
    rfsoc_io.send_msg('INFO', 'Finished tuning detectors', output)
    rfsoc_io.send_msg('INFO', f"Final Tone Frequencies set to {R.cfg['rfsoc_tones']['tone_freqs']} MHz", output)

    # Wait for light source to be plugged in
    #rfsoc_io.send_msg('INFO', 'Waiting for light source to be plugged in...')
    #with tqdm(range(60), desc = f'TIME') as secs:
        #for sec in secs:
            #time.sleep(1)
    
    # take one last target sweep for data analysis purposes
    rfsoc_io.send_msg('INFO', '========================Taking Calibration Sweep==============================', output)
    #R.take_target_sweep(write_tones=True)

    # Rotate grid and take timestream data
    rfsoc_io.send_msg('INFO', '========================Starting Data Acquisition==============================', output)
    acquire_data(P, R, args)
    
def tune_detectors(R, cfg, cfg_io, args):
    # Find detectors and set tones
    # ----------------------------
    R.find_detectors(new_sweep = True, N_steps = 400, peak_prom_db = 0.07, peak_dis = 8800, peak_width_min = 25, peak_width_max = 400)
    det_freqs = R.find_detectors_fine(N_steps = 300, bandwidth = 4)
    
    # Makes sure that the correct number of resonators are found
    assert len(det_freqs) == cfg['rfsoc_tones']['num_tones'], 'Number of found detectors do not match nominal number of detectors'
    
    ## Removed shifts for now. 
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

def generate_sweep(args):
    # generates a list of coordinates for map making
    center = args.center
    resolution = args.res
    num_pixels = args.size
    coordinates = []

    if args.beamtype == 'rect':
        center_x, center_y = center
        num_pixels_x, num_pixels_y = num_pixels
        
        # Calculate the starting points
        start_x = center_x - (num_pixels_x * resolution) / 2
        start_y = center_y - (num_pixels_y * resolution) / 2
        
        for i in range(num_pixels_x):
            for j in range(num_pixels_y):
                x = start_x + i * resolution
                y = start_y + j * resolution
                coordinates.append((x, y))
    
        return coordinates
    
    if args.beamtype == 'radial':
        pass

def acquire_data(P, R, args):
    '''
    Rotates polarizing grid to specified angles and collects timestream data.

    Parameters:
        R: RFSoC data acquisition object
        t: Length of timestreams in seconds
    '''

    # Create list of positions to create map over
    # -------------------------------------
    positions = generate_sweep(args)
    P.changeStageSpeed(speed=6000)

    # Edit rfsoc config with the arguments inputed
    # -------------------------------------
    R.edit_main_config('center', args.center)
    R.edit_main_config('beam_type', args.beamtype)
    R.edit_main_config('beam_size', args.res)

    # Iterate over angles and take timestream at each angle
    # -----------------------------------------------------
    with tqdm(range(len(positions)), desc = f'DAQ:') as pbar:
        for i in pbar:
            # Update/add angle to config file
            R.edit_main_config('position', positions[i], append = True)
            pbar.set_postfix_str(f'Position {positions[i]}')

            R.take_timestream(args.t, write_tones = False)

            # Move xy stage to next position
            if i == len(positions) - 1: pass
            else: P.moveStageTo(x=positions[i+1][0], y=positions[i+1][1])

            # Wait before taking next timestream
            time.sleep(args.wait)

def eval_args():
    # Initialize arg parser
    parser = argparse.ArgumentParser(prog='measure_pol',
                                     description='''Perform beam mapping on 
                                     microwave kinetic inductance detectors (MKID) using a automatic
                                     XY stage''')
    parser.add_argument('t', type = float, help='Length of timestreams in seconds')
    parser.add_argument('-T', "--beamtype", default='rect', type = str, help='type of beam map')
    parser.add_argument('-c', "--center", nargs = 2, type = float,
                         default = (0, 0), metavar=('X', 'Y'), 
                         help='Center of Map')
    parser.add_argument('-s', "--size", nargs = 2, type = float,
                         default = (4, 4), metavar=('X/R', 'Y/Theta'), 
                         help='how many pixals in each direction')
    parser.add_argument('-r', "--res", type= float, default = 1,
                         help = 'resolution of beam map in mm. If radial beam map then applied in radial direction')
    parser.add_argument('-w', "--wait", type= float, default = 1,
                         help = 'Time to wait after moving xy stage before taking timestream.')
    parser.add_argument('-C', "--cfg", type = str, default='./beam_config.yaml', help='Path of a custom config file.')
    return parser.parse_args()

def get_dets(sweep, rfsoc_dir, dets=6):
    for i in range(dets):
        sweep_id = int(get_sweep_id(sweep))
        cfg_file = [file for file in rfsoc_dir.glob(f'*{sweep_id}*.yaml')][0]
        yield Sweep(sweep, i, cfg_file)      

def get_sweep_id(sweep_path):
    sweep = Path(sweep_path).name
    return sweep[5:15]


if __name__ == '__main__':
    main()
