# Imports
from ocs.ocs_client import OCSClient
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
import numpy as np
import time

# Local imports
from ccatkidlib.rfsoc.rfsoc_daq import R
import ccatkidlib.rfsoc_io as rfsoc_io

def main():
    '''
    Main method run when 'fts.py' is called directly
    '''

    # Parse command line arguments
    args = eval_args()
   
    # Initialize PCS Agents
    # ---------------------
    # Initialize RFSoC control agent
    RC = R(args.cfg)
    output = RC.io_cfg['io']['terminal_output']

    # Initialize polarized beam mapper agent
    fts = OCSClient(RC.io_cfg['pcs_agents']['fts_agent'], args=[])
    rfsoc_io.send_msg('INFO', 'Initialized FTS PCS agent!', output)
    fts.init_stage()
    fts.home()


    # Tune detectors before taking data
    # ---------------------------------
    if args.tune: 
        rfsoc_io.send_msg('INFO', 'Tuning Detectors', output)
        tune_detectors(RC, args)
        rfsoc_io.send_msg('INFO', 'Finished tuning detectors', output)
        
    turned_on = False
    # Wait for IR source to be plugged in
    # -----------------------------------
    rfsoc_io.send_msg('INFO', 'Waiting for chopper to be moved...')

    while not turned_on:
        response = input('Is the Chopper in Place? (y/n): ')
        if response.lower() == 'yes' or response.lower() == 'y':
            turned_on = True   

    # Start beam mapping
    # ------------------
    rfsoc_io.send_msg('INFO', 'Starting FTS Mirror Sweep', output)
    acquire_data(fts, RC, output, args)
   
def acquire_data(fts, RC, output, args):
    '''
    Move FTS mirror to specified coordinates and collect timestream data.

    Parameters:
        R: RFSoC data acquisition object
        t: Length of timestreams in seconds
    '''

    # Generate map 
    # -------------------------------------
    pos_array = generate_passes(output, args)
    rfsoc_io.send_msg('INFO', f'Generated Mirror Positions:\n{pos_array}', output)

    RC.edit_config(RC.io_cfg, "pos_array", pos_array.tolist())
    rfsoc_io.save_config(RC.log_dir / f"fts_config_io_{RC.timestamp}.yaml", RC.io_cfg, RC.save_cfg)

    # Iterate over positions and take timestreams
    # -------------------------------------------
    with tqdm(range(len(pos_array)), desc = f'FTS Mirror Sweep:') as pbar:
        for i in pbar:
            pos = pos_array[i]

            # Move xy stage to next position
            fts.move_to(position=pos)
            if i == 0:
                time.sleep(15)

            rfsoc_io.send_msg('INFO', f'Finished moving FTS mirror to {pos} mm!', output)
            
            # Wait before taking timestream
            time.sleep(args.wait)

            # Update/add position to config file
            RC.edit_config(RC.ext_cfg, 'position', float(pos))
            pbar.set_postfix_str(f'Current Mirror Position: {pos} mm')

            if args.t > 0: RC.take_timestream(args.t, write_comb = False, setup = False)
    #RC.take_timestream(1, write_comb = False, turn_off = True, save_data = False, setup = False, reset = False)

def generate_passes(output, args):
    # Generates a list of positions for the FTS mirror
    # ----------------------------------------------
    low, up = args.positions  # Lower and upper positions of FTS mirror
    res = args.res # Resolution of mirror spacing
    num = args.N 
    passes = args.sweeps

    if num:
        single_pass = np.linspace(low, up, res)
    else:
        single_pass = np.arange(low*100, (up + res)*100, res*100)
        single_pass = single_pass/100
    
    pos_array = single_pass
    for _ in range(passes-1):
        single_pass = single_pass[::-1]
        pos_array = np.concatenate([pos_array, single_pass], axis=0)

    return pos_array

###################
# Other Functions #
###################

def tune_detectors(RC, args):
    # Find detectors and set tones
    # ----------------------------
    RC.find_detectors(new_sweep = True)
    det_freqs = RC.find_detectors_fine(new_sweep = True, write_comb = False)
    RC.take_target_sweep(write_comb = False)

def eval_args():
    # Initialize arg parser
    parser = argparse.ArgumentParser(prog='FTS', description='''Perform FTS measurement on microwave kinetic inductance detectors (MKID) using a FTS with wire grids and moving central mirror.''')
    
    # Add arguments
    parser.add_argument('t', type = float, help='Length of timestreams in seconds')
    parser.add_argument('-p', "--positions", nargs=2, type=float, default=(-40, 40), metavar=('Lower_Position', 'Upper_Position'),
                        help='Positions to move the FTS mirror between.')
    parser.add_argument('-r', "--res", type= float, default = 1,
                         help = 'Resolution of mirror spacing in mm. Number of points if -N flag used.')
    parser.add_argument('-N', action='store_true', help='Flag indicating that resolution is number of points instead of mirror spacing.')
    parser.add_argument('-s', '--sweeps', type=int, default = 1, help='Number of sweeps/passes of FTS mirror. ')
    parser.add_argument('-w', "--wait", type= float, default = 1,
                         help = 'Time to wait after moving FTS mirror before taking timestream.')
    parser.add_argument('--tune', action = 'store_true', help = 'Flag indicating detectors should be tuned.')
    parser.add_argument('-C', "--cfg", type = str, default='./fts_system_config.yaml', help='Path of a custom config file.')
    return parser.parse_args()


# def tune_detectors(R, cfg, cfg_io, args):
#     # Find detectors and set tones
#     # ----------------------------
#     R.find_detectors(new_sweep = True, N_steps = 400, peak_prom_db = 0.07, peak_dis = 8800, peak_width_min = 25, peak_width_max = 400)
#     det_freqs = R.find_detectors_fine(N_steps = 300, bandwidth = 4)
    
#     # Makes sure that the correct number of resonators are found
#     assert len(det_freqs) == cfg['rfsoc_tones']['num_tones'], 'Number of found detectors do not match nominal number of detectors'
    
#     ## Removed shifts for now. 
#     # Write a tone at all det_freqs
#     # --------------------------------------
#     tone_freqs = np.array(det_freqs)
#     tone_powers = np.array(cfg['rfsoc_tones']['tone_powers'])
#     tone_phis = np.array(cfg['rfsoc_tones']['tone_phis'])
    
#     data_dir = Path(cfg_io['file_paths']['base_data_dir'])

#     # Use tone powers and phis found from VNA sweep custom parameters not provided 
#     if len(tone_powers) == 0 or len(tone_powers) != len(det_freqs):
#         tone_powers = np.load(data_dir / "tmp" / "a_tones_comb_cust.npy")
#     if len(tone_phis) == 0 or len(tone_phis) != len(det_freqs):
#         tone_phis = np.load(data_dir / "tmp" / "p_tones_comb_cust.npy")

#     # Edit tones in RFSoC main config file
#     R.edit_main_config('tone_freqs', tone_freqs.tolist())
#     R.edit_main_config('tone_powers', tone_powers.tolist())
#     R.edit_main_config('tone_phis', tone_phis.tolist())


# def get_dets(sweep, rfsoc_dir, dets=6):
#     for i in range(dets):
#         sweep_id = int(get_sweep_id(sweep))
#         cfg_file = [file for file in rfsoc_dir.glob(f'*{sweep_id}*.yaml')][0]
#         yield Sweep(sweep, i, cfg_file)      

# def get_sweep_id(sweep_path):
#     sweep = Path(sweep_path).name
#     return sweep[5:15]


if __name__ == '__main__':
    main()
