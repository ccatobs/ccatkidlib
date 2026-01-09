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
    Main method run when 'measure_beam.py' is called directly
    '''

    # Parse command line arguments
    args = eval_args()
   
    # Initialize PCS Agents
    # ---------------------
    # Initialize RFSoC control agent
    RC = R(args.cfg)
    output = RC.io_cfg['io']['terminal_output']

    # Initialize polarized beam mapper agent
    P = OCSClient(RC.io_cfg['pcs_agents']['beam_map_agent'], args=[])
    rfsoc_io.send_msg('INFO', 'Initialized polarized beam mapper PCS agent!', output)

    # Wait for IR source to be plugged in
    # -----------------------------------
    rfsoc_io.send_msg('INFO', 'Waiting for light source to be plugged in...')
    plugged_in = False

    while not plugged_in:
        response = input('Is the IR source plugged in? (y/n): ')
        if response.lower() == 'yes' or response.lower() == 'y':
            plugged_in = True

    # Tune detectors before taking data
    # ---------------------------------
    if args.tune: 
        rfsoc_io.send_msg('INFO', 'Tuning Detectors', output)
        tune_detectors(RC, args)
        rfsoc_io.send_msg('INFO', 'Finished tuning detectors', output)

    # Start beam mapping
    # ------------------
    rfsoc_io.send_msg('INFO', 'Starting Beam Map', output)
    acquire_data(P, RC, output, args)
   
def acquire_data(P, RC, output, args):
    '''
    Move beam mapper to specified coordinates and collect timestream data.

    Parameters:
        R: RFSoC data acquisition object
        t: Length of timestreams in seconds
    '''

    # Generate map 
    # -------------------------------------
    map, mesh = generate_map(output, args)
    rfsoc_io.send_msg('INFO', f'Generated Beam Map:\n{map}', output)

    RC.edit_config(RC.io_cfg, "map", map.tolist())
    if mesh is not None:
        RC.edit_config(RC.io_cfg, "mesh_X", mesh[0].tolist())
        RC.edit_config(RC.io_cfg, "mesh_Y", mesh[1].tolist())
    rfsoc_io.save_config(RC.log_dir / f"beammap_config_io_{RC.timestamp}.yaml", RC.io_cfg, RC.save_cfg)
    coords = map.reshape(-1, 2)

    # Iterate over positions and take timestreams
    # -------------------------------------------
    # P.changeStageSpeed(speed=4000) # Set beam mapper speed
    with tqdm(range(len(coords)), desc = f'Beam Mapping:') as pbar:
        for i in pbar:
            x, y = coords[i]

            # Move xy stage to next position
            P.moveStageTo(x=x,y=y)
            if i == 0:
                time.sleep(40)

            rfsoc_io.send_msg('INFO', f'Finished moving beam mapper to {coords[i]} mm!', output)
            
            # Wait before taking timestream
            time.sleep(args.wait)

            # Update/add position to config file
            RC.edit_config(RC.ext_cfg, 'coords', coords[i].tolist())
            pbar.set_postfix_str(f'Current Coordinate: {coords[i]} mm')

            if args.t > 0: RC.take_timestream(args.t, write_comb = False, setup = False)
    RC.take_timestream(1, write_comb = False, turn_off = True, save_data = False, setup = False, reset = False)

def generate_map(output, args):
    # Generates a list of coordinates for map making
    # ----------------------------------------------
    center = args.center  # Center position of map
    resolution = args.res # Resolution of map
    size = args.size      # Size of map in mm
    xystage_max = 110     # Maximum amount xy stage can move (in mm)

    if args.maptype == 'rect':
        center_x, center_y = center
        size_x, size_y = size

        # Ensure that an odd number of points is used to generate the map grid
        if size_x/resolution % 2 == 0: 
            size_x +=resolution 
            rfsoc_io.send_msg('INFO', 'Cannot generate a centered map with an even number of points in the X direction. Using an odd number instead!', output)
        if size_y/resolution % 2 == 0:
            size_y +=resolution
            rfsoc_io.send_msg('INFO', 'Cannot generate a centered map with an even number of points in the Y direction. Using an odd number instead!', output) 

        # Calculate corner coordinates
        min_x = center_x - (size_x-resolution)/2
        max_x = center_x + (size_x-resolution)/2

        min_y = center_y - (size_y-resolution)/2
        max_y = center_y + (size_y-resolution)/2

        # Ensure that the map is within the limits of the XY stage
        if min_x < -xystage_max or max_x > xystage_max or min_y < -xystage_max or max_y > xystage_max:
            rfsoc_io.send_msg('CRITICAL', f'Map exceeds the {xystage_max} mm limit of the XY stage. Please choose a smaller size or move the center.', output)
            sys.exit(1)

        # Create a map starting in the upper left corner
        x_arr = np.arange(min_x, max_x + resolution, resolution)
        y_arr = np.arange(max_y, min_y - resolution, -resolution)

        X, Y = np.meshgrid(x_arr, y_arr)
        mesh = (X, Y)

        map = list(zip(X.ravel(), Y.ravel()))        # Create pairs of points using meshgrid
        map  = np.reshape(map, [X.shape[0], -1, 2])  # Reshape pairs to match meshgrid dimensions
        return map, mesh
    
    if args.beamtype == 'radial':
        pass

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
    parser = argparse.ArgumentParser(prog='Beam Map', description='''Perform beam mapping on microwave kinetic inductance detectors (MKID) using a motorized XY stage''')
    
    # Add arguments
    parser.add_argument('t', type = float, help='Length of timestreams in seconds')
    parser.add_argument('-T', "--maptype", default='rect', type = str, help='type of beam map')
    parser.add_argument('-c', "--center", nargs = 2, type = float,
                         default = (0, 0), metavar=('X', 'Y'), 
                         help='Center of Map')
    parser.add_argument('-s', "--size", nargs = 2, type = float,
                         default = (4, 4), metavar=('X/R', 'Y/Theta'), 
                         help='Size of beam map (X mm by Y mm) or (R mm by Theta degrees)')
    parser.add_argument('-r', "--res", type= float, default = 1,
                         help = 'Resolution of beam map in mm. If radial beam map then applied in radial direction')
    parser.add_argument('-w', "--wait", type= float, default = 1,
                         help = 'Time to wait after moving xy stage before taking timestream.')
    parser.add_argument('--tune', action = 'store_true', help = 'Flag indicating detectors should be tuned.')
    parser.add_argument('-C', "--cfg", type = str, default='./beam_map_modcam_system_config.yaml', help='Path of a custom config file.')
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
