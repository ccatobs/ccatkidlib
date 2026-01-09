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
    Main method run when 'measure_beam.py' is called directly
    '''
    
    # Import RFSoC control module
    from rfsoc_daq import R

    # Parse command line arguments
    args = eval_args()
   
    # Initialize PCS Agents
    # ---------------------
    # Initialize RFSoC control agent
    R = R(args.cfg)
    output = R.io_cfg['io']['terminal_output']

    # Initialize polarized beam mapper agent
    P = OCSClient(R.io_cfg['pcs_agents']['beam_map_agent'], args=[])
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
        tune_detectors(R, args)
        rfsoc_io.send_msg('INFO', 'Finished tuning detectors', output)

    # Start beam mapping
    # ------------------
    rfsoc_io.send_msg('INFO', 'Starting Polarization Data Collection', output)
    acquire_data(P, R, output, args)

def acquire_data(P, R, output, args):
    '''
    Move beam mapper to specified coordinates and collect timestream data.

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
    
    R.edit_config(R.io_cfg, "angles", angles.tolist())
    rfsoc_io.save_config(R.log_dir / f"pol_config_io_{R.timestamp}.yaml", R.io_cfg, R.save_cfg)
    
    # Take initial timestream to initialize drones
    # --------------------------------------------
    R.parallel = True
    R.take_timestream(1, write_tones = False, turn_off = False, save_data = False)

    # Iterate over positions and take timestreams
    # -------------------------------------------
    # P.changeStageSpeed(speed=4000) # Set beam mapper speed
    with tqdm(range(len(angles) - 1), desc = f'Taking Polarization Data:') as pbar:
        for i in pbar:
            print(angles[i])
            R.edit_config(R.ext_cfg, "angle", float(angles[i]))
            pbar.set_postfix_str(f'Current Angle: {angles[i]}')
            
            # Take timestream data (also saves config file)
            for i in tqdm(range(args.trials), desc = "Trial"):
                if args.t > 0: R.take_timestream(args.t, write_tones = False, reset = False, turn_off = False, setup = False)

            # Rotate polarizing grid to next angle
            dtheta = angles[i+1] - angles[i]
            P.moveStepper(deg=dtheta)
            
            rfsoc_io.send_msg('INFO', f'Finished moving grid to {angles[i]} degrees!', output)

            # Wait before taking next timestream
            time.sleep(args.wait)

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

def tune_detectors(R, args):
    # Find detectors and set tones
    # ----------------------------
    R.find_detectors(new_sweep = True)
    det_freqs = R.find_detectors_fine(new_sweep = True)
    R.take_target_sweep(write_tones = False)

def eval_args():
    # Initialize arg parser
    parser = argparse.ArgumentParser(prog='measure_pol',
                                     description='''Perform polarization measurements on 
                                     microwave kinetic inductance detectors (MKID) using a rotating
                                     polarizing grid.''')
    parser.add_argument('t', type = float, help='Length of timestreams in seconds')
    parser.add_argument('-a', "--angles", nargs = 3, type = int,
                         default = (0, 360, 1), metavar=('Start', 'End', 'Spacing/Num'), 
                         help='Angles for which to collect data.')
    parser.add_argument('-w', "--wait", type= float, default = 1,
                         help = 'Time to wait after moving grid before taking timestream.')
    parser.add_argument('-c', "--cfg", type = str, default='./pol_system_config.yaml', help='Path of a custom config file.')
    parser.add_argument('-t', "--trials", type = int, default=1, help='Number of timestreams to take per angle.')
    parser.add_argument('--tune', action = 'store_true', help = 'Flag indicating detectors should be tuned.')
    parser.add_argument("--num", action = 'store_true', 
                        help='Flag indicating that last value in -a argument is number of angles')
    return parser.parse_args()

if __name__ == '__main__':
    main()
