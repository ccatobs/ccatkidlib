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
    args = eval_args()
    cfg = rfsoc_io.load_config(args.cfg)  
    fig_dir = Path(cfg['file_paths']['base_data_dir']) / 'fig'
    
def eval_args():
    # Initialize arg parser
    parser = argparse.ArgumentParser(prog='measure_pol',
                                     description='''Quick Data Analyzer for Polarization Measurements of MKIDS''')
    parser.add_argument('d', type = str, help='Date of data taken in YYYYMMDD')
    parser.add_argument('bd', "--board_drone", nargs = 2, type = int,
                         default = (1,1), metavar=('Board', 'Drone'), 
                         help='the board and drone to analyze data from')
    parser.add_argument('ts', "--timestamp", type = int, 
                        help='timestamp of the data taken')
    
    return parser.parse_args()
    
if __name__ == '__main__':
    main()    

