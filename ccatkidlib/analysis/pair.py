#=================================#
# rfsoc_io.py               2025 #
# Darshan Patel dp649@cornell.edu #
#=================================#

'''
Library of helper functions for getting output data files of rfsoc_daq.py and pairing with corresponding configuration files.
'''

from pathlib import Path
import numpy as np
import sys

# Local Imports
import ccatkidlib.rfsoc_io as rfsoc_io

def get_timestamp(path) -> int:
    '''
    Extract the timestamp from a file name.

    Parameters:
        path (str | Path): Path of the file
    Returns:
        timestamp (int): Timestamp of file. -1 if no valid timestamp is found
    '''

    try: # Check that the passed path is a string or Path object
        path = Path(path) # Cast to Path object
        file = path.stem  # Get the file stem (without extension) 

        parts = file.split('_') # Split file into parts
        for i in [-1, -2, 0]: # Timestamp should only be at the 0, -1, or -2 index of the file part list
            try:
                tstamp = int(parts[i]) # Try casting part of file to int
                if tstamp > 1.7e9: return tstamp # Check that integer is a valid timestamp
            except:
                pass
        return -1 # Return -1 if no valid timestamp was found
    except Exception as e: # If exception is thrown, return -1 to represent invalid path
        return -1

def get_data_file(com_to, timestamp, data_dir = '**', date = '**', sess_id = '**', data_type = '**', root_data_dir = '/'):
    root_data_dir = Path(root_data_dir)

    bid, drid = com_to.split('.')
    com_str = f'B{bid}D{drid}'

    file_trees = [Path(data_dir) / date / sess_id / data_type / com_str, Path(data_dir) / data_type / date / sess_id / com_str]

    for tree in file_trees:
        tree = str(tree / f'*{timestamp}*')
        data_files = sorted(root_data_dir.glob(tree))
        if len(data_files) > 0: return data_files
    return []

def get_config(path, all_cfg = False) -> list:
    '''
    Get the config file associated with the specified VNA sweep, target sweep, or timestream data file.
    Parameters:
        path (str | Path): Path of a VNA sweep, target sweep, or timestream data file
    Returns:
        cfg_path (list): List of config file paths (io_cfg, drone_cfg(s), and ext_cfg) associated with the specified data file
    '''

    timestamp = get_timestamp(path) # Get timestamp of data file

    if not timestamp == -1: # Make sure a valid data file is passed
        data_names = ['vna', 'targ', 'timestream'] # Directory names of main three data file types
        ind = -1 

        # Check 'config' directory for matching config file. If not found, check 'rfsoc' directory in case a legacy data set is used
        # --------------------------------------------------------------------------------------------------------------------------
        for name in ['config', 'rfsoc']:
            # Get parent directory and split path into parts. Path casting is safe since already validated in get_timestamp function
            parts = list(Path(path).parts)[:-1] 
            
            # Replace instance of 'vna', 'targ', or 'timestream' in data file path with 'config' or 'rfsoc' to get file path of config files
            # ------------------------------------------------------------------------------------------------------------------------------
            if not ind == -1:
                parts[ind] = name # If not the first iteration through loop, part of path that needs to be replaced is already known
            else:
                for i, part in enumerate(parts): # If first iteration through loop, find part in path that needs to be replaced
                    if any(part == data_name for data_name in data_names): # Check if part name matches any of the data_names
                        parts[i] = name
                        ind = i
                        break # Should only be one part that needs to be replaced so break out of loop
            
            # Check config directory for matching config files
            # ------------------------------------------------
            cfg_path = Path(*parts)
            io_cfg = rfsoc_io.get_most_recent_file(cfg_path.parent, '*io*.yaml', time_past = np.inf) # Get io config file
            if io_cfg.exists(): # If io config exists, then config directory is correct
                if all_cfg: cfg_path = cfg_path.parent # If all_cfg, change to parent directory so that configs for all drones are found
                cfg_files = sorted(cfg_path.rglob(f'*{timestamp}*.yaml')) 
                cfgs = [io_cfg] + cfg_files
                if not all_cfg: # Find ext config if all_cfg is False (since it is not in the drone config directory)
                    ext_cfg = rfsoc_io.get_most_recent_file(cfg_path.parent, f'*{timestamp}*.yaml', time_past = np.inf) 
                    cfgs += [ext_cfg]
                return cfgs # Return found config files
        return [] # Return an empty list if none of the searched config directories contain config files matching data file
    else: # Return an empty list if invalid data file is passed
        return []