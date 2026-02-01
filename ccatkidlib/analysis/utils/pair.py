#=================================#
# pair.py                    2025 #
# Darshan Patel dp649@cornell.edu #
#=================================#

'''
Library of helper functions for getting ccatkidlib data files and pairing with corresponding configuration files.
'''

import numpy as np
import sys
import pathlib

from pathlib import Path
from tqdm import tqdm

# Local Imports
import ccatkidlib.io as io
import ccatkidlib.log as log

def get_sess_dir(sess_id, data_dir: str = '**', date: str = '**', root_data_dir: str = '/') -> str:
    root_data_dir = Path(root_data_dir)
    
    file_tree = Path(data_dir) / date / sess_id
    try:
        # The session ID directory is unique so we can stop searching after it is found
        # Will raise a StopIteration Error if directory is not found
        sess_dir = str(next(root_data_dir.glob(str(file_tree))))
    except StopIteration:
        sess_dir = "invalid/path" 
    return sess_dir

def get_data_file(com_to: str, timestamp: str | int, data_type: str, data_dir: str = '**', date: str = '**', sess_id: str = '**', root_data_dir: str = '/') -> list[str]:
    '''Get a ccatkidlib data file based on provided path information.

    Args:
        com_to (str): Drone that took the data. In form 'Board.Drone'
        timestamp (str | int): Timestamp of data file
        data_type (str): Type of data file. Should be one of 'vna', 'targ', 'timestream'. 
        data_dir (str, optional): Directory where data is stored. Defaults to wildcard '**'
        date (str, optional): Date data was taken. Defaults to wildcard '**'
        sess_id (str, optional): ccatkidlib session ID of data. Defaults to wildcard '**'
        root_data_dir (str, optional): Root directory where data is stored. Defaults to '/'
    Returns:
        str: Path of found data file. Returns 'invalid/path' if data file not found.
    '''
    
    root_data_dir = Path(root_data_dir)
    bid, drid = com_to.split('.')
    com_str = f'B{bid}D{drid}'

    file_trees = [Path(data_dir) / date / sess_id / data_type / com_str,
                  Path(data_dir) / data_type / date / sess_id / com_str]

    for tree in file_trees:
        tree = str(tree / f'*{timestamp}*')
        data_files = sorted(root_data_dir.glob(tree))
        if len(data_files) > 0: return data_files # Return data files, return as list because multiple timestream files could be found
    return ["invalid/path"]

def get_config(path: str | pathlib.PosixPath, all_cfg: bool = False) -> list[str]:
    ''' Get the config files associated with the specified data file.
    
    Args:
        path (str | pathlib.PosixPath): Path of data file
        all_cfg (bool, optional): Whether to return config files for all drones. Defaults to False. 
    Returns:
        list[str]: List of config file paths (io_cfg, drone_cfg(s), and ext_cfg) associated with the specified data file
    '''

    timestamp = io.get_timestamp(path) # Get timestamp of data file

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

            # TODO: Should probably just use str.replace() here
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
            io_cfg = io.get_most_recent_file(cfg_path.parent, '*io*.yaml', time_past = np.inf) # Get io config file
            if io_cfg.exists(): # If io config exists, then config directory is correct
                if all_cfg: cfg_path = cfg_path.parent # If all_cfg, change to parent directory so that configs for all drones are found
                cfg_files = sorted(cfg_path.rglob(f'*{timestamp}*.yaml')) 
                cfgs = [io_cfg] + cfg_files
                if not all_cfg: # Find ext config if all_cfg is False (since it is not in the drone config directory)
                    ext_cfg = io.get_most_recent_file(cfg_path.parent, f'*{timestamp}*.yaml', time_past = np.inf) 
                    cfgs += [ext_cfg]
                return list(map(str, cfgs)) # Return found config files
        return [] # Return an empty list if none of the searched config directories contain config files matching data file
    else: # Return an empty list if invalid data file is passed
        return []

def get_sweep(path: str | pathlib.PosixPath, **kwargs):
    timestamp = io.get_timestamp(path) # Get timestamp of data file

    if not timestamp == -1: # Make sure a valid data file is passed
        # Get parent directory and split path into parts. Path casting is safe since already validated in get_timestamp function
        parts = list(Path(path).parts)[:-1] 
        
        # Replace instance of 'vna', 'targ', or 'timestream' in data file path with 'config' or 'rfsoc' to get file path of config files
        # ------------------------------------------------------------------------------------------------------------------------------
        for i, part in enumerate(parts): # If first iteration through loop, find part in path that needs to be replaced
            if part == 'timestream':
                targ_parts = parts.copy()
                vna_parts = parts.copy()

                targ_parts[i] = 'targ'
                vna_parts[i] = 'vna'

                parts_list = [targ_parts, vna_parts]
                break
            elif part == 'targ':
                parts[i] = 'vna'
                parts_list = [parts]
                break
        
        # Check config directory for matching config files
        # ------------------------------------------------

        sweeps = [None]*len(parts_list)
        for i, parts in enumerate(parts_list):
            sweep_path = Path(*parts)
            sweeps[i] = io.get_most_recent_file(sweep_path, '*.npy', time_past = np.inf, time_ref = timestamp, ccatkidlib_file = True) 

        recent_sweeps = sorted(sweeps, key = io.get_timestamp, reverse = True)
        if 'targ' in recent_sweeps[0].parts:
            for sweep in recent_sweeps:
                if 'vna' in sweep.parts:
                    return str(sweep), str(recent_sweeps[0])
            else:
                return 'invalid/path', recent_sweeps[0]
        elif 'vna' in recent_sweeps[0].parts:
            return  str(recent_sweeps[0]), 'invalid/path'
        else:
            return 'invalid/path', 'invalid/path'

def replace_root(path: str | pathlib.PosixPath, old_root: str, new_root: str):
    '''Replace the root directory of a file path with a new root

    Args:
        path (str | pathlib.PosixPath): Original file path
        old_root (str): Old root directory of file path to be replaced
        new_root (str): New root directory to replace the old root
    Returns:
        return (str): New file path with the root directory replaced. If the new file path does not exist, returns the original path.
    '''
    path = str(path).strip()
    new_path = path.replace(old_root, new_root)
    if Path(new_path).exists(): 
        return new_path
    else:
        log.log('ERROR', 'Could not find file %s with original root directory %s replaced with %s', new_path, old_root, new_root)
        return path
