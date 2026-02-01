#=================================#
# io.py               2025 #
# Darshan Patel dp649@cornell.edu #
#=================================#

'''
Library of helper functions for general file and directory read/write operations as well as logging.
'''

# Import Python modules
import os
import sys
import ast
import time
import yaml
import subprocess
import numpy as np

from pathlib import Path
from tqdm import tqdm
from fabric import Connection, Config
from jinja2 import Environment, FileSystemLoader

# Local imports
from ccatkidlib.utils import function_timer
import ccatkidlib.utils as utils
import ccatkidlib.log as log

#========================#
# Directory IO Functions #
#========================#

def create_tree(com_to: list[str], curr_date: str, sess_id: int, data_dir: str) -> tuple[list[str], list[str], list[str], list[str]]:
    '''
    Create file tree for storage of timestream, sweep, and other (e.g., config) data.

    Args:
        com_to (list[str]) : List of drones to create directories for
        curr_date    (str) : Current date
        sess_id      (int) : ID of observing session
        data_dir     (str) : Where to create directory tree
    Returns:
        tuple[list[str], list[str], list[str], list[str]]: Config directories, target directories, timestream directories, and VNA sweep directories in that order.
    '''
    data_dir = Path(data_dir)

    config_dirs = []
    targ_dirs = []
    timestream_dirs = []
    vna_dirs = []

    for com in com_to:
        # Split into board id, drone id
        board, drone = com.split('.')
        com_str = f'B{board}D{drone}'

        # Define data directories
        timestream_dir = data_dir   / curr_date / sess_id / 'timestream' / com_str 
        vna_dir        = data_dir   / curr_date / sess_id / 'vna'        / com_str
        targ_dir       = data_dir   / curr_date / sess_id / 'targ'       / com_str 
        config_dir     = data_dir   / curr_date / sess_id / 'config'     / com_str
        comb_dir       = config_dir / 'combs'
        res_dir        = config_dir / 'res'

        # Create timestream directory
        create_dir(timestream_dir)
        timestream_dirs.append(timestream_dir)

        # Create vna sweep directory
        create_dir(vna_dir)
        vna_dirs.append(vna_dir)

        # Create targ sweep directory
        create_dir(targ_dir)
        targ_dirs.append(targ_dir)

        # Create config directory
        create_dir(config_dir)
        config_dirs.append(config_dir)

        # Create comb directory in config directory
        create_dir(comb_dir)

        # Create res directory in config directory
        create_dir(res_dir)

    return config_dirs, targ_dirs, timestream_dirs, vna_dirs

def create_tmp(com_to: list[str], tmp_dir: str) -> list[str]:
    '''
    Create *ccatkidlib* tmp directory and files

    Args:
        com_to (list[str]): _description_
        tmp_dir (str): _description_
    '''
    tmp_dir = Path(tmp_dir)
    create_dir(tmp_dir) # Create tmp directory if it does not exist

    noise_files = ['invalid/path']*len(com_to)
    for i, com in enumerate(com_to):
        board, drone = com.split('.')
        com_str = f'{board}_{drone}'

        noise_file = tmp_dir / f'noise_tones_{com_str}.npy'
        if not (noise_file).exists(): np.save(noise_file, [])
        noise_files[i] = str(noise_file)
    return noise_files

def create_dir(dir_path: str) -> None:
    '''
    Create directory at the specified path.

    Parameters:
        dir_path (str) : Path of the directory that is to be created
    '''

    # Attempt to make the directory
    try:
        dir_path = Path(dir_path)
        # Check if directory already exists, if not make directory
        if not dir_path.exists():
            dir_path.mkdir(parents = True, exist_ok = False)
            log.log('DEBUG', "The directory '%s' was successfully created!", dir_path)
        else:
            log.log('DEBUG', "The directory '%s' already exists! Directory was not overwritten.", dir_path)
    except FileNotFoundError:
        log.log('ERROR', "The directory '%s' could not be created! Ensure that the file path is valid!", dir_path)
        raise FileNotFoundError(f"The directory '{dir_path}' could not be created! Ensure that the file path is valid!")
    except PermissionError:
        log.log('ERROR', f"The directory '%s' could not be created! Ensure that the parent directory has suitable write permissions!", dir_path)
        raise PermissionError(f"The directory '{dir_path}' could not be created! Ensure that the parent directory has suitable write permissions!")

def add_dir(dir_name: str, 
            data_dir: str, 
            save_root: str | None = None,
            data_root: str | None = None,
            sub_dirs: list[str] = [],
            timestamp: str | None = None):
    '''
    Add a new directory to a pre-existing ccatkidlib data file tree
    
    '''
    if not dir_name[-1] == '/': dir_name += '/'
    if not dir_name[0] == '/': dir_name = '/' + dir_name

    for data_type in ('/targ/', '/timestream/', '/vna/'): 
        add_dir = data_dir.replace(data_type, dir_name, 1)
        if not add_dir == data_dir: 
            add_dir = Path(add_dir).parent 
            add_dir = add_dir / Path(*sub_dirs) if sub_dirs else add_dir / data_type[1:]
            if timestamp is not None: add_dir = add_dir / timestamp
            break
    
    if save_root is not None and data_root is not None:
        if not data_root[-1] == '/': data_root += '/'
        if not save_root[-1] == '/': save_root += '/' 
        add_dir = str(add_dir).strip().replace(data_root, save_root)

    create_dir(add_dir)
    return str(add_dir)

#=====================#
# Config IO Functions #
#=====================#

def load_config(config):
    '''
    Load config file.

    Parameters:
        config (str) : File path of config file to load
    Returns:
        cfg   (dict) : List of dictionaries loaded from config file
    '''
    cfg_path = Path(config)
    if not cfg_path.exists(): raise FileNotFoundError(f"Could not find config file: {config}!") # Raise error if config file does not exist

    env = Environment(loader = FileSystemLoader(f"{str(cfg_path.parent)}"))
    template = env.get_template(f"{cfg_path.name}")

    config = template.render()

    # Load config file
    cfg = [file for file in yaml.safe_load_all(config)]
    # Return loaded dictionary(ies)
    if len(cfg) == 1:
        return cfg[0] # If only one dictionary, do not return array
    else:
        return cfg

def save_config(cfg_path, cfg_dic, save = True):
    '''
    Save configuration file.

    Parameters:
        cfg_path (str) : File path where the config file should be saved
        cfg_dic (dict) : Dictionary to save as config file
        save    (bool) : Whether to save config file
    Returns:
        cfg_dic (dict) : Returns dictionary that was saved to config file
    '''
    if save:
        # Save config file
        with open(cfg_path, 'w') as config:
            yaml.safe_dump(cfg_dic, config, sort_keys=False, default_flow_style=None)
        
        log.log('DEBUG', f"Saved configuration file: '{cfg_path}'!")
        #log.log('DEBUG', f'Configuration file contents: {cfg_dic}')

        # Load new config file
        with open(cfg_path, 'r') as config:
            return yaml.safe_load(config)
    else:
        return cfg_dic

def edit_config(cfg, key, value, append = False):
    '''
    Update key in specified configuration file with the specified value.

    Parameters:
        cfg    (dict) : Configuration file to update
        key     (str) : Key that should be updated
        value   (Any) : Value with which to update key
        append (bool) : Whether to append a new key, value pair to config file if key is not found
    Returns:
        done   (bool) : True if key was successfully created or updated.
    '''
    # Edit config file dictionary
    # ---------------------------
    done = utils.dict_set(cfg, key, value)

    # Check if key was successfully updated
    # -------------------------------------
    if done: # If matching key was updated
        log.log('DEBUG', f'Updated key "{key}" with value "{value}" in config file"!')
    elif append: # If key was not found and append=True, add key value pair to dictionary
        cfg[key] = value
        done = True
        log.log('DEBUG', f'Added key "{key}" with value "{value}" to config file!')
    else: # If key was not found and append=False
        log.log('DEBUG', f'Failed to update key "{key}" with value "{value}" in config file!')
    return done

#===================#
# File IO Functions #
#===================#

#@function_timer
def get_most_recent_file(path, file_identifier, time_past = 60*60, time_ref = None, ccatkidlib_file: bool = False):
    '''
    Fetch the most recent file in a directory with the desired file identifier.

    Parameters:
        path            (Path) : Directory in which the file is located 
        file_identifier (str) : Substring included in the file name
        time_past     (float) : How far in the past to look for files (in seconds)
    Returns:
        file            (str) : File path of most recent file (returns "invalid/path" if no valid files found)
    '''
    if time_ref is None: time_ref = time.time()

    try:
        path = Path(path)
        # Attempt to get most recent file in directory using glob
        if isinstance(file_identifier, str): file_identifier = [file_identifier]

        files = []
        for file_id in file_identifier:
            files.extend(path.glob(file_id))
        
        get_time = get_timestamp if ccatkidlib_file else get_creation_time
        files = sorted(files, key = get_time, reverse = True)

        # Find file with time closest to time_ref using a binary search
        min_ind, max_ind = 0, len(files) - 1
        curr_ind, shifted_time, num_its = None, None, 0 
        while max_ind >= min_ind:
            num_its += 1
            curr_ind = min_ind + (max_ind - min_ind)//2

            shifted_time = time_ref - get_time(files[curr_ind])
            if shifted_time < 0: # If time difference is negative, then file is too new
                min_ind = curr_ind + 1
            elif shifted_time > 0 and not (max_ind == min_ind): # If time difference is positive, then file is a candidate but there could be more recent files
                max_ind = curr_ind
            else: # If no time difference between file time and reference time, there cannot be a more recent file so break out of loop
                break 
        # Check that most recent file satisfies time constraints
        if 0 <= shifted_time < time_past:
            file = files[curr_ind]
            log.log('DEBUG', f"Found most recent file '{file}' in {path}.")
            return file
        else:
            raise Exception("No files found within specified time range!")
    except Exception as e:
        log.log('DEBUG', f"Failed to fetch most recent file in '{path}' with identifier '{file_identifier}' with Exception:\n{e}")
        return Path("invalid/path")

def get_timestamp(path: str) -> int:
    '''Extract the timestamp from a ccatkidlib file name.

    Args:
        path (str | pathlib.PosixPath): Path of the file
    Returns:
        int: Timestamp of the file. -1 if no valid timestamp is found
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
        raise ValueError(f'The file {file} has no valid timestamp.') # Raise ValueError if timestamp could not be determined
    except Exception as e: # If exception is thrown, return -1 to represent invalid path
        log.log('ERROR', 'Failed to determine timestamp of file %s with Exception %s', path, e)
        return -1

def get_creation_time(file_path):
    '''
    Get the creation time of a file. Helper method for get_most_recent_file()

    Parameters:
        file_path (str) : Path of the file of which to get creation time
    Returns:
        creation_time (float): Creation time of file (returns -1 if creation time could not be determined)
    '''
    try:
        file = Path(file_path)
        # Get creation time of file
        creation_time = file.stat().st_ctime
        log.log('DEBUG', f"Creation time of file '{file_path}' is {creation_time}.")
        return creation_time
    except:
        log.log('DEBUG', f"Error getting creation time of file: '{file_path}'")
        return -1

def get_array(src_path, dest_path, action = 'cp', load = True, timestamp = False):
    # Convert path objects to str
    src_path = str(src_path)
    dest_path = str(dest_path)
    if Path(src_path).exists(): # Ensure that path to numpy array exists on RFSoC board
        # Copy numpy file from board and load locally
        # -------------------------------------------
        # Get name of file to be copied
        file = Path(src_path).stem
        ext = Path(src_path).suffix
        if timestamp: file = '_'.join(file.split('_')[:-1]) # Trim timestamp off file if desired

        if Path(dest_path).suffix == '': dest_path += f'/{file}{ext}' # If local path is a directory, use same file name as on RFSoC board

        # Define command for copying/moving array for src to dest dir.
        cmd = [action, src_path, dest_path]
        
        # Try to copy/move the array 
        # --------------------------
        try:
            subprocess.run(cmd, check=True)

            # Load array if desired
            loaded_array = np.load(dest_path).tolist() if load else str(dest_path)
        except Exception as e:
            # Send error message if failed to copy or load array
            log.log('ERROR', f'Failed to copy/load array with error {e}!')
            loaded_array = None
        
        log.log('DEBUG', f"Copied array from '{src_path}' to '{dest_path}'.")
    else:
        # Send error message if specified path does not exist on the RFSoC board
        log.log('ERROR', f'Failed to locate array at path {src_path}!')
        loaded_array = None
    return loaded_array

def increment_file(dir_path, file_prefix, file_suffix, overwrite=False):
    if overwrite:
        return Path(dir_path) / f'{file_prefix[:-1]}{file_suffix}', None
    else:
        file_count = 0
        full_path = Path(dir_path) / f'{file_prefix}{file_count}{file_suffix}' 
        while full_path.exists(): 
            file_count += 1
            full_path = Path(dir_path) / f'{file_prefix}{file_count}{file_suffix}'
        return full_path, file_count

def combine_npy(files, num, com = None, fname_out = None):
    files = list(files)
    num_zipped = len(files)/num 
    num_zipped = int(num_zipped) if num_zipped == int(num_zipped) else int(num_zipped) + 1
    
    fname = str(Path(files[0]).parent / '_'.join(str(Path(files[0]).stem).split('_')[:-1]))
    fnames = []
    for i in range(num_zipped):
        name = f'{fname}_{i:03d}.npz'
        fnames.append(name)
        loaded_files = {str(Path(file).stem):np.load(file, mmap_mode='r') for file in files[i*num:(i+1)*num]}
        np.savez(name, **loaded_files)
    for file in files: os.remove(str(file))

    if fname_out is not None: fname_out[com] = fnames
    
    return fnames

#=====================#
# Remote IO Functions #
#=====================#

def get_connection(ip, ssh_key):
    '''
    Create Fabric Connection to RFSoC board with specified IP address. 

    Parameters:
        ip      (str) : IP address to connect to
        ssh_key (str) : File path of private ssh key
    Returns:
        connection (Connection) : Fabric Connection object to specified IP address
    '''

    # Get Fabric Connection to RFSoC board
    connect = Connection(f'xilinx@{ip}', connect_kwargs = {'key_filename': ssh_key})
    log.log('DEBUG', f'Created Fabric Connection to {ip}.')
    return connect

#@function_timer
def get_array_board(c, ip, ssh_key, remote_path, local_path, load = True, timestamp = False):
    '''
    Load numpy array from RFSoC board.

    Parameters:
        c    (Connection) : Fabric Connection object of RFSoC board
        ip          (str) : IP address of RFSoC board
        ssh_key     (str) : Path to private ssh key
        remote_path (str) : Path of numpy array on RFSoC board
        local_path  (str) : Local file path where numpy array should be copied
        load       (bool) : Whether to load and return the contents of the numpy array
        timestamp  (bool) : Whether to remove the timestamp from numpy array file name (False if no timestamp)
    Returns:
        array (ndarray | str) : Loaded numpy array or file path of array if not loaded
    '''
    # Convert path objects to str
    remote_path = str(remote_path)
    local_path = str(local_path)
    if path_exists(c, remote_path): # Ensure that path to numpy array exists on RFSoC board
        # Copy numpy file from board and load locally
        # -------------------------------------------
        # Get name of file to be copied
        file = Path(remote_path).name.split('.')[0]
        if timestamp: file = '_'.join(file.split('_')[:-1]) # Trim timestamp off file if desired

        # Define scp command for copying numpy array to local directory.
        cmd = ['scp', '-q', '-i', f'{ssh_key}', f'xilinx@{ip}:{remote_path}', f'{local_path}']

        if Path(local_path).suffix == '': 
            local_path += f'/{file}.npy' # If local path is a directory, use same file name as on RFSoC board
            cmd = ['rsync', '-a', '--no-times', '--inplace', '-e', f"ssh -i {ssh_key}", f'xilinx@{ip}:{remote_path}', f'{local_path}']
        
        # Try to copy the array from remote
        # --------------------------------
        try:
            subprocess.run(cmd, check=True)

            # Load array if desired
            loaded_array = np.load(local_path).tolist() if load else str(local_path)
        except Exception as e:
            # Send error message if failed to copy or load array
            log.log('ERROR', f'Failed to copy/load array with error {e}!')
            loaded_array = None
        
        log.log('DEBUG', f"Copied array from '{remote_path}' to '{local_path}'.")
    else:
        # Send error message if specified path does not exist on the RFSoC board
        log.log('ERROR', f'Failed to locate array at path {remote_path}!')
        loaded_array = None
    return loaded_array

#@function_timer
def save_array_board(ip, ssh_key, path, saved_array, tmp_dir):
    '''
    Save numpy array to RFSoC board.

    Parameters:
        c        (Connection) : Fabric Connection object of RFSoC board
        path            (str) : File path where numpy array should be saved on RFSoC board
        saved_array (ndarray) : Numpy array to save to RFSoC board
    Returns:
        result          (str) : Standard out of save command
    ''' 

    save_path = Path(tmp_dir) / Path(path).name
    np.save(save_path, saved_array)

    # Define command for saving numpy array to RFSoC board
    cmd = ['rsync', '-a', '--no-times', '--inplace', '-e', f"ssh -i {ssh_key}", f'{save_path}', f'xilinx@{ip}:{path}']
    try:
        result = subprocess.run(cmd, check=True)
    except Exception as e:
        # Send error message if failed to copy or load array
        log.log('ERROR', f'Failed to copy/load array with error {e}!')
        result = None

    log.log('DEBUG', f"Saved array to '{path}'.")
    return result

#@function_timer
def get_most_recent_file_board(c: Connection, dir: str, file_identifier: str = "*", time_past: int = 60*60):
    '''
    Get most recent file in specified directory on RFSoC board.

    Args:
        c (Connection): Fabric Connection object of RFSoC board. Can be created using ``get_connection`` function
        dir (str): Directory in which the file is located
        file_identifier (str, optional): Substring included in the file name. Defaults to wildcard: *"\*"*
        time_past (int, optional): How old the file can be in seconds. Files older than ``time_past`` will be ignored. Defaults to 3600 seconds
    Returns:
        return (str): File path of most recent file. Returns *"invalid/path"* if no valid file found
    '''

    # Define command string for finding most recent file, use find to get most recent file
    if not time_past == np.inf:
        time_past = time_past/60
        time_past = int(time_past) if time_past == int(time_past) else int(time_past) + 1
    cmd = f"find {dir}/* -type f {f'-cmin -{time_past} ' if not time_past == np.inf else ''}-name '*{file_identifier}*' | xargs --no-run-if-empty ls -rt | tail -1"

    # Try to find the most recent file
    # --------------------------------
    try:
        file = c.run(cmd, hide = 'out').stdout
        file = file.rstrip('\r\n') # Remove trailing characters from str
        log.log('DEBUG', f"Found most recent file '{file}' in {dir}.")
        return file
    except:
        # Send warning message if failed to fetch most recent file
        log.log('WARNING', f"Failed to fetch most recent file in '{dir}' with identifier '{file_identifier}'")
        return "invalid/path"

def path_exists(c: Connection, path: str) -> bool:
    '''
    Check if specified path exists on RFSoC board.

    Args:
        c (Connection): Fabric Connection object of RFSoC board. Can be created using ``get_connection`` function
        path (str | pathlib.PosixPath): File path to check existence of
    Returns:
        return (bool): Whether the file path exists
    '''

    path = str(path) # Convert path objects to str

    cmd = f"[ -f {path} ] && echo True || echo False" # Define command str to check if path exists
    return ast.literal_eval(c.run(cmd, hide = 'out').stdout) # Run command on RFSoC board and get command stdout
