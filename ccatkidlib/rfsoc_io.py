#=================================#
# rfsoc_io.py               2025 #
# Darshan Patel dp649@cornell.edu #
#=================================#

'''
Library of helper functions for file and directory read/write operations as well as logging.
'''

# Import Python modules
import os
import sys
import ast
import time
import yaml
import logging
import subprocess
import numpy as np

from pathlib import Path
from tqdm import tqdm
import tqdm.contrib.logging as tqdm_logging
from functools import partial, partialmethod, wraps
from fabric import Connection, Config
from jinja2 import Environment, FileSystemLoader


# Local imports
from ccatkidlib.style import Style
from ccatkidlib.utils import function_timer
import ccatkidlib.utils as utils

#========================#
# Directory IO Functions #
#========================#

def create_book(curr_date, sess_id, com_to, data_dir):
    '''
    Create book for storage of timestream, sweep, and other (e.g., config) data.

    Parameters:
        curr_date      (str) : Current date
        sess_id        (int) : ID of current observing session
        com_to (list of str) : List of board and drone IDs of RFSoC in form board.drone
        data_dir       (str) : Path of directory in which to store data
    Returns:
        config_dirs     (list of str) : Directories where log and config files are saved
        targ_dirs      (list of str) : Directories where target sweeps are saved
        timestream_dirs (list of str): Directories where timestreams are saved
        vna_dirs       (list of str) : Directories where VNA sweeps are saved
    '''

    # Create tmp directory
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

def create_dir(dir_path):
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
            send_msg('DEBUG', f"The directory '{dir_path}' was successfully created!")
        else:
            send_msg('DEBUG', f"The directory '{dir_path}' already exists! Directory was not overwritten.")
    except FileNotFoundError:
        send_msg('ERROR', f"The directory '{dir_path}' could not be created! Ensure that the file path is valid!")
        raise FileNotFoundError(f"The directory '{dir_path}' could not be created! Ensure that the file path is valid!")
    except PermissionError:
        send_msg('ERROR', f"The directory '{dir_path}' could not be created! Ensure that the parent directory has suitable write permissions!")
        raise PermissionError(f"The directory '{dir_path}' could not be created! Ensure that the parent directory has suitable write permissions!")

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
        
        send_msg('DEBUG', f"Saved configuration file '{cfg_path}'!")
        send_msg('DEBUG', f'Configuration file contents: {cfg_dic}')

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
        send_msg('DEBUG', f'Updated key "{key}" with value "{value}" in config file"!')
    elif append: # If key was not found and append=True, add key value pair to dictionary
        cfg[key] = value
        done = True
        send_msg('DEBUG', f'Added key "{key}" with value "{value}" to config file!')
    else: # If key was not found and append=False
        send_msg('DEBUG', f'Failed to update key "{key}" with value "{value}" in config file!')
    return done

#===================#
# File IO Functions #
#===================#

#@function_timer
def get_most_recent_file(path, file_identifier, time_past = 60*60, time_ref = None):
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
        files = sorted(files, key = get_creation_time, reverse = True)

        # Check if creation time is within the specified time_past 
        for file in files:
            if 0 <= time_ref - get_creation_time(file) < time_past:
                send_msg('DEBUG', f"Found most recent file '{file}' in {path}.")
                return file
        else:
            raise Exception("No files found within specified time range!")
    except Exception as e:
        send_msg('WARNING', f"Failed to fetch most recent file in '{path}' with identifier '{file_identifier}'")
        return Path("invalid/path")

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
        send_msg('DEBUG', f"Creation time of file '{file_path}' is {creation_time}.")
        return creation_time
    except:
        send_msg('DEBUG', f"Error getting creation time of file: '{file_path}'")
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
            send_msg('ERROR', f'Failed to copy/load array with error {e}!')
            loaded_array = None
        
        send_msg('DEBUG', f"Copied array from '{src_path}' to '{dest_path}'.")
    else:
        # Send error message if specified path does not exist on the RFSoC board
        send_msg('ERROR', f'Failed to locate array at path {src_path}!')
        loaded_array = None
    return loaded_array

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

#======================#
# Logging IO Functions #
#======================#

def setup_logging(log_path, file_level, terminal_level, name = __name__):
    '''
    Setup logger and logger config.

    Args:
        log_path  (str) : File path of the logger including log name
        level     (str) : Level at which to log (messages below this level are ignored)
        name      (str) : Name of the logger
    '''

    def _addLevel(name, num):
        '''
        Adds a custom logging level to the logger.

        Parameters:
            num  (int): Logging level
            name (str): Name of logging level
        '''
        
        # Convert passed name to lowercase 
        method_name = name.lower()

        logging.addLevelName(num, name) # Add new logging level to logger
        setattr(logging, name, num)     # Add new attribute to the logging class corresponding to custom logging level

        # Add new methods to relevant loggging classes
        setattr(logging.getLoggerClass(), method_name, partialmethod(logging.getLoggerClass().log, num)) 
        setattr(logging, method_name, partial(logging.log, num))

    # Get logger
    logger = logging.getLogger(name)

    # Add custom logging levels
    # -------------------------
    custom_levels = [['HEADER', int((logging.INFO + logging.WARNING)/2)],
                     ['FOOTER', int((logging.INFO + logging.WARNING)/2)],
                     ['PCS', int(logging.DEBUG - 1)],
                     ['TIMER', int(logging.INFO)]]
    
    for lvl in custom_levels: _addLevel(*lvl)

    # Setup logger config
    # -------------------

    # Setup logging to file
    file_log = logging.FileHandler(log_path, mode='a')

    file_level = logging.getLevelName(file_level)
    file_log.setLevel(file_level)

    file_format = logging.Formatter(fmt='%(asctime)s | %(message)s', datefmt="%m/%d/%Y %I:%M:%S %p")
    file_log.setFormatter(file_format)

    # Setup logging to terminal
    terminal_log = logging.StreamHandler(sys.stdout)

    terminal_level = logging.getLevelName(terminal_level)
    terminal_log.setLevel(terminal_level)

    terminal_format = logging.Formatter(fmt='%(asctime)s | %(message)s', datefmt="%m/%d/%Y %I:%M:%S %p")
    terminal_log.setFormatter(terminal_format)
    
    # Set logger level and add handlers
    logger.setLevel(min(file_level, terminal_level)) # Set logger level to the minimum of file and terminal levels
    logger.addHandler(file_log)
    logger.addHandler(terminal_log)

    # Test logging/confirm successful logger setup
    send_msg('DEBUG', "Successfully initialized logger: %s", name, name = name)

def send_msg(level: str, msg: str, *args, name: str = __name__) -> None:
    '''
    Log message and print message to terminal. 

    Args:
        level   (str) : Level of message at which to log (One of: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        msg     (str) : Message to log
        name    (str) : Name of logger 


    '''
    # Get logger
    logger = logging.getLogger(name)

    # Fetch the level of the stream handler logging to terminal
    # ---------------------------------------------------------
    terminal_level = logging.getLevelName('CRITICAL') # Default to critical level if no stream handler
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler):
            terminal_level = handler.level

    # Try logging message
    # -------------------
    try:
        log_level = logging.getLevelName(level)
        style = Style()

        # Log message with given level. Redirect stdout logs to tqdm
        with tqdm_logging.logging_redirect_tqdm(loggers=[logger]):
            # TQDM log handler does not respect original streamHandler level (see https://github.com/tqdm/tqdm/issues/1272) so we need to override it
            for handler in logger.handlers:
                if isinstance(handler, tqdm_logging._TqdmLoggingHandler):
                    handler.setLevel(terminal_level)
            msg = f'{style.log_begin(level, getattr(style, level))} {msg}'
            logger.log(log_level, msg, *args)
    except Exception as e:
        # Log error message
        with tqdm_logging.logging_redirect_tqdm(loggers=[logger]):
            logger.log(logging.ERROR, 'Failed to log message %s with error %s!', msg, e)

def wait(t_sec, desc = ""):
    '''
    Wait for t_sec seconds with progress bar.

    Parameters:
        t_sec   (int) : Number of seconds to wait
    '''    

    start_time = time.time()
    time_diff = 0

    with tqdm(total=t_sec, colour='BLUE', desc = f"{Style().log_begin('WAIT', Style.WAIT)} {desc}") as pbar:
        while time_diff < t_sec:
            pbar.update(int(time_diff - pbar.n))
            time.sleep(0.1)
            time_diff = time.time() - start_time
        pbar.update(t_sec - pbar.n)

def header(func):
    '''
    Decorator for wrapping rfsoc_daq methods. Provides error handling and logs HEADER and FOOTER messages.

    Parameters:
        func (func): Function to decorate    
    '''
    @wraps(func) # Help calls on func will print help message of func instead of help message of header
    def _wrapper(self, *args, **kwargs):
        name = func.__name__ # Get method name
        fmt = Style().func_name(name) # Add style to function name

        # Try to execute func
        # -------------------
        try:
            send_msg('HEADER', f"Executing {fmt}...")
            rtn = func(self, *args, **kwargs)
            send_msg('FOOTER', f"{fmt} executed successfully!")
            return rtn
        except Exception as e:
            import traceback 
            # Print error traceback if func failed to execute and exit out of program
            send_msg('CRITICAL', f"TERMINATING PROGRAM -- {fmt} failed to execute with error:\n{traceback.format_exc()}", True)
            sys.exit()
    return _wrapper

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
    send_msg('DEBUG', f'Created Fabric Connection to {ip}.')
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
            send_msg('ERROR', f'Failed to copy/load array with error {e}!')
            loaded_array = None
        
        send_msg('DEBUG', f"Copied array from '{remote_path}' to '{local_path}'.")
    else:
        # Send error message if specified path does not exist on the RFSoC board
        send_msg('ERROR', f'Failed to locate array at path {remote_path}!')
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
        send_msg('ERROR', f'Failed to copy/load array with error {e}!')
        result = None

    send_msg('DEBUG', f"Saved array to '{path}'.")
    return result

#@function_timer
def get_most_recent_file_board(c, dir, file_identifier = "*", time_past = 60*60):
    '''
    Get most recent file in directory on RFSoC board.

    Parameters:
        c        (Connection) : Fabric Connection object of RFSoC board
        dir             (str) : Directory in which the file is located
        file_identifier (str) : Substring included in the file name 
    Returns:
        file            (str) : File path of most recent file (returns "invalid/path" if no valid files found)
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
        send_msg('DEBUG', f"Found most recent file '{file}' in {dir}.")
        return file
    except:
        # Send warning message if failed to fetch most recent file
        send_msg('WARNING', f"Failed to fetch most recent file in '{dir}' with identifier '{file_identifier}'")
        return "invalid/path"

def path_exists(c, path) -> bool:
    '''
    Check if path exists on RFSoC board.

    Parameters:
        c      (Connection) : Fabric Connection object of RFSoC board. Use get_connection to create object
        path          (str) : File path to check existence of
    Returns:
        exists       (bool) : Whether the file path exists
    '''

    path = str(path) # Convert path objects to str

    cmd = f"[ -f {path} ] && echo True || echo False" # Define command str to check if path exists
    return ast.literal_eval(c.run(cmd, hide = 'out').stdout) # Run command on RFSoC board and get command stdout
