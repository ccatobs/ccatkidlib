#=================================#
# rfsoc_io.py               2025 #
# Darshan Patel dp649@cornell.edu #
#=================================#

'''
Library of helper functions for file and directory read/write operations as well as logging.
'''

# Import Python modules
from pathlib import Path
from tqdm import tqdm
from functools import partial, partialmethod, wraps
from fabric import Connection, Config
import logging
import yaml
import time
import sys
import ast
import subprocess
import numpy as np

# Local imports
from style import Style
from utils import function_timer

##########################
# Directory IO Functions #
##########################

def create_book(curr_date, sess_id, com_to, data_dir, output = False):
    '''
    Create book for storage of timestream, sweep, and other (e.g., config) data.

    Parameters:
        curr_date      (str) : Current date
        sess_id        (int) : ID of current observing session
        com_to (list of str) : List of board and drone IDs of RFSoC in form board.drone
        data_dir       (str) : Path of directory in which to store data
        output        (bool) : Whether to print logging output to terminal
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
        create_dir(timestream_dir, output = output)
        timestream_dirs.append(timestream_dir)

        # Create vna sweep directory
        create_dir(vna_dir, output = output)
        vna_dirs.append(vna_dir)

        # Create targ sweep directory
        create_dir(targ_dir, output = output)
        targ_dirs.append(targ_dir)

        # Create config directory
        create_dir(config_dir, output = output)
        config_dirs.append(config_dir)

        # Create comb directory in config directory
        create_dir(comb_dir, output = output)

        # Create res directory in config directory
        create_dir(res_dir, output = output)

    return config_dirs, targ_dirs, timestream_dirs, vna_dirs

def create_dir(dir_path, output = False):
    '''
    Create directory at the specified path.

    Parameters:
        dir_path (str) : Path of the directory that is to be created
        output  (bool) : Whether to print logging output to terminal
    '''

    # Attempt to make the directory
    try:
        dir_path = Path(dir_path)
        # Check if directory already exists, if not make directory
        if not dir_path.exists():
            dir_path.mkdir(parents = True, exist_ok = False)
            send_msg('INFO', f"The directory '{dir_path}' was successfully created!", output = output)
        else:
            send_msg('INFO', f"The directory '{dir_path}' already exists! Directory was not overwritten.", output = output)
    except FileNotFoundError:
        send_msg('ERROR', f"The directory '{dir_path}' could not be created! Ensure that the file path is valid.", output = output)

#####################
# File IO Functions #
#####################

def load_config(config):
    '''
    Load config file.

    Parameters:
        config (str) : File path of config file to load
    Returns:
        cfg   (dict) : List of dictionaries loaded from config file
    '''
    cfg_path = Path(config)
    assert cfg_path.exists(), "Could not find config file!" # Check that config file exists
    
    # Load config file
    with open(cfg_path, 'r') as config:
        cfg = [file for file in yaml.safe_load_all(config)]
    
    # Return loaded dictionary(ies)
    if len(cfg) == 1:
        return cfg[0] # If only one dictionary, do not return array
    else:
        return cfg

def save_config(cfg_path, cfg_dic, save = True, output = False):
    '''
    Save configuration file.

    Parameters:
        cfg_path (str) : File path where the config file should be saved
        cfg_dic (dict) : Dictionary to save as config file
        save    (bool) : Whether to save config file
        output  (bool) : Whether to print logging output to terminal 
    Returns:
        cfg_dic (dict) : Returns dictionary that was saved to config file
    '''

    if save:
        # Save config file
        with open(cfg_path, 'w') as config:
            yaml.safe_dump(cfg_dic, config, sort_keys=False, default_flow_style=None)
        
        send_msg('INFO', f"Saved configuration file '{cfg_path}'!", output = output)
        send_msg('DEBUG', f'Configuration file contents: {cfg_dic}', output = output)

        # Load new config file
        with open(cfg_path, 'r') as config:
            return yaml.safe_load(config)
    else:
        return cfg_dic

#@function_timer
def get_most_recent_file(path, file_identifier, output = False, time_past = 5*60):
    '''
    Fetch the most recent file in a directory with the desired file identifier.

    Parameters:
        path            (Path) : Directory in which the file is located 
        file_identifier (str) : Substring included in the file name
        output         (bool) : Whether to print logging output to terminal
        time_past     (float) : How far in the past to look for files (in seconds)
    Returns:
        file            (str) : File path of most recent file (returns "invalid/path" if no valid files found)
    '''

    try:
        path = Path(path)
        # Attempt to get most recent file in directory using glob
        if isinstance(file_identifier, str): file_identifier = [file_identifier]

        files = []
        for file_id in file_identifier:
            files.extend(path.glob(file_id))
        file = Path(sorted(files, key = get_creation_time, reverse = True)[0])
        
        # Check if creation time is within the specified time_past 
        if abs(get_creation_time(file) - time.time()) < time_past:
            send_msg('DEBUG', f"Found most recent file '{file}' in {path}.", output = output)
            return file
        else:
            raise Exception("No files found within specified time range!")
    except Exception as e:
        send_msg('WARNING', f"Failed to fetch most recent file in '{path}' with identifier '{file_identifier}'", output = output)
        return Path("invalid/path")

def get_creation_time(file_path, output = False):
    '''
    Get the creation time of a file. Helper method for get_most_recent_file()

    Parameters:
        file_path (str) : Path of the file of which to get creation time
        output   (bool) : Whether to print logging output to terminal
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
        send_msg('WARNING', f"Error getting creation time of file: '{file_path}'", output=output)
        return -1

def get_array(src_path, dest_path, action = 'cp', load = True, output = False, timestamp = False):
    # Convert path objects to str
    src_path = str(src_path)
    dest_path = str(dest_path)
    if Path(src_path).exists(): # Ensure that path to numpy array exists on RFSoC board
        # Copy numpy file from board and load locally
        # -------------------------------------------
        # Get name of file to be copied
        file = Path(src_path).name.split('.')[0]
        if timestamp: file = '_'.join(file.split('_')[:-1]) # Trim timestamp off file if desired

        if Path(dest_path).suffix == '': dest_path += f'/{file}.npy' # If local path is a directory, use same file name as on RFSoC board

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
        
        send_msg('DEBUG', f"Copied array from '{src_path}' to '{dest_path}'.", output = output)
    else:
        # Send error message if specified path does not exist on the RFSoC board
        send_msg('ERROR', f'Failed to locate array at path {src_path}!')
        loaded_array = None
    return loaded_array

########################
# Logging IO Functions #
########################

def setup_logging(log_path , level, name = __name__, output = False):
    '''
    Setup logger and logger config.

    Parameters:
        log_path  (str) : File path of the logger including log name
        level     (str) : Level at which to log (messages below this level are ignored)
        name      (str) : Name of the logger
        output   (bool) : Whether to print messages to terminal
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
                     ['PCS', int(logging.DEBUG - 1)]]
    
    for lvl in custom_levels: _addLevel(*lvl)

    # Setup logger config
    # -------------------
    logging.basicConfig(filename=log_path, filemode = "w",
    format='%(levelname)s | %(asctime)s | %(message)s', datefmt="%m/%d/%Y %I:%M:%S %p", level = logging.getLevelName(level))

    # Test logging/confirm successful logger setup
    send_msg('INFO', f"Successfully initialized logger: {name}", output = output, name = name)

def send_msg(level, msg, output = True, name = __name__):
    '''
    Log message and print message to terminal. 

    Parameters:
        level   (str) : Level of message at which to log (One of: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        msg     (str) : Message to log
        output (bool) : Whether or not to print message to terminal
        name    (str) : Name of logger  
    '''
    # Get logger
    logger = logging.getLogger(name)

    # Try logging message
    # -------------------
    try:
        log_level = logging.getLevelName(level)
        # Log message with given level
        logger.log(log_level, msg)
        
        # Write message to terminal
        if output and logger.isEnabledFor(log_level): tqdm.write(f'{Style().log_begin(level, getattr(Style, level))} {msg}')
    except Exception as e:
        print(e)
        # Log error message
        logger.log(logging.ERROR, 'Error logging message. Ensure that the message is a string!')

        # Write error message to terminal
        if output: tqdm.write(f"{Style().log_begin('ERROR', Style.ERROR)} Error logging message. Ensure that the message is a string!")

def wait(t_sec, output = True, desc = ""):
    '''
    Wait for t_sec seconds with progress bar.

    Parameters:
        t_sec   (int) : Number of seconds to wait
        output (bool) : Whether to print progress bar to terminal
    '''

    # If terminal output is True, use tqdm progress bar
    iterator = range(int(t_sec)) # Create iterator

    # Wrap iterator in tqdm progress bar if output = True
    if output: iterator = tqdm(iterator, colour='BLUE', desc = f"{Style().log_begin('WAIT', Style.WAIT)} {desc}")

    # Wait t_sec seconds
    for _ in iterator: time.sleep(1)

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
            send_msg('HEADER', f"Executing {fmt}...", self.output)
            rtn = func(self, *args, **kwargs)
            send_msg('FOOTER', f"{fmt} executed successfully!")
            return rtn
        except Exception as e:
            import traceback 
            # Print error traceback if func failed to execute and exit out of program
            send_msg('CRITICAL', f"TERMINATING PROGRAM -- {fmt} failed to execute with error:\n{traceback.format_exc()}", True)
            sys.exit()
    return _wrapper

#######################
# Remote IO Functions #
#######################

def get_connection(ip, ssh_key, sudo = False, output = False):
    '''
    Create Fabric Connection to RFSoC board with specified IP address. 

    Parameters:
        ip      (str) : IP address to connect to
        ssh_key (str) : File path of private ssh key
        output (bool) : Whether to print logging output to terminal
    Returns:
        connection (Connection) : Fabric Connection object to specified IP address
    '''

    # MOVE PASSWORD AT SOME POINT
    config = Config(overrides={'sudo': {'password': 'xilinx'}}) if sudo else Config()

    # Get Fabric Connection to RFSoC board
    connect = Connection(f'xilinx@{ip}', config = config, connect_kwargs = {'key_filename': ssh_key})
    send_msg('DEBUG', f'Created Fabric Connection to {ip}.', output = output)
    return connect

#@function_timer
def get_array_board(c, ip, ssh_key, remote_path, local_path, load = True, timestamp = False, output = False):
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
        output     (bool) : Whether to print logging output to terminal
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
        
        send_msg('DEBUG', f"Copied array from '{remote_path}' to '{local_path}'.", output = output)
    else:
        # Send error message if specified path does not exist on the RFSoC board
        send_msg('ERROR', f'Failed to locate array at path {remote_path}!')
        loaded_array = None
    return loaded_array

#@function_timer
def save_array_board(c, path, saved_array, output = False):
    '''
    Save numpy array to RFSoC board.

    Parameters:
        c        (Connection) : Fabric Connection object of RFSoC board
        path            (str) : File path where numpy array should be saved on RFSoC board
        saved_array (ndarray) : Numpy array to save to RFSoC board
        output         (bool) : Whether to print logging output to terminal
    Returns:
        result          (str) : Standard out of save command
    ''' 

    # Define command for saving numpy array to RFSoC board
    cmd = f'python3 -c \'import numpy as np; np.save(\"{path}\", {saved_array})\''

    # Save numpy array on board
    result = c.run(cmd, hide = 'out').stdout
    send_msg('DEBUG', f"Saved array to '{path}'.", output = output)
    return result

#@function_timer
def get_most_recent_file_board(c, dir, file_identifier = "*", output = False, time_past = 5*60):
    '''
    Get most recent file in directory on RFSoC board.

    Parameters:
        c        (Connection) : Fabric Connection object of RFSoC board
        dir             (str) : Directory in which the file is located
        file_identifier (str) : Substring included in the file name 
    Returns:
        file            (str) : File path of most recent file (returns "invalid/path" if no valid files found)
    '''

    # Define command string for finding most recent file, use grep to get most recent file
    cmd = f'find {dir}/* -type f | grep {file_identifier} | xargs ls -rt | tail -1'

    # Try to find the most recent file
    # --------------------------------
    try:
        file = c.run(cmd, hide = 'out').stdout
        file = file.rstrip('\r\n') # Remove trailing characters from str
        send_msg('DEBUG', f"Found most recent file '{file}' in {dir}.", output = output)
        return file
    except:
        # Send warning message if failed to fetch most recent file
        send_msg('WARNING', f"Failed to fetch most recent file in '{dir}' with identifier '{file_identifier}'", output = output)
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
