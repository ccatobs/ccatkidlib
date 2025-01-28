#=================================#
# rfsoc_io.py               2024 #
# Darshan Patel dp649@cornell.edu #
#=================================#

'''
Helper functions for file and directory read/write operations as well as logging.
'''

from pathlib import Path
from tqdm import tqdm
from functools import partial, partialmethod
from fabric import Connection
import logging
import yaml
import time
import sys


##########################
# Directory IO Functions #
##########################

def create_book(curr_date, sess_id, com_to, data_dir, output = False):
    '''
    Create book for storage of timestream, sweep, and other (e.g., config) data.

    Parameters:
        curr_date (str): Current date
        sess_id (int): ID of current observing session
        com_to: List of board and drone IDs of RFSoC in form board.drone
        data_dir (str): Path of directory in which to store data
        output (bool): Whether to print logging output to terminal
    Returns:
        rfsoc_dirs (str): Directories where log and config files are saved
        targ_dirs (str): Directories where target sweeps are saved
        timestream_dirs (str): Directories where timestreams are saved
        vna_dirs (str): Directories where VNA sweeps are saved
    '''

    rfsoc_dirs = []
    targ_dirs = []
    timestream_dirs = []
    vna_dirs = []

    for com in com_to:
        # Split into board id, drone id
        board, drone = com.split('.')
        com_str = f'B{board}D{drone}'

        # Define data directories
        data_dir = Path(data_dir)
        timestream_dir = data_dir / 'timestream' / curr_date / sess_id /  com_str 
        vna_dir = data_dir / 'vna' / curr_date / sess_id / com_str
        targ_dir = data_dir / 'targ' / curr_date / sess_id / com_str 
        rfsoc_dir = data_dir / 'rfsoc'  / curr_date / sess_id / com_str

        # Create timestream directory
        create_dir(timestream_dir, output = output)
        timestream_dirs.append(timestream_dir)

        # Create vna sweep directory
        create_dir(vna_dir, output = output)
        vna_dirs.append(vna_dir)

        # Create targ sweep directory
        create_dir(targ_dir, output = output)
        targ_dirs.append(targ_dir)

        # Create rfsoc directory
        create_dir(rfsoc_dir, output = output)
        rfsoc_dirs.append(rfsoc_dir)

    return rfsoc_dirs, targ_dirs, timestream_dirs, vna_dirs

def create_dir(dir_path, output = False):
    '''
    Create directory at the specified path.

    Parameters:
        dir_path (str): Path of the directory that is to be created
        output (bool): Whether to print logging output to terminal
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
        config (str): File path of config file to load
    Returns:
        cfg (dict): List of dictionaries loaded from config file
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
        cfg_path (str): File path where the config file should be saved
        cfg_dic (dict): Dictionary to save as config file
        save (bool): Whether to save config file
        output (bool): Whether to print logging output to terminal 
    Returns:
        cfg_dic (dict): Returns dictionary that was saved to config file
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
    
def get_most_recent_file(dir, file_identifier, output = False, time_past = 5*60):
    '''
    Fetch the most recent file in a directory with the desired file identifier.

    Parameters:
        dir (Path): Directory in which the file is located 
        file_identifier (str): Substring included in the file name
        output (bool): Whether to print logging output to terminal
        time_past (float): How far in the past to look for files (in seconds)
    Returns:
        file (str): File path of most recent file (returns "invalid/path" if no valid files found)
    '''

    try:
        dir = Path(dir)
        # Attempt to get most recent file in directory using glob
        file = Path(sorted(dir.glob(file_identifier), key = get_creation_time, reverse = True)[0])

        # Check if creation time is within the specified time_past 
        if abs(get_creation_time(file) - time.time()) < time_past:
            send_msg('DEBUG', f"Found most recent file '{file}' in {dir}.", output = output)
            return file
        else:
            raise Exception("No files found within specified time range!")
    except:
        send_msg('WARNING', f"Failed to fetch most recent file in '{dir}' with identifier '{file_identifier}'", output = output)
        return Path("invalid/path")

def get_creation_time(file_path, output = False):
    '''
    Get the creation time of a file. Helper method for get_most_recent_file()

    Parameters:
        file_path (str): Path of the file of which to get creation time
        output (bool): Whether to print logging output to terminal
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

########################
# Logging IO Functions #
########################

def setup_logging(log_path, level, output = False, name = __name__):
    '''
    Setup logger and logger config.

    Parameters:
        log_path: File path of the logger including log name
        level: Level at which to log (messages below this level are ignored)
        name: Name of the logger
    '''
    # Get logger
    logger = logging.getLogger(name)

    # Setup logger config
    logging.basicConfig(filename=log_path, filemode = "w",
    format='%(levelname)s | %(asctime)s | %(message)s', datefmt="%m/%d/%Y %I:%M:%S %p", level = logging.getLevelName(level))

    # Add custom logging levels
    custom_levels = [['HEADER', int((logging.INFO + logging.WARNING)/2)], 
                     ['PCS', int(logging.DEBUG - 1)]]
    
    for level in custom_levels:
        _addLevel(level[1], level[0])

    # Test logging/confirm successful logger setup
    send_msg('INFO', f"Successfully initialized logger: {name}", output = output, name = name)

def send_msg(level, msg, output = True, name = __name__):
    '''
    Log message and print message to terminal. 

    Parameters:
        level (str): Level of message at which to log (One of: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        msg (str): Message to log
        output (boolean): Whether or not to print message to terminal
        name (str): Name of logger  
    '''
    # Get logger
    logger = logging.getLogger(name)

    try:
        log_level = logging.getLevelName(level)
        # Log message with given level
        logger.log(log_level, msg)
        
        # Write message to terminal
        if output and logger.isEnabledFor(log_level):
            color_mapping = {'PCS':     '\033[95m',
                             'DEBUG':   '\033[96m',
                             'INFO':    '\033[32m',
                             'HEADER':  '\033[92m', 
                             'WARNING': '\033[93m',
                             'ERROR':   '\033[31m',
                             'CRITICAL':'\033[91m',
                             'DEFAULT': '\033[0m'}
            
            tqdm.write(f'{color_mapping[level]}{level} {color_mapping["DEFAULT"]}| {msg}')
    except:
        logger.log(logging.WARNING, 'Error logging message. Ensure that the message is a string!')

        if output:
            tqdm.write("Error logging message. Ensure that the message is a string!")

def wait(t_sec, output = True, desc = ""):
    '''
    Wait for t_sec seconds with progress bar.

    Parameters:
        t_sec (int): Number of seconds to wait
        output (bool): Whether to print progress bar to terminal
    '''

    # If terminal output is True, use tqdm progress bar
    iterator = range(int(t_sec))
    if output:
        iterator = tqdm(iterator, desc = desc)

    for t in iterator:
        time.sleep(1)

def header(func):
    def wrapper(self, *args, **kwargs):
        name = func.__name__
        send_msg('HEADER', f"Executing {name}...", self.output)
        try:
            res = func(self, *args, **kwargs)
            send_msg('HEADER', f"{name} executed successfully!")
            return res
        except Exception as e:
            send_msg('CRITICAL', f"TERMINATING PROGRAM -- {name} failed to execute with error:\n{e}", True)
            sys.exit()
    return wrapper

def _addLevel(num, name):
    '''
    Adds a custom logging level to the logger.
    '''

    method_name = name.lower()

    logging.addLevelName(num, name)
    setattr(logging, name, num)
    setattr(logging.getLoggerClass(), method_name, partialmethod(logging.getLoggerClass().log, num))
    setattr(logging, method_name, partial(logging.log, num))

#######################
# Remote IO Functions #
#######################

def get_connection(ip, key, output = False):
    '''
    Create Fabric Connection to RFSoC board with specified IP address. 

    Parameters:
        ip (str): IP address to connect to
        key (str): File path of private RSA key
        output (bool): Whether to print logging output to terminal
    Returns:
        connection: Fabric Connection object to specified IP address
    '''

    # Get Fabric Connection to RFSoC board
    connect = Connection(f'xilinx@{ip}', connect_kwargs = {'key_filename': key})
    send_msg('DEBUG', f'Created Fabric Connection to {ip}.', output = output)
    return connect

def load_array_board(c, path, output = False):
    '''
    Load numpy array from RFSoC board.

    Parameters:
        c: Fabric Connection object of RFSoC board
        path (str): File path of numpy array on RFSoC board
        output (bool): Whether to print logging output to terminal
    Returns:
        array: Loaded numpy array
    '''

    cmd = f'python3 -c \'import numpy as np; print(np.load(\"{path}\").tolist())\''
    loaded_array = eval(c.run(cmd, hide = 'out').stdout)
    send_msg('DEBUG', f"Loaded array {loaded_array} from '{path}'.", output = output)
    return loaded_array

def save_array_board(c, path, saved_array, output = False):
    '''
    Save numpy array to RFSoC board.

    Parameters:
        c: Fabric Connection object of RFSoC board
        path (str): File path where numpy array should be saved on RFSoC board
        saved_array: Numpy array to save to RFSoC board
        output (bool): Whether to print logging output to terminal
    Returns:
        array: Numpy array saved to RFSoC board
    ''' 

    cmd = f'python3 -c \'import numpy as np; np.save(\"{path}\", {saved_array})\''
    result = c.run(cmd, hide = 'out').stdout
    send_msg('DEBUG', f"Saved array {saved_array} to '{path}'.", output = output)
    return load_array_board(c, path)

def get_most_recent_file_board(c, dir, file_identifier = "*", output = False):
    '''
    Get most recent file in directory on RFSoC board.

    Parameters:
        c: Fabric Connection object of RFSoC board
        dir: Directory in which the file is located
        file_identifier: Substring included in the file name 
    Returns:
        file (str): File path of most recent file (returns "invalid/path" if no valid files found)
    '''

    # Use grep to get most recent file
    cmd = f'find {dir}/* -type f | grep {file_identifier} | xargs ls -rt | tail -1'
    try:
        file = c.run(cmd, hide = 'out').stdout
        file = file.rstrip('\r\n')
        send_msg('DEBUG', f"Found most recent file '{file}' in {dir}.", output = output)
        return file
    except:
        send_msg('WARNING', f"Failed to fetch most recent file in '{dir}' with identifier '{file_identifier}'", output = output)
        return "invalid/path"

def path_exists(c, path):
    '''
    Check if path exists on RFSoC board.

    Parameters:
        c: Fabric Connection object of RFSoC board
        path (str): File path to check existence of
    Returns:
        exists (bool): Whether the file path exists
    '''

    cmd = f"[ -f {path} ] && echo True || echo False"
    return eval(c.run(cmd, hide = 'out').stdout)