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
import logging
import yaml
import time


##########################
# Directory IO Functions #
##########################

def create_book(curr_date, sess_id, com_to, data_dir, output = False):
    '''
    Create book for storage of timestream, sweep, and other (e.g., config) data.

    Parameters:
        sess_id (int): ID of current observing session
        com_to (float): Board and drone ID of RFSoC in form board.drone
        data_dir (str): Path of directory in which to store data
    '''

    # Try to split com_to into board and drone
    try:
        board, drone = com_to.split('.')
        com_str = f'B{board}D{drone}'
    # If can't split com_to, assume com_to is just the a board id
    except:
        com_str = com_to

    # Define data directories
    data_dir = Path(data_dir)
    timestream_dir = data_dir / 'timestream' / curr_date / com_str / sess_id
    vna_dir = data_dir / 'vna' / curr_date / com_str / sess_id
    targ_dir = data_dir / 'targ' / curr_date / com_str / sess_id
    rfsoc_dir = data_dir / 'rfsoc'  / curr_date / com_str / sess_id

    # Create timestream directory
    create_dir(timestream_dir, output = output)

    # Create vna sweep directory
    create_dir(vna_dir, output = output)

    # Create targ sweep directory
    create_dir(targ_dir, output = output )

    # Create rfsoc directory
    create_dir(rfsoc_dir, output = output)

    # Create tmp directory
    create_dir(data_dir / 'tmp')

    return rfsoc_dir, targ_dir, timestream_dir, vna_dir

def create_dir(dir_path, output = False):
    '''
    Create directory at the specified path.

    Parameters:
        dir_path (str): Path of the directory that is to be created
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
    except:
        send_msg('ERROR', f"The directory '{dir_path}' could not be created! Ensure that the file path is valid.", output = output)

#####################
# File IO Functions #
#####################

def load_config(config):
    cfg_path = Path(config)
    assert cfg_path.exists(), "Could not find config file!" # Check that config file exists
    
    # Load config file
    with open(cfg_path, 'r') as config:
        cfg = [file for file in yaml.safe_load_all(config)]
    
    if len(cfg)  == 1:
        return cfg[0]
    else:
        return cfg

def save_config(cfg_path, cfg_dic, save = True):
    '''
    Save configuration file.
    '''
    if save:
        # Save config file
        with open(cfg_path, 'w') as config:
            yaml.safe_dump(cfg_dic, config, sort_keys=False, default_flow_style=None)

        # Load new config file
        with open(cfg_path, 'r') as config:
            return yaml.safe_load(config)
    else:
        return cfg_dic
    
def get_most_recent_file(dir, file_identifier, output = False, time_past = 60):
    '''
    Fetch the most recent file in a directory with the desired file identifier.

    Parameters:
        dir (Path): Directory in which the file is located 
        file_identifier (str): Substring included in the file name
        time_past (float): How far in the past to look for files (in seconds)
    '''

    try:
        dir = Path(dir)
        file = Path(sorted(dir.glob(file_identifier), key = get_creation_time, reverse = True)[0])
        if abs(get_creation_time(file) - time.time()) < time_past:
            return file
        else:
            raise Exception("No files found within specified time range!")
    except:
        send_msg('WARNING', f"Failed to fetch most recent file in {dir} with identifier '{file_identifier}'", output = output)
        return Path("invalid/path")

def get_creation_time(file_path, output = False):
    '''
    Get the creation time of a file. Helper method for get_most_recent_file()

    Parameters:
        file_path (str): Path of the file of which to get creation time
    '''
    try:
        # Get and return creation time of the file
        file = Path(file_path)
        return file.stat().st_ctime
    except:
        send_msg('WARNING', f"Error getting creation time of file: '{file_path}'", output=output)
        return -1

########################
# Logging IO Functions #
########################

def addLevel(num, name):
    '''
    Adds a custom logging level to the logger.
    '''

    method_name = name.lower()

    logging.addLevelName(num, name)
    setattr(logging, name, num)
    setattr(logging.getLoggerClass(), method_name, partialmethod(logging.getLoggerClass().log, num))
    setattr(logging, method_name, partial(logging.log, num))

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
    addLevel(int((logging.getLevelName('INFO') + logging.getLevelName('WARNING'))/2), 'HEADER')

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
        # Log message with given level
        logger.log(logging.getLevelName(level), msg)
        
        # Write message to terminal
        if output and logger.isEnabledFor(logging.getLevelName(level)):
            tqdm.write(f'{level} | {msg}')
    except:
        logger.log(logging.WARNING, 'Error logging message. Ensure that the message is a string!')

        if output:
            tqdm.write("WARNING | Error logging message. Ensure that the message is a string!")


