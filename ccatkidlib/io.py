'''
Library of helper functions for general directory/file IO operations as well as communication and IO with |RFSoC| boards.

.. codeauthor:: Darshan Patel <dp649@cornell.edu>

'''

import os
import sys
import ast
import time
import yaml
import subprocess
import numpy as np

from pathlib import Path
from tqdm import tqdm
from typing import Any, Literal
from fabric import Connection, Config
from jinja2 import Environment, FileSystemLoader

import ccatkidlib.utils as utils
import ccatkidlib.log as log

#========================#
# Directory IO Functions #
#========================#

def create_tree(com_to: list[str], curr_date: str, sess_id: int, data_dir: str) -> tuple[list[str], list[str], list[str], list[str]]:
    '''
    Create file tree for storage of sweep, timestream, comb, and configuration files

    Args:
        com_to: List of |RFSoC| drones to create directories for
        curr_date: Current date
        sess_id: Session ID of measurement
        data_dir: Directory in which to create file tree
    Returns:
        Tuple of */config* directories, */target* directories, */timestream* directories, and */vna* directories in that order.
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

def create_dir(dir: str) -> None:
    '''
    Create directory at the specified path. Will not overwrite if it already exists.

    Args:
        dir: Path of directory to be created

    Raises:
        FileNotFoundError: If an invalid ``dir`` is specified
        PermissionError: If specified ``dir`` does not have suitable write permissions to be created
    '''

    # Attempt to make the directory
    try:
        dir = Path(dir)
        # Check if directory already exists, if not make directory
        if not dir.exists():
            dir.mkdir(parents = True, exist_ok = False)
            log.log('DEBUG', "The directory '%s' was successfully created!", dir)
        else:
            log.log('DEBUG', "The directory '%s' already exists! Directory was not overwritten.", dir)
    except FileNotFoundError:
        log.log('ERROR', "The directory '%s' could not be created! Ensure that the file path is valid!", dir)
        raise FileNotFoundError(f"The directory '{dir}' could not be created! Ensure that the file path is valid!")
    except PermissionError:
        log.log('ERROR', "The directory '%s' could not be created! Ensure that the parent directory has suitable write permissions!", dir)
        raise PermissionError(f"The directory '{dir}' could not be created! Ensure that the parent directory has suitable write permissions!")

def add_dir(dir_name: str, 
            data_dir: str, 
            save_root: str | None = None,
            data_root: str | None = None,
            sub_dirs: list[str] = [],
            timestamp: str | None = None) -> str:
    '''
    Add a new directory within/mimicing a pre-existing *ccatkidlib* file tree. The directory will be created in the already existing
    file tree structure if ``save_root`` is not specified, otherwise it will mimic the already existing structure within the specified ``save_root`` directory.

    Args:
        dir_name: Name of directory to add
        data_dir: Data directory of *ccatkidlib* file tree
        save_root: Root data directory where new directory should be created
        data_root: Root data directory of *ccatkidlib* file tree
        sub_dirs: Sub-directories to create within new directory. If no sub-directories specified, will mimic the sub-directories of the already existing file tree.
        timestamp: Unix timestamp. If specified, will create a directory with the timestamp as the name within the specified ``sub_dirs`` (at the deepest level).
    Returns:
        Path of the newly added directory
    
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

def create_noise_files(com_to: list[str], tmp_dir: str) -> list[str]:
    '''
    Create *ccatkidlib* */tmp* directory and populate with empty noise |tone| files.
    Will not overwrite if directory or noise tone files already exist.

    Args:
        com_to: List of drones for which to create noise tone files
        tmp_dir: Temporary directory where noise tone files should be created
    Returns:
        List of noise tone file paths in order of ``com_to``
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

#=====================#
# Config IO Functions #
#=====================#

def load_config(cfg_path: str) -> dict | list[dict]:
    '''
    Load configuration file from specified ``cfg_path``

    Args:
        cfg_path: File path of configuration file to load
    Returns:
        Loaded configuration file or list of loaded configuration files if ``cfg_path`` contains more than one
    '''
    cfg_path = Path(cfg_path)
    if not cfg_path.exists(): raise FileNotFoundError(f"Could not find config file: '{cfg_path}'") # Raise error if config file does not exist

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

def save_config(cfg_path: str, cfg: dict, save: bool = True) -> dict:
    '''
    Save ``cfg`` configuration file to the specified ``cfg_path``

    Args:
        cfg_path: File path where the configuration file should be saved
        cfg: Configuration file to save
        save: Whether to save configuration file 
    Returns:
        Configuration file that was saved
    '''
    if save:
        with open(cfg_path, 'w') as config:
            yaml.safe_dump(cfg, config, sort_keys=False, default_flow_style=None)
        log.log('DEBUG', f"Saved configuration file: '{cfg_path}'")
    return cfg

def edit_config(cfg: dict, key: str, value: Any, append: bool = False) -> bool:
    '''
    Update ``key`` in ``cfg`` configuration file with the specified ``value``.

    Args:
        cfg: Configuration file to update
        key: Key that should be updated
        value: Value with which to update ``key``
        append: Whether to add a new ``key``, ``value`` pair to configuration file if ``key`` is not found
    Returns:
        done: **True** if key was successfully updated or created, otherwise **False**
    '''
    # Edit config file dictionary
    # ---------------------------
    done = utils.dict_set(cfg, key, value)

    # Check if key was successfully updated
    # -------------------------------------
    if done: # If matching key was updated
        log.log('DEBUG', f'Updated key "{key}" with value "{value}" in config file"!')
    elif append: # If key was not found and append=True, add key value pair to dictionary
        if isinstance(key, list):
            for k in key[:0:-1]: value = {k:value}
            key = key[0]
        cfg[key] = value
        done = True
        log.log('DEBUG', f'Added key "{key}" with value "{value}" to config file!')
    else: # If key was not found and append=False
        log.log('DEBUG', f'Failed to update key "{key}" with value "{value}" in config file!')
    return done

#===================#
# File IO Functions #
#===================#

def get_most_recent_file(dir: str, file_identifier: str | list[str] = '*', time_past: float = 60*60, time_ref: float | None = None, ccatkidlib_file: bool = False) -> str:
    '''
    Fetch the most recent file in the ``dir`` directory

    Args:
        dir: Directory from which to get most recent file
        file_identifier: List of sub-strings to use for identifying/filtering files. A file will be identified as valid if it contains any of the sub-strings in the list. 
        time_past: How old the file can be in seconds. Files older than ``time_past`` will be ignored.
        time_ref: Unix time to reference creation time of files against for determining if file is too old/new. 
        ccatkidlib_file: Whether or not the files in ``dir`` are *ccatkidlib* data files. If **True**, will use the timestamp in the file name as the creation time.
    Returns:
        File path of the most recent file if a vaild file is found, otherwise returns *"invalid/path"*
    '''
    if time_ref is None: time_ref = time.time()

    try:
        path = Path(dir)
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
        log.log('DEBUG', f"Failed to fetch most recent file in '{dir}' with identifier '{file_identifier}' with Exception:\n{e}")
        return Path("invalid/path")

def get_timestamp(path: str) -> int:
    '''
    Extract the timestamp from a *ccatkidlib* data file name.

    Args:
        path: Path of the *ccatkidlib* data file
    Returns:
        Timestamp of the file or -1 if no valid timestamp can be determined
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

def get_creation_time(path: str) -> float:
    '''
    Get the creation time of a file.

    Note:
        The time reported is the *ctime* of the file, which usually corresponds to the creation time of the file.
        The *ctime* can, however, change with certain file modifications so care must be taken when interpreting the time reported.

    Args:
        path: Path of the file to get creation time of 
    Returns:
        Creation time of file or -1 if creation time could not be determined
    '''
    try:
        file = Path(path)
        # Get creation time of file
        creation_time = file.stat().st_ctime
        log.log('DEBUG', f"Creation time of file '{path}' is {creation_time}.")
        return creation_time
    except:
        log.log('DEBUG', f"Error getting creation time of file: '{path}'")
        return -1

def get_array(src_path: str, dest_path: str, action: Literal['cp', 'mv'] = 'cp', load: bool = True, timestamp: bool = False) -> np.ndarray | str | None:
    '''
    Copy or move a *NumPy* array and load its contents
    
    Args:
        src_path: Path of the *NumPy* array
        dest_path: Destination path where the *NumPy* array should be copied or moved. If path does not include a file name, the same file name will be used.
        action: Whether to copy **'cp'** or move **'mv'** the *NumPy* array
        load: Whether to load and return the contents of the *NumPy* array
        timestamp: Whether to remove timestamp from *NumPy* array file name (Use **False** if no timestamp)
    Returns:
        Loaded *NumPy* array if ``load`` is **True** otherwise the file path where the *NumPy* array was copied or moved to. 
        Returns **None** if failed to copy/move/load *NumPy* array

    '''
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

def increment_file(dir: str, file_prefix: str, file_suffix: str, overwrite: bool = False) -> tuple[str, int | None]:
    '''
    Increment the file count in a file name. Used when saving files to prevent overwriting files with duplicate names.
    
    Args:
        dir: Directory where file will be saved
        file_prefix: Sub-string of the file name before the file count
        file_suffix: Sub-string of the file name after the file count
        overwrite: Whether overwriting files is allowed. Will not add file count to file name if **True**
    Returns:
        The new file name with file count included and the file count of the file.
        If overwrite is **True**, will return **None** instead of the file count 
    '''

    if overwrite:
        return Path(dir) / f'{file_prefix[:-1]}{file_suffix}', None
    else:
        file_count = 0
        full_path = Path(dir) / f'{file_prefix}{file_count}{file_suffix}' 
        while full_path.exists(): 
            file_count += 1
            full_path = Path(dir) / f'{file_prefix}{file_count}{file_suffix}'
        return full_path, file_count

def combine_npy(files: list[str], num: int, com: str | None = None, fname_out: dict | None = None) -> list[str]:
    '''
    Combine *NumPy* into **NpzFile** zipped archive files.

    Args:
        files: List of *NumPy* files to combine
        num: Number of **NpzFile** files to create
        com: 
        fname_out: 
    Returns:
        File names of the **NpzFile** files that were created
    '''

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

def get_connection(ip: str, ssh_key: str) -> Connection:
    '''
    Create *Fabric* **Connection** object to the |RFSoC| board with the specified ``ip`` address

    .. important::
        This function requires a SSH key pair to exist between the local machine and the |RFSoC| board for passwordless login

    Args:
        ip: IP address of |RFSoC| board
        ssh_key: File path of private SSH key corresponding to public key on the |RFSoC| board
    Returns:
        *Fabric* **Connection** object to specified |RFSoC| board
    '''

    # Get Fabric Connection to RFSoC board
    connect = Connection(f'xilinx@{ip}', connect_kwargs = {'key_filename': ssh_key})
    log.log('DEBUG', f'Created Fabric Connection object to {ip}.')
    return connect

def get_array_board(c: Connection, ip: str, ssh_key: str, remote_path: str, local_path: str, load: bool = True, timestamp: bool = False) -> np.ndarray | str | None:
    '''
    Load *NumPy* array from |RFSoC| board

    Parameters:
        c: *Fabric* **Connection** object of |RFSoC| board
        ip: IP address of |RFSoC| board
        ssh_key: File path of private SSH key corresponding to public key on the |RFSoC| board
        remote_path: File path of *NumPy* array on |RFSoC| board
        local_path: Path on local machine where *NumPy* array should be copied. If it does not contain a file name, the same file name will be used as that on the |RFSoC| board
        load: Whether to load and return the contents of the *NumPy* array
        timestamp: Whether to remove timestamp from *NumPy* array file name (Use **False** if no timestamp)
    Returns:
        Loaded *NumPy* array if ``load`` is **True** otherwise the file path on the local machine where *NumPy* array was copied to. 
        Returns **None** if failed to copy/load *NumPy* array
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

def save_array_board(ip: str, ssh_key: str, path: str, array: np.ndarray, tmp_dir: str) -> str | None:
    '''
    Save *NumPy* array to |RFSoC| board

    Args:
        ip: IP address of |RFSoC| board
        ssh_key: File path of private SSH key corresponding to public key on the |RFSoC| board
        path: File path where *NumPy* array should be saved on |RFSoC| board
        array: *NumPy* array to save to |RFSoC| board
        tmp_dir: Temporary directory where *NumPy* array is saved on local machine before copying to |RFSoC| board
    Returns:
        Returns output string of *rsync* command if successful, otherwise returns **None**
    ''' 

    save_path = Path(tmp_dir) / Path(path).name
    np.save(save_path, array)

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

def get_most_recent_file_board(c: Connection, dir: str, file_identifier: str = "*", time_past: float = 60*60) -> str:
    '''
    Get most recent file in specified ``dir`` directory on |RFSoC| board

    Args:
        c: *Fabric* **Connection** object of |RFSoC| board
        dir: Directory from which to get most recent file from
        file_identifier: Sub-string used to identify/filter files
        time_past: How old the file can be in seconds. Files older than ``time_past`` will be ignored.
    Returns:
        File path of the most recent file if valid file exists otherwise returns *"invalid/path"*
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
    Check if specified ``path`` exists on |RFSoC| board.

    Args:
        c: *Fabric* **Connection** object of |RFSoC| board
        path: Path to check existence of
    Returns:
        **True** if ``path`` exists, otherwise **False**
    '''

    path = str(path) # Convert path objects to str

    cmd = f"[ -f {path} ] && echo True || echo False" # Define command str to check if path exists
    return ast.literal_eval(c.run(cmd, hide = 'out').stdout) # Run command on RFSoC board and get command stdout
