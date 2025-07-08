import ccatkidlib.rfsoc_io as rfsoc_io
import ccatkidlib.utils as utils
import numpy as np

from ..utils import function_timer

#@function_timer
def group_args(com_to, *args):
    arg_list = []
    for arg in zip(*args):
        arg = ', '.join([str(a) for a in list(arg)])
        arg_list.append(arg)

    sweep_dict = {}
    for com, arg in zip(com_to, arg_list):
        com_list = sweep_dict.setdefault(arg, [])
        com_list.append(com)

    return sweep_dict

def parse_args(R, com_to, arg):
    '''
    Parse key word drone arguments

    Parameters:
        com_to       (list of str) : List of drones
        arg    (Any | list of Any) : Arugment to parse
    Returns:
        args         (list of Any) : List of parsed args
    '''

    args = None
    try: # Check if argument is a list
        if len(arg) == len(com_to): # Length of argument list should match number of drones
            args = arg
        else:
            rfsoc_io.send_msg('WARNING', f'{arg} is not a valid argument. Must be a single value or match the length of {com_to}!')
            return None # Return None if argument is invalid
    except:
        try: # Assume argument is a single value and create a list of arugments with length equal to the number of drones
            args = [arg] * len(com_to)
        except:
            rfsoc_io.send_msg('WARNING', f'{arg} is not a valid argument. Must be a single value or match the length of {com_to}!')
            return None # Return None if argument is invalid
    return args # Return parsed argument

def get_drone_args(R, com_to, key):
    '''
    Get drone arguments from drone config files

    Parameters:
        com_to (list of str) : List of drones
        key    (list of str) : List of dictionary key(s) of argument to retrieve from drone config files
    Returns:
        args   (list of Any) : List of values from drone config files corresponding to dictionary key
    '''

    inds = [R.drone_list.index(com) for com in com_to] # Get list of indicies corresponding to drones in com_to
    return [utils.dict_get(R.drone_cfg[ind], key) for ind in inds] # Get dictionary value corresponding to key for each drone config file

def set_drone_args(R, com_to, key, args):
    '''
    Set drone arguments in drone config files

    Parameters:
        com_to (list of str) : List of drones
        key            (str) : Dictionary key to set value of
        args   (list of Any) : List of values to set in drone config files
    Returns:
        rtn_list (list of bool) : List of returns from edit_config for each drone
    '''

    # Iterate over drones and args
    rtn_list = []
    for com, arg in zip(com_to, args):
        ind = R.drone_list.index(com) # Get index corresponding to drone in com_to
        rtn = rfsoc_io.edit_config(R.drone_cfg[ind], key, arg) # Set config value of specified key
        rtn_list.append(rtn)
    return rtn_list

#@function_timer
def get_com_to(R, **kwargs):
    '''
    Parses a list of drone com_to and sets up these drones. The drone_list specified in the system config is used if no com_to is passed as a key word argument.
    If a com_to is passed as a key word argument, makes sure that the com_to is a list and that all drones are included in the system config drone_list.

    Parameters:
        com_to (list of str): String or list of strings specifying drone com_to
    Returns:
        com_to (list of str): List of drone com_to
        bids   (list of str): List of boards in com_to
    '''

    # Set com_to to drone list specified in system config (make a copy so that it does not point to the class drone_list attribute)
    com_to = np.copy(R.drone_list).tolist()
    bids = R.board_list
    setup = True

    # Override com_to with that passed as key word argument (if any)
    for key, value in kwargs.items():
        if key == 'com_to':
            com_to = value

            # If com_to is not a list, make it a list
            if not isinstance(com_to, list): com_to = [com_to]

            # Get list of boards used
            bids = set()
            for com in com_to[::-1]:
                split_str = com.split('.') # Split drone com_to into bid and drid
                bids.add(split_str[0]) # Add bid to set of board ids

                # Replace any board only com_to (e.g. '1') with bid.drid for all four drones
                if len(split_str) == 1:
                    com_to.remove(com)
                    for i in range(4):
                        com_to.append(com + f'.{i + 1}')

            # Remove any duplicate entries and sort list of drones
            com_to = sorted(list(set(com_to)))

            # Check that all drones are in initialized drone list
            extra_drones = set(com_to) - set(R.drone_list) # Get drones in com_to that are not in drone_list
            if len(extra_drones) > 0: raise ValueError(f'The drones {sorted(list(extra_drones))} are not in system config drone list!')
        elif key == 'setup':
            setup = value

    # Set up drone with specified com_to list
    kwargs['restart'] = False
    if setup: R.setup_drones(**kwargs)

    # Return list of drones and list of boards in use
    return com_to, sorted(list(bids))
