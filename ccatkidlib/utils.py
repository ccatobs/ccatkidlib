'''
Various utility functions for MKID data collection and analysis.
'''

import numpy as np
from datetime import datetime
import pytz
from tqdm import tqdm
from functools import wraps
from .style import Style

def convert_timestamp(timestamp, timezone = 'America/New_York'):
    timestamp = int(timestamp)
    return datetime.fromtimestamp(timestamp, pytz.timezone(timezone)).strftime("%Y-%m-%d %H:%M:%S")


def dict_get(dic, keys):
    '''
    Get value from dictionary using provided dictionary keys.

    Parameters:
        dic: Dictionary to pull value from
        keys: Dictionary keys
    Returns:
        value: Value corresponding to dictionary keys (returns None if invalid key is encountered)
    '''

    def _dict_get_r(dic, key):
        '''
        Recursively get value in dictionary using specified key.
        Assumes key is unique, otherwise gets first matching key.

        Parameters:
            cfg (dict): Dictionary to get value from
            key (str): Entry in dictionary to retrieve
        '''

        done = False
        for k, v in dic.items():
            if isinstance(v, dict):
                done, value = _dict_get_r(v, key)
                if done: return done, value
            elif k == key:
                value = dic[k]
                return True, value
        return done, value

    if not isinstance(keys, list): keys = [keys]

    for key in keys[:-1]:
        try:
            dic = dic[key]
        except KeyError:
            return None
    done, value = _dict_get_r(dic, keys[-1])
    return value if done else None

def dict_set(dic, keys, value):
    '''
    Set value in dictionary using provided dictionary keys.

    Parameters:
        dic: Dictionary to set value in
        keys: Dictionary keys
    Returns:
        value: Value corresponding to dictionary keys (returns False if invalid key is encountered)
    '''
    def _dict_set_r(dic, key, value):
        '''
        Recursively edit value in dictionary using specified key.
        Assumes key is unique, otherwise edits first matching key.

        Parameters:
            cfg (dict): Dictionary to edit
            key (str): Entry in dictionary to be edited
            value: Value to replace current value in dictionary
        '''
        
        done = False
        for k, v in dic.items():
            if isinstance(v, dict):
                done = _dict_set_r(v, key, value)
                if done: return done
            elif k == key:
                dic[k] = value
                return True
        return done

    if not isinstance(keys, list): keys = [keys]

    for key in keys[:-1]:
        try:
            dic = dic[key]
        except KeyError:
            return False
    return _dict_set_r(dic, keys[-1], value)

def convert_from_dB(power):
    '''
    Convert a power from dB into normal units.
    '''
    try:
        return 10**(np.array(power)/20)
    except:
        return 10**(power/20)

def convert_to_dB(power):
    '''
    Convert a power from normal units into dB.
    '''
    try:
        return 20*np.log10(np.array(power))
    except:
        return 20*np.log10(power)

def method_timer(func):
    @wraps(func)
    def _wrapper(self, *args, **kwargs):
        import time
        name = func.__name__

        start_time = time.time()
        rtn = func(self, *args, **kwargs)
        time_diff = time.time() - start_time

        s = Style()
        tqdm.write(f'{s.log_begin("TIMER", Style.TIMER)} Method {s.func_name(name)} executed in {time_diff} seconds.')
        return rtn
    return _wrapper

def function_timer(func):
    @wraps(func)
    def _wrapper(*args, **kwargs):
        import time
        name = func.__name__

        start_time = time.time()
        rtn = func(*args, **kwargs)
        time_diff = time.time() - start_time

        s = Style()
        tqdm.write(f'{s.log_begin("TIMER", Style.TIMER)} Function {s.func_name(name)} executed in {time_diff} seconds.')
        return rtn
    return _wrapper