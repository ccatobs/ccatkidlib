'''
Various utility functions for MKID data collection and analysis.
'''

import numpy as np

def edit_dic(dic, key, value):
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
            done = edit_dic(v, key, value)
            if done: return done
        elif k == key:
            dic[k] = value
            return True
    return done

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