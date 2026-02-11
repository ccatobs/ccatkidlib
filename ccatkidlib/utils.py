'''
General utility functions

.. codeauthor:: Darshan Patel <dp649@cornell.edu>
'''

from typing import Any

def dict_get(dic: dict, keys: str | list[str]) -> Any | None:
    '''
    Fetch value from a nested dictionary corresponding to the last key specified in ``keys``. 
    An attempt will be made to reduce the dictionary using the ``keys`` in order until the final key is found or a **KeyError** is encountered.
    If a **KeyError** is encountered, the reduced dictionary will be recursively searched for the final key.

    Args:
        dic: Dictionary to get value from
        keys: List of dictionary keys (in dictionary nesting order)
    Returns:
        Value of ``dic`` corresponding to the final key in ``keys`` or **None** if final key is not found
    '''

    def _dict_get_r(dic: dict, key: str) -> tuple[bool, Any]:
        '''
        Recursively get value in dictionary using specified key.
        Assumes key is unique, otherwise gets first matching key.

        Args:
            dic: Dictionary to get value from
            key: Entry in dictionary to retrieve
        '''

        done = False
        value = None
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

def dict_set(dic: dict, keys: str | list[str], value: Any) -> bool:
    '''
    Set value of the last key specified in ``keys`` in the specified dictionary. 
    An attempt will be made to reduce the dictionary using the ``keys`` in order until the final key is found or a **KeyError** is encountered.
    If a **KeyError** is encountered, the reduced dictionary will be recursively searched for the final key.

    Args:
        dic: Dictionary to set value in
        keys: List of dictionary keys (in dictionary nesting order)
        value: Value to set 
    Returns:
        **True** if final key in ``keys`` was successfully set with the specified ``value`` otherwise **False**
    '''

    def _dict_set_r(dic: dict, key: str, value: Any) -> bool:
        '''
        Recursively edit value in dictionary using specified key.
        Assumes key is unique, otherwise edits first matching key.

        Args:
            dic: Dictionary to edit
            key: Entry in dictionary to be edited
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