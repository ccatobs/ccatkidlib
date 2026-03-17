'''
Library for standardized console and terminal logging across module

.. codeauthor:: Darshan Patel <dp649@cornell.edu>

'''

import sys
import time
import logging
import collections
import tqdm.contrib.logging as tqdm_logging

from tqdm import tqdm
from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, field
from functools import partial, partialmethod, wraps

from typing import Literal, Callable, Any

Level = Literal['PCS', 'DEBUG', 'TIMER', 'INFO', 'HEADER', 'WARNING', 'ERROR', 'CRITICAL']

def setup_logging(log_path: str, 
                  file_level: str, terminal_level: str, 
                  max_file_size: int = 100, 
                  name: str = __name__) -> None:
    '''
    Setup levels and handlers for logger of the specified ``name``

    Args:
        log_path: Path of file that logger should log to
        file_level: Level at which to log messages to log file
        terminal_level: Level at which to log messages to terminal
        max_file_size: Maximum size of log file in Megabytes. If exceeded, a new log file will be created.
        name: Name of logger
    '''

    def _addLevel(name: str, num: int) -> None:
        '''
        Add a custom logging level to the logger

        Args:
            name: Name of the custom level
            num: Log level of the custom level
        '''
        
        # Convert passed name to lowercase 
        method_name = name.lower()

        if not hasattr(logging, name):
            logging.addLevelName(num, name) # Add new logging level to logger
            setattr(logging, name, num)     # Add new attribute to the logging class corresponding to custom logging level

            # Add new methods to relevant loggging classes
            setattr(logging.getLoggerClass(), method_name, partialmethod(logging.getLoggerClass().log, num)) 
            setattr(logging, method_name, partial(logging.log, num))
    
    # Configure logging to capture warnings
    # -------------------------------------
    logging.captureWarnings(True)

    # Get logger
    logger = logging.getLogger(name)

    # Add custom logging levels
    # -------------------------
    custom_levels = [['HEADER', int((logging.INFO + logging.WARNING)/2)],
                     ['FOOTER', int((logging.INFO + logging.WARNING)/2)],
                     ['PCS', int(logging.DEBUG - 5)],
                     ['TIMER', int(logging.INFO - 5)]]
    
    for lvl in custom_levels: _addLevel(*lvl)

    # Setup logger config
    # -------------------

    # Setup logging to file
    if not any(isinstance(handler, RotatingFileHandler) for handler in logger.handlers):
        file_log = RotatingFileHandler(log_path, 
                                       mode='a', 
                                       maxBytes= max_file_size * 1024**2,
                                       backupCount=1)

        file_level = logging.getLevelName(file_level)
        file_log.setLevel(file_level)

        file_format = logging.Formatter(fmt='{asctime} | {name:>17} | {message}', datefmt="%m/%d/%Y %I:%M:%S %p", style='{')
        file_log.setFormatter(file_format)
        logger.addHandler(file_log)

    # Setup logging to terminal
    if not any(isinstance(handler, tqdm_logging._TqdmLoggingHandler) for handler in logger.handlers):
        terminal_log = tqdm_logging._TqdmLoggingHandler()

        terminal_level = logging.getLevelName(terminal_level)
        terminal_log.setLevel(terminal_level)

        terminal_format = logging.Formatter(fmt='{asctime} | {name:>17} | {message}', datefmt="%m/%d/%Y %I:%M:%S %p", style='{')
        terminal_log.setFormatter(terminal_format)
        logger.addHandler(terminal_log)
        
    # Set logger level and add handlers
    logger.setLevel(min(file_level, terminal_level)) # Set logger level to the minimum of file and terminal levels

    # Test logging/confirm successful logger setup
    log('DEBUG', "Successfully initialized logger: %s", name, name = name)

def log(level: Level, msg: str, *args, name: str = __name__) -> None:
    '''
    Log message to logger of the specified ``name`` with custom formatting

    Args:
        level: Level of message at which to log
        msg: Message to log
        args: Positional arguments to pass to ``logger.log`` method for *print-f* style message formatting
        name: Name of logger
    '''
    # Get logger
    logger = logging.getLogger(name)

    # Try logging message
    # -------------------
    try:
        log_level = logging.getLevelName(level)

        msg = f'{Style.style_level(level)} {msg}'
        logger.log(log_level, msg, *args) if not len(args) == 0 else logger.log(log_level, msg)
    except Exception as e:
        # Log error message
        logger.log(logging.ERROR, 'Failed to log message %s with error %s!', msg, e)

def wait(t: float, desc: str = "", interval: float = 0.01) -> None:
    '''
    Sleep for ``t`` seconds with a *tqdm* progress bar. This is done by monitoring the difference between the current and start time every ``interval`` seconds.

    Args:
        t: Number of seconds to sleep
        desc: Description to add to progress bar
        interval: Number of seconds to sleep before checking the difference between the current and start time
    '''    

    time_diff = 0
    start_time = time.time()
    with tqdm(total=t, colour='BLUE', desc = f"{Style.style_level('WAIT')} {desc}") as pbar:
        while time_diff < t:
            pbar.update(int(time_diff - pbar.n))
            time.sleep(interval)
            time_diff = time.time() - start_time
        pbar.update(t - pbar.n)

def header(method: Callable) -> Any:
    '''
    Decorator that logs **HEADER** and **FOOTER** messages and implements error handling for the wrapped method

    Args:
        method: Method to decorate  
    Returns:
        The return of ``method`` if executed successfully otherwise **None**
    '''
    @wraps(method) # Needed for help() calls on ``method`` to print help string of ``method`` instead of help string of header
    def _wrapper(self, *args, **kwargs):
        name = method.__name__ 
        fmt = Style.style_name(name)

        # Try to execute func
        # -------------------
        try:
            log('HEADER', "Executing %s...", fmt)
            rtn = method(self, *args, **kwargs)
            log('FOOTER', "%s executed successfully", fmt)
            return rtn
        except Exception as e:
            import traceback 
            log('ERROR', "Method %s failed to execute with error:\n%s", fmt, traceback.format_exc())
            return None
    return _wrapper

def method_timer(method: Callable) -> Any:
    '''
    Decorator that logs method execution time
    
    Args:
        method: Method to decorate
    Returns:
        The return of ``method``
    '''

    @wraps(method)
    def _wrapper(self, *args, **kwargs):
        import time
        name = method.__name__

        start_time = time.time()
        rtn = method(self, *args, **kwargs)
        time_diff = time.time() - start_time

        log('TIMER', f'Method {Style.style_name(name)} executed in {time_diff} seconds.')
        return rtn
    return _wrapper

def function_timer(func: Callable) -> Any:   
    '''
    Decorator that logs function execution time
    
    Args:
        func: Function to decorate
    Returns:
        The return of ``func``
    '''
    @wraps(func)
    def _wrapper(*args, **kwargs):
        import time
        name = func.__name__

        start_time = time.time()
        rtn = func(*args, **kwargs)
        time_diff = time.time() - start_time

        log('TIMER', f'Method {Style.style_name(name)} executed in {time_diff} seconds.')
        return rtn
    return _wrapper

@dataclass
class Style:
    # Font styles
    ITALICS:  str = '\033[3m'
    INVERT:   str = '\033[7m'

    # Logging Colors
    PCS:      str = '\033[92m'
    DEBUG:    str = '\033[34m'
    INFO:     str = '\033[32m'
    HEADER:   str = '\033[48;5;177m'
    FOOTER:   str = '\033[48;5;177m'
    WARNING:  str = '\033[93m'
    ERROR:    str = '\033[31m'
    CRITICAL: str = '\033[91m' 
    
    # Other Colors
    DEFAULT:  str = '\033[0m'
    WAIT:     str = '\033[94;7m' 
    TIMER:    str = '\033[96;7m'

    # Style Properties
    LONGEST_DESC: int = 8
    
    def style_level(level: str) -> str:
        '''
        Apply style to specified log ``level``
        
        Args:
            level: Log level to applying styling to
        Returns:
            String with styling applied to log level
        '''

        return f'{getattr(Style, level)}{level:>{Style.LONGEST_DESC}} {Style.DEFAULT}|'

    def style_name(name: str) -> str:
        '''
        Apply italics style to specified function/method ``name``
        
        Args:
            name: Function/method name to apply styling to
        Returns:
            String with styling applied to function/method name
        '''
        return f'{Style.ITALICS}{name}{Style.DEFAULT}'