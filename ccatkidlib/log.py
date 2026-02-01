import collections
import logging

import tqdm.contrib.logging as tqdm_logging

from logging.handlers import RotatingFileHandler
from dataclasses import dataclass, field
from functools import partial, partialmethod, wraps
from tqdm import tqdm

def setup_logging(log_path, file_level, terminal_level, max_file_size = 100, name = __name__):
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

def log(level: str, msg: str, *args, name: str = __name__) -> None:
    '''
    Log message and print message to terminal. 

    Args:
        level (str) : Level of message at which to log (One of: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        msg   (str) : Message to log
        *args: Additional arguments to the ``logger.log`` method for message formatting

        name  (str, optional) : Name of logger
    '''
    # Get logger
    logger = logging.getLogger(name)

    # Try logging message
    # -------------------
    try:
        log_level = logging.getLevelName(level)
        style = Style()

        msg = f'{style.log_begin(level, getattr(style, level))} {msg}'
        logger.log(log_level, msg, *args) if not len(args) == 0 else logger.log(log_level, msg)
    except Exception as e:
        # Log error message
        logger.log(logging.ERROR, 'Failed to log message %s with error %s!', msg, e)

def wait(t_sec: float, desc: str = "") -> None:
    '''
    Wait for ``t_sec`` seconds with a progress bar

    Parameters:
        t_sec (int) : Number of seconds to wait
        desc (str, optional): Description to add to progress bar. Defaults to ""
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
            log.log('HEADER', "Executing %s...", fmt)
            rtn = func(self, *args, **kwargs)
            log.log('FOOTER', "%s executed successfully!", fmt)
            return rtn
        except Exception as e:
            import traceback 
            # Print error traceback if func failed to execute and exit out of program
            log.log('CRITICAL', "TERMINATING PROGRAM -- %s failed to execute with error:\n%s", fmt, traceback.format_exc())
            sys.exit()
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

    # Create stack for changing header/footer colors
    HEADER_STACK: collections.deque = field(default_factory=collections.deque)

    # Style Properties
    LONGEST_DESC: int = 8
    
    def log_begin(self, level, style):
        return f'{style}{level:>{self.LONGEST_DESC}} {self.DEFAULT}|'

    def func_name(self, name):
        return f'{self.ITALICS}{name}{self.DEFAULT}'

    def __getattribute__(self, name):
        try:
            if name == 'HEADER':
                curr_header = super().__getattribute__("HEADER")
                self.HEADER_STACK.append(curr_header)

                styles = curr_header.split(';')
                color = styles[-1][:-1]
                Style.HEADER = f"{';'.join(styles[:-1])};{int(color)-1}m"
                return curr_header
            elif name == 'FOOTER':
                curr_footer = self.HEADER_STACK.pop()
                Style.HEADER = curr_footer
                return curr_footer
        except Exception as e:
            #print(e)
            pass
        return super().__getattribute__(name)