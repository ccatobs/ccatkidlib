from dataclasses import dataclass

@dataclass
class Style:
    # Font styles
    ITALICS:  str = '\033[3m'
    INVERT:   str = '\033[7m'

    # Logging Colors
    PCS:      str = '\033[92m'
    DEBUG:    str = '\033[34m'
    INFO:     str = '\033[32m'
    HEADER:   str = '\033[95m'
    FOOTER:   str = '\033[95m'
    WARNING:  str = '\033[93m'
    ERROR:    str = '\033[31m'
    CRITICAL: str = '\033[91m' 
    
    # Other Colors
    DEFAULT:  str = '\033[0m'
    WAIT:     str = '\033[94;7m' 
    TIMER:    str = '\033[96;7m'

    # Style Properties
    LONGEST_DESC: int = 8
    
    def log_begin(self, desc, style):
        return f'{style}{desc:>{self.LONGEST_DESC}} {self.DEFAULT}|'

    def func_name(self, name):
        return f'{self.ITALICS}{name}{self.DEFAULT}'