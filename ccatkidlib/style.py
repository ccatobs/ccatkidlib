from dataclasses import dataclass, field
import collections

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