from sweep import Sweep
from pathlib import Path
import sys

# Local Imports
sys.path.append(str(Path(__file__).parent / '..' / '..' / 'rfsoc'))
import rfsoc_io
import pair



class VNA(Sweep):
    '''
    Class representing a vector network analyzer (VNA) esque sweep taken with a Radio Frequency System on a Chip (RFSoC). 
    Subclass of the Sweep class.
    '''

    def __init__(self, com_to, analysis_cfg=str(Path(__file__).parent / 'analysis_config.yaml'), **kwargs):
        super().__init__(com_to, analysis_cfg, **kwargs)
    


                
                

