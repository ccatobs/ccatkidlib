from pathlib import Path
import sys

# Local Imports
sys.path.append(str(Path(__file__).parent / '..' / '..' / 'rfsoc'))
import rfsoc_io
import pair


class Sweep:
    '''
    Class representing a sweep over a range of frequencies.
    '''

    def __init__(self, com_to, analysis_cfg = str(Path(__file__).parent / 'analysis_config.yaml'), **kwargs):
        # Define sweep attributes
        # -----------------------
        self.bid, self.drid = com_to.split('.') # Baard and drone sweep was taken with

        # Sweep frequency and complex S21 data
        self.freqs = []
        self.s21z  = []

        self.sweep_path = None # File path of sweep
        self.analysis_cfg = rfsoc_io.load_config(analysis_cfg) # File path of analysis config

        # Parse key word arguments
        for key, value in kwargs.items():
            if key == 'sweep_path':
                self.sweep_path = value
            elif key == 'freqs':
                freqs = value
            elif key == 's21z':
                s21z = value

        # If full sweep path is not provided, find sweep data file based on timestamp and (optional) file path parts
        # ----------------------------------------------------------------------------------------------------------
        if self.sweep_path is None:
            # Find sweep data file using 
            data_type  = None
            timestamp  = None

            # Parse sweep data file part key word arguments
            # ---------------------------------------------
            root_data_dir = self.analysis_cfg['data_load']['root_data_dir']
            data_dir = '**'
            date = '**'
            sess_id = '**'

            for key, value in kwargs.items():
                if key == 'data_type':
                    data_type = value
                elif key == 'timestamp':
                    timestamp = value
                elif key == 'root_data_dir':
                    root_data_dir = value
                elif key == 'data_dir':
                    data_dir = value
                elif key == 'date':
                    date = value
                elif key == 'sess_id':
                    sess_id = value

            # Ensure that timestamp and type of sweep are provided to uniquely find sweep data file
            assert (timestamp is not None and data_type is not None), "Need to provide either the full path to the sweep or the sweep timestamp and type ('vna' or 'targ')!"
            
            # Try to find sweep data file using given information
            try:
                self.sweep_path = pair.get_data_file(com_to, timestamp, data_dir = data_dir, date = date, sess_id = sess_id, data_type = data_type, root_data_dir=root_data_dir)[0]
            except:
                raise FileNotFoundError(f'Could not find {data_type} file for board {self.bid}, drone {self.drid} with timestamp {timestamp}! Check that all optional file path segments are correct!')
        
        # Get io, ext, and drone configs associated with the sweep data file
        self.sweep_configs = pair.get_config(self.sweep_path, all_cfg=False)

