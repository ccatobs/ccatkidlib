from pathlib import Path

# Local Imports
import ccatkidlib.rfsoc_io as rfsoc_io
import ccatkidlib.utils as utils
import ccatkidlib.analysis.pair as pair
import ccatkidlib.analysis.plot_utils as putils

class Timestream():
    '''
    Class representing a timestream taken with a Radio Frequency System on a Chip
    '''

    def __init__(self, com_to, res_num = None, analysis_cfg = str(Path(__file__).parent / 'analysis_config.yaml'), **kwargs):
        # Define timestream attributes
        # -----------------------
        self.bid, self.drid = com_to.split('.') # Baard and drone timestream was taken with

        self.res_num = res_num # Resonators timestream correspond to

        self.timestream_path = None
        self.analysis_cfg, self.plot_cfg = rfsoc_io.load_cfg(analysis_cfg)

        for key, value in kwargs.items():
            if key == 'timestream_path':
                self.timestream_path = value

        if self.timestream_path is None:
            # Find timstream data file 
            timestamp = None
            data_type = 'timestream'

            # Parse sweep data file part key word arguments
            # ---------------------------------------------
            root_data_dir = self.analysis_cfg['data_load']['root_data_dir']
            data_dir = '**'
            date = '**'
            sess_id = '**'

            for key, value in kwargs.items():
                if key == 'timestamp':
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
            assert (timestamp is not None), "Need to provide either the full path to the timestream or the timestream timestamp!"
            
            # Try to find sweep data file using given information
            try:
                self.timstream_path = pair.get_data_file(com_to, timestamp, data_dir = data_dir, date = date, sess_id = sess_id, data_type = data_type, root_data_dir=root_data_dir)[0]
                self.timestamp = timestamp
            except:
                raise FileNotFoundError(f'Could not find {data_type} file for board {self.bid}, drone {self.drid} with timestamp {timestamp}! Check that all optional file path segments are correct!')
        else:
            self.timestamp = pair.get_timestamp(self.sweep_path)
        # Get io, ext, and drone configs associated with the sweep data file
        self.sweep_configs = pair.get_config(self.sweep_path, all_cfg=False)
        self.io_cfg = None
        self.ext_cfg = None
        self.drone_cfg = None

    def _load_timestream():
        return




    #################
    # Magic Methods #
    #################

    def __getattribute__(self, name):
        if name == 'freqs' or name == 's21z':
            if super().__getattribute__("freqs")  is None: self.freqs, self.s21z = self._load_sweep()
        elif name == 'res_freqs':
            if super().__getattribute__("res_freqs") is None: self.res_freqs = self._load_res_freqs()
        elif name == 'res_s21z':
            if super().__getattribute__("res_s21z") is None: self.res_s21z = self._get_res_s21z()
        elif name == 'io_cfg':
            if super().__getattribute__("io_cfg") is None: self.io_cfg = self._load_cfg('_io_')
        elif name == 'ext_cfg':
            if super().__getattribute__("ext_cfg") is None: self.ext_cfg = self._load_cfg('_ext_')
        elif name == 'drone_cfg':
            if super().__getattribute__("drone_cfg") is None: self.drone_cfg = self._load_cfg('_drone_')

        return super().__getattribute__(name)
