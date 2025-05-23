import gc
from pathlib import Path
from numba import njit
import sys
import numpy as np
import pandas as pd

# Local Imports
import ccatkidlib.rfsoc_io as rfsoc_io
import ccatkidlib.utils as utils
import ccatkidlib.analysis.pair as pair
import ccatkidlib.analysis.plot_utils as putils
from ccatkidlib.utils import method_timer


class Data:
    '''
    Class representing a RFSoC output data file (VNA sweep, target sweep, or timestream)
    '''
    def __init__(self, com_to, analysis_cfg = str(Path(__file__).parent / 'analysis_config.yaml'), **kwargs):
        # Define data attributes
        # -----------------------
        self.bid, self.drid = com_to.split('.') # Baard and drone data was taken with

        self.data = None

        self.data_path = None # Path of data file
        self.analysis_cfg, self.plot_cfg = rfsoc_io.load_config(analysis_cfg) # File path of analysis config

        # Parse key word arguments
        for key, value in kwargs.items():
            if key == 'data_path':
                self.data_path = value

        # If full data path is not provided, find data file based on timestamp and (optional) file path parts
        # ----------------------------------------------------------------------------------------------------------
        if self.data_path is None:
            # Find sweep data file using 
            data_type  = None
            timestamp  = None

            # Parse data file part key word arguments
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

            # Ensure that timestamp and type of data are provided to uniquely find data file
            assert (timestamp is not None and data_type is not None), "Need to provide either the full path to the data file or the data file timestamp (and the data file data type ('targ', 'vna', or 'timestream')!"
            
            # Try to find data file using given information
            try:
                self.data_path = pair.get_data_file(com_to, timestamp, data_dir = data_dir, date = date, sess_id = sess_id, data_type = data_type, root_data_dir=root_data_dir)
                self.timestamp = timestamp
            except:
                raise FileNotFoundError(f'Could not find {data_type} file for board {self.bid}, drone {self.drid} with timestamp {timestamp}! Check that all optional file path segments are correct!')
        else:
            self.timestamp = pair.get_timestamp(self.data_path)
        
        if not isinstance(self.data_path, list): self.data_path=[data_path]
        
        # Get io, ext, and drone configs associated with the sweep data file
        # ------------------------------------------------------------------
        self.configs = pair.get_config(self.data_path[0], all_cfg=False)
        self.io_cfg = None
        self.ext_cfg = None
        self.drone_cfg = None

        # Initialize comb attribute
        # --------------------------
        self.comb = None

    ##############################
    # Data Getter/Setter Methods # 
    ##############################

    #@method_timer
    def transform(self, name, func, res_num = None):
        res_slice = self._res_slice(res_num)
        res_slice_idx = pd.Index(res_slice)
        sort_levels = []
        if name in self.data.columns.levels[0]:
            res_slice_idx = res_slice_idx.difference(self.data[name].columns)
        else:
            sort_levels.append(0)
        sort_levels.append(1)
        res_slice_idx = list(res_slice_idx)
        if not len(res_slice_idx) == 0:
            transformed_data = func(res_num, res_slice_idx)
            transformed_data.columns = pd.MultiIndex.from_product([[name], res_slice_idx]) 
            self.data = pd.concat([self.data, transformed_data], axis=1).sort_index(axis=1, level=sort_levels)

        return self.data.loc[:, (name, res_slice)][name]

    def I(self, res_num = None, res_slice = None):
        if res_slice is None and not res_num is None: res_slice = self._res_slice(res_num)
        return self.data.loc[:, ('I', res_slice)]['I'] if res_slice is not None else self.data['I']
    
    def Q(self, res_num = None, res_slice = None):
        if res_slice is None and not res_num is None: res_slice = self._res_slice(res_num)
        return self.data.loc[:, ('Q', res_slice)]['Q'] if res_slice is not None else self.data['Q']

    def phase(self, res_num = None):
        @njit
        def _calc_phase(I, Q):
            return np.arctan2(Q, I)

        def _phase(res_num, res_slice):
            I = self.I(res_num, res_slice).to_numpy()
            Q = self.Q(res_num, res_slice).to_numpy()
            #return pd.DataFrame(_calc_phase(I, Q))
            return np.arctan2(self.Q(res_num, res_slice), self.I(res_num, res_slice))
        
        return self.transform('Phase', _phase, res_num = res_num)

    def mag(self, res_num = None):
        def _mag(res_num, res_slice):
            return np.sqrt(self.I(res_num, res_slice)**2 + self.Q(res_num, res_slice)**2)
        
        return self.transform('Magnitude', _mag, res_num = res_num)
 
    def resonator(self, res_num = None):
        return self.data.loc[:, (slice(None), self._res_slice(res_num))]
    
    #############################
    # Auxiliary Loading Methods #
    #############################
    
    def _load_cfg(self, id):
        '''
        Load io, ext, or drone cfg file corresponding to data file. 
        '''
        cfg = None
        for i, config_path in enumerate(self.configs):
            if id in str(config_path): 
                cfg = rfsoc_io.load_config(config_path)
                self.configs.pop(i)
                break
        return cfg

    def _load_comb(self):
        '''
        Load comb frequencies, powers, and phases
        '''
        comb = {'tone_freqs': None, 'tone_powers': None, 'tone_phis': None}
        for key in comb.keys():
            value = self.drone_cfg['tones'][key]
            if isinstance(value, list):
                value = value.real
            else:
                try:
                    value = np.load(value).real
                except:
                    value = None
            comb[key] = value
        return comb
    
    def _res_slice(self, res_num):
        if res_num is not None:
            if isinstance(res_num, int): res_num = [res_num]
        else:
            res_num = self.res_num if self.res_num is not None else range(self.drone_cfg['tones']['num_tones'])
        return [f'R_{res:04d}' for res in res_num]


    #################
    # Magic Methods #
    #################
    def __getattribute__(self, name):
        if name == 'io_cfg':
            if super().__getattribute__("io_cfg") is None: self.io_cfg = self._load_cfg('_io_')
            gc.collect()
        elif name == 'ext_cfg':
            if super().__getattribute__("ext_cfg") is None: self.ext_cfg = self._load_cfg('_ext_')
            gc.collect()
        elif name == 'drone_cfg':
            if super().__getattribute__("drone_cfg") is None: self.drone_cfg = self._load_cfg('_drone_')
            gc.collect()
        elif name == 'comb':
            if super().__getattribute__("comb") is None: self.comb = self._load_comb()
            gc.collect()

        return super().__getattribute__(name)