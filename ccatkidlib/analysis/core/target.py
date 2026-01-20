import gc
import sys
import numpy as np
import polars as pl
from numba import njit

from functools import cached_property
from collections.abc import Iterable
from pathlib import Path

from bokeh.layouts import layout
from bokeh.io import show
from bokeh.plotting import curdoc


# Local Imports
import ccatkidlib.rfsoc_io as rfsoc_io
import ccatkidlib.utils as utils
import ccatkidlib.analysis.utils.pair as pair
import ccatkidlib.analysis.utils.dataframe as ccat_df

from ccatkidlib.analysis.core.sweep import Sweep
from ccatkidlib.analysis.fit.fit import linear_fit


class Target(Sweep):
    '''Class representing a target sweep taken with a Radio Frequency System on a Chip (RFSoC)
    
    Subclasses Sweep.
    '''

    def __init__(self, com_to: str, tones: int | list[int] | None = None, noise_tones: int | list[int] | None = None, analysis_cfg: str = str(Path(__file__).parents[1] / 'analysis_config.yaml'), **kwargs):
        '''Subclass of Sweep with additional arguments

        Args:
            tones (int | list[int] | None, optional): Which tones to load. None for loading all data without splitting into individual tones. -1 for all data split into individual tones. Defaults to None
        '''
        kwargs['data_type'] = 'targ'
        super().__init__(com_to, analysis_cfg, **kwargs)
        
        # Parse 'tone' argument specifying which tones should be loaded
        # -------------------------------------------------------------
        if isinstance(tones, int): 
            if tones >= 0: 
                tones = [tones]
            else:
                tones = list(range(self.num_tones))
        if tones is None or (isinstance(tones, Iterable) and all([isinstance(tone, int) for tone in tones])):
            self.tones = tones
        else:
            error = f"Invalid type {type(tones)} for argument 'tones'. Should be int, list[int], or None."
            rfsoc_io.send_msg('CRITICAL', error)
            raise ValueError(error)
        
        # Define list of noise tones
        # --------------------------
        if noise_tones is not None:
            if isinstance(noise_tones, int): 
                noise_tones = [noise_tones]
            elif not isinstance(noise_tones, Iterable) or not all([isinstance(noise_tone, int) for noise_tone in noise_tones]):
                noise_tones = None
                rfsoc_io.send_msg('CRITICAL', f"Invalid type {type(noise_tones)} for argument 'noise_tones'. Should be int, list[int], or None.")
        else:
            noise_tones = utils.dict_get(self.drone_cfg, ['tones', 'noise_tones'])
        self.noise_tones = noise_tones
        
        self._properties = {f'det_{tone:0{self.padding}d}': {} for tone in self.tones} if tones is not None else {}
        self._properties_df = pl.DataFrame({'det': self.tones}).with_columns(pl.lit(f'{self.bid}.{self.drid}').alias('com_to'),
                                                                             pl.lit(self.timestamp).alias('timestamp'))

    # ================ #
    # Analysis Methods #
    # ================ #

    #==========================#
    # Lazily Loaded Attributes #
    #==========================#
    
    @property
    def data(self):
        if self._data is None:
            data = super().data # Get DataFrame of sweep data 
            tones = self.tones # Define local variable since it will be used often

            if tones is not None: # Run if a specific tone(s) is specified
                data = data.to_numpy().T
                try:
                    sweep_steps = self.drone_cfg['tones']['sweep_steps']
                except KeyError:
                    sweep_steps = self.drone_cfg['tones']['N_step']    

                data = data[1:].reshape((3, -1, sweep_steps))
                num_tones = len(tones)

                try:
                    fs, Is, Qs = [None]*num_tones, [None]*num_tones, [None]*num_tones
                    for i, t in enumerate(tones):
                        f, I, Q = data[:, t, :]
                        fs[i] = f
                        Is[i] = I
                        Qs[i] = Q
                    fs, Is, Qs = np.array(fs), np.array(Is), np.array(Qs)
                except Exception as e:
                    fs, Is, Qs = [None]*num_tones, [None]*num_tones, [None]*num_tones
                    rfsoc_io.send_msg('ERROR', 'Failed to reshape data array with error %s.', e)

                data_dict = {'sample': range(sweep_steps)}
                for t, f, I, Q in zip(tones, fs, Is, Qs):
                    data_dict[(f'f_{t:0{self.padding}d}')] = f
                    data_dict[(f'I_{t:0{self.padding}d}')] = I
                    data_dict[(f'Q_{t:0{self.padding}d}')] = Q
                df = pl.DataFrame(data_dict)
            else:
                df = data
            self._data = df 
        return self._data

    @data.setter
    def data(self, value: pl.lazyframe.frame.LazyFrame | None): 
        if value is None or isinstance(value, pl.dataframe.frame.DataFrame): 
            self._data = value
        else:
            rfsoc_io.send_msg('ERROR', 'Cannot set data with type %s. Must be a Polars DataFrame!')
    
    @property
    def properties(self):
        if self.tones is None: return self._properties_df
        
        # Reshape properties dictionary to have resonator properties as primary keys
        new_dict = {'det': []}

        props_dict = self._properties
        self._properties = {f'det_{tone:0{self.padding}d}': {} for tone in self.tones}

        all_props = set([prop for props in props_dict.values() for prop in props.keys()])
        if len(all_props) == 0: return self._properties_df
        
        for det, props in props_dict.items():
            new_dict['det'].append(int(det.split('_')[-1]))
            for prop in all_props:
                curr = new_dict.get(prop, [])
                value = props.get(prop, None)
                if curr: 
                    curr.append(value)
                else:
                    new_dict[prop] = [value]

        new_df = pl.DataFrame(new_dict)
        shared_cols = set(self._properties_df.columns) & set(new_df.columns) - {'det'}
        self._properties_df = ccat_df.coalesce_join(self._properties_df, new_df, 'det', shared_cols)
        return self._properties_df
    
    @properties.setter
    def properties(self, value):
        if isinstance(value, pl.DataFrame):
            self._properties_df = value

    @cached_property
    def cable_delay(self) -> dict | None:
        '''Get the cable delay of the RF chain using the phase data

        Returns:
            float: The cable delay in nanoseconds
        '''

        @njit(cache=True)
        def _calc_cable_delay(f, phase):
            window = int(0.1*len(f))
            cable_delay_low, _ = linear_fit(f[:window], phase[:window])
            cable_delay_high, _ = linear_fit(f[-window:], phase[-window:])
            cable_delay = (cable_delay_high + cable_delay_low)/2
            return cable_delay*1e9/(2*np.pi)

        tones = self.tones
        if tones is None: 
            cable_delay = None
        else:
            self.phase()
            cable_delay = {f'det_{tone:0{self.padding}d}': self.data.select(pl.struct([f'f_{tone:0{self.padding}d}', f'phase_{tone:0{self.padding}d}'])
                                                       .map_batches(lambda arrs: _calc_cable_delay(arrs.struct.field(f'f_{tone:0{self.padding}d}').to_numpy(), arrs.struct.field(f'phase_{tone:0{self.padding}d}').to_numpy()),
                                                       returns_scalar=True, return_dtype=pl.Float64,)).item() for tone in tones}
        return cable_delay
    
    def join(self, other, in_place=False):
        new_data = super().join(other, in_place=in_place)
        left_prop, right_prop = self.properties, other.properties
        new_data._properties_df = pl.concat([left_prop, right_prop], how='diagonal').with_columns(pl.Series('det', new_data.tones))
        return new_data
