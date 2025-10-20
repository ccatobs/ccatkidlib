''' Module for analyzing kinetic inductance detector (KID) composite data

Authors:
    - Darshan Patel <dp649@cornell.edu>

TODO:
    - Change data args to string enums
'''

import os
import sys

import numpy as np
import polars as pl
import pathlib
import concurrent.futures
import lmfit

from collections.abc import Iterable
from pathlib import Path
from functools import cached_property

# local imports
import ccatkidlib
import ccatkidlib.rfsoc_io as rfsoc_io
import ccatkidlib.analysis.utils.pair as pair
import ccatkidlib.analysis.fit.fit as ccat_fit
import ccatkidlib.analysis.utils.mp as ccat_mp

from ccatkidlib.analysis.core.timestream import Timestream
from ccatkidlib.analysis.core.vna import VNA
from ccatkidlib.analysis.core.target import Target

# Plotting functions
import holoviews as hv
import hvplot.polars
from holoviews import opts

class Detector:
    '''Class representing kinetic inductance detectors (KIDs). Used for KID analyses requiring fitting and/or multiple types of data files (e.g., timestream and target sweep data).

    Attributes:
        bid  (str): RFSoC board that took the detector data
        drid (str): RFSoC drone that took the detector data
        dets (list[int]): List of detectors

        stream (Timestream | None): ``Timestream`` object of detector timestream
        targ (Target): ``Target`` object of detector target sweep
        vna (VNA | None): ``VNA`` object of detector VNA sweep

        analysis_cfg (dict): Config file with parameters used for data analysis
        viz_cfg (dict): Config file with paramateres used for data visualization

        cable_delay (float | None): Cable delay of the network (in nanoseconds)
        properties  (polars.DataFrame): Polars dataframe with detector properties
    '''

    def __init__(self, 
                 com_to: str,
                 analysis_cfg: str = str(Path(__file__).parents[1] / 'analysis_config.yaml'),
                 dets: int | list[int] = -1,
                 noise_tones: int | list[int] | None = None,
                 cable_delay: float | None = None,
                 stream: Timestream | None = None, stream_path: str | pathlib.PosixPath | list[str] | list[pathlib.PosixPath] | None = None, stream_timestamp: int | str | None = None,
                 targ: Target = None, targ_path: str | pathlib.PosixPath | None = None, targ_timestamp: int | str | None = None,
                 vna: VNA = None, vna_path: str | pathlib.PosixPath | None = None, vna_timestamp: int | str | None = None,
                 **kwargs):
        '''
        Constructor for Detector. Creates *ccatkidlib* data objects (``VNA``, ``Target``, ``Timsetream``)

        Note:
            - A Detector object can be initialized without a ``Timestream`` object but must **always** have a ``Target`` object.
            - If only the information to load a timestream is provided, an attempt will be made to find the target sweep file corresponding to the timestream
        
        Args:
        
        '''

        # Create Timestream, Target, and VNA data objects based provided arguments
        # ------------------------------------------------------------------------
        if not isinstance(stream, Timestream): stream = Detector._load_data(Timestream, com_to, analysis_cfg, dets, noise_tones, stream_timestamp, stream_path, **kwargs)
        if not isinstance(targ, Target): targ = Detector._load_data(Target, com_to, analysis_cfg, dets, noise_tones, targ_timestamp, targ_path, **kwargs)
        if not isinstance(vna, VNA): vna = Detector._load_data(VNA, com_to, analysis_cfg, None, None, vna_timestamp, vna_path, **kwargs)
   
        # Must have a sweep to do meaningful data analysis
        # ------------------------------------------------
        if not isinstance(targ, Target):
            if isinstance(stream, Timestream): # If timestream provided, try to find associated sweep
                self.dets = stream.tones
                vna_path, targ_path = pair.get_sweep(stream.data_path[0], **kwargs)

                # Load found sweep
                if  Path(vna_path).exists(): vna = Detector._load_data(VNA, com_to, analysis_cfg, None, None, vna_timestamp, vna_path, **kwargs)
                if Path(targ_path).exists(): targ = Detector._load_data(Target, com_to, analysis_cfg, dets, noise_tones, targ_timestamp, targ_path, **kwargs)

                if not isinstance(targ, Target): # and not isinstance(vna, VNA):
                    error = 'Failed to find target sweep associated with timestream. If there is no target sweep, create a Timestream object instead.'
                    rfsoc_io.send_msg('CRITICAL', error)
                    raise RuntimeError(error)

            else: # Error of no sweep or timestream provided
                error = 'A timestream, target sweep or both need to be specified!'
                rfsoc_io.send_msg('CRITICAL', error)
                raise RuntimeError(error)
        else:
            self.dets = targ.tones
            if vna is None: 
                vna_path, _ = pair.get_sweep(targ.data_path[0], **kwargs)
                if Path(vna_path).exists(): vna = Detector._load_data(VNA, com_to, analysis_cfg, None, None, vna_timestamp, vna_path, **kwargs)

        self.bid, self.drid = com_to.split('.')
        self.analysis_cfg, self.viz_cfg = rfsoc_io.load_config(analysis_cfg)

        self.stream = stream
        self.targ = targ
        self.vna = vna

        # Create internal attributes corresponding to lazily loaded attributes
        # ====================================================================
        self._cable_delay = cable_delay
        self._properties_df = self.targ.comb

    #==========================#
    # Lazily Loaded Attributes #
    #==========================#

    @property
    def properties(self):
        '''
        
        '''
        def _merge_properties(new_properties_df):
            shared_cols = (set(self._properties_df.columns) & set(new_properties_df.columns)) - {'det'}
            self._properties_df = self._properties_df.drop(list(shared_cols))
            self._properties_df = self._properties_df.join(new_properties_df, on='det', how='full', coalesce=True)
        
        if self.targ._properties_df is not None: _merge_properties(self.targ.properties)
        if self.stream is not None and self.stream._properties_df is not None: _merge_properties(self.stream.properties)
        
        return self._properties_df
    
    @properties.setter
    def properties(self, value):
        if isinstance(value, pl.DataFrame):
            self._properties_df = value

    @cached_property
    def cable_delay(self):
        self._cable_delay = self.vna.cable_delay if self._cable_delay is None and isinstance(self.vna, VNA) else self._cable_delay
        self.properties

        # Get cable delays for individual detectors using target sweep data. The target sweep cable delays tend to be too large so average with the overall network cable delay 
        self.targ._properties = {det: {'det_cable_delay': 0.4*delay + 0.6*self._cable_delay} for det, delay in self.targ.cable_delay.items()}
        
        # Replace cable delays that are far from the overall network cable delay with the network cable delay
        threshold = pl.lit(100) # TODO: Make this accessible
        self._properties_df = (self.properties.lazy().with_columns(pl.when((pl.col('det_cable_delay') - self._cable_delay).abs() > threshold)
                                                                     .then(pl.lit(self._cable_delay))
                                                                     .otherwise(pl.col('det_cable_delay'))
                                                                    .alias('det_cable_delay'))
                                              .with_columns(pl.lit(self._cable_delay).alias('network_cable_delay')).collect())
        return self._cable_delay

    @cached_property
    def fit_dir(self):
        return self.analysis_cfg['file_paths']['fit_dir']

    #=====================#
    # Data Getter Methods #
    #=====================#

    def nonlinear_fit(self, 
                      nonlinear: bool = False, 
                      asymm: bool = False, 
                      fix_cable: bool = False, 
                      fix_thetaQ: bool = False,
                      save_model_result: bool = False,
                      include: int | list[int] | None = None, 
                      exclude: int | list[int] | None = None, 
                      recalc: bool = False, 
                      max_workers: int = 1) -> pl.DataFrame:
        '''
        Fit target sweep using complex data (*I + iQ*)

        '''
        
        if not self.fit_dir in sys.path: sys.path.append(self.fit_dir)
        import resonator_model_v3
        globals()['resonator_model_v3'] = resonator_model_v3
        
        col_name = ['f', 'I', 'Q', 'nonlinear_fit']
        
        args = [[self, nonlinear, asymm, fix_cable, fix_thetaQ, ccat_mp.check_max_workers(max_workers), save_model_result]]
        self.targ.transform(Detector.calc_nonlinear_fit, *args, include=include, exclude=exclude, recalc = recalc, col_name = col_name, batch_size=len(self.targ.tones))
        self.targ.data = self.targ._unnest('struct_' + col_name[-1])

        # Calculate Q_c and Q_i
        self.properties = (self.properties.lazy().with_columns([((pl.col(f'{col_name[-1]}_Q_e_real')/(pl.col(f'{col_name[-1]}_Q_e_real')**2 + pl.col(f'{col_name[-1]}_Q_e_imag')**2))**-1).alias(f'{col_name[-1]}_Q_c')])
                                                 .with_columns(((1/pl.col(f'{col_name[-1]}_Q') - 1/pl.col(f'{col_name[-1]}_Q_c'))**-1).alias(f'{col_name[-1]}_Q_i'))
                                                 .with_columns((-1e9*pl.col(f'{col_name[-1]}_delay')).alias(f'{col_name[-1]}_delay_ns')).collect()) # Convert cable delay to nanoseconds

        return self.targ.get_data(col_name=col_name[-1], include=include, exclude=exclude)

    def phase_fit(self, 
                  prefix: str | list[str] = 'mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', 
                  circle_fit_col: str = 'circle_fit_unwind_rotate', 
                  nonlinear: bool = False, 
                  method: str = 'least_squares', 
                  params: lmfit.Parameters = None,
                  window: float = 1,
                  save_model_result: bool = True,
                  include: int | list[int] | None = None, 
                  exclude: int | list[int] | None = None, 
                  recalc: bool = False, 
                  max_workers: int = 1) -> pl.DataFrame:
        ''' 
        Fit target sweep using phase data (*arctan(Q/I)*)
        
        
        '''
        
        col_name = ['f', 'I', 'Q', 'phase', 'phase_fit']

        if isinstance(prefix, str): prefix = [prefix]
        num_prefix = len(prefix)

        if isinstance(params, lmfit.parameter.Parameters) or not isinstance(params, Iterable) or not len(params) == num_prefix: params = [params]
        if not isinstance(nonlinear, Iterable) or not len(nonlinear) == num_prefix: nonlinear = [nonlinear]
        if not isinstance(window, Iterable) or not len(window) == num_prefix: window = [window]
        if isinstance(circle_fit_col, str) or not len(circle_fit_col) == num_prefix: circle_fit_col = [circle_fit_col]

        col_names = [[]]*num_prefix
        for i, pre in enumerate(prefix):
            col_names[i] = [col_name[0]] + [f"{pre}{'_' if pre else ''}{name}" for name in col_name[1:-1]] + [col_name[-1]]
        
        args = [[self, pre, circle, nonlin, method, param, win, ccat_mp.check_max_workers(max_workers), save_model_result] for pre, circle, nonlin, param, win in zip(prefix, circle_fit_col, nonlinear, params, window)]
        self.targ.transform([Detector.calc_phase_fit]*num_prefix, *args, include=include, exclude=exclude, recalc=recalc, col_name = col_names, batch_size=len(self.targ.tones))        
        self.targ.data = self.targ._unnest(['struct_' + col_name[-1] for col_name in col_names])

        # Calculate Q_c, Q_i, and nonlinearity parameter 'a'
        for pre, circle in zip(prefix, circle_fit_col):
            self.properties = (self.properties.lazy().with_columns([(1e-8*(pl.col(f'{col_name[-1]}_{pre}_beta')*(2*pl.col(f'{col_name[-1]}_{pre}_R'))**2)/(pl.col(f'{col_name[-1]}_{pre}_f_0')/pl.col(f'{col_name[-1]}_{pre}_Qr'))).alias(f'{col_name[-1]}_{pre}_a'),
                                                                    (pl.col(f'{col_name[-1]}_{pre}_Qr')*(pl.col(f'{circle}_center_mag') + pl.col(f'{col_name[-1]}_{pre}_R'))/(2*pl.col(f'{col_name[-1]}_{pre}_R'))).alias(f'{col_name[-1]}_{pre}_Q_c')])
                                                    .with_columns(((1/pl.col(f'{col_name[-1]}_{pre}_Qr') - 1/pl.col(f'{col_name[-1]}_{pre}_Q_c'))**-1).alias(f'{col_name[-1]}_{pre}_Q_i')).collect())
        
        return self.targ.get_data(col_name=[col_name[-1] for col_name in col_names], include=include, exclude=exclude)

    def IQ_unwind(self, 
                  prefix: str | list[str] = '',
                  data: str = 'both',
                  delay_col: str = 'cable_delay', 
                  include: int | list[int] | None = None, 
                  exclude: int | list[int] | None = None, 
                  recalc: bool = False) -> pl.DataFrame:
        '''
        Remove cable delay from target sweep and/or timestream I & Q data

        Args:
            prefix (str | list[str]): Prefix of I & Q data to remove cable delay from
            data (str, optional): Which type of data to remove cable delay from. Options are 'targ' or 'timestream'. Defaults to 'targ'.
            delay_col (str): Column in ``properties`` DataFrame with detector cable delays
            include (int | list[int] | None, optional): Detector(s) for which to perform calculation
            exclude (int | list[int] | None, optional): Detector(s) for which not to perform calculation
            recalc (bool): Whether to recalculate if data is already in ``data`` DataFrame. Defaults to False

        Returns:
            polars.Dataframe: Polars DataFrame with unwound I & Q data
        '''
        col_name = ['f', 'I', 'Q', 'unwind']
        if isinstance(prefix, str): prefix = [prefix]
        num_prefix = len(prefix)

        data_objs, fs = self._get_data_obj(data)
        if not data_objs:
            error = f"Invalid data type {data}, must be 'targ', 'timestream', or 'both'."
            rfsoc_io.send_msg('ERROR', error)
            raise ValueError(error)

        # Ensure that the cable delay has been calculated
        if delay_col == 'cable_delay': self.cable_delay
        
        coeff = -2*np.pi*1e-9
        unwind_dfs = []
        for data_obj, f in zip(data_objs, fs):
            angle = ([coeff*delay*tone_freq for tone_freq, delay in zip(self.targ.get_data('f', strict=True).to_numpy().T, self.properties[delay_col].to_numpy().T)] if f is None 
                     else [coeff*delay*tone_freq for tone_freq, delay in zip(f, self.properties[delay_col].to_numpy().T)])
            for pre in prefix:
                data_obj.IQ_rotate(prefix=pre, angle=angle, name=col_name[-1], include = include, exclude=exclude, recalc=recalc)
            unwind_dfs.append(data_obj.get_data(col_name=([f"{col_name[-1]}_rotate_{pre}{'_' if pre else ''}{col_name[1]}" for pre in prefix] +
                                                          [f"{col_name[-1]}_rotate_{pre}{'_' if pre else ''}{col_name[2]}" for pre in prefix]), include=include, exclude=exclude))
        return unwind_dfs

    def IQ_norm(self, 
                prefix: str | list[str] = '', 
                data: str = 'both', 
                norm_col: str = 'cable_nonlinear_fit', 
                include: int | list[int] | None = None, 
                exclude: int | list[int] | None = None, 
                recalc: bool = False) -> pl.DataFrame:
        '''
        Divide out cable baseline from target sweep and/or timestream I & Q data
        
        '''
        
        
        col_name = ['f', 'I', 'Q', 'mag', 'norm']
        if isinstance(prefix, str): prefix = [prefix]
        num_prefix = len(prefix)
        
        data_objs, fs = self._get_data_obj(data)
        if not data_objs:
            error = f"Invalid data type {data}, must be 'targ', 'timestream', or 'both'."
            rfsoc_io.send_msg('ERROR', error)
            raise ValueError(error)

        self.targ.mag(prefix=norm_col, include=include, exclude=exclude, recalc=recalc)
        norm_dfs = []
        for data_obj, f in zip(data_objs, fs):
            cable_mags = [1/norm for norm in self.targ.get_data(f'{norm_col}_{col_name[-2]}', strict=True)] 
            scale = cable_mags if f is None else [self.targ.data.select(pl.col(f'{col_name[0]}_{tone:04d}').alias('f'), cable_mag.alias('cable'))
                                                                .sort((pl.col('f') - tone_freq).abs())
                                                                .select(pl.col('cable'))
                                                                .item(0, 0) for cable_mag, tone_freq, tone in zip(cable_mags, f, self.dets)]
            for pre in prefix:
                data_obj.IQ_scale(prefix=pre, scale=scale, name=col_name[-1], include = include, exclude=exclude, recalc=recalc)
            norm_dfs.append(data_obj.get_data(col_name=([f"{col_name[-1]}_scale_{pre}{'_' if pre else ''}{col_name[1]}" for pre in prefix] +
                                                          [f"{col_name[-1]}_scale_{pre}{'_' if pre else ''}{col_name[2]}" for pre in prefix]), include=include, exclude=exclude))
        return norm_dfs

    def IQ_trim(self, 
                prefix: str | list[str] = '', 
                window: float | list[float] = 1.5, 
                use_fit: bool = False,
                f_0_col: str = 'nonlinear_fit_f_0',
                Q_col: str = 'nonlinear_fit_Q',  
                mean_points: int = 10,
                include: int | list[int] | None = None, 
                exclude: int | list[int] | None = None, 
                recalc: bool = False) -> pl.DataFrame:
        '''
        Trim off-resonance target sweep I & Q data 

        Args:
            prefix:
            window (float | list[float], optional): How many linewidths of data to include around the magnitude minimum. Defaults to 1.5
            use_fit (bool): Whether to use resonant frequency and total quality factor from a fit to determine linewidth. Defaults to False
            f_0_col (str): Name of column in ``properties`` DataFrame with resonant frequencies to use if ``use_fit`` is True.
            Q_col (str): Name of column in ``properties`` DataFrame with total quality factors to use if ``use_fit`` is True.
            mean_points (int): Number of points to average to determine max magnitude of resonator profile. Used if ``use_fit`` is False. Defaults to 10
        Returns:
            return (pl.DataFrame): Polars DataFrame with trimmed I & Q data
        '''
        
        col_name = ['I', 'Q', 'tail']
        if isinstance(prefix, str): prefix = [prefix]
        num_prefix = len(prefix)

        if isinstance(window, (int, float)): window = [window]*num_prefix

        if f_0_col in self.properties.schema and use_fit:
            pass
        else:
            fwhm_col_name = ['sample', 'mag'] # Data columns used for estimating the FWHM

            if recalc or not ('HM_low' in self.properties.schema):
                # Get detector magnitudes and sample numbers and unpivot DataFrame from wide to long format
                mag_df = self.targ.get_data(col_name=fwhm_col_name, strict=True, include=include, exclude=exclude)
                mag_df = (mag_df.unpivot(on=list(set(mag_df.columns) - {fwhm_col_name[0]}),
                                        index=fwhm_col_name[0],
                                        variable_name='det',
                                        value_name=fwhm_col_name[1])
                                .with_columns(pl.col('det').str.strip_prefix(f'{fwhm_col_name[1]}_').cast(pl.Int32))
                                .sort(fwhm_col_name[1], descending=True))

                # Get minimum magnitude values for each detector and corresponding sample numbers
                min_df = (mag_df.filter((pl.col(fwhm_col_name[1]) == pl.col(fwhm_col_name[1]).min()).over('det'))
                                .rename({fwhm_col_name[0]: f'min_{fwhm_col_name[0]}', fwhm_col_name[1]: f'min_{fwhm_col_name[1]}'}))
                if 'HM_mid' in self._properties_df.schema: self._properties_df = self._properties_df.drop('HM_mid')
                self._properties_df = self._properties_df.join(min_df.select('det', pl.col(f'min_{fwhm_col_name[0]}').alias('HM_mid')), on='det', how='full', coalesce=True)
                mag_min_df = (mag_df.join(min_df, on='det', how='left', coalesce=True)
                                    .with_columns((pl.col(fwhm_col_name[0]) < pl.col(f'min_{fwhm_col_name[0]}')).alias('low')))
                # Get mean maximum magnitude values for both the low and high frequency sides of each detector
                max_df = (mag_min_df.group_by(['low', 'det'], maintain_order=True)
                                    .agg(pl.col(fwhm_col_name[1]).head(mean_points).mean())
                                    .pivot(on='low',
                                            index='det',
                                            values=fwhm_col_name[1])
                                    .sort('det')
                                    .rename({'true': f'max_{fwhm_col_name[1]}_low', 'false': f'max_{fwhm_col_name[1]}_high'}))
                min_max_df = (mag_min_df.join(max_df, on='det', how='left', coalesce=True)
                            .with_columns([(pl.col(fwhm_col_name[1]) - ((pl.col(f'min_{fwhm_col_name[1]}').abs() + pl.col(f'max_{fwhm_col_name[1]}_{side}'))/2)).alias(f'HM_{side}') for side in ['low', 'high']]))
                # Get the samples corresponding to the half max on the low and high frequency sides of each detector
                for side in ('low', 'high'):
                    HM_df = (min_max_df.lazy()
                                       .filter(pl.col('low') == ('low' == side))
                                       .sort(f'HM_{side}')
                                       .select('det', pl.col(fwhm_col_name[0]).first().over('det'))
                                       .unique().sort('det').rename({fwhm_col_name[0]: f'HM_{side}'}).collect())
                    if f'HM_{side}' in self._properties_df.schema: self._properties_df = self._properties_df.drop(f'HM_{side}')
                    self._properties_df = self._properties_df.join(HM_df, on='det', how='full', coalesce=True)
            HM_low, HM_mid, HM_high = self.properties.select('HM_low', 'HM_mid', 'HM_high').to_numpy().T

        for pre, win in zip(prefix, window):
            lower_bound = (HM_mid - (HM_mid - HM_low)*win).astype(int)
            upper_bound = (HM_mid + (HM_high - HM_mid)*win).astype(int)
            self.targ.IQ_trim(prefix=pre, lower_bound = lower_bound, upper_bound=upper_bound, name=col_name[-1], include = include, exclude=exclude, recalc=recalc)
        return self.targ.get_data(col_name=([f"{col_name[-1]}_trim_{pre}" for pre in prefix]), include=include, exclude=exclude)

    def IQ_circle_fit(self, 
                      prefix: str | list[str] = 'unwind_rotate', 
                      bounds = None, 
                      loss: str = 'soft_l1', 
                      f_scale: float = 1, 
                      method: str = 'trf',
                      include: int | list[int] | None = None, 
                      exclude: int | list[int] | None = None, 
                      recalc: bool = False, 
                      max_workers=1) -> pl.DataFrame:
        '''
        Fit the target sweep circle in the IQ plane
        
        '''
        col_name = ['I', 'Q', 'circle_fit']

        if isinstance(prefix, str): prefix = [prefix]
        num_prefix = len(prefix)

        col_names = [[]]*num_prefix
        for i, pre in enumerate(prefix):
            col_names[i] = [f"{pre}{'_' if pre else ''}{name}" for name in col_name[:-1]] + [col_name[-1]]
        args = [[self, pre, bounds, loss, f_scale, method, ccat_mp.check_max_workers(max_workers)] for pre in prefix]
        self.targ.transform([Detector.calc_IQ_circle_fit]*num_prefix, *args, include=include, exclude=exclude, recalc=recalc, col_name = col_names, batch_size=len(self.targ.tones))        
        self.targ.data = self.targ._unnest(['struct_' + col_name[-1] for col_name in col_names])
        return self.targ.get_data(col_name=[col_name[-1] for col_name in col_names], include=include, exclude=exclude)

    def IQ_circle_real(self, 
                       prefix: str | list[str] = 'unwind_rotate', 
                       data: str = 'both',
                       loc: str = 'origin',  
                       circle_fit_col='circle_fit_unwind_rotate', 
                       use_fit=True, 
                       include: int | list[int] | None = None, 
                       exclude: int | list[int] | None = None, 
                       recalc: bool = False) -> pl.DataFrame:
        '''
        Rotate and center the target sweep circle in the IQ plane onto the real axis


        '''

        dest_I = None
        if loc == 'origin':
            dest_I = 0

        col_names = ['I', 'Q']
        fit_names = [f'{circle_fit_col}_center_I', f'{circle_fit_col}_center_Q', f'{circle_fit_col}_center_angle', f'{circle_fit_col}_center_mag']

        center_angle = []
        center_mag = []
        if dest_I is None: # Do not transform the circle if no destination provided
            shift = np.zeros(len(self.dets))
            center_angle = np.zeros(len(self.dets))
        else:
            use_mean = True
            if fit_names[0] in self.properties.schema and use_fit:
                #TODO: Use pl.when to avoid recalculating every time
                self._properties_df = self._properties_df.with_columns((pl.arctan2(pl.col(fit_names[1]), pl.col(fit_names[0]))).alias(fit_names[2]),
                                                                       ((pl.col(fit_names[0])**2 + pl.col(fit_names[1])**2).sqrt()).alias(fit_names[3]))
                center_angle, center_mag = self._properties_df.select(pl.col(fit_names[2:])).to_numpy().T
                if not center_angle is None and not center_mag is None: use_mean = False

            if use_mean:
                # TODO: Use add_data_to_properties here
                center_I = np.array([self.targ.data.select(pl.col(f"{prefix}{'_' if prefix else ''}{col_names[0]}_{tone:04d}").mean()).item() for tone in self.targ.tones]) 
                center_Q = np.array([self.targ.data.select(pl.col(f"{prefix}{'_' if prefix else ''}{col_names[1]}_{tone:04d}").mean()).item() for tone in self.targ.tones])
                center_angle = np.arctan2(center_Q, center_I)
                center_mag = np.sqrt(center_I**2 + center_Q**2)

            center_angle = np.pi - center_angle        
            shift = center_mag + dest_I

        data_objs, _ = self._get_data_obj(data)
        if not data_objs:
            error = f"Invalid data type {data}, must be 'targ', 'timestream', or 'both'."
            rfsoc_io.send_msg('ERROR', error)
            raise ValueError(error)

        shift_dfs = []
        for data_obj in data_objs:
            data_obj.IQ_rotate(prefix=prefix, angle=center_angle, name=loc, include = include, exclude=exclude, recalc=recalc)
            data_obj.IQ_shift(prefix=f"{loc}{'_' if loc else ''}rotate_" + prefix, shift_I = shift, name=loc, include=include, exclude=exclude, recalc=recalc)
            shift_dfs.append(data_obj.get_data(col_name=f"{loc}{'_' if loc else ''}shift{'_' if loc else ''}{loc}_rotate", include=include, exclude=exclude))
        return shift_dfs

    def IQ_circle_rotate(self, 
                         prefix: str | list[str] = 'origin_shift_origin_rotate_unwind_rotate', 
                         data: str = 'both', 
                         rotation: str ='mismatch', 
                         include: int | list[int] | None = None, 
                         exclude: int | list[int] | None = None, 
                         recalc: bool = False, 
                         **kwargs) -> pl.DataFrame:
        '''
        Rotate the target sweep circle in the IQ plane around its center
        
        '''
        
        col_name = ['I', 'Q', rotation]
        if isinstance(prefix, str): prefix = [prefix]

        angles = []
        for pre in prefix:
            if rotation == 'mismatch':
                mismatch_col_name = f'{pre}_{col_name[-1]}_angle'
                if recalc or not mismatch_col_name in self.properties.schema:
                    # TODO: Optimize to get first and last values before doing evaluation
                    pi = pl.lit(np.pi)
                    mismatch_df = self.targ.data.lazy().select([(pi - pl.arctan2(pl.col(f"{pre}{'_' if pre else ''}{col_name[1]}_{tone:04d}").first(), pl.col(f"{pre}{'_' if pre else ''}{col_name[0]}_{tone:04d}").first() % 2*pi +
                                                                      pl.arctan2(pl.col(f"{pre}{'_' if pre else ''}{col_name[1]}_{tone:04d}").last(), pl.col(f"{pre}{'_' if pre else ''}{col_name[0]}_{tone:04d}").last()) % 2*pi)/2).alias(f'{tone:04d}') for tone in self.targ.tones]).collect()

                    self.add_data_to_properties(mismatch_df, mismatch_col_name)
                mismatch_angle = self._properties_df[mismatch_col_name].to_numpy()
                angles.append(mismatch_angle)
            elif rotation == 'timestream':
                timestream_col_name = f'{pre}_{col_name[-1]}_angle'
                if recalc or not timestream_col_name in self.properties.schema:
                    pi = pl.lit(np.pi)
                    timestream_df = self.stream.data.select([(pl.arctan2(pl.col(f"{pre}{'_' if pre else ''}{col_name[1]}_{tone:04d}").median(),
                                                                         pl.col(f"{pre}{'_' if pre else ''}{col_name[0]}_{tone:04d}").median()) % 2*pi).alias(f'{tone:04d}') if tone in include else pl.lit(0).alias(f'{tone:04d}') for tone in self.stream.tones])
                    
                    self.add_data_to_properties(timestream_df, timestream_col_name)
                
                timestream_angle = self._properties_df[timestream_col_name].to_numpy().T
                angles.append(timestream_angle)
            else:
                error = f"Invalid rotation '{rotation}' specified; Must be one of 'mismatch' or 'timestream'."
                rfsoc_io.send_msg('ERROR', error)
                raise ValueError(error)
        
        data_objs, _ = self._get_data_obj(data)
        if not data_objs:
            error = f"Invalid data type {data}, must be 'targ', 'timestream', or 'both'."
            rfsoc_io.send_msg('ERROR', error)
            raise ValueError(error)

        rotation_dfs = []
        for data_obj in data_objs:
            data_obj.IQ_rotate(prefix=prefix, angle=angles, name=f'{col_name[-1]}', include = include, exclude=exclude, recalc=recalc)
            rotation_dfs.append(data_obj.get_data(col_name=f"{col_name[-1]}_rotate_", include=include, exclude=exclude))
        return rotation_dfs

    def IQ_noise(self,
                 prefix: str | list[str] = 'mismatch_rotate_origin_shift_origin_rotate',
                 use_noise_tones: bool = True,
                 include: int | list[int] | None = None, 
                 exclude: int | list[int] | None = None, 
                 recalc: bool = False) -> pl.DataFrame:
        '''
        Transform timestream data to isolate readout noise. 
        Will shift nearest frequency noise tone onto detector tone if ``use_noise_tones``, otherwise will rotate detector tone by ninety degrees around its center

        Args:
            prefix (str | list[str]): 
            use_noise_tones (bool): Whether to use noise tones. Defaults to True
        '''
        def _get_medians(prefix):
            # TODO: Figure out how to deal with include and exclude
            if recalc or not (f'median_{prefix}{'_' if prefix else ''}{col_name[0]}' in self.properties.schema):
                median_I_df = self.stream.data.select([pl.col(f'{prefix}{'_' if prefix else ''}{col_name[0]}_{tone:04d}').median().alias(f'{tone:04d}') for tone in self.stream.tones])
                median_Q_df = self.stream.data.select([pl.col(f'{prefix}{'_' if prefix else ''}{col_name[1]}_{tone:04d}').median().alias(f'{tone:04d}') for tone in self.stream.tones])

                self.add_data_to_properties(median_I_df, f'median_{prefix}{'_' if prefix else ''}{col_name[0]}')
                self.add_data_to_properties(median_Q_df, f'median_{prefix}{'_' if prefix else ''}{col_name[1]}')
            
            median_I = self.properties[f'median_{prefix}{'_' if prefix else ''}{col_name[0]}'].to_numpy()
            median_Q = self.properties[f'median_{prefix}{'_' if prefix else ''}{col_name[1]}'].to_numpy()
            return median_I, median_Q

        col_name = ['I', 'Q']

        if isinstance(prefix, str): prefix = [prefix]
        num_prefix = len(prefix)
        noise_tones = self.stream.noise_tones

        if use_noise_tones and not noise_tones is None: # Use noise tones
            col_name += ['noise_shift']
            if recalc or not ('closest_noise_tone' in self.properties.schema):
                self._properties_df = self.properties.with_columns([((pl.col('tone_freqs') - pl.col('tone_freqs').abs().gather(tone))/pl.lit(1e6)).alias(f'noise_tone_dist_{tone:04d}') for tone in noise_tones])
                unpivot_cols = set(self._properties_df.select('det', '^noise_tone_dist_\d.*$').columns) - {'det'}
                closest_tones = (self._properties_df.unpivot(on=unpivot_cols,
                                                            index='det',
                                                            variable_name='closest_noise_tone',
                                                            value_name='dist')
                                                    .with_columns(pl.col('closest_noise_tone').str.split('_').list[-1].cast(pl.Int32))
                                                    .filter(pl.col('dist') == pl.col('dist').min().over('det'))
                                                    .sort('det'))['closest_noise_tone']
                self._properties_df = self._properties_df.with_columns(closest_tones)
            closest_tones = self.properties['closest_noise_tone'].to_numpy()
            noise_median_I, noise_median_Q = _get_medians('')

            col_names = [[]]*num_prefix
            median_Is, median_Qs = [[]]*num_prefix, [[]]*num_prefix
            for i, pre in enumerate(prefix):
                col_names[i] = col_name[:-1] + [f'{col_name[-1]}{'_' if pre else ''}{pre}']
                median_Is[i], median_Qs[i] = _get_medians(pre)
            args = [[median_I, median_Q, noise_median_I, noise_median_Q, self.stream.tones, closest_tones] for median_I, median_Q in zip(median_Is, median_Qs)]
            self.stream.transform([Detector.calc_noise_shift]*num_prefix, *args, include=include, exclude=exclude, recalc=recalc, col_name = col_names)
            return self.stream.get_data(col_name=[col_name[-1] for col_name in col_names], include=include, exclude=exclude)
        else: 
            # TODO: A bit slow, figure out why
            col_name += ['noise_rotate']
        
            for pre in prefix:
                median_I, median_Q = _get_medians(pre)

                self.stream.IQ_shift(prefix=pre, shift_I = -1*median_I, shift_Q = -1*median_Q, name='', recalc=recalc)
                self.stream.IQ_rotate(prefix=f'shift_{pre}', angle=pl.lit(np.pi/2), name='noise', recalc=recalc)
                self.stream.IQ_shift(prefix=f'noise_rotate_shift_{pre}', shift_I = median_I, shift_Q = median_Q, name='', recalc=recalc)

                self.stream.data = self.stream.data.with_columns([pl.col(col).alias(col.replace('shift_noise_rotate_shift', col_name[-1])) for col in self.stream.data.select(pl.col('^shift_noise_rotate_shift_.*$')).columns])
            return self.stream.get_data(col_name=[f'{col_name[-1]}{'_' if pre else ''}{pre}' for pre in prefix], include=include, exclude=exclude)

    def phase_spline(self, 
                     prefix: str | list[str] = 'mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', 
                     phase_low: float = -3.14, 
                     phase_up: float = 3.14, 
                     k: int = 3, 
                     include: int | list[int] | None = None, 
                     exclude: int | list[int] | None = None, 
                     recalc: bool = False, 
                     max_workers=1, 
                     **kwargs) -> pl.DataFrame:
        '''Interpolate target sweep phase vs. frequency data and add interpolating splines to ``properties`` attribute

        Args:
            phase_low (float): Lower bound of phase to use for interpolation. Defaults to -pi
            phase_up (float): Upper bound of phase to use for interpolation. Defaults to +pi
            k (int): Degree of polynomials to use for interpolation. Defaults to degree 3.
        '''
        col_name = ['f', 'phase', 'f', 'phase']

        if isinstance(prefix, str): prefix = [prefix]
        num_prefix = len(prefix)

        if not isinstance(phase_low, Iterable) or not len(phase_low) == num_prefix: phase_low = [phase_low]
        if not isinstance(phase_up, Iterable) or not len(phase_up) == num_prefix: phase_up = [phase_up]

        col_names = [[]]*num_prefix
        for i, pre in enumerate(prefix):
            col_names[i] = [col_name[0], f"{pre}{'_' if pre else ''}{col_name[1]}", 'to_' + col_name[-2] + '_spline', f"to_{pre}{'_' if pre else ''}{col_name[-1]}_spline"]
        
        stream_timestamp = self.stream.timestamp
        args = [[self, low, up, k, stream_timestamp, ccat_mp.check_max_workers(max_workers)] for low, up in zip(phase_low, phase_up)]
        self.targ.transform([Detector.calc_phase_spline]*num_prefix, *args, include=include, exclude=exclude, recalc=recalc, col_name = col_names, batch_size=len(self.targ.tones))   
        self.targ.data = self.targ._unnest(['struct_' + col_name[-1] for col_name in col_names])
        return self.targ.get_data(col_name=([col_name[-1] for col_name in col_names] + [col_name[-2] for col_name in col_names]), include=include, exclude=exclude)

    def phase_to_f(self,
                   prefix: str | list[str] = 'mismatch_rotate_origin_shift_origin_rotate_unwind_rotate',
                   spline_col: str = 'mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', 
                   phase_bounds: float = 0.2, k: int = 3, 
                   include: int | list[int] | None = None, 
                   exclude: int | list[int] | None = None, 
                   recalc: bool = False, 
                   max_workers=1, 
                   **kwargs) -> pl.DataFrame:
        '''
        Convert timestream phase data to frequency using target sweep phase vs. frequency interpolating spline

        Args:
            prefix (str | list[str]): Prefix of phase data to convert to frequency
            spline_col (str): Prefix of phase data used to construct spline
            phase_bounds (float): Amount to add to min and max timestream phase to determine phase vs. frequency spline bounds (i.e., ``phase_low = min_stream_phase - phase_bounds`` & ``phase_up = max_stream_phase + phase_bounds``)
            k (int): Order of polynomials to use for phase vs. frequency spline
            include (int | list[int] | None, optional): Detector(s) for which to perform calculation
            exclude (int | list[int] | None, optional): Detector(s) for which not to perform calculation
            recalc (bool): Whether to recalculate if data is already in ``data`` DataFrame. Defaults to False
            max_workers (int): Number of processor cores to use for calculation. Defaults to 1.         
        '''
        
        #TODO: Does not work when include is specified
        col_name = ['phase', 'f']
        if isinstance(prefix, str): prefix = [prefix]
        num_prefix = len(prefix)

        col_names, min_phases, max_phases = [[]]*num_prefix, [[]]*num_prefix, [[]]*num_prefix
        for i, pre in enumerate(prefix):
            prefix_names = [f"{pre}{'_' if pre else ''}{name}" for name in col_name]
            col_names[i] = prefix_names
            min_phases[i] = self.stream.data.select(pl.col(f'^{prefix_names[0]}_.*$').min().name.prefix('min_')).to_numpy()[0] - phase_bounds
            max_phases[i] = self.stream.data.select(pl.col(f'^{prefix_names[0]}_.*$').max().name.prefix('max_')).to_numpy()[0] + phase_bounds
        self.phase_spline(prefix=spline_col, phase_low = min_phases, phase_up = max_phases, k = k, include=include, exclude=exclude, recalc=recalc, max_workers=max_workers, **kwargs)

        args = [[self, spline_col, ccat_mp.check_max_workers(max_workers)]]*num_prefix
        self.stream.transform([Detector.calc_phase_to_f]*num_prefix, *args, include=include, exclude=exclude, recalc=recalc, col_name = col_names, batch_size=len(self.stream.tones))
        self.stream.data = self.stream._unnest(['struct_' + col_name[-1] for col_name in col_names])
        return self.stream.get_data(col_name=[col_name[-1] for col_name in col_names], include=include, exclude=exclude)
    
    def frac_f(self, 
               prefix: str | list[str] = 'mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', 
               f_0 = None, 
               name='', 
               include: int | list[int] | None = None, 
               exclude: int | list[int] | None = None, 
               recalc: bool = False, 
               **kwargs) -> pl.DataFrame:        
        '''
        Convert timestream frequency data to fractional frequency shift 

        '''
        
        col_name = ['f', f"{name}{'_' if name else ''}frac"]
        if isinstance(prefix, str): prefix = [prefix]
        num_prefix = len(prefix)

        if f_0 is None: f_0 = self.properties.select(pl.col('tone_freqs')).to_numpy().T[0]
        if not isinstance(f_0, Iterable) or not len(f_0) == num_prefix: f_0 = [f_0]

        col_names = [[]]*num_prefix
        for i, pre in enumerate(prefix):
            col_names[i] = [f"{pre}{'_' if pre else ''}{col_name[0]}", col_name[-1]]

        args = [[f, self.dets] for f in f_0]
        self.stream.transform([Detector.calc_frac_f]*num_prefix, *args, include=include, exclude=exclude, recalc=recalc, col_name = col_names)        
        return self.stream.get_data(col_name=[f"{col_name[-1]}_{col_name[0]}" for col_name in col_names], include=include, exclude=exclude)

    #==================#
    # Analysis Methods #
    #==================#

    @staticmethod
    def calc_nonlinear_fit(schema, *args, tones: list[int], recalc: bool = False, col_name = ['f', 'I', 'Q', 'nonlinear_fit']):
        ''' Fit using resonator_model_v3
        '''

        def _nonlinear_fit(df):
            struct = df.struct

            self.properties
            results_dict = {}
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {executor.submit(resonator_model_v3.nonlinear_fit,
                                                   struct.field(f_col).to_numpy(),
                                                   struct.field(I_col).to_numpy(),
                                                   struct.field(Q_col).to_numpy(),
                                                   nonlinear=nonlinear,
                                                   asymm=asymm,
                                                   fix_cable=fix_cable,
                                                   fix_thetaQ=fix_thetaQ):  (tone, f_col, I_col, Q_col) for tone, (f_col, I_col, Q_col) in zip(to_calc, batches)}
                
                for future in concurrent.futures.as_completed(future_to_batch):
                    tone, f_col, I_col, Q_col = future_to_batch[future]
                    try:
                        result = future.result()
                        best_fit = result.best_fit
                        cable_fit = resonator_model_v3.fine_s21_model(struct.field(f_col).to_numpy(), result.params, cable=True)
                        best_vals_dict = {f'{col_name[-1]}_{k}': float(v) for k, v in result.best_values.items()}
                        if save_model_result: best_vals_dict[f'{col_name[-1]}_model_result'] = result
                        self.targ._properties[f'det_{tone:04d}'] = best_vals_dict
                    except Exception as e:
                        rfsoc_io.send_msg('WARNING', 'Fit failed for tone %s with exception: %s', tone, e)
                        best_fit = np.zeros(df.len())
                        cable_fit = np.zeros(df.len())
                        self.targ._properties[f'det_{tone:04d}'] = {}
                    results_dict[f'{col_name[-1]}_{I_col}'] = best_fit.real
                    results_dict[f'{col_name[-1]}_{Q_col}'] = best_fit.imag
                    results_dict[f'cable_{col_name[-1]}_{I_col}'] = cable_fit.real
                    results_dict[f'cable_{col_name[-1]}_{Q_col}'] = cable_fit.imag

            df = pl.DataFrame(results_dict)
            return pl.Series(df.select(pl.struct(df.columns)))
            
        if len(args) == 7:
            self, nonlinear, asymm, fix_cable, fix_thetaQ, max_workers, save_model_result = args
        else:
            error = 'nonlinear, asymm, fix_cable, and fix_thetaQ are required arguments.'
            rfsoc_io.send_msg('ERROR', error)
            raise ValueError(error)
        
        return_col = [f'{col_name[-1]}_{col_name[1]}', f'{col_name[-1]}_{col_name[2]}', f'cable_{col_name[-1]}_{col_name[1]}', f'cable_{col_name[-1]}_{col_name[2]}']
        return_type = [pl.Float64, pl.Float64, pl.Float64, pl.Float64]
        expr, to_calc, calc_col, batches = ccat_mp.batch_calc(_nonlinear_fit, tones, col_name, schema, return_col=return_col, return_type=return_type, recalc=recalc)
        return expr

    @staticmethod
    def calc_phase_fit(schema, *args, tones: list[int], recalc: bool = False, col_name = ['f', 'I', 'Q', 'phase', 'phase_fit']):
        def _phase_fit(df):
            struct = df.struct

            R, = self.properties.select(pl.col(f'{circle_col}_R'))       

            results_dict = {}
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {executor.submit(ccat_fit.phase_fit,
                                                   struct.field(f_col).to_numpy(),
                                                   struct.field(phase_col).to_numpy(),
                                                   I = struct.field(I_col).to_numpy(),
                                                   Q = struct.field(Q_col).to_numpy(),
                                                   nonlinear = nonlinear[tones.index(tone)],
                                                   method=method,
                                                   params = params[tones.index(tone)],
                                                   R = R.item(tones.index(tone)),
                                                   window=window[tones.index(tone)]):  (tone, f_col, phase_col) for tone, (f_col, I_col, Q_col, phase_col) in zip(to_calc, batches)}
                
                for future in concurrent.futures.as_completed(future_to_batch):
                    tone, f_col, phase_col = future_to_batch[future]
                    try:
                        result = future.result()
                        best_fit = np.full(df.len(), np.nan)
                        mask = result.mask
                        best_fit[mask] = result.best_fit
                        best_vals_dict = {f'{col_name[-1]}_{prefix}_{k}': float(v) for k, v in result.best_values.items()}
                        init_vals_dict = {f'{col_name[-1]}_{prefix}_init_{k}': float(v) for k, v in result.init_values.items()}
                        if save_model_result: 
                            best_vals_dict[f'{col_name[-1]}_{prefix}_model_result'] = result
                            best_vals_dict[f'{col_name[-1]}_{prefix}_params'] = result.params
                        self.targ._properties[f'det_{tone:04d}'] = best_vals_dict | init_vals_dict
                    except Exception as e:
                        rfsoc_io.send_msg('WARNING', 'Fit failed for tone %s with exception: %s', tone, e)
                        best_fit = np.full(df.len(), np.nan)
                        self.targ._properties[f'det_{tone:04d}'] = {}
                    results_dict[f'{col_name[-1]}_{phase_col}'] = best_fit

            df = pl.DataFrame(results_dict)

            return pl.Series(df.select(pl.struct(df.columns)))
            
        if len(args) == 9:
            self, prefix, circle_col, nonlinear, method, params, window, max_workers, save_model_result = args
            if isinstance(params, lmfit.parameter.Parameters) or not isinstance(params, Iterable): params = len(tones)*[params]
            if isinstance(nonlinear, bool): nonlinear = len(tones)*[nonlinear]
            if isinstance(window, (int, float)): window = len(tones)*[window]
        else:
            error = 'nonlinear, prefix, params, window, max_workers, and save_model_result are required arguments.'
            rfsoc_io.send_msg('ERROR', error)
            raise ValueError(error)
        
        return_col = [f'{col_name[-1]}_{col_name[-2]}']
        return_type = [pl.Float64]
        expr, to_calc, calc_col, batches = ccat_mp.batch_calc(_phase_fit, tones, col_name, schema, return_col=return_col, return_type=return_type, recalc=recalc)
        return expr 

    @staticmethod
    def calc_IQ_circle_fit(schema, *args, tones: list[int], recalc: bool = False, col_name = ['I', 'Q', 'circle_fit']):
        def _circle_fit(df):
            struct = df.struct
            
            self.properties
            angles = np.linspace(0, 2*np.pi, df.len())
            sin = np.sin(angles)
            cos = np.cos(angles)

            property_keys = ['center_I', 'center_Q', 'R', 'A', 'D', 'theta', 'optimality', 'nfev', 'njev']
            results_dict = {}
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {executor.submit(ccat_fit.circle_fit,
                                                   struct.field(I_col).to_numpy(),
                                                   struct.field(Q_col).to_numpy(),
                                                   full_output=True,
                                                   bounds=bounds,
                                                   loss=loss,
                                                   f_scale=f_scale,
                                                   method=method):  (tone, I_col, Q_col) for tone, (I_col, Q_col) in zip(to_calc, batches)}
                
                for future in concurrent.futures.as_completed(future_to_batch):
                    tone, I_col, Q_col = future_to_batch[future]
                    try:
                        I_c, Q_c, R, result = future.result()
                        if not result.success: 
                            raise RuntimeError(f'Fit failed with exit code {result.status}')
                        elif result.optimality < 1:
                            raise RuntimeError(f'Fit converged with low optimality {result.optimality}.')
                    
                        fit_I, fit_Q = R*cos + I_c, R*sin + Q_c
                        
                        A, D, theta = result.x
                        property_vals = [I_c, Q_c, R, A, D, theta, result.optimality, result.nfev, result.njev]
                        self.targ._properties[f'det_{tone:04d}'] = {f'{col_name[-1]}_{prefix}_{k}': v for k, v in zip(property_keys, property_vals)}
                    except Exception as e:
                        rfsoc_io.send_msg('WARNING', 'Fit failed for tone %s with exception: %s', tone, e)
                        fit_I, fit_Q = np.zeros(df.len()), np.zeros(df.len())
                        self.targ._properties[f'det_{tone:04d}'] = {}
                    results_dict[f'{col_name[-1]}_{I_col}'] = fit_I
                    results_dict[f'{col_name[-1]}_{Q_col}'] = fit_Q

            df = pl.DataFrame(results_dict)
            return pl.Series(df.select(pl.struct(df.columns)))
            
        if len(args) == 7:
            self, prefix, bounds, loss, f_scale, method, max_workers = args
        else:
            error = 'self, prefix, bounds, loss, f_scale, method, and max_workers are required arguments.'
            rfsoc_io.send_msg('ERROR', error)
            raise ValueError(error)
        
        return_col = [f'{col_name[-1]}_{col_name[1]}', f'{col_name[-1]}_{col_name[2]}']
        return_type = [pl.Float64, pl.Float64]
        expr, to_calc, calc_col, batches = ccat_mp.batch_calc(_circle_fit, tones, col_name, schema, return_col = return_col, return_type=return_type, recalc=recalc)
        return expr

    @staticmethod
    def calc_noise_shift(schema, *args, tones: list[int], recalc: bool = False, col_name = ['I', 'Q', 'noise']):
        if tones is not None:
            tone = tones[0]
        if len(args) == 6:
            tone_med_I, tone_med_Q, noise_med_I, noise_med_Q, tone_list, noise_tone = args

            noise_tone = noise_tone[tone_list.index(tone)]
            tone_med_I, tone_med_Q = tone_med_I[tone_list.index(tone)], tone_med_Q[tone_list.index(tone)]
            noise_med_I, noise_med_Q = noise_med_I[tone_list.index(noise_tone)], noise_med_Q[tone_list.index(noise_tone)]
            
            I_shift, Q_shift = noise_med_I - tone_med_I, noise_med_Q - tone_med_Q
        else:
            rfsoc_io.send_msg('ERROR', 'I_shift, Q_shift, tone_list, and noise_tone are required arguments.')

        col_name = [f'{name}_{noise_tone:04d}' for name in col_name[:-1]] + [col_name[-1]]
        I_col_noise, Q_col_noise, shift_col = col_name
        I_col_tone, Q_col_tone = f'{I_col_noise.split('_')[0]}_{tone:04d}', f'{Q_col_noise.split('_')[0]}_{tone:04d}'
        
        if recalc or not (f'{shift_col}_{I_col_tone}' in schema):
            return [(pl.col(I_col_noise) - I_shift).alias(f'{shift_col}_{I_col_tone}'),
                    (pl.col(Q_col_noise) - Q_shift).alias(f'{shift_col}_{Q_col_tone}')]
        else:
            return pl.col(f'{shift_col}_{I_col_tone}')

    @staticmethod
    def calc_phase_spline(schema, *args, tones: list[int], recalc: bool = False, col_name = ['f', 'phase', 'to_f', 'to_phase']):
        def _phase_spline(df):
            struct = df.struct

            self.properties
            results_dict = {}
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {executor.submit(ccat_fit.y_to_x_spline,
                                                   struct.field(f_col).to_numpy(),
                                                   struct.field(phase_col).to_numpy(),
                                                   k=k,
                                                   y_low = phase_low[tones.index(tone)],
                                                   y_up = phase_up[tones.index(tone)]):  (tone, f_col, phase_col) for tone, (f_col, phase_col) in zip(to_calc, batches)}
                
                for future in concurrent.futures.as_completed(future_to_batch):
                    tone, f_col, phase_col = future_to_batch[future]
                    try:
                        y_to_x, x_to_y = future.result()
                    
                        if y_to_x is not None:
                            y_to_x.extrapolate=False
                            to_phase = np.zeros(df.len())
                            to_f = y_to_x(struct.field(phase_col).to_numpy())
                        elif x_to_y is not None:
                            x_to_y.extrapolate=False
                            to_phase = x_to_y(struct.field(f_col).to_numpy())
                            to_f = np.zeros(df.len())
                        else:
                            raise RuntimeError('No spline was calculated.')
                        
                        property_vals = [y_to_x, x_to_y]
                        property_dict = {k: v for k, v in zip(interp_names, property_vals)}
                    except Exception as e:
                        rfsoc_io.send_msg('WARNING', 'Spline calculation for tone %s failed with exception: %s', tone, e)
                        property_dict = {}
                        to_phase, to_f = np.zeros(df.len()), np.zeros(df.len())
                    self.stream._properties[f'det_{tone:04d}'] = property_dict
                    results_dict[f'{interp_names[0]}_{stream_timestamp}_{tone:04d}'] = to_f
                    results_dict[f'{interp_names[1]}_{stream_timestamp}_{tone:04d}'] = to_phase
            
            df = pl.DataFrame(results_dict)
            return pl.Series(df.select(pl.struct(df.columns)))

        if len(args) == 6:
            self, phase_low, phase_up, k, stream_timestamp, max_workers = args

            if isinstance(phase_low, float): phase_low = len(tones)*[phase_low]
            if isinstance(phase_up, float): phase_up = len(tones)*[phase_up]
        else:
            error = 'self, phase_low, phase_up, k, max_workers are required arguments.'
            rfsoc_io.send_msg('ERROR', error)
            raise ValueError(error)      

        data_col_name = [col_name[0], col_name[1], col_name[-1]]
        interp_names = [f'{col_name[1]}_{col_name[-2]}', f'{col_name[0]}_{col_name[-1]}']
        calc_col = [f'{interp_names[0]}_{tone:04d}' for tone in tones]

        return_col = [f'{interp_names[0]}', f'{interp_names[1]}']
        return_type = [pl.Float64, pl.Float64]
        expr, to_calc, calc_col, batches = ccat_mp.batch_calc(_phase_spline, tones, data_col_name, schema, return_col=return_col, return_type=return_type, recalc=recalc, calc_col = calc_col)
        return expr

    @staticmethod
    def calc_phase_to_f(schema, *args, tones: list[int], recalc: bool = False, col_name = ['phase', 'f']):
        def _phase_to_f(df):
            struct = df.struct
            y_to_x, x_to_y = self.properties.select(pl.col([f'{spline_col}_phase_to_f_spline', f'f_to_{spline_col}_phase_spline']))            
            results_dict = {}
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {executor.submit(ccat_fit.y_to_x_interp,
                                                   struct.field(phase_col).to_numpy(),
                                                   y_to_x_spline = y_to_x.item(tones.index(tone)),
                                                   x_to_y_spline = x_to_y.item(tones.index(tone))):  (tone, phase_col) for tone, phase_col in zip(to_calc, batches)}
                
                for future in concurrent.futures.as_completed(future_to_batch):
                    tone, phase_col = future_to_batch[future]
                    try:
                        f = future.result()
                    except Exception as e:
                        rfsoc_io.send_msg('WARNING', 'Interpolation for tone %s failed with exception: %s', tone, e)
                        f = np.zeros(df.len())
                    results_dict[f'{col_name[-1]}_{tone:04d}'] = f

            df = pl.DataFrame(results_dict)
            return pl.Series(df.select(pl.struct(df.columns)))
        
        if len(args) == 3:
            self, spline_col, max_workers = args
        else:
            error = 'self, spline_col, and max_workers are required arguments.'
            rfsoc_io.send_msg('ERROR', error)
            raise ValueError(error) 

        calc_col = [f'{col_name[-1]}_{tone:04d}' for tone in tones]
        return_col = [f'{col_name[-1]}']
        return_type = [pl.Float64]
        expr, to_calc, calc_col, batches = ccat_mp.batch_calc(_phase_to_f, tones, col_name, schema, return_col=return_col, return_type=return_type, recalc=recalc, calc_col = calc_col)
        return expr
    
    @staticmethod
    def calc_frac_f(schema, *args, tones: list[int], recalc: bool = False, col_name = ['f', 'frac']):
        if tones is not None:
            tone = tones[0]
            col_name = [f'{name}_{tone:04d}' for name in col_name[:-1]] + [col_name[-1]]

        f_col, frac_f_col = col_name

        if len(args) == 2:
            f_0, tone_list = args
            if isinstance(f_0, Iterable):
                if tones is not None: 
                    f_0 = f_0[tone_list.index(tone)]
                else:
                    error = 'Cannot use an array of f_0 when there are no tones.'
                    rfsoc_io.send_msg('ERROR', error)
                    raise ValueError(error)
        else:
            rfsoc_io.send_msg('ERROR', 'f_0 and tone_list are required arguments.')

        if recalc or not (f'{frac_f_col}_{f_col}' in schema):
            return ((pl.col(f_col) - f_0)/f_0).name.prefix(frac_f_col + '_')
        else:
            return pl.col(f'{frac_f_col}_{f_col}')
    
    def properties_histogram(self, col_name, bins=None, mad_filter = True, num_mads = 10, recalc=False, **kwargs):
        ''' Calculate histogram of a detector property

        Args:
            col_name (str): Name of property
            bins (int | None, optional): Number of bins to use for histogram. Defaults to number of tones used in histogram divided by five
            mad_filter (bool, optional): Whether to use the median absolute deviation (MAD) to filter outliers before creating histogram. Defaults to True
            num_mads (int, optional): Number of MADs data must be within to keep. Defaults to 10
            recalc (bool, optional): Whether to recalculate histogram if data already exists. Defaults to False
        Returns:
            return (pl.DataFrame): Polars DataFrame with histogram counts and bin edges
        '''
        hist_col_name = ['counts', 'edges']
        hist_col_name = [f'hist_{name}_{col_name}' for name in hist_col_name]

        properties = self.properties

        if recalc or not (hist_col_name[0] in properties.schema):
            num_tones = properties.height
            df = properties.select(col_name).filter(~pl.col(col_name).is_nan()) # Filter out NaN values

            if mad_filter:
                df = (df.with_columns(pl.col(col_name).median().alias('median'))
                        .with_columns((pl.col(col_name) - pl.col('median')).abs().median().alias('MAD'))
                        .filter((pl.col(col_name) > (pl.col('median') - num_mads*pl.col('MAD'))) & (pl.col(col_name) < (pl.col('median') + num_mads*pl.col('MAD'))))) 
            
            bins = df.height // 5 if bins is None else min(bins, num_tones)
            data = df[col_name].to_numpy()

            counts, edges = np.histogram(data, bins)
            counts, edges = np.pad(np.array(counts, dtype=float), (0, int(num_tones - len(counts))), constant_values=None), np.pad(np.array(edges, dtype=float), (0, int(num_tones - len(edges))), constant_values=None)

            self.properties = properties.with_columns([pl.Series(name, value) for name, value in zip(hist_col_name, [counts, edges])])
        else:
            counts, edges = properties.select(hist_col_name).to_numpy().T
        return counts, edges

    #==================#
    # Plotting Methods #
    #==================#

    def properties_histogram_plot(self, col_name, plot_median=True, label='',  **kwargs):
        ''' Plot histogram of a detector property

        Args:
            col_name (str): Name of property
            plot_median (bool): Whether to plot a vertical line at the median
        Returns:
            return (hv.Histogram | hv.Overlay): Holoviews histogram figure
        '''

        counts, edges = self.properties_histogram(col_name, **kwargs)
        hist = hv.Histogram((edges, counts), label=label)
        
        if plot_median:
            median = self.properties[col_name].median()
            vline, spike = hv.VLine(median), hv.Curve(([median, median], [0, 1]), label=f'Median: {median:0.2e}')
            hist = hist*spike*vline
            hist.opts(opts.Curve(color=hv.Cycle(), linewidth=3, linestyle='--'),
                      opts.VLine(color=hv.Cycle(), linewidth=3, linestyle='--'))
        
        cfg = self.targ.drone_cfg['det_config']
        title = rf"${cfg['detector_type']}\ {cfg['network']}$: {len(counts)} Detectors"
        hist.opts(opts.Histogram(xlabel=col_name,
                                 ylabel='Count',
                                 title=title,
                                 aspect=self.viz_cfg['plot']['width']/self.viz_cfg['plot']['height'],
                                 fig_size=250,
                                 show_grid=True,
                                 show_legend=True))
        return hist

    #================#
    # Helper Methods #
    #================#
    
    def add_data_to_properties(self, df, col_name) -> None:
        '''
        Add a quantity calculated with a data object's ``data`` DataFrame to the ``properties`` DataFrame
        
        Note:
            - The ``df`` DataFrame does not necessarily need to derive from a data object's ``data`` DataFrame, but the structure of this method is designed specifically for that use case

        Example:
            -
        
        Args:
            df (pl.DataFrame): Polars DataFrame with the data to be added to the ``properties`` DataFrame. The DataFrame must be in wide format with the column names being tone numbers (e.g., '0000', '0001', etc.)
            col_name (str): Name of column to add to ``properties`` DataFrame 
        '''

        unpivot_cols = df.select(pl.all()).columns
        df = df.unpivot(on=unpivot_cols,
                        variable_name='det',
                        value_name=col_name).with_columns(pl.col('det').cast(int)).unique()
        if col_name in self.properties.schema: self._properties_df = self._properties_df.drop(col_name)
        self._properties_df = self._properties_df.join(df, on='det', how='full', coalesce=True)
        
    def _get_data_obj(self, data):
        '''
        Get data object (Target, Timestream, or both) corresponding to string 

        data (

        
        '''
        data_objs = []
        f = []
        if data == 'targ' or data == 'both':
            data_objs.append(self.targ)
            f.append(None)
        if data == 'timestream' or data == 'both':
            data_obj = self.stream
            data_objs.append(data_obj)
            f.append(data_obj.comb.select('tone_freqs').to_numpy().T[0])
        return data_objs, f
    
    @staticmethod
    def _load_data(data_class, com_to, analysis_cfg, dets, noise_tones, timestamp, data_path, **kwargs):
        '''
        Load *ccatkidlib* data file into VNA, Target, or Timestream data object 

        Args:
            data_class (VNA | Target | Timestream): Class corresponding to the type of data to load. Must be one of VNA, Target, or Timestream
            com_to (str): 
            analysis_cfg (str): Path to analysis configuration file
            dets (int | list[int]): Subset of detectors to load
            timestamp (int | str | None): Timestamp of data file
            data_path (str | list[str] | pathlib.PosixPath | list[pathlib.PosixPath] | None): Path of data file. Can pass a list of file paths for a G3 timestream split into multiple files.
        '''

        data = None
        if data_path is not None or timestamp is not None:
            try:
                data = data_class(com_to = com_to, analysis_cfg = analysis_cfg, tones = dets, noise_tones = noise_tones, timestamp = timestamp, data_path = data_path, **kwargs)
            except Exception as e:
                rfsoc_io.send_msg('ERROR', 'Failed to load %s with exception: %s.', data_class.__name__, e)
                data = None
        return data
