''' Module for analyzing kinetic inductance detector (KID) composite data

Authors:
    - Darshan Patel <dp649@cornell.edu>

TODO:
    - Change data args to string enums
'''

import os
import sys
import time

import copy
import numpy as np
import polars as pl
import pathlib
import concurrent.futures
import lmfit

from collections.abc import Iterable
from typing import Any, TypeAlias, Literal
from pathlib import Path
from functools import cached_property

# local imports
import ccatkidlib
import ccatkidlib.io as io
import ccatkidlib.log as log
import ccatkidlib.analysis.utils.pair as pair
import ccatkidlib.analysis.fit.fit as ccat_fit
import ccatkidlib.analysis.utils.multiprocess as ccat_mp
import ccatkidlib.analysis.utils.dataframe as ccat_df
import ccatkidlib.analysis.viz.viz_utils as viz_utils

from ccatkidlib.log import header
from ccatkidlib.analysis.core.data import Data
from ccatkidlib.analysis.core.timestream import Timestream
from ccatkidlib.analysis.core.vna import VNA
from ccatkidlib.analysis.core.target import Target

# Plotting functions
import holoviews as hv
import hvplot.polars
from holoviews import opts

Format: TypeAlias = Literal['png', 'jpeg', 'pdf']

class Detector:
    '''Class representing kinetic inductance detectors (KIDs). Used for KID analyses requiring fitting and/or multiple types of data files (e.g., timestream and target sweep data).

    Attributes:
        bid  (str): RFSoC board that took the detector data
        drid (str): RFSoC drone that took the detector data
        tones (list[int]): List of detectors

        stream (Timestream | None): ``Timestream`` object of detector timestream
        targ (Target): ``Target`` object of detector target sweep
        vna (VNA | None): ``VNA`` object of detector VNA sweep

        analysis_cfg (dict): Config file with parameters used for data analysis
        viz_cfg (dict): Config file with paramateres used for data visualization

        cable_delay (float | None): Cable delay of the network (in nanoseconds)
        properties  (pl.DataFrame): Polars dataframe with detector properties
    '''

    def __init__(self,
                 com_to: str,
                 cfg_path: str = str(Path(__file__).parents[1] / 'analysis_config.yaml'),
                 dets: int | list[int] = -1,
                 noise_tones: int | list[int] | None = None,
                 cable_delay: float | None = None,
                 stream: Timestream | None = None, stream_path: str | pathlib.PosixPath | list[str] | list[pathlib.PosixPath] | None = None, stream_timestamp: int | str | None = None,
                 targ: Target | None = None, targ_path: str | pathlib.PosixPath | None = None, targ_timestamp: int | str | None = None,
                 vna: VNA | None = None, vna_path: str | pathlib.PosixPath | None = None, vna_timestamp: int | str | None = None,
                 analysis_cfg: dict | None = None,
                 viz_cfg: dict | None = None,
                 **kwargs):
        '''
        Constructor for Detector. Creates *ccatkidlib* data objects (``VNA``, ``Target``, ``Timsetream``)

        Note:
            - A Detector object can be initialized without a ``Timestream`` object but must **always** have a ``Target`` object.
            - If only the information to load a timestream is provided, an attempt will be made to find the target sweep file corresponding to the timestream

        Args:
            com_to (str): Drone that took the data. In form *'\<board>.\<drone>'*
            analysis_cfg (str, optional): Path to analysis configuration file. Defaults to analysis configuration file in *ccatkidlib/ccatkidlib/analysis*
            dets (int | list[int], optional): Which detectors to load; -1 to load all detectors. Defaults to -1
            noise_tones (int | list[int] | None, optional): Indicies of noise tones (tones not placed on detectors). Defaults to *None*
            cable_delay (float | None, optional): Cable delay of full network. Defaults to *None*

            stream (Timestream | None, optional): Detector ``Timestream`` object. Defaults to *None*
            stream_path (str | pathlib.PosixPath | list[str] | list[pathlib.PosixPath] | None, optional): Data path to detector timestream files. Defaults to *None*
            stream_timestamp (int | str | None, optional): Timestamp of detector timestream files. Defaults to *None*

            targ (Target | None, optional): Detector ``Target`` object. Defaults to *None*
            targ_path (str | pathlib.PosixPath | None, optional): Data path to detector target sweep file. Defaults to *None*
            targ_timestamp (int | str | None, optional): Timestamp of detector target sweep file. Defaults to *None*

            vna (VNA | None, optional): Detector ``VNA`` object. Defaults to *None*
            vna_path (str | pathlib.PosixPath | None, optional): Data path to detector VNA sweep file. Defaults to *None*
            vna_timestamp (int | str | None, optional): Timestamp of detector VNA sweep file. Defaults to *None*

            **kwargs: Key word arguments for finding data files. See below:
            root_data_dir (str, optional): Root directory where data is stored. Defaults to that specified in analysis config
            data_dir (str, optional): Directory where data is stored
            date (str, optional): Date data was taken
            sess_id (str, optional): ccatkidlib session ID of data

        '''

        self.analysis_cfg, self.viz_cfg = io.load_config(cfg_path)
        if analysis_cfg is not None: self.analysis_cfg = analysis_cfg
        if viz_cfg is not None: self.viz_cfg = viz_cfg

        # Create Timestream, Target, and VNA data objects based provided arguments
        # ------------------------------------------------------------------------
        if not isinstance(stream, Timestream): stream = Detector._load_data(Timestream, com_to, cfg_path, self.analysis_cfg, self.viz_cfg, dets, noise_tones, stream_timestamp, stream_path, **kwargs)
        if not isinstance(targ, Target): targ = Detector._load_data(Target, com_to, cfg_path, self.analysis_cfg, self.viz_cfg, dets, noise_tones, targ_timestamp, targ_path, **kwargs)
        if not isinstance(vna, VNA): vna = Detector._load_data(VNA, com_to, cfg_path, self.analysis_cfg, self.viz_cfg, None, None, vna_timestamp, vna_path, **kwargs)

        # Must have a sweep to do meaningful data analysis
        # ------------------------------------------------
        if not isinstance(targ, Target):
            if isinstance(stream, Timestream): # If timestream provided, try to find associated sweep
                self.tones = stream.tones
                vna_path, targ_path = pair.get_sweep(stream.data_path[0], **kwargs)

                # Load found sweep
                if Path(vna_path).exists(): vna = Detector._load_data(VNA, com_to, cfg_path, self.analysis_cfg, self.viz_cfg, None, None, vna_timestamp, vna_path, **kwargs)
                if Path(targ_path).exists(): targ = Detector._load_data(Target, com_to, cfg_path, self.analysis_cfg, self.viz_cfg, dets, noise_tones, targ_timestamp, targ_path, **kwargs)

                if not isinstance(targ, Target): # and not isinstance(vna, VNA):
                    error = 'Failed to find target sweep associated with timestream. If there is no target sweep, create a Timestream object instead.'
                    log.log('CRITICAL', error)
                    raise RuntimeError(error)

            else: # Error of no sweep or timestream provided
                error = 'A timestream, target sweep or both need to be specified!'
                log.log('CRITICAL', error)
                raise RuntimeError(error)
        else:
            self.tones = targ.tones
            if vna is None:
                vna_path, _ = pair.get_sweep(targ.data_path[0], **kwargs)
                if Path(vna_path).exists(): vna = Detector._load_data(VNA, com_to, cfg_path, self.analysis_cfg, self.viz_cfg, None, None, vna_timestamp, vna_path, **kwargs)

        self.bid, self.drid = com_to.split('.')

        self.stream = stream
        self.targ = targ
        self.vna = vna

        self.timestamp = self.stream.timestamp if self.stream is not None else self.targ.timestamp

        # Create internal attributes corresponding to lazily loaded attributes
        # --------------------------------------------------------------------
        self._cable_delay = cable_delay
        self._properties_df = None

        log_dir = io.add_dir('log', 
                             str(self.targ.data_path[0]), 
                             save_root = self.analysis_cfg['io']['file_logging']['logging_root_dir'],
                             data_root = self.targ._root_dir,
                             sub_dirs=[""])
        log.setup_logging(Path(log_dir) / self.analysis_cfg['io']['file_logging']['logging_fname'], 
                          self.analysis_cfg['io']['file_logging']['detector_level'], 
                          self.analysis_cfg['io']['terminal_logging']['detector_level'],
                          name='analysis.detector')

    #==========================#
    # Lazily Loaded Attributes #
    #==========================#

    @property
    def properties(self) -> pl.DataFrame:
        '''

        '''
        def _merge_properties(new_properties_df: pl.DataFrame) -> None:
            '''
            Merge two ``properties`` Polars DataFrames

            Args:
                new_properties_df (pl.DataFrame):
            '''
            shared_cols = (set(self._properties_df.columns) & set(new_properties_df.columns)) - {'det'}
            self._properties_df = ccat_df.coalesce_join(self._properties_df, new_properties_df, 'det', shared_cols)

        if isinstance(self._properties_df, pl.LazyFrame): 
            self._properties_df = self._properties_df.collect()
        elif self._properties_df is None:
            self._properties_df = self.targ.comb
        if self.targ.properties is not None: _merge_properties(self.targ._properties_df)
        if self.stream is not None and self.stream.properties is not None: _merge_properties(self.stream._properties_df)

        return self._properties_df

    @properties.setter
    def properties(self, value):
        if isinstance(value, (pl.DataFrame, pl.LazyFrame)):
            self._properties_df = value

    @property
    def cable_delay(self):
        self._cable_delay = self.vna.cable_delay if self._cable_delay is None and isinstance(self.vna, VNA) else self._cable_delay
        if not isinstance(self._cable_delay, Iterable):
            self._properties_df = self.properties.with_columns(pl.lit(self._cable_delay).alias('network_cable_delay'))
            # Get cable delays for individual detectors using target sweep data. The target sweep cable delays tend to be too large so average with the overall network cable delay
            #self.targ._properties = {det: {'det_cable_delay': 0.4*delay + 0.6*self._cable_delay} for det, delay in self.targ.cable_delay.items()}

            # Replace cable delays that are far from the overall network cable delay with the network cable delay
            #threshold = pl.lit(100) # TODO: Make this accessible
            #self._properties_df = (self.properties.lazy()#.with_columns(pl.when((pl.col('det_cable_delay') - self._cable_delay).abs() > threshold)
            #                                             #                .then(pl.lit(self._cable_delay))
            #                                             #                .otherwise(pl.col('det_cable_delay'))
            #                                             #               .alias('det_cable_delay'))
            #                                            .with_columns(pl.lit(self._cable_delay).alias('network_cable_delay'))
            #                                            .collect())
        return self._cable_delay

    @cached_property
    def fig_dir(self) -> str:
        '''
        Directory where figures should be saved. Create if it does not already exist.
        '''

        return io.add_dir('fig',
                          str(self.targ.data_path[0]),
                          save_root = self.viz_cfg['save']['fig_root_dir'],
                          data_root = self.targ._root_dir,
                          sub_dirs = ['detector'],
                          timestamp = str(self.timestamp))

    @cached_property
    def pickle_dir(self) -> str:
        '''
        Directory where pickle files should be saved. Create if it does not already exist.
        '''

        pickle_dir =  io.add_dir('pickle',
                                 str(self.targ.data_path[0]),
                                 save_root = self.analysis_cfg['io']['pickle']['pickle_root_dir'],
                                 data_root = self.targ._root_dir,
                                 sub_dirs = ['detector'],
                                 timestamp = str(self.timestamp))
        io.create_dir(Path(pickle_dir) / 'dataframe')
        return pickle_dir

    @cached_property
    def fit_dir(self):
        return self.analysis_cfg['file_paths']['fit_dir']

    #=====================#
    # Data Getter Methods #
    #=====================#

    def get_properties(self,
                       col_name: str | list[str] = '.*',
                       include: int | list[int] | None = None,
                       exclude: int | list[int] | None = None,
                       strict: bool = False):
        ''' Get the specified data columns and rows from the ``properties`` Polars DataFrame

        Args:
            col_name (str | list[str], optional): Defaults to all columns
            include (int | list[int] | None, optional): Defaults to *None*
            exclude (int | list[int] | None, optional): Defaults to *None*
            strict (bool, optional): Defaults to *False*

        '''
        return ccat_df.get_properties(self, col_name = col_name, include=include, exclude=exclude, strict=strict)

    # Fitting
    # -------
    def complex_fit(self,
                    nonlinear: bool = False,
                    asymm: bool = False,
                    fix_cable: bool = False,
                    fix_thetaQ: bool = False,
                    include: int | list[int] | None = None,
                    exclude: int | list[int] | None = None,
                    recalc: bool = False,
                    max_workers: int = 1,
                    ex = None) -> pl.DataFrame:
        '''
        Fit target sweep using complex forward transmission data (*z = I + iQ*)

        Args:
            nonlinear (bool, optional): Whether to perform a nonlinear fit. Defaults to *False*
            asymm (bool, optional): Whether to perform a asymmetric fit. Defaults to *False*
            fix_cable (bool, optional): Whether to vary cable parameters for fit. Defaults to *False*: parameters are varied
            fix_thetaQ (bool, optional): Whether to vary impedance mismatch angle and coupling quality factor for fit. Defaults to *False*: parameters are varied
        Returns:
            return (pl.DataFrame): Polars DataFrame with fit I and Q data
        '''

        if not self.fit_dir in sys.path: sys.path.append(self.fit_dir)
        import resonator_model_v3
        globals()['resonator_model_v3'] = resonator_model_v3

        col_name = ['f', 'I', 'Q', 'complex_fit']

        self.properties # Load properties
        args = [[self, nonlinear, asymm, fix_cable, fix_thetaQ, ccat_mp.check_max_workers(max_workers), ex]]
        self.targ.transform(Detector.calc_complex_fit, *args, include=include, exclude=exclude, recalc = recalc, col_name = col_name)
        self.targ.data = self.targ._unnest('struct_' + col_name[-1])

        # Calculate Q_c and Q_i. Convert cable delay into nanaseconds
        self.properties = (self.properties.lazy()
                                          .with_columns(((pl.col(f'{col_name[-1]}_Q_e_real')/(pl.col(f'{col_name[-1]}_Q_e_real')**2 + pl.col(f'{col_name[-1]}_Q_e_imag')**2))**-1).alias(f'{col_name[-1]}_Q_c')) # Calculate Q_c
                                          .with_columns([((1/pl.col(f'{col_name[-1]}_Q') - 1/pl.col(f'{col_name[-1]}_Q_c'))**-1).alias(f'{col_name[-1]}_Q_i'), # Calculate Q_i
                                                         (-1e9*pl.col(f'{col_name[-1]}_delay')).alias(f'{col_name[-1]}_delay_ns')]) # Convert cable delay to nanoseconds
                                          .collect())

        return self.targ.get_data(col_name=[f'{col_name[-1]}_{col_name[1]}', f'{col_name[-1]}_{col_name[2]}'], include=include, exclude=exclude, strict=True)

    def phase_fit(self,
                  prefix: str | list[str] = 'mismatch_rotate_origin_shift_origin_rotate_unwind_rotate',
                  circle_fit_col: str = 'circle_fit_unwind_rotate',
                  nonlinear: bool = False,
                  method: str = 'least_squares',
                  params: lmfit.Parameters = None,
                  window: float = 1,
                  include: int | list[int] | None = None,
                  exclude: int | list[int] | None = None,
                  recalc: bool = False,
                  max_workers: int = 1,
                  ex=None) -> pl.DataFrame:
        '''
        Fit target sweep using phase data (*arctan(Q/I)*)

        '''

        col_name = ['f', 'I', 'Q', 'phase', 'phase_fit']

        if isinstance(prefix, str): prefix = [prefix]
        num_prefix = len(prefix)

        if isinstance(params, lmfit.parameter.Parameters) or not isinstance(params, Iterable) or not len(params) == num_prefix: params = [params]*num_prefix
        if not isinstance(nonlinear, Iterable) or not len(nonlinear) == num_prefix: nonlinear = [nonlinear]*num_prefix
        if not isinstance(window, Iterable) or not len(window) == num_prefix: window = [window]*num_prefix
        if isinstance(circle_fit_col, str) or not len(circle_fit_col) == num_prefix: circle_fit_col = [circle_fit_col]*num_prefix

        col_names = [[]]*num_prefix
        for i, pre in enumerate(prefix):
            col_names[i] = [col_name[0]] + [f"{pre}{'_' if pre else ''}{name}" for name in col_name[1:-1]] + [col_name[-1]]

        radii = [self.get_properties(col_name = f'{col}_R', include=include, exclude=exclude, strict=True).to_numpy().T[1] for col in circle_fit_col]
        args = [[self, pre, radius, nonlin, method, param, win, ccat_mp.check_max_workers(max_workers), ex] for pre, radius, nonlin, param, win in zip(prefix, radii, nonlinear, params, window)]
        self.targ.transform([Detector.calc_phase_fit]*num_prefix, *args, include=include, exclude=exclude, recalc=recalc, col_name = col_names)
        self.targ.data = self.targ._unnest(['struct_' + col_name[-1] for col_name in col_names])

        # Calculate Q_c, Q_i, and nonlinearity parameter 'a'
        for pre, circle in zip(prefix, circle_fit_col):
            self.properties = (self.properties.lazy()
                                              .with_columns([(1e-8*(pl.col(f'{col_name[-1]}_{pre}_beta')*(2*pl.col(f'{col_name[-1]}_{pre}_R'))**2)/(pl.col(f'{col_name[-1]}_{pre}_f_0')/pl.col(f'{col_name[-1]}_{pre}_Qr'))).alias(f'{col_name[-1]}_{pre}_a'),
                                                             (pl.col(f'{col_name[-1]}_{pre}_Qr')*(pl.col(f'{circle}_center_mag') + pl.col(f'{col_name[-1]}_{pre}_R'))/(2*pl.col(f'{col_name[-1]}_{pre}_R'))).alias(f'{col_name[-1]}_{pre}_Q_c')])
                                              .with_columns(((1/pl.col(f'{col_name[-1]}_{pre}_Qr') - 1/pl.col(f'{col_name[-1]}_{pre}_Q_c'))**-1).alias(f'{col_name[-1]}_{pre}_Q_i'))
                                              .collect())

        return self.targ.get_data(col_name=[col_name[-1] for col_name in col_names], include=include, exclude=exclude)

    def IQ_circle_fit(self,
                      prefix: str | list[str] = 'unwind_rotate',
                      bounds = None,
                      loss: str = 'soft_l1',
                      f_scale: float = 1,
                      method: str = 'trf',
                      include: int | list[int] | None = None,
                      exclude: int | list[int] | None = None,
                      recalc: bool = False,
                      max_workers=1,
                      ex=None) -> pl.DataFrame:
        '''
        Fit the target sweep circle in the IQ plane

        '''

        col_name = ['I', 'Q', 'circle_fit']

        if isinstance(prefix, str): prefix = [prefix]
        num_prefix = len(prefix)

        col_names = [[]]*num_prefix
        for i, pre in enumerate(prefix):
            col_names[i] = [f"{pre}{'_' if pre else ''}{name}" for name in col_name[:-1]] + [col_name[-1]]

        self.properties # load properties
        args = [[self, pre, bounds, loss, f_scale, method, ccat_mp.check_max_workers(max_workers), ex] for pre in prefix]
        self.targ.transform([Detector.calc_IQ_circle_fit]*num_prefix, *args, include=include, exclude=exclude, recalc=recalc, col_name = col_names)
        self.targ.data = self.targ._unnest(['struct_' + col_name[-1] for col_name in col_names])
        return self.targ.get_data(col_name=[col_name[-1] for col_name in col_names], include=include, exclude=exclude)

    # IQ transformations
    # ------------------

    def IQ_unwind(self,
                  prefix: str | list[str] = '',
                  data: str = 'both',
                  delay_col: str = 'network_cable_delay',
                  include: int | list[int] | None = None,
                  exclude: int | list[int] | None = None,
                  recalc: bool = False) -> list[pl.DataFrame]:
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
            pl.Dataframe: Polars DataFrame with unwound I & Q data
        '''
        col_name = ['f', 'I', 'Q', 'unwind']
        if isinstance(prefix, str): prefix = [prefix]

        data_objs, data_types = self._get_data_obj(data)
        if not data_objs:
            error = f"Invalid data type {data}, must be 'targ', 'timestream', or 'both'."
            log.log('ERROR', error)
            raise ValueError(error)

        # Ensure that the cable delay has been calculated
        if delay_col == 'network_cable_delay': self.cable_delay
        delays = self.get_properties(delay_col, include=include, exclude=exclude, strict=True).to_numpy().T[1]

        delays = -2*np.pi*1e-9*delays
        unwind_dfs = []
        for data_obj, data_type in zip(data_objs, data_types):
            angle = ([delay*tone_freq for tone_freq, delay in zip(self.targ.get_data('f', include=include, exclude=exclude, strict=True).to_numpy().T, delays)] if data_type == 'targ' else
                     [delay*tone_freq for tone_freq, delay in zip(self.stream.comb['tone_freqs'].to_numpy().T, delays)])
            data_obj.IQ_rotate(prefix=prefix, angle=angle, name=col_name[-1], include=include, exclude=exclude, recalc=recalc)
            unwind_dfs.append(data_obj.get_data(col_name=([f"{col_name[-1]}_rotate_{pre}{'_' if pre else ''}{col_name[1]}" for pre in prefix] +
                                                          [f"{col_name[-1]}_rotate_{pre}{'_' if pre else ''}{col_name[2]}" for pre in prefix]), include=include, exclude=exclude, strict=True))
        return unwind_dfs

    def IQ_norm(self,
                prefix: str | list[str] = '',
                data: str = 'both',
                norm_col: str = 'cable_complex_fit',
                include: int | list[int] | None = None,
                exclude: int | list[int] | None = None,
                recalc: bool = False) -> list[pl.DataFrame]:
        '''
        Divide out cable baseline from target sweep and/or timestream I & Q data

        '''

        col_name = ['f', 'I', 'Q', 'norm']
        if isinstance(prefix, str): prefix = [prefix]

        data_objs, data_types = self._get_data_obj(data)
        if not data_objs:
            error = f"Invalid data type {data}, must be 'targ', 'timestream', or 'both'."
            log.log('ERROR', error)
            raise ValueError(error)

        tone_freqs = self.get_properties('tone_freqs', include=include, exclude=exclude, strict=True)
        norm_dfs = []
        for data_obj, data_type in zip(data_objs, data_types):
            cable_mags = self.targ.get_data(f"{norm_col}{'_' if norm_col else ''}mag", include=include, exclude=exclude, strict=True)
            if data_type == 'targ':
                scale = 1/cable_mags.to_numpy().T
            else:
                f_df = (self.targ.get_data(col_name[0], include=include, exclude=exclude, strict=True)
                                 .unpivot(variable_name='det', value_name=col_name[0])
                                 .with_columns(pl.col('det').str.strip_prefix(f'{col_name[0]}_').cast(int)))
                cable_df = cable_mags.unpivot(variable_name='temp', value_name='cable').drop('temp')
                f_cable_df = pl.concat([f_df, cable_df], how='horizontal')
                f_cable_df = f_cable_df.join(tone_freqs, on='det', how='left', coalesce=True)
                scale = 1/(f_cable_df.lazy()
                                     .sort((pl.col(col_name[0]) - pl.col('tone_freqs')).abs())
                                     .select(pl.col('cable').first().over('det'))
                                     .collect()).to_numpy().T
            for pre in prefix:
                data_obj.IQ_scale(prefix=pre, scale=scale, name=col_name[-1], include = include, exclude=exclude, recalc=recalc)
            norm_dfs.append(data_obj.get_data(col_name=([f"{col_name[-1]}_scale_{pre}{'_' if pre else ''}{col_name[1]}" for pre in prefix] +
                                                        [f"{col_name[-1]}_scale_{pre}{'_' if pre else ''}{col_name[2]}" for pre in prefix]), include=include, exclude=exclude))
        return norm_dfs

    def IQ_trim(self,
                prefix: str | list[str] = '',
                window: float | list[float] = 1.5,
                use_fit: bool = False,
                f_0_col: str = 'complex_fit_f_0',
                Q_col: str = 'complex_fit_Q',
                mean_points: int = 10,
                mag_prefix: str = '',
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

        if use_fit:
            if f_0_col in self.properties.schema:
                pass
        else:
            fwhm_col_name = ['sample', f"{mag_prefix}{'_' if mag_prefix else ''}mag"] # Data columns used for estimating the FWHM

            include_subset = ccat_df.check_properties(self, 'HM_mid', include=include, exclude=exclude, recalc=recalc)
            if not len(include_subset) == 0:
                # Get detector magnitudes and sample numbers and unpivot DataFrame from wide to long format
                mag_df = self.targ.get_data(col_name=fwhm_col_name, strict=True, include=include_subset)
                mag_df = (mag_df.unpivot(index=fwhm_col_name[0],
                                         variable_name='det',
                                         value_name=fwhm_col_name[1])
                                .lazy()
                                .with_columns(pl.col('det').str.strip_prefix(f'{fwhm_col_name[1]}_').cast(pl.Int32))
                                .sort(fwhm_col_name[1], descending=True))

                # Get minimum magnitude values for each detector and corresponding sample numbers
                min_df = (mag_df.filter((pl.col(fwhm_col_name[1]) == pl.col(fwhm_col_name[1]).min()).over('det'))
                                .rename({fwhm_col_name[0]: f'min_{fwhm_col_name[0]}', fwhm_col_name[1]: f'min_{fwhm_col_name[1]}'})
                                .collect())
                shared_cols = 'HM_mid' if 'HM_mid' in self._properties_df.schema else []
                self._properties_df = ccat_df.coalesce_join(self._properties_df, min_df.select(['det', pl.col(f'min_{fwhm_col_name[0]}').alias('HM_mid')]), 'det', shared_cols)

                mag_min_df = (mag_df.collect()
                                    .join(min_df, on='det', how='left', coalesce=True)
                                    .lazy()
                                    .with_columns((pl.col(fwhm_col_name[0]) < pl.col(f'min_{fwhm_col_name[0]}')).alias('low')))
                # Get mean maximum magnitude values for both the low and high frequency sides of each detector
                max_df = (mag_min_df.group_by(['low', 'det'], maintain_order=True)
                                    .agg(pl.col(fwhm_col_name[1]).head(mean_points).mean())
                                    .collect()
                                    .pivot(on='low',
                                           index='det',
                                           values=fwhm_col_name[1])
                                    .lazy()
                                    .sort('det')
                                    .rename({'true': f'max_{fwhm_col_name[1]}_low', 'false': f'max_{fwhm_col_name[1]}_high'})
                                    .collect())
                min_max_df = (mag_min_df.collect()
                                        .join(max_df, on='det', how='left', coalesce=True)
                                        .lazy()
                                        .with_columns([(pl.col(fwhm_col_name[1]) - ((pl.col(f'min_{fwhm_col_name[1]}') + pl.col(f'max_{fwhm_col_name[1]}_{side}'))/2)).abs().alias(f'HM_{side}') for side in ['low', 'high']]))
                # Get the samples corresponding to the half max on the low and high frequency sides of each detector
                for side in ('low', 'high'):
                    HM_df = (min_max_df.filter(pl.col('low') == ('low' == pl.lit(side)))
                                       .sort(f'HM_{side}')
                                       .select('det', pl.col(fwhm_col_name[0]).first().over('det'))
                                       .unique()
                                       .sort('det')
                                       .rename({fwhm_col_name[0]: f'HM_{side}'})
                                       .collect())
                    shared_cols = f'HM_{side}' if f'HM_{side}' in self._properties_df.schema else []
                    self._properties_df = ccat_df.coalesce_join(self._properties_df, HM_df, 'det', shared_cols)
            HM_low, HM_mid, HM_high = self.get_properties(['HM_low', 'HM_mid', 'HM_high'], include=include, exclude=exclude, strict=True).to_numpy().T[1:4]
        for pre, win in zip(prefix, window):
            lower_bound = (HM_mid - (HM_mid - HM_low)*win).astype(int)
            upper_bound = (HM_mid + (HM_high - HM_mid)*win).astype(int)
            self.targ.IQ_trim(prefix=pre, lower_index = lower_bound, upper_index=upper_bound, name=col_name[-1], include = include, exclude=exclude, recalc=recalc)
        return self.targ.get_data(col_name=([f"{col_name[-1]}_trim_{pre}{'_' if pre else ''}{col_name[0]}" for pre in prefix] +
                                            [f"{col_name[-1]}_trim_{pre}{'_' if pre else ''}{col_name[1]}" for pre in prefix]), include=include, exclude=exclude, strict=True)

    def IQ_circle_real(self,
                       prefix: str | list[str] = 'unwind_rotate',
                       data: str = 'both',
                       loc: str = 'origin',
                       circle_fit_col='circle_fit_unwind_rotate',
                       use_fit=True,
                       include: int | list[int] | None = None,
                       exclude: int | list[int] | None = None,
                       recalc: bool = False) -> list[pl.DataFrame]:
        '''
        Rotate and center the target sweep circle in the IQ plane onto the real axis
        '''
        if isinstance(prefix, str): prefix = [prefix]

        dest_I = None
        if loc == 'origin':
            dest_I = 0

        col_name = ['I', 'Q', f"{loc}{'_' if loc else ''}shift{'_' if loc else ''}{loc}_rotate"]

        if dest_I is None: # Do not transform the circle if no destination provided
            shift = np.zeros(len(self.tones))
            center_angle = np.zeros(len(self.tones))
        else:
            if not use_fit: # Use median I & Q values of target sweep IQ circle if not using center from circle fit
                property_names = [f"targ_median_{prefix}{'_' if prefix else ''}{col_name[0]}",
                                  f"targ_median_{prefix}{'_' if prefix else ''}{col_name[0]}",
                                  f"targ_median_{prefix}{'_' if prefix else ''}angle",
                                  f"targ_median_{prefix}{'_' if prefix else ''}mag"]

                # Calculate target sweep median I & Q values if not in ``properties`` DataFrame already
                include_subset = ccat_df.check_properties(self, property_names[0], include=include, exclude=exclude, recalc=recalc)
                if not len(include_subset) == 0:
                    median_I_df = self.targ.get_data(f"{prefix}{'_' if prefix else ''}{col_name[0]}", include=include_subset, strict=True).select(pl.all().median().name.map(lambda s: s.split('_')[-1]))
                    median_Q_df = self.targ.get_data(f"{prefix}{'_' if prefix else ''}{col_name[0]}", include=include_subset, strict=True).select(pl.all().median().name.map(lambda s: s.split('_')[-1]))

                    ccat_df.add_data_to_properties(self, median_I_df, property_names[0])
                    ccat_df.add_data_to_properties(self, median_Q_df, property_names[1])
            else:
                property_names = [f"{circle_fit_col}_center_I",
                                  f"{circle_fit_col}_center_Q",
                                  f"{circle_fit_col}_center_angle",
                                  f"{circle_fit_col}_center_mag"]

            include_subset = ccat_df.check_properties(self, property_names[2], include=include, exclude=exclude, recalc=recalc)
            if not len(include_subset) == 0:
                circle_fit_df = (self.get_properties(property_names[0:2] , include=include_subset, strict=True)
                                     .select(['det',
                                             (pl.arctan2(pl.col(property_names[1]), pl.col(property_names[0]))).alias(property_names[2]), # Calculate angle
                                            ((pl.col(property_names[0])**2 + pl.col(property_names[1])**2).sqrt()).alias(property_names[3])])) # Calculate magnitude
                shared_cols = property_names[2:4] if property_names[2] in self._properties_df.schema else []
                self._properties_df = ccat_df.coalesce_join(self._properties_df, circle_fit_df, on = 'det', shared_cols = shared_cols)

            center_angle, center_mag = self.get_properties(property_names[2:4], include=include, exclude=exclude, strict=True).to_numpy().T[1:3]
            center_angle = np.pi - center_angle
            shift = center_mag + dest_I

        data_objs, _ = self._get_data_obj(data)
        if not data_objs:
            error = f"Invalid data type {data}, must be 'targ', 'timestream', or 'both'."
            log.log('ERROR', error)
            raise ValueError(error)

        shift_dfs = []
        for data_obj in data_objs:
            data_obj.IQ_rotate(prefix=prefix, angle=center_angle, name=loc, include = include, exclude=exclude, recalc=recalc)
            data_obj.IQ_shift(prefix=[f"{loc}{'_' if loc else ''}rotate_{pre}" for pre in prefix], shift_I = shift, name=loc, include=include, exclude=exclude, recalc=recalc)
            shift_dfs.append(data_obj.get_data(col_name=[f"{col_name[-1]}_{pre}{'_' if pre else ''}{col_name[0]}" for pre in prefix] +
                                                        [f"{col_name[-1]}_{pre}{'_' if pre else ''}{col_name[1]}" for pre in prefix], include=include, exclude=exclude, strict=True))
        return shift_dfs

    def IQ_circle_rotate(self,
                         prefix: str | list[str] = 'origin_shift_origin_rotate_unwind_rotate',
                         data: str = 'both',
                         rotation: str ='mismatch',
                         mean_points: int = 10,
                         include: int | list[int] | None = None,
                         exclude: int | list[int] | None = None,
                         recalc: bool = False,
                         **kwargs) -> list[pl.DataFrame]:
        '''
        Rotate the target sweep circle in the IQ plane around its center
        '''

        col_name = ['I', 'Q', rotation]
        if isinstance(prefix, str): prefix = [prefix]
        num_prefix=len(prefix)

        angles = [[]]*num_prefix
        for i, pre in enumerate(prefix):
            if rotation == 'mismatch':
                mismatch_col_name = f'{pre}_{col_name[-1]}_angle'
                include_subset = ccat_df.check_properties(self, mismatch_col_name, include=include, exclude=exclude, recalc=recalc)
                if not len(include_subset) == 0:
                    pi = pl.lit(np.pi)
                    I_df = self.targ.get_data(f"{pre}{'_' if pre else ''}{col_name[0]}", include=include_subset, strict=True)
                    Q_df = self.targ.get_data(f"{pre}{'_' if pre else ''}{col_name[1]}", include=include_subset, strict=True)
                    I_cols, Q_cols = I_df.columns, Q_df.columns
                    IQ_df = pl.concat([I_df, Q_df], how='horizontal')

                    mismatch_df = (IQ_df.lazy()
                                        .select(pl.all().head(mean_points).mean().name.prefix('first_'), pl.all().tail(mean_points).mean().name.prefix('last_'))
                                        .select([(pi - (pl.arctan2(pl.col(f"first_{Q_col}"), pl.col(f"first_{I_col}")) % (2*pi) +
                                                        pl.arctan2(pl.col(f"last_{Q_col}"),  pl.col(f"last_{I_col}")) % (2*pi))/2).alias(I_col.split('_')[-1]) for I_col, Q_col in zip(I_cols, Q_cols)])
                                        .collect())
                    ccat_df.add_data_to_properties(self, mismatch_df, mismatch_col_name)
                mismatch_angle = self.get_properties(mismatch_col_name, include=include, exclude=exclude, strict=True).to_numpy().T[1]
                angles[i] = mismatch_angle
            elif rotation == 'timestream':
                timestream_col_name = f'{pre}_{col_name[-1]}_angle'

                # Calculate angle of center of timestream (center determined using I and Q medians)
                include_subset = ccat_df.check_properties(self, timestream_col_name, include=include, exclude=exclude, recalc=recalc)
                if not len(include_subset) == 0:
                    pi = pl.lit(np.pi)
                    I_df = self.stream.get_data(f"{pre}{'_' if pre else ''}{col_name[0]}", include=include_subset, strict=True)
                    Q_df = self.stream.get_data(f"{pre}{'_' if pre else ''}{col_name[1]}", include=include_subset, strict=True)
                    I_cols, Q_cols = I_df.columns, Q_df.columns
                    IQ_df = pl.concat([I_df, Q_df], how='horizontal')
                    timestream_df = (IQ_df.lazy()
                                          .select(pl.all().median())
                                          .select([(pl.arctan2(pl.col(Q_col), pl.col(I_col)) % (2*pi)).alias(I_col.split('_')[-1]) for I_col, Q_col in zip(I_cols, Q_cols)])
                                          .collect())
                    ccat_df.add_data_to_properties(self, timestream_df, timestream_col_name)
                timestream_angle = self.get_properties(timestream_col_name, include=include, exclude=exclude, strict=True).to_numpy().T[1]
                angles[i] = timestream_angle
            else:
                error = f"Invalid rotation '{rotation}' specified; Must be one of 'mismatch' or 'timestream'."
                log.log('ERROR', error)
                raise ValueError(error)

        data_objs, _ = self._get_data_obj(data)
        if not data_objs:
            error = f"Invalid data type {data}, must be 'targ', 'timestream', or 'both'."
            log.log('ERROR', error)
            raise ValueError(error)

        rotation_dfs = []
        for data_obj in data_objs:
            data_obj.IQ_rotate(prefix=prefix, angle=angles, name=f'{col_name[-1]}', include = include, exclude=exclude, recalc=recalc)
            rotation_dfs.append(data_obj.get_data(col_name=[f"{col_name[-1]}_rotate_{pre}{'_' if pre else ''}{col_name[0]}" for pre in prefix] +
                                                           [f"{col_name[-1]}_rotate_{pre}{'_' if pre else ''}{col_name[1]}" for pre in prefix], include=include, exclude=exclude, strict=True))
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
            med_col = ['stream_median']

            include_subset = ccat_df.check_properties(self, f"{med_col[0]}_{prefix}{'_' if prefix else ''}{col_name[0]}", include=include, exclude=exclude, recalc=recalc)
            if not len(include_subset) == 0:
                median_I_df = self.stream.get_data(f"{prefix}{'_' if prefix else ''}{col_name[0]}", include=include_subset, strict=True).select(pl.all().median().name.map(lambda s: s.split('_')[-1]))
                median_Q_df = self.stream.get_data(f"{prefix}{'_' if prefix else ''}{col_name[1]}", include=include_subset, strict=True).select(pl.all().median().name.map(lambda s: s.split('_')[-1]))

                ccat_df.add_data_to_properties(self, median_I_df, f"{med_col[0]}_{prefix}{'_' if prefix else ''}{col_name[0]}")
                ccat_df.add_data_to_properties(self, median_Q_df, f"{med_col[0]}_{prefix}{'_' if prefix else ''}{col_name[1]}")
            median_I, median_Q = self.get_properties([f"{med_col[0]}_{prefix}{'_' if prefix else ''}{col_name[0]}",
                                                      f"{med_col[0]}_{prefix}{'_' if prefix else ''}{col_name[1]}"], include=include, exclude=exclude, strict=True).to_numpy().T[1:3]
            return median_I, median_Q

        col_name = ['I', 'Q']

        if isinstance(prefix, str): prefix = [prefix]
        num_prefix = len(prefix)
        noise_tones = self.stream.noise_tones

        if use_noise_tones and not (noise_tones is None): # Use noise tones
            col_name += ['noise_shift']

            include_subset = ccat_df.check_properties(self, f'closest_noise_tone', include=include, exclude=exclude, recalc=recalc)
            if not len(include_subset) == 0:
                noise_freqs = self.get_properties('tone_freqs', include=noise_tones, strict=True).to_numpy().T[1]
                closest_tones = (self.get_properties('tone_freqs', include=include_subset, strict=True)
                                     .lazy()
                                     .select(['det'] + [((pl.col('tone_freqs') - freq).abs()/pl.lit(1e6)).alias(f'{tone:0{self.stream.padding}d}') for tone, freq in zip(noise_tones, noise_freqs)])
                                     .collect()
                                     .unpivot(index='det', variable_name='closest_noise_tone', value_name='dist')
                                     .lazy()
                                     .with_columns(pl.col('closest_noise_tone').cast(pl.Int32))
                                     .filter(pl.col('dist') == pl.col('dist').min().over('det'))
                                     .sort('det')
                                     .select(['det', 'closest_noise_tone'])
                                     .collect())
                shared_cols = 'closest_noise_tone' if 'closest_noise_tone' in self._properties_df.schema else []
                self._properties_df = ccat_df.coalesce_join(self._properties_df, closest_tones, on='det', shared_cols = shared_cols)
            closest_tones = self.get_properties('closest_noise_tone', include=include, exclude=exclude, strict=True).to_numpy().T[1]
            noise_median_I, noise_median_Q = _get_medians('')

            col_names = [[]]*num_prefix
            median_Is, median_Qs = [[]]*num_prefix, [[]]*num_prefix
            for i, pre in enumerate(prefix):
                col_names[i] = col_name[:-1] + [f"{col_name[-1]}{'_' if pre else ''}{pre}"]
                median_Is[i], median_Qs[i] = _get_medians(pre)
            args = [[median_I, median_Q, noise_median_I, noise_median_Q, closest_tones] for median_I, median_Q in zip(median_Is, median_Qs)]
            self.stream.transform([Detector.calc_noise_shift]*num_prefix, *args, include=include, exclude=exclude, recalc=recalc, col_name = col_names)
            return self.stream.get_data(col_name=[f'{col_name[-1]}_{col_name[0]}' for col_name in col_names] +
                                                 [f'{col_name[-1]}_{col_name[1]}' for col_name in col_names], include=include, exclude=exclude)
        else:
            col_name += ['noise_rotate']

            for pre in prefix:
                median_I, median_Q = _get_medians(pre)

                self.stream.IQ_shift(prefix=pre, shift_I = -1*median_I, shift_Q = -1*median_Q, name='', include=include, exclude=exclude, recalc=recalc)
                self.stream.IQ_rotate(prefix=f'shift_{pre}', angle=np.pi/2, name='noise', include=include, exclude=exclude, recalc=recalc)
                self.stream.IQ_shift(prefix=f'noise_rotate_shift_{pre}', shift_I = median_I, shift_Q = median_Q, name='', include=include, exclude=exclude, recalc=recalc)

                self.stream.data = self.stream.data.with_columns([pl.col(col).alias(col.replace('shift_noise_rotate_shift', col_name[-1])) for col in self.stream.data.select(pl.col('^shift_noise_rotate_shift_.*$')).columns])
            return self.stream.get_data(col_name=[f"{col_name[-1]}{'_' if pre else ''}{pre}" for pre in prefix], include=include, exclude=exclude)

    # Timestream conversion
    # ---------------------
    def phase_spline(self,
                     prefix: str | list[str] = 'mismatch_rotate_origin_shift_origin_rotate_unwind_rotate',
                     phase_low: float = -3.14,
                     phase_up: float = 3.14,
                     k: int = 3,
                     include: int | list[int] | None = None,
                     exclude: int | list[int] | None = None,
                     recalc: bool = False,
                     max_workers=1,
                     ex = None,
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

        if not isinstance(phase_low, Iterable) or not len(phase_low) == num_prefix: phase_low = [phase_low]*num_prefix
        if not isinstance(phase_up, Iterable) or not len(phase_up) == num_prefix: phase_up = [phase_up]*num_prefix

        col_names = [[]]*num_prefix
        for i, pre in enumerate(prefix):
            col_names[i] = [col_name[0], f"{pre}{'_' if pre else ''}{col_name[1]}", 'to_' + col_name[-2] + '_spline', f"to_{pre}{'_' if pre else ''}{col_name[-1]}_spline"]

        self.properties
        stream_timestamp = self.stream.timestamp
        args = [[self, low, up, k, stream_timestamp, ccat_mp.check_max_workers(max_workers), ex] for low, up in zip(phase_low, phase_up)]
        self.targ.transform([Detector.calc_phase_spline]*num_prefix, *args, include=include, exclude=exclude, recalc=recalc, col_name = col_names)
        self.targ.data = self.targ._unnest(['struct_' + col_name[-1] for col_name in col_names])
        return self.targ.get_data(col_name=([col_name[-1] for col_name in col_names] + [col_name[-2] for col_name in col_names]), include=include, exclude=exclude)

    def phase_to_f(self,
                   prefix: str | list[str] = 'mismatch_rotate_origin_shift_origin_rotate_unwind_rotate',
                   spline_col: str = 'mismatch_rotate_origin_shift_origin_rotate_unwind_rotate',
                   phase_bounds: float = 0.2,
                   k: int = 3,
                   include: int | list[int] | None = None,
                   exclude: int | list[int] | None = None,
                   recalc: bool = False,
                   max_workers=1,
                   ex = None,
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

        col_name = ['phase', 'f']
        if isinstance(prefix, str): prefix = [prefix]
        num_prefix = len(prefix)

        if isinstance(spline_col, str) or not len(spline_col) == num_prefix: spline_col = [spline_col]*num_prefix

        col_names, min_phases, max_phases = [[]]*num_prefix, [[]]*num_prefix, [[]]*num_prefix
        for i, (pre, spline) in enumerate(zip(prefix, spline_col)):
            col_names[i] = [f"{pre}{'_' if pre else ''}{name}" for name in col_name]
            spline_names = [f"{spline}{'_' if spline else ''}{name}" for name in col_name]

            include_subset = ccat_df.check_properties(self, f'min_{spline_names[0]}', include=include, exclude=exclude, recalc=recalc)
            if not len(include_subset) == 0:
                phase_df = self.stream.get_data(spline_names[0], include=include_subset, strict=True)
                min_df, max_df = phase_df.select([pl.all().min().name.map(lambda s: s.split('_')[-1])]), phase_df.select([pl.all().max().name.map(lambda s: s.split('_')[-1])])
                ccat_df.add_data_to_properties(self, min_df, f'min_{spline_names[0]}'), ccat_df.add_data_to_properties(self, max_df, f'max_{spline_names[0]}')
            min_phase, max_phase = self.get_properties([f'min_{spline_names[0]}', f'max_{spline_names[0]}'], include=include, exclude=exclude, strict=True).to_numpy().T[1:3]
            min_phases[i], max_phases[i] = min_phase - phase_bounds, max_phase + phase_bounds

        self.phase_spline(prefix=spline_col, phase_low = min_phases, phase_up = max_phases, k = k, include=include, exclude=exclude, recalc=recalc, max_workers=1, ex=ex, **kwargs)
        spline_dict = self.stream.spline_dict
        y_to_x_spline = [self.get_properties(col_name = f'{col}_phase_to_f_spline', include=include, exclude=exclude, strict=True).to_numpy().T[1] for col in spline_col]
        x_to_y_spline = [self.get_properties(col_name = f'f_to_{col}_phase_spline', include=include, exclude=exclude, strict=True).to_numpy().T[1] for col in spline_col]

        args = [[[spline_dict[spline] for spline in y_to_x],
                 [spline_dict[spline] for spline in x_to_y],
                 ccat_mp.check_max_workers(max_workers),
                 ex] for y_to_x, x_to_y in zip(y_to_x_spline, x_to_y_spline)]
        self.stream.transform([Detector.calc_phase_to_f]*num_prefix, *args, include=include, exclude=exclude, recalc=recalc, col_name = col_names)
        self.stream.data = self.stream._unnest(['struct_' + col_name[-1] for col_name in col_names])
        return self.stream.get_data(col_name=[col_name[-1] for col_name in col_names], include=include, exclude=exclude)

    def frac_f(self,
               prefix: str | list[str] = 'mismatch_rotate_origin_shift_origin_rotate_unwind_rotate',
               f_0: str | list[float] = 'tone_freqs',
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

        if isinstance(f_0, str): f_0 = self.get_properties(f_0, include=include, exclude=exclude, strict=True).to_numpy().T[1]
        if not isinstance(f_0, Iterable) or not len(f_0) == num_prefix: f_0 = [f_0]*num_prefix

        col_names = [[]]*num_prefix
        for i, pre in enumerate(prefix):
            col_names[i] = [f"{pre}{'_' if pre else ''}{col_name[0]}", col_name[-1]]

        args = [[f] for f in f_0]
        self.stream.transform([Detector.calc_frac_f]*num_prefix, *args, include=include, exclude=exclude, recalc=recalc, col_name = col_names)
        return self.stream.get_data(col_name=[f"{col_name[-1]}_{col_name[0]}" for col_name in col_names], include=include, exclude=exclude)

    # High Level Analysis Methods
    # ---------------------------

    def mag_min(self,
                include: int | list[int] | None = None,
                exclude: int | list[int] | None = None,
                recalc: bool = False) -> list[pl.DataFrame]:
        col_name = ['f', 'mag', 'min']
        prop_names = [f'{col_name[-1]}_{col_name[1]}_{col_name[0]}', f'{col_name[-1]}_{col_name[1]}']

        include_subset = ccat_df.check_properties(self, prop_names[0], include=include, exclude=exclude, recalc=recalc)
        if not len(include_subset) == 0:
            # Get detector magnitudes and frequencies and unpivot DataFrame from wide to long format
            f_df = self.targ.get_data(col_name=col_name[0], strict=True, include=include_subset)
            mag_df = self.targ.get_data(col_name=col_name[1], strict=True, include=include_subset)

            mag_df = (mag_df.unpivot(variable_name='det',
                                     value_name=col_name[1])
                            .with_columns(pl.col('det').str.strip_prefix(f'{col_name[1]}_').cast(pl.Int32)))

            f_df = (f_df.unpivot(variable_name='tmp',
                                 value_name=col_name[0])
                        .drop('tmp'))

            mag_f_df = pl.concat([mag_df, f_df], how='horizontal')

            # Get minimum magnitude values for each detector and corresponding sample numbers

            min_df = (mag_f_df.filter((pl.col(col_name[1]) == pl.col(col_name[1]).min()).over('det'))
                              .rename({col_name[0]: prop_names[0], col_name[1]: prop_names[1]}))
            shared_cols = prop_names if prop_names[0] in self._properties_df.schema else []
            self._properties_df = ccat_df.coalesce_join(self._properties_df, min_df, 'det', shared_cols)
        return self.get_properties(col_name = prop_names, include=include, exclude=exclude, strict=True)

    def IQ_circle_center(self):
        '''
        Remove cable delay from IQ circle and center at the origin
        '''
        return

    def IQ_max_dist(self,
                    trim_window: int = 2,
                    trim_savgol_window: int = 9,
                    diff_savgol_window: int = 21,
                    trim_savgol_k: int = 1,
                    diff_savgol_k: int = 1,
                    include=None,
                    exclude=None,
                    recalc=False,
                    max_workers=1,
                    ex=None):
        '''
        Get the frequency corresponding to the max geometric distance between adjacent points in IQ space
        '''
        trim_savgol, diff_savgol = trim_savgol_window > 1, diff_savgol_window > 1
        # Calculate the geometric distance between adjacent points in IQ space
        self.cable_delay
        self.targ.mag(include=include, exclude=exclude, recalc=recalc)

        if trim_savgol: self.targ.savgol(col_name='mag', prefix='', window=trim_savgol_window, k=trim_savgol_k, deriv=0, max_workers=max_workers, ex=ex, include=include, exclude=exclude, recalc=recalc)
        self.IQ_unwind(prefix='', data='targ', delay_col = 'network_cable_delay', include=include, exclude=exclude, recalc=recalc)
        self.IQ_trim(prefix='unwind_rotate',  window=trim_window, use_fit=False, mag_prefix=f"{'savgol0' if trim_savgol else ''}", include=include, exclude=exclude, recalc=recalc)

        if diff_savgol:
            self.targ.savgol(col_name='I', prefix='tail_trim_unwind_rotate', window=diff_savgol_window, k=diff_savgol_k, deriv=1, include=include, exclude=exclude, recalc=recalc, max_workers=max_workers, ex=ex)
            self.targ.savgol(col_name='Q', prefix='tail_trim_unwind_rotate', window=diff_savgol_window, k=diff_savgol_k, deriv=1, include=include, exclude=exclude, recalc=recalc, max_workers=max_workers, ex=ex)
        else:
            self.targ.diff(col_name='I', prefix='tail_trim_unwind_rotate', include=include, exclude=exclude, recalc=recalc)
            self.targ.diff(col_name='Q', prefix='tail_trim_unwind_rotate', include=include, exclude=exclude, recalc=recalc)
        self.targ.mag(prefix=f"{'savgol1' if diff_savgol else 'diff'}_tail_trim_unwind_rotate", include=include, exclude=exclude, recalc=recalc)

        # Get the frequencies corresponding to the max distance in IQ space (same as frequency with steepest phase gradient)
        include_subset = ccat_df.check_properties(self, 'max_IQ_dist_f', include=include, exclude=exclude, recalc=recalc)
        if not len(include_subset) == 0:
            diff_IQ = (self.targ.get_data(['sample', f"{'savgol1' if diff_savgol else 'diff'}_tail_trim_unwind_rotate_mag"], strict=True, include=include_subset)
                                .rechunk()
                                .lazy()
                                .unpivot(index='sample', value_name='IQ', variable_name='temp')
                                .drop('temp')
                                .collect())
            f = (self.targ.get_data(['f'], strict=True, include=include_subset)
                          .lazy()
                          .unpivot(value_name='f', variable_name='det')
                          .with_columns((pl.col('det').str.strip_prefix('f_')).cast(pl.Int32))
                          .collect())
            full_df = pl.concat([f, diff_IQ], how='horizontal')
            max_sample = (full_df.lazy()
                                 .filter(~pl.col('IQ').is_nan())
                                 .filter((pl.col('IQ') == pl.col('IQ').max()).over('det'))
                                 .select('det', pl.col('sample').alias('max_sample'))
                                 .group_by('det').agg(pl.col('max_sample').first())
                                 .collect())

            full_df = full_df.join(max_sample, on='det', how='left')
            max_IQ = (full_df.lazy()
                             .filter(pl.col('sample') == pl.col('max_sample'))
                             .select('det', pl.col('sample').alias('max_IQ_dist_sample'), pl.col('f').alias('max_IQ_dist_f'), pl.col('IQ').alias('max_IQ_dist'))
                             .collect())
            adj_IQ = (full_df.lazy()
                             .filter(pl.col('sample') == pl.col('max_sample') - 1)
                             .select('det', pl.col('f').alias('max_IQ_dist_adj_f'))
                             .collect())

            max_IQ = max_IQ.join(adj_IQ, on='det', how='left')

            # Use tone frequencies for detectors where finding the max distance frequency failed
            tone_freq_df = self.get_properties('tone_freqs', strict=True, include=include_subset)
            max_IQ = (max_IQ.join(tone_freq_df, on='det', how='right', coalesce=True)
                            .lazy()
                            .with_columns(pl.when(pl.col('max_IQ_dist_f').is_null())
                                            .then(pl.col('tone_freqs'))
                                            .otherwise(pl.col('max_IQ_dist_f')).alias('max_IQ_dist_f'))
                            .drop('tone_freqs')
                            .collect())
            shared_cols = ['max_IQ_dist_f', 'max_IQ_dist', 'max_IQ_dist_sample', 'max_IQ_dist_adj_f'] if 'max_IQ_dist_f' in self._properties_df.schema else []
            self._properties_df = ccat_df.coalesce_join(self.properties, max_IQ, 'det', shared_cols)
            self.targ._properties_df = ccat_df.coalesce_join(self.targ.properties, max_IQ, 'det', shared_cols)

        max_IQ_f = self.get_properties('max_IQ_dist_f', include=include, exclude=exclude, strict=True)
        return max_IQ_f

    def is_bifurcated(self,
                      bifurcation_threshold=60,
                      qifurcation_threshold=50,
                      trim_window: int = 2,
                      trim_savgol_window: int = 9,
                      trim_savgol_k: int = 1,
                      include=None,
                      exclude=None,
                      recalc=False,
                      max_workers=1,
                      ex=None):
        '''
        Determine if a detector is bifurcated or has a high quasiparticle nonlinearity using the angle of the maximally seperated points in IQ space


        '''



        # Get maximally distant points in IQ space
        # ----------------------------------------
        self.IQ_max_dist(diff_savgol_window=1,
                         trim_window=trim_window,
                         trim_savgol_window=trim_savgol_window,
                         trim_savgol_k=trim_savgol_k,
                         include=include,
                         exclude=exclude,
                         recalc=recalc,
                         max_workers=max_workers,
                         ex=ex)

        # Fit IQ circle to get radius
        # ---------------------------
        self.IQ_circle_fit(prefix='tail_trim_unwind_rotate',
                           include=include,
                           exclude=exclude,
                           recalc=recalc,
                           max_workers=max_workers,
                           ex=ex)

        # Get frequency corresponding to the |S_21| minimum
        # -------------------------------------------------
        self.mag_min(include=include, exclude=exclude, recalc=recalc)

        include_subset = ccat_df.check_properties(self, 'bifurcated', include=include, exclude=exclude, recalc=recalc)

        added_cols = ['bifurcated', 'qifurcated', 'sin_half_max_IQ_angle', 'chord_length_ratio', 'max_IQ_angle_rad', 'max_IQ_angle_deg']
        if not len(include_subset) == 0:
            df = self.get_properties(['max_IQ_dist',
                                      'max_IQ_dist_f',
                                      'max_IQ_dist_adj_f',
                                      'circle_fit_tail_trim_unwind_rotate_R',
                                      'min_mag_f'], include=include_subset, strict=True)

            bif_df = (df.lazy()
                        .with_columns((0.5*(pl.col('max_IQ_dist')/pl.col('circle_fit_tail_trim_unwind_rotate_R'))).alias('sin_half_max_IQ_angle'),
                                    (0.5*(4 - (pl.col('max_IQ_dist')/pl.col('circle_fit_tail_trim_unwind_rotate_R'))**2).sqrt()).alias('chord_length_ratio'))
                        .with_columns((2*pl.col('sin_half_max_IQ_angle').arcsin()).alias('max_IQ_angle_rad'))
                        .with_columns(((180/np.pi)*pl.col('max_IQ_angle_rad')).alias('max_IQ_angle_deg'))
                        .with_columns(((pl.col('max_IQ_angle_deg') >= bifurcation_threshold) & (pl.col('max_IQ_dist_adj_f') < pl.col('min_mag_f'))).alias('bifurcated'),
                                    ((pl.col('max_IQ_angle_deg') >= qifurcation_threshold) & (pl.col('max_IQ_dist_f') > pl.col('min_mag_f'))).alias('qifurcated'))
                        .select(['det'] + added_cols)
                        .collect())
            shared_cols = added_cols if 'bifurcated' in self._properties_df.schema else []
            self._properties_df = ccat_df.coalesce_join(self.properties, bif_df, 'det', shared_cols)

        bif_df = self.get_properties(['bifurcated', 'qifurcated'], include=include, exclude=exclude, strict=True)
        return bif_df

    #==================#
    # Analysis Methods #
    #==================#

    @staticmethod
    def calc_complex_fit(schema, *args, tones: list[int], padding: int = 4, recalc: bool = False, col_name = ['f', 'I', 'Q', 'complex_fit']):
        ''' Fit using resonator_model_v3
        '''

        def _complex_fit(df):
            data = ccat_mp.struct_batches(df, 3, batch_len, max_workers)

            results_dict = {}
            with ccat_mp.optional_executor(max_workers, ex=ex) as executor:
                future_to_batch = {executor.submit(ccat_mp.process_batches,
                                                   resonator_model_v3.nonlinear_fit,
                                                   data[i][0],
                                                   data[i][1],
                                                   data[i][2],
                                                   nonlinear=nonlinear[inds],
                                                   asymm=asymm[inds],
                                                   fix_cable=fix_cable[inds],
                                                   fix_thetaQ=fix_thetaQ[inds]):  (i, tones, cols) for i, (tones, inds, cols) in enumerate(zip(to_calc, calc_ind, batches))}

                for future in concurrent.futures.as_completed(future_to_batch):
                    i, tones, cols = future_to_batch[future]
                    f_cols = data[i][0]
                    fit_cols = future.result()
                    for tone, f_col, (_, I_col, Q_col), fit_col in zip(tones, f_cols, cols, fit_cols):
                        if isinstance(fit_col, Exception):
                            log.log('DEBUG', 'Fit failed for tone %s with exception: %s', tone, fit_col)
                            best_fit, cable_fit = np.zeros(df.len()), np.zeros(df.len())
                            self.targ._properties[f'det_{tone:0{padding}d}'] = {}
                        else:
                            best_fit = fit_col.best_fit
                            cable_fit = resonator_model_v3.fine_s21_model(f_col, fit_col.params, cable=True)
                            best_vals_dict = {f'{col_name[-1]}_{k}': float(v) for k, v in fit_col.best_values.items()}
                            self.targ._properties[f'det_{tone:0{padding}d}'] = best_vals_dict

                        results_dict[f'{col_name[-1]}_{I_col}'] = best_fit.real
                        results_dict[f'{col_name[-1]}_{Q_col}'] = best_fit.imag
                        results_dict[f'cable_{col_name[-1]}_{I_col}'] = cable_fit.real
                        results_dict[f'cable_{col_name[-1]}_{Q_col}'] = cable_fit.imag

            return ccat_mp.package_results(results_dict)

        if len(args) == 7:
            self, nonlinear, asymm, fix_cable, fix_thetaQ, max_workers, ex = np.array(args)
            if tones is not None: self, max_workers, ex = self[0], int(max_workers[0]), ex[0]
        else:
            error = 'nonlinear, asymm, fix_cable, and fix_thetaQ are required arguments.'
            log.log('ERROR', error)
            raise ValueError(error)

        return_col = [f'{col_name[-1]}_{col_name[1]}', f'{col_name[-1]}_{col_name[2]}', f'cable_{col_name[-1]}_{col_name[1]}', f'cable_{col_name[-1]}_{col_name[2]}']
        return_type = [pl.Float64, pl.Float64, pl.Float64, pl.Float64]
        expr, to_calc, calc_ind, calc_col, batches, batch_len = ccat_mp.create_batches(_complex_fit, tones, col_name, schema, return_col=return_col, return_type=return_type, padding=padding, max_workers=max_workers, recalc=recalc)
        return expr

    @staticmethod
    def calc_phase_fit(schema, *args, tones: list[int], padding: int = 4, recalc: bool = False, col_name = ['f', 'I', 'Q', 'phase', 'phase_fit']):
        def _phase_fit(df):
            data = ccat_mp.struct_batches(df, 4, batch_len, max_workers)

            results_dict = {}
            with ccat_mp.optional_executor(max_workers, ex=ex) as executor:
                future_to_batch = {executor.submit(ccat_mp.process_batches,
                                                   ccat_fit.phase_fit,
                                                   data[i][0],
                                                   data[i][3],
                                                   I = data[i][1],
                                                   Q = data[i][2],
                                                   nonlinear = nonlinear[inds],
                                                   method=method[inds],
                                                   params = params[inds],
                                                   R = radius[inds],
                                                   window=window[inds]):  (tones, cols) for i, (tones, inds, cols) in enumerate(zip(to_calc, calc_ind, batches))}

                for future in concurrent.futures.as_completed(future_to_batch):
                    tones, cols = future_to_batch[future]
                    fit_cols = future.result()
                    for tone, (_, _, _, phase_col), fit_col in zip(tones, cols, fit_cols):
                        if isinstance(fit_col, Exception):
                            log.log('DEBUG', 'Fit failed for tone %s with exception: %s', tone, fit_col)
                            best_fit = np.full(df.len(), np.nan)
                            self.targ._properties[f'det_{tone:0{padding}d}'] = {}
                        else:
                            best_fit = np.full(df.len(), np.nan)
                            mask = fit_col.mask
                            best_fit[mask] = fit_col.best_fit
                            best_vals_dict = {f'{col_name[-1]}_{prefix}_{k}': float(v) for k, v in fit_col.best_values.items()}
                            init_vals_dict = {f'{col_name[-1]}_{prefix}_init_{k}': float(v) for k, v in fit_col.init_values.items()}
                            best_vals_dict[f'{col_name[-1]}_{prefix}_params'] = fit_col.params
                            self.targ._properties[f'det_{tone:0{padding}d}'] = best_vals_dict | init_vals_dict
                        results_dict[f'{col_name[-1]}_{phase_col}'] = best_fit
            return ccat_mp.package_results(results_dict)

        if len(args) == 9:
            self, prefix, radius, nonlinear, method, params_list, window, max_workers, ex = args
            params = np.empty(len(params_list), dtype=object)
            for i in range(len(params_list)): params[i] = params_list[i]
            radius, nonlinear, method, window = np.array(radius), np.array(nonlinear), np.array(method), np.array(window)
            if tones is not None: self, prefix, max_workers, ex = self[0], prefix[0], int(max_workers[0]), ex[0]
        else:
            error = 'nonlinear, prefix, params, window, and max_workers are required arguments.'
            log.log('ERROR', error)
            raise ValueError(error)

        return_col, return_type = [f'{col_name[-1]}_{col_name[-2]}'], [pl.Float64]
        expr, to_calc, calc_ind, calc_col, batches, batch_len = ccat_mp.create_batches(_phase_fit, tones, col_name, schema, padding=padding, return_col=return_col, return_type=return_type, max_workers=max_workers, recalc=recalc)
        return expr

    @staticmethod
    def calc_IQ_circle_fit(schema, *args, tones: list[int], padding: int = 4, recalc: bool = False, col_name = ['I', 'Q', 'circle_fit']):
        def _circle_fit(df):
            data = ccat_mp.struct_batches(df, 2, batch_len, max_workers)

            angles = np.linspace(0, 2*np.pi, df.len())
            sin = np.sin(angles)
            cos = np.cos(angles)

            property_keys = ['center_I', 'center_Q', 'R', 'A', 'D', 'theta', 'optimality', 'nfev', 'njev']
            results_dict = {}
            with ccat_mp.optional_executor(max_workers, ex=ex) as executor:
                future_to_batch = {executor.submit(ccat_mp.process_batches,
                                                   ccat_fit.circle_fit,
                                                   data[i][0],
                                                   data[i][1],
                                                   full_output=[True]*len(tones),
                                                   bounds=bounds[inds],
                                                   loss=loss[inds],
                                                   f_scale=f_scale[inds],
                                                   method=method[inds]):  (tones, cols) for i, (tones, inds, cols) in enumerate(zip(to_calc, calc_ind, batches))}

                for future in concurrent.futures.as_completed(future_to_batch):
                    tones, cols = future_to_batch[future]
                    fit_cols = future.result()
                    for tone, (I_col, Q_col), fit_col in zip(tones, cols, fit_cols):
                        if isinstance(fit_col, Exception):
                            log.log('DEBUG', 'Fit failed for tone %s with exception: %s', tone, fit_col)
                            fit_I, fit_Q = np.full(df.len(), np.nan), np.full(df.len(), np.nan)
                            self.targ._properties[f'det_{tone:0{padding}d}'] = {}
                        else:
                            I_c, Q_c, R, result = fit_col
                            fit_I, fit_Q = R*cos + I_c, R*sin + Q_c
                            A, D, theta = result.x
                            property_vals = [I_c, Q_c, R, A, D, theta, result.optimality, result.nfev, result.njev]
                            self.targ._properties[f'det_{tone:0{padding}d}'] = {f'{col_name[-1]}_{prefix}_{k}': v for k, v in zip(property_keys, property_vals)}
                        results_dict[f'{col_name[-1]}_{I_col}'] = fit_I
                        results_dict[f'{col_name[-1]}_{Q_col}'] = fit_Q
            return ccat_mp.package_results(results_dict)

        if len(args) == 8:
            self, prefix, bounds, loss, f_scale, method, max_workers, ex = np.array(args)
            if tones is not None: self, prefix, max_workers, ex = self[0], prefix[0], int(max_workers[0]), ex[0]
        else:
            error = 'self, prefix, bounds, loss, f_scale, method, and max_workers are required arguments.'
            log.log('ERROR', error)
            raise ValueError(error)

        return_col, return_type = [f'{col_name[-1]}_{col_name[1]}', f'{col_name[-1]}_{col_name[2]}'], [pl.Float64, pl.Float64]
        expr, to_calc, calc_ind, calc_col, batches, batch_len = ccat_mp.create_batches(_circle_fit, tones, col_name, schema, padding=padding, return_col = return_col, return_type=return_type, max_workers=max_workers, recalc=recalc)
        return expr

    @staticmethod
    def calc_noise_shift(schema, *args, tones: list[int], padding: int = 4, recalc: bool = False, col_name = ['I', 'Q', 'noise']):
        if tones is None:
            error = 'ERROR HERE'
            log.log('ERROR', error, name='analysis.detector')
            raise RuntimeError(error)
        
        if not len(args) == 5: log.log('ERROR', "'tone_med_I', 'tone_med_Q', 'noise_med_I', 'noise_med_Q', and 'noise_tone' are required arguments")
        tone_med_Is, tone_med_Qs, noise_med_Is, noise_med_Qs, noise_tones = np.array(args)
        I_shifts, Q_shifts = noise_med_Is - tone_med_Is, noise_med_Qs - tone_med_Qs

        col_names = [[f'{name}_{noise_tone:0{padding}d}' for name in col_name[:-1]] + [col_name[-1]] for noise_tone in noise_tones]

        exprs = []
        for i, (col_name, tone, I_shift, Q_shift) in enumerate(zip(col_names, tones, I_shifts, Q_shifts)):
            I_col_noise, Q_col_noise, shift_col = col_name
            I_col_tone, Q_col_tone = f"{I_col_noise.split('_')[0]}_{tone:0{padding}d}", f"{Q_col_noise.split('_')[0]}_{tone:0{padding}d}"

            if recalc or not (f'{shift_col}_{I_col_tone}' in schema):
                exprs += [(pl.col(I_col_noise) - I_shift).alias(f'{shift_col}_{I_col_tone}'),
                          (pl.col(Q_col_noise) - Q_shift).alias(f'{shift_col}_{Q_col_tone}')]
            else:
                exprs.append(pl.col(f'{shift_col}_{I_col_tone}'))
        return exprs

    @staticmethod
    def calc_phase_spline(schema, *args, tones: list[int], padding: int = 4, recalc: bool = False, col_name = ['f', 'phase', 'to_f', 'to_phase']):
        def _phase_spline(df):
            data = ccat_mp.struct_batches(df, 2, batch_len, max_workers)

            results_dict = {}
            with ccat_mp.optional_executor(max_workers, ex=ex) as executor:
                future_to_batch = {executor.submit(ccat_mp.process_batches,
                                                   ccat_fit.y_to_x_spline,
                                                   data[i][0],
                                                   data[i][1],
                                                   k = k[inds],
                                                   y_low = phase_low[inds],
                                                   y_up = phase_up[inds]):  (i, tones) for i, (tones, inds, _) in enumerate(zip(to_calc, calc_ind, batches))}

                for future in concurrent.futures.as_completed(future_to_batch):
                    i, tones = future_to_batch[future]
                    f_cols, phase_cols = data[i][0], data[i][1]
                    spline_cols = future.result()
                    for tone, f_col, phase_col, spline_col in zip(tones, f_cols, phase_cols, spline_cols):
                        property_dict = {name: 'None' for name in interp_names}
                        spline_data = 2*[np.full(df.len(), np.nan)]
                        if isinstance(spline_col, Exception):
                            log.log('DEBUG', 'Spline calculation for tone %s failed with exception: %s', tone, spline_col)
                        else:
                            for i, (name, spline, data) in enumerate(zip(interp_names, spline_col, [phase_col, f_col])):
                                if spline is not None:
                                    spline.extrapolate = False
                                    spline_data[i] = spline(data)
                                    self.stream.spline_dict[str(spline)] = spline
                                    property_dict[name] = str(spline)
                        self.stream._properties[f'det_{tone:0{padding}d}'] = property_dict
                        for name, data in zip(interp_names, spline_data): results_dict[f'{name}_{stream_timestamp}_{tone:0{padding}d}'] = data
            return ccat_mp.package_results(results_dict)

        if len(args) == 7:
            self, phase_low, phase_up, k, stream_timestamp, max_workers, ex = np.array(args)
            if tones is not None: self, stream_timestamp, max_workers, ex = self[0], stream_timestamp[0], int(max_workers[0]), ex[0]
        else:
            error = 'self, phase_low, phase_up, k, max_workers are required arguments.'
            log.log('ERROR', error)
            raise ValueError(error)

        data_col_name = [col_name[0], col_name[1], col_name[-1]]
        interp_names = [f'{col_name[1]}_{col_name[-2]}', f'{col_name[0]}_{col_name[-1]}']
        calc_col = [f'{interp_names[0]}_{stream_timestamp}_{tone:0{padding}d}' for tone in tones]

        return_col = [f'{interp_names[0]}', f'{interp_names[1]}']
        return_type = [pl.Float64, pl.Float64]
        expr, to_calc, calc_ind, calc_col, batches, batch_len = ccat_mp.create_batches(_phase_spline, tones, data_col_name, schema, padding=padding, calc_col = calc_col, return_col=return_col, return_type=return_type, max_workers=max_workers, recalc=recalc)
        return expr

    @staticmethod
    def calc_phase_to_f(schema, *args, tones: list[int], padding: int = 4, recalc: bool = False, col_name = ['phase', 'f']):
        def _phase_to_f(df):
            data = ccat_mp.struct_batches(df, 1, batch_len, max_workers)

            results_dict = {}
            with ccat_mp.optional_executor(max_workers, ex=ex) as executor:
                future_to_batch = {executor.submit(ccat_mp.process_batches,
                                                   ccat_fit.y_to_x_interp,
                                                   data[i][0],
                                                   y_to_x_spline = y_to_x_spline[inds],
                                                   x_to_y_spline = x_to_y_spline[inds]): tones for i, (tones, inds, _) in enumerate(zip(to_calc, calc_ind, batches))}

                for future in concurrent.futures.as_completed(future_to_batch):
                    tones = future_to_batch[future]
                    f_cols = future.result()
                    for tone, f_col in zip(tones, f_cols):
                        if isinstance(f_col, Exception):
                            log.log('DEBUG', 'Interpolation for tone %s failed with exception: %s', tone, f_col)
                            f_col = np.full(df.len(), np.nan)
                        results_dict[f'{col_name[-1]}_{tone:0{padding}d}'] = f_col
            return ccat_mp.package_results(results_dict)

        if len(args) == 4:
            y_to_x_spline, x_to_y_spline, max_workers, ex = np.array(args)
            if tones is not None: max_workers, ex = int(max_workers[0]), ex[0]
        else:
            error = 'self, spline_col, and max_workers are required arguments.'
            log.log('ERROR', error)
            raise ValueError(error)
        calc_col = [f'{col_name[-1]}_{tone:0{padding}d}' for tone in tones]
        return_col, return_type = [f'{col_name[-1]}'], [pl.Float64]
        expr, to_calc, calc_ind, calc_col, batches, batch_len = ccat_mp.create_batches(_phase_to_f, tones, col_name, schema, padding=padding, calc_col = calc_col, return_col=return_col, return_type=return_type, max_workers=max_workers, recalc=recalc)
        return expr

    @staticmethod
    def calc_frac_f(schema, *args, tones: list[int], padding: int = 4, recalc: bool = False, col_name = ['f', 'frac']):
        if not len(args) == 1: log.log('ERROR', "'f_0' is a required argument")
        f_0s = args[0]

        col_names = [[f'{name}_{tone:0{padding}d}' for name in col_name[:-1]] + [col_name[-1]] for tone in tones] if tones is not None else [col_name]

        exprs = [None]*len(col_names)
        for i, (col_name, f_0) in enumerate(zip(col_names, f_0s)):
            f_col, frac_f_col = col_name

            if recalc or not (f'{frac_f_col}_{f_col}' in schema):
                exprs[i] = ((pl.col(f_col) - f_0)/f_0).name.prefix(frac_f_col + '_')
            else:
                exprs[i] = pl.col(f'{frac_f_col}_{f_col}')
        return exprs

    def properties_histogram(self, col_name, bins=None, mad_filter = True, num_mads = 10, filter_exprs = [], recalc=False, **kwargs):
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
            if filter_exprs: df = df.filter(filter_exprs)

            if mad_filter:
                df = (df.with_columns(pl.col(col_name).median().alias('median'))
                        .with_columns((pl.col(col_name) - pl.col('median')).abs().median().alias('MAD'))
                        .filter((pl.col(col_name) > (pl.col('median') - num_mads*pl.col('MAD'))) & (pl.col(col_name) < (pl.col('median') + num_mads*pl.col('MAD')))))
    
            bins = df.height // 8 if bins is None else min(bins, num_tones)
            data = df[col_name].to_numpy()

            counts, edges = np.histogram(data, bins)
            counts, edges = np.pad(np.array(counts, dtype=float), (0, int(num_tones - len(counts))), constant_values=None), np.pad(np.array(edges, dtype=float), (0, int(num_tones - len(edges))), constant_values=None)

            self.properties = properties.with_columns([pl.Series(name, value) for name, value in zip(hist_col_name, [counts, edges])])
        return self.properties.select(hist_col_name)

    #==================#
    # Plotting Methods #
    #==================#

    @staticmethod
    def _plot_histogram(df, plot_opts, *args, **kwargs):
        dynamic = kwargs['dynamic'] if 'dynamic' in kwargs else True
        by, plot_median = args

        if by is None: df = df.with_columns(pl.lit(True).alias('tmp'))

        hist_dict = {}
        for *vals, counts, edges, median in df.group_by(by if by is not None else ['tmp']).agg('counts', 'edges', 'median').iter_rows():
            label = ','.join([f'{name}={value}' for name, value in zip(by, vals)]) if by is not None else ''
            hist = hv.Histogram((edges, counts), label=label).relabel(group='Detector')
            if plot_median:
                median = median[0]
                vline = hv.VLine(median, label='Median').relabel(group='Detector')
                spike = hv.Curve(([median, median], [0, 1]), label=f'Median: {median:0.2e}').relabel(group='Detector')
                hist = hist*spike*vline
            hist_dict[tuple(vals)] = hist

        hist = hv.HoloMap(hist_dict, kdims=by)
        hist.opts(*plot_opts)

        if dynamic and by is not None: hist = hv.util.Dynamic(hist)
        return hist

    def properties_histogram_plot(self, 
                                  col_name, 
                                  plot_median=None, 
                                  xlabel = '' , 
                                  title='', 
                                  filter_exprs = [], 
                                  save_fig: bool | None = None, 
                                  figs_per_file: int | None = None,
                                  overwrite: bool | None = None, 
                                  save_dir: str | Path | None = None,
                                  save_name: str = None, 
                                  save_fmt: Format | None = None,
                                  return_fig=True, 
                                  return_df=False, 
                                  df = None, 
                                  by=None, 
                                  **kwargs):
        ''' Plot histogram of a detector property

        Args:
            col_name (str): Name of property
            plot_median (bool): Whether to plot a vertical line at the median
        Returns:
            return (hv.Histogram | hv.Overlay): Holoviews histogram figure
        '''

        if df is None:
            df = self.properties_histogram(col_name, **kwargs)
            df = (df.rename({col: name for col, name in zip(df.columns, ['counts', 'edges'])})
                    .with_columns(pl.lit(self.properties[col_name].median()).alias('median')))
            if filter_exprs: df = df.filter(filter_exprs)
        if not return_fig: return df, None

        if plot_median is None: plot_median = self.viz_cfg['static_plot']['histogram']['plot_median']
        if isinstance(by, str): by = [by]
        args = [by, plot_median]

        cfg = self.targ.drone_cfg['det_config']
        title = title if title else rf"${cfg['detector_type']}\ {cfg['network']}$"

        linewidth = kwargs['linewidth'] if 'linewidth' in kwargs else self.viz_cfg['static_plot']['histogram']['linewidth']
        linestyle = kwargs['linestyle'] if 'linestyle' in kwargs else self.viz_cfg['static_plot']['histogram']['linestyle']
        plot_opts = (opts.Histogram(xlabel=xlabel if xlabel else col_name,
                                    ylabel='Count',
                                    title=title,
                                    aspect=kwargs['aspect'] if 'aspect' in kwargs else self.viz_cfg['static_plot']['histogram']['aspect'],
                                    fig_size=kwargs['fig_size'] if 'aspect' in kwargs else self.viz_cfg['static_plot']['histogram']['fig_size'],
                                    show_grid=True,
                                    show_legend=True),
                    opts.Curve(linewidth=linewidth,
                               linestyle=linestyle),
                    opts.VLine(linewidth=linewidth,
                               linestyle=linestyle))

        # Create plot for immediate visualization
        # ---------------------------------------
        plot = Detector._plot_histogram(df, plot_opts, *args, **kwargs)

        # Save plot in background
        # -----------------------
        if save_name is None: save_name = f'hist_{col_name}'
        viz_utils.save_fig(self, Detector._plot_histogram, df, plot_opts, *args, 
                           save_fig = save_fig, figs_per_file = figs_per_file, overwrite=overwrite, save_dir = save_dir, save_name=save_name, save_fmt = save_fmt,
                           **kwargs)

        if return_df:
            return plot, df
        else:
            return plot

    #================#
    # Helper Methods #
    #================#

    def _get_data_obj(self, data: str):
        '''
        Get data object (Target, Timestream, or both) corresponding to string

        data (

        '''
        data_objs = []
        data_types = []
        if data == 'targ' or data == 'both':
            data_objs.append(self.targ)
            data_types.append('targ')
        if data == 'timestream' or data == 'both':
            data_objs.append(self.stream)
            data_types.append('stream')
        return data_objs, data_types

    @staticmethod
    def _load_data(data_class, com_to, cfg_path, analysis_cfg, viz_cfg, dets, noise_tones, timestamp, data_path, **kwargs):
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
                data = data_class(com_to = com_to, cfg_path = cfg_path, analysis_cfg = analysis_cfg, viz_cfg = viz_cfg, tones = dets, noise_tones = noise_tones, timestamp = timestamp, data_path = data_path, **kwargs)
            except Exception as e:
                log.log('ERROR', 'Failed to load %s with exception: %s.', data_class.__name__, e)
                data = None
        return data

    def join(self, other, in_place=False):
        def _join_consts(left_const: Any | list[Any], right_const: Any | list[Any]) -> list[Any]:
            if not isinstance(left_const, list): left_const = [left_const]
            if not isinstance(right_const, list): right_const = [right_const]
            return left_const + right_const

        if not isinstance(other, Detector):
            error = f'Cannot join with object of type {type(other)}. Must be of type Detector.'
            log.log('ERROR', error)
            raise ValueError(error)

        # Create a copy of the Detector object
        new_data = self if in_place else copy.deepcopy(self)
        new_data._cable_delay = _join_consts(self.cable_delay, other.cable_delay)

        # Join Target, Timestream, and VNA objects
        # ----------------------------------------
        new_data.targ = self.targ.join(other.targ, in_place=in_place)
        new_data.vna = _join_consts(self.vna, other.vna)

        left_stream, right_stream = self.stream, other.stream

        if not (left_stream is None or right_stream is None):
            new_data.stream = self.stream.join(other.stream, in_place=in_place)
        elif bool(left_stream is None) ^ bool(right_stream is None):
            error = f'Cannot join Detector objects where one has a Timestream and the other does not. Either both or neither Detector objects must have a Timestream.'
            log.log('ERROR', error)
            raise ValueError(error)

        # Join properties
        # ---------------
        left_prop, right_prop = self.properties, other.properties
        new_data._properties_df = pl.concat([left_prop, right_prop], how='diagonal').with_columns(pl.Series('det', new_data.targ.tones))

        return new_data

    # ============= #
    # Magic Methods #
    # ============= #

    def __str__(self):
        return f'detector_{self.timestamp}'

    # =============================== #
    # Define Custom Pickling Behavior #
    # =============================== #

    def __getstate__(self):
        state = self.__dict__.copy()
        if not self.analysis_cfg['io']['pickle']['pickle_dataframes'] and (properties := self._properties_df) is not None:
            del state['_properties_df']
            save_path, file_count = io.increment_file(Path(self.pickle_dir) / 'dataframe',
                                                      f'properties_',
                                                      '.parquet',
                                                      overwrite=self.analysis_cfg['io']['pickle']['overwrite'])
            state['pickle_count'] = file_count
            if isinstance(properties, pl.DataFrame):
                properties.write_parquet(save_path)
            elif isinstance(properties, pl.LazyFrame):
                properties.sink_parquet(save_path)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not self.analysis_cfg['io']['pickle']['pickle_dataframes'] and getattr(self, '_properties_df', True):
            file_name = 'properties' if (pickle_count := self.pickle_count) is None else f'properties_{pickle_count}'
            
            analysis_cfg, _ = io.load_config(str(Path(__file__).parents[1] / 'analysis_config.yaml'))
            if (curr_dir := analysis_cfg['io']['pickle']['curr_pickle_root_dir']): self.analysis_cfg['io']['pickle']['pickle_root_dir'] = curr_dir
            
            self.properties = pl.scan_parquet(Path(self.pickle_dir) / 'dataframe' / f'{file_name}.parquet')

