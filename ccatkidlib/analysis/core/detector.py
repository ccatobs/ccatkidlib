import numpy as np
import polars as pl
import pathlib
import sys
import concurrent.futures
import lmfit

from collections.abc import Iterable
from pathlib import Path
from functools import cached_property

# local imports
import ccatkidlib
import ccatkidlib.rfsoc_io as rfsoc_io
import ccatkidlib.analysis.pair as pair
import ccatkidlib.analysis.fit.fit as ccat_fit

from ccatkidlib.analysis.core.timestream import Timestream
from ccatkidlib.analysis.core.vna import VNA
from ccatkidlib.analysis.core.target import Target


class Detector:
    '''Class representing kinetic inductance detectors (KIDs). Used for KID analyses requiring fitting and multiple types of data files (e.g., timestream and target sweep data).

    Attributes:
        bid  (str): RFSoC board that took the detector data
        drid (str): RFSoC drone that took the detector data
        dets (list[int]): List of detectors

        stream (ccatkidlib.analysis.timestream.Timestream | None): Timestream object of detector timestream
        targ (ccatkidlib.analysis.target.Target | None): Target object of detector target sweep
        vna (ccatkidlib.analysis.vna.VNA | None): VNA object of detector VNA sweep

        analysis_cfg (dict): Config file with parameters used for data analysis
        viz_cfg (dict): Config file with paramateres used for data visualization

        cable_delay (float | None): Cable delay of the network (in nanoseconds)
        properties  (polars.dataframe.frame.DataFrame): Polars dataframe with detector properties extracted from fits
    '''

    def __init__(self, com_to: str,
                 analysis_cfg: str = str(Path(__file__).parents[1] / 'analysis_config.yaml'),
                 dets: int | list[int] = -1,
                 cable_delay: float | None = None,
                 stream: ccatkidlib.analysis.core.timestream.Timestream | None = None, stream_path: str | pathlib.PosixPath | list[str] | list[pathlib.PosixPath] | None = None, stream_timestamp: int | str | None = None,
                 targ: ccatkidlib.analysis.core.target.Target = None, targ_path: str | pathlib.PosixPath | None = None, targ_timestamp: int | str | None = None,
                 vna: ccatkidlib.analysis.core.vna.VNA = None, vna_path: str | pathlib.PosixPath | None = None, vna_timestamp: int | str | None = None,
                 **kwargs):

        # Create Timestream, Target, and VNA data objects based provided arguments
        # ------------------------------------------------------------------------
        if not isinstance(stream, Timestream): stream = Detector._load_data(Timestream, com_to, analysis_cfg, dets, stream_timestamp, stream_path, **kwargs)
        if not isinstance(targ, Target): targ = Detector._load_data(Target, com_to, analysis_cfg, dets, targ_timestamp, targ_path, **kwargs)
        if not isinstance(vna, VNA): vna = Detector._load_data(VNA, com_to, analysis_cfg, None, vna_timestamp, vna_path, **kwargs)
   
        # Must have a sweep to do meaningful data analysis
        # ------------------------------------------------
        if not isinstance(targ, Target):
            if isinstance(stream, Timestream): # If timestream provided, try to find associated sweep
                self.dets = stream.tones
                vna_path, targ_path = pair.get_sweep(stream.data_path[0], **kwargs)

                # Load found sweep
                if  Path(vna_path).exists(): vna = Detector._load_data(VNA, com_to, analysis_cfg, None, vna_timestamp, vna_path, **kwargs)
                if Path(targ_path).exists(): targ = Detector._load_data(Target, com_to, analysis_cfg, dets, targ_timestamp, targ_path, **kwargs)

                if not isinstance(vna, VNA) and not isinstance(targ, Target):
                    error = 'Failed to find target sweep or VNA sweep associated with timestream.'
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
                if Path(vna_path).exists(): vna = Detector._load_data(VNA, com_to, analysis_cfg, None, vna_timestamp, vna_path, **kwargs)

        self.bid, self.drid = com_to.split('.')
        self.analysis_cfg, self.viz_cfg = rfsoc_io.load_config(analysis_cfg)

        self.stream = stream
        self.targ = targ
        self.vna = vna

        self._cable_delay = cable_delay

        # Fitting 
        self._properties = {f'det_{det:04d}': {} for det in self.dets}
        self._properties_df = self.targ.comb

    #==========================#
    # Lazily Loaded Attributes #
    #==========================#

    @property
    def properties(self):
        # Reshape properties dictionary to have resonator properties as primary keys
        new_dict = {'det': []}

        props_dict = self._properties
        self._properties = {f'det_{det:04d}': {} for det in self.dets}

        all_props = set([prop for props in props_dict.values() for prop in props.keys()])
        
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
        self._properties_df = self._properties_df.drop(list(shared_cols))
        self._properties_df = self._properties_df.join(pl.DataFrame(new_dict), on='det', how='full', coalesce=True)
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
        self._properties = {det: {'cable_delay': 0.4*delay + 0.6*self._cable_delay} for det, delay in self.targ.cable_delay.items()}
        
        # Replace cable delays that are far from the overall network cable delay with the network cable delay
        self._properties_df = (self.properties.with_columns(pl.when(np.abs(pl.col('cable_delay') - self._cable_delay) > 100)
                                                          .then(self._cable_delay)
                                                          .otherwise(pl.col('cable_delay'))
                                                          .alias('cable_delay')))
        return self._cable_delay

    @cached_property
    def fit_dir(self):
        return self.analysis_cfg['file_paths']['fit_dir']

    #=====================#
    # Data Getter Methods #
    #=====================#

    def nonlinear_fit(self, include: int | list[int] | None = None, exclude: int | list[int] | None = None, recalc: bool = False, nonlinear=False, asymm = False, fix_cable = False, fix_thetaQ = False, max_workers=1, save_model_result = False):
        if not self.fit_dir in sys.path: sys.path.append(self.fit_dir)
        import resonator_model_v3
        globals()['resonator_model_v3'] = resonator_model_v3
        
        col_name = ['f', 'I', 'Q', 'nonlinear_fit']
        
        args = [[self, nonlinear, asymm, fix_cable, fix_thetaQ, max_workers, save_model_result]]
        self.targ.transform(Detector.calc_nonlinear_fit, *args, include=include, exclude=exclude, recalc = recalc, col_name = col_name, batch_size=len(self.targ.tones))
        self.targ.data = self.targ._unnest('struct_' + col_name[-1])

        # Calculate Q_c and Q_i
        self.properties = (self.properties.lazy().with_columns([((pl.col(f'{col_name[-1]}_Q_e_real')/(pl.col(f'{col_name[-1]}_Q_e_real')**2 + pl.col(f'{col_name[-1]}_Q_e_imag')**2))**-1).alias(f'{col_name[-1]}_Q_c')])
                                                 .with_columns(((1/pl.col(f'{col_name[-1]}_Q') - 1/pl.col(f'{col_name[-1]}_Q_c'))**-1).alias(f'{col_name[-1]}_Q_i'))
                                                 .with_columns((-1e9*pl.col(f'{col_name[-1]}_delay')).alias(f'{col_name[-1]}_delay_ns')).collect()) # Convert cable delay to nanoseconds

        return self.targ.get_data(col_name=col_name[-1], include=include, exclude=exclude)

    def phase_fit(self, prefix: str | list[str] = 'mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', circle_fit_col='circle_fit_unwind_rotate', include: int | list[int] | None = None, exclude: int | list[int] | None = None, recalc: bool = False, nonlinear=False, method = 'least_squares', params = None, window=1, max_workers=1, save_model_result = True):
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
        
        args = [[self, pre, circle, nonlin, method, param, win, max_workers, save_model_result] for pre, circle, nonlin, param, win in zip(prefix, circle_fit_col, nonlinear, params, window)]
        self.targ.transform([Detector.calc_phase_fit]*num_prefix, *args, include=include, exclude=exclude, recalc=recalc, col_name = col_names, batch_size=len(self.targ.tones))        
        self.targ.data = self.targ._unnest(['struct_' + col_name[-1] for col_name in col_names])

        # Calculate Q_c, Q_i, and nonlinearity parameter 'a'
        for pre, circle in zip(prefix, circle_fit_col):
            self.properties = (self.properties.lazy().with_columns([(1e-8*(pl.col(f'{col_name[-1]}_{pre}_beta')*(2*pl.col(f'{col_name[-1]}_{pre}_R'))**2)/(pl.col(f'{col_name[-1]}_{pre}_f_0')/pl.col(f'{col_name[-1]}_{pre}_Qr'))).alias(f'{col_name[-1]}_{pre}_a'),
                                                                    (pl.col(f'{col_name[-1]}_{pre}_Qr')*(pl.col(f'{circle}_center_mag') + pl.col(f'{col_name[-1]}_{pre}_R'))/(2*pl.col(f'{col_name[-1]}_{pre}_R'))).alias(f'{col_name[-1]}_{pre}_Q_c')])
                                                    .with_columns(((1/pl.col(f'{col_name[-1]}_{pre}_Qr') - 1/pl.col(f'{col_name[-1]}_{pre}_Q_c'))**-1).alias(f'{col_name[-1]}_{pre}_Q_i')).collect())
        
        return self.targ.get_data(col_name=[col_name[-1] for col_name in col_names], include=include, exclude=exclude)

    def IQ_unwind(self, prefix = '', data: str = 'both', delay_col = 'cable_delay', include: int | list[int] | None = None, exclude: int | list[int] | None = None, recalc: bool = False):
        '''Unwind (remove cable delay from) target sweep or timestream IQ data

        Args:
            data (str, optional): Which type of data to remove cable delay from. Options are 'targ' or 'timestream'. Defaults to 'targ'.
            cable_delay (float, optional): Cable delay of the network in nanoseconds
        Returns:
            polars.dataframe.frame.Dataframe: Polars DataFrame with unwound IQ data
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
        
        unwind_dfs = []
        for data_obj, f in zip(data_objs, fs):
            angle = ([-2*np.pi*delay*1e-9*pl.col(f'{col_name[0]}_{tone:04d}') for tone, delay in zip(self.dets, self.properties.select(delay_col).to_numpy().T[0])] if f is None 
                     else [-2*np.pi*delay*1e-9*tone_freq for tone_freq, delay in zip(f, self.properties.select(delay_col).to_numpy().T[0])])
            for pre in prefix:
                data_obj.IQ_rotate(prefix=pre, angle=angle, name=col_name[-1], include = include, exclude=exclude, recalc=recalc)
            unwind_dfs.append(data_obj.get_data(col_name=([f"{col_name[-1]}_rotate_{pre}{'_' if pre else ''}{col_name[1]}" for pre in prefix] +
                                                          [f"{col_name[-1]}_rotate_{pre}{'_' if pre else ''}{col_name[2]}" for pre in prefix]), include=include, exclude=exclude))
        return unwind_dfs

    def IQ_norm(self, prefix: str | list[str] = '', data: str = 'both', norm_col = 'cable_nonlinear_fit', include: int | list[int] | None = None, exclude: int | list[int] | None = None, recalc: bool = False):
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
            cable_mags = [1/pl.col(f'{norm_col}_{col_name[-2]}_{tone:04d}') for tone in self.dets] 
            scale = cable_mags if f is None else [self.targ.data.select(pl.col(f'{col_name[0]}_{tone:04d}').alias('f'), cable_mag.alias('cable'))
                                                                .sort(np.abs(pl.col('f') - tone_freq))
                                                                .select(pl.col('cable'))
                                                                .item(0, 0) for cable_mag, tone_freq, tone in zip(cable_mags, f, self.dets)]
            for pre in prefix:
                data_obj.IQ_scale(prefix=pre, scale=scale, name=col_name[-1], include = include, exclude=exclude, recalc=recalc)
            norm_dfs.append(data_obj.get_data(col_name=([f"{col_name[-1]}_scale_{pre}{'_' if pre else ''}{col_name[1]}" for pre in prefix] +
                                                          [f"{col_name[-1]}_scale_{pre}{'_' if pre else ''}{col_name[2]}" for pre in prefix]), include=include, exclude=exclude))
        return norm_dfs

    def IQ_trim(self, prefix: str | list[str] = '', f_0_col = 'nonlinear_fit_f_0', Q_col = 'nonlinear_fit_Q', window=1, use_fit=False, include: int | list[int] | None = None, exclude: int | list[int] | None = None, recalc: bool = False):
        col_name = ['I', 'Q', 'FWHM']
        if isinstance(prefix, str): prefix = [prefix]
        num_prefix = len(prefix)

        if f_0_col in self.properties.schema and use_fit:
            pass
        else:
            f_df = self.targ.data.select(pl.col('^f_0.*$'))
            f_df = (f_df.unpivot(on=f_df.columns,
                                 variable_name='det',
                                 value_name='f')
                        .with_columns(pl.col('det').str.strip_prefix('f_').cast(int)))

            self.targ.mag()
            mag_df = self.targ.data.select(pl.col('^mag_0.*$'))
            mag_df = (mag_df.unpivot(on=mag_df.columns,
                                     variable_name='det',
                                     value_name='mag')
                            .with_columns(pl.col('det').str.strip_prefix('mag_').cast(int))
                            .drop('det'))

            long_df = pl.concat([f_df, mag_df], how='horizontal')
            min_df = (long_df.lazy().filter((pl.col('mag') == pl.col('mag').min()).over('det'))
                                    .unique()
                                    .select('det', pl.col('f').alias('min_mag_f'), pl.col('mag').alias('min_mag'))
                                    .sort('det').collect())
            
            
        
        for pre in prefix:
            self.targ.IQ_scale(prefix=pre, lower_bound = lower_bound, upper_bound=upper_bound , name=col_name[-1], include = include, exclude=exclude, recalc=recalc)
        return self.targ.get_data(col_name=([f"{col_name[-1]}_trim_{pre}" for pre in prefix]), include=include, exclude=exclude)

    def IQ_circle_fit(self, prefix='unwind_rotate', include: int | list[int] | None = None, exclude: int | list[int] | None = None, recalc: bool = False, bounds = None, loss = 'soft_l1', f_scale=1, method = 'trf', max_workers=1):
        
        col_name = ['I', 'Q', 'circle_fit']

        if isinstance(prefix, str): prefix = [prefix]
        num_prefix = len(prefix)

        col_names = [[]]*num_prefix
        for i, pre in enumerate(prefix):
            col_names[i] = [f"{pre}{'_' if pre else ''}{name}" for name in col_name[:-1]] + [col_name[-1]]
        args = [[self, pre, bounds, loss, f_scale, method, max_workers] for pre in prefix]
        self.targ.transform([Detector.calc_IQ_circle_fit]*num_prefix, *args, include=include, exclude=exclude, recalc=recalc, col_name = col_names, batch_size=len(self.targ.tones))        
        self.targ.data = self.targ._unnest(['struct_' + col_name[-1] for col_name in col_names])
        return self.targ.get_data(col_name=[col_name[-1] for col_name in col_names], include=include, exclude=exclude)

    def IQ_circle_real(self, prefix: str | list[str] = 'unwind_rotate', data: str = 'both', loc: str = 'origin',  circle_fit_col='circle_fit_unwind_rotate', use_fit=True, include: int | list[int] | None = None, exclude: int | list[int] | None = None, recalc: bool = False):
        '''Rotate and center the IQ circle on the real axis
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
                self._properties_df = self._properties_df.with_columns((np.arctan2(pl.col(fit_names[1]), pl.col(fit_names[0]))).alias(fit_names[2]),
                                                                        (np.sqrt(pl.col(fit_names[0])**2 + pl.col(fit_names[1])**2)).alias(fit_names[3]))
                center_angle, center_mag = self._properties_df.select(pl.col(fit_names[2:])).to_numpy().T
                if not center_angle is None and not center_mag is None: use_mean = False

            if use_mean:
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

    def IQ_circle_rotate(self, prefix: str | list[str] = 'origin_shift_origin_rotate_unwind_rotate', data: str = 'both', rotation: str ='mismatch', include: int | list[int] | None = None, exclude: int | list[int] | None = None, recalc: bool = False, **kwargs):
        def _add_angles_to_properties(df, rotation_col_name, targ_angle = 0):
            unpivot_cols = df.select(pl.all()).columns
            df = df.unpivot(on=unpivot_cols,
                            variable_name='det',
                            value_name=rotation_col_name).with_columns(pl.col('det').cast(int))
            df = df.with_columns((targ_angle +  -1*pl.col(rotation_col_name)).alias(rotation_col_name))
            if rotation_col_name in self.properties.schema: self._properties_df = self._properties_df.drop(rotation_col_name)
            self._properties_df = self._properties_df.join(df, on='det', how='full', coalesce=True)
        
        col_name = ['I', 'Q', rotation]
        if isinstance(prefix, str): prefix = [prefix]
        num_prefix = len(prefix)

        angles = []
        for pre in prefix:
            if rotation == 'mismatch':
                mismatch_col_name = f'{pre}_{col_name[-1]}_angle'
                if recalc or not mismatch_col_name in self.properties.schema:
                    mismatch_df = self.targ.data.select([((np.mod(np.arctan2(pl.col(f"{pre}{'_' if pre else ''}{col_name[1]}_{tone:04d}").first(), pl.col(f"{pre}{'_' if pre else ''}{col_name[0]}_{tone:04d}").first()), 2*np.pi) +
                                                            np.mod(np.arctan2(pl.col(f"{pre}{'_' if pre else ''}{col_name[1]}_{tone:04d}").last(), pl.col(f"{pre}{'_' if pre else ''}{col_name[0]}_{tone:04d}").last()), 2*np.pi))/2).alias(f'{tone:04d}') for tone in self.targ.tones])
                    _add_angles_to_properties(mismatch_df, mismatch_col_name, targ_angle=np.pi)
                
                mismatch_angle = self._properties_df.select(mismatch_col_name).to_numpy().T[0]
                angles.append(mismatch_angle)
            elif rotation == 'timestream':
                timestream_col_name = f'{pre}_{col_name[-1]}_angle'
                if recalc or not timestream_col_name in self.properties.schema:
                    timestream_df = self.stream.data.select([(np.mod(np.arctan2(pl.col(f"{pre}{'_' if pre else ''}{col_name[1]}_{tone:04d}").median(),
                                                                                pl.col(f"{pre}{'_' if pre else ''}{col_name[0]}_{tone:04d}").median()), 2*np.pi)).alias(f'{tone:04d}') for tone in self.stream.tones])
                    
                    _add_angles_to_properties(timestream_df, timestream_col_name, targ_angle=0)
                
                timestream_angle = self._properties_df.select(timestream_col_name).to_numpy().T[0]
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

    def phase_spline(self, prefix: str | list[str] = 'mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', phase_low: float = -3.14, phase_up: float = 3.14, k: int = 3, include: int | list[int] | None = None, exclude: int | list[int] | None = None, recalc: bool = False, max_workers=1, **kwargs):
        '''Interpolate target sweep phase data and add interpolating splines to propreties attribute

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

        args = [[self, low, up, k, max_workers] for low, up in zip(phase_low, phase_up)]
        self.targ.transform([Detector.calc_phase_spline]*num_prefix, *args, include=include, exclude=exclude, recalc=recalc, col_name = col_names, batch_size=len(self.targ.tones))   
        self.targ.data = self.targ._unnest(['struct_' + col_name[-1] for col_name in col_names])
        return self.targ.get_data(col_name=([col_name[-1] for col_name in col_names] + [col_name[-2] for col_name in col_names]), include=include, exclude=exclude)

    def phase_to_f(self, prefix: str | list[str] = 'mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', phase_bounds: float = 0.2, k: int = 3, include: int | list[int] | None = None, exclude: int | list[int] | None = None, recalc: bool = False, max_workers=1, **kwargs):
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
        self.phase_spline(prefix=prefix, phase_low = min_phases, phase_up = max_phases, k = k, include=include, exclude=exclude, recalc=recalc, max_workers=max_workers, **kwargs)

        args = [[self, max_workers]]*num_prefix
        self.stream.transform([Detector.calc_phase_to_f]*num_prefix, *args, include=include, exclude=exclude, recalc=recalc, col_name = col_names, batch_size=len(self.stream.tones))
        self.stream.data = self.stream._unnest(['struct_' + col_name[-1] for col_name in col_names])
        return self.stream.get_data(col_name=[col_name[-1] for col_name in col_names], include=include, exclude=exclude)
    
    def frac_f(self, prefix: str | list[str] = 'mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', f_0 = None, name='', include: int | list[int] | None = None, exclude: int | list[int] | None = None, recalc: bool = False, **kwargs):        
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
                        self._properties[f'det_{tone:04d}'] = best_vals_dict
                    except Exception as e:
                        rfsoc_io.send_msg('WARNING', 'Fit failed for tone %s with exception: %s', tone, e)
                        best_fit = np.zeros(df.len())
                        cable_fit = np.zeros(df.len())
                        self._properties[f'det_{tone:04d}'] = {}
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
        expr, to_calc, calc_col, batches = Detector._batch_calc(_nonlinear_fit, tones, col_name, schema, return_col=return_col, return_type=return_type, recalc=recalc)
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
                        self._properties[f'det_{tone:04d}'] = best_vals_dict | init_vals_dict
                    except Exception as e:
                        rfsoc_io.send_msg('WARNING', 'Fit failed for tone %s with exception: %s', tone, e)
                        best_fit = np.full(df.len(), np.nan)
                        self._properties[f'det_{tone:04d}'] = {}
                    results_dict[f'{col_name[-1]}_{phase_col}'] = best_fit

            df = pl.DataFrame(results_dict)

            return pl.Series(df.select(pl.struct(df.columns)))
            
        if len(args) == 9:
            self, prefix, circle_col, nonlinear, method, params, window, max_workers, save_model_result = args
            if isinstance(params, lmfit.parameter.Parameters) or not isinstance(params, Iterable): params = len(tones)*[params]
            if isinstance(nonlinear, bool): nonlinear = len(tones)*[nonlinear]
            if isinstance(window, (float, int)): window = len(tones)*[window]
        else:
            error = 'nonlinear, prefix, params, window, max_workers, and save_model_result are required arguments.'
            rfsoc_io.send_msg('ERROR', error)
            raise ValueError(error)
        
        return_col = [f'{col_name[-1]}_{col_name[-2]}']
        return_type = [pl.Float64]
        expr, to_calc, calc_col, batches = Detector._batch_calc(_phase_fit, tones, col_name, schema, return_col=return_col, return_type=return_type, recalc=recalc)
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
                        self._properties[f'det_{tone:04d}'] = {f'{col_name[-1]}_{prefix}_{k}': v for k, v in zip(property_keys, property_vals)}
                    except Exception as e:
                        rfsoc_io.send_msg('WARNING', 'Fit failed for tone %s with exception: %s', tone, e)
                        fit_I, fit_Q = np.zeros(df.len()), np.zeros(df.len())
                        self._properties[f'det_{tone:04d}'] = {}
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
        expr, to_calc, calc_col, batches = Detector._batch_calc(_circle_fit, tones, col_name, schema, return_col = return_col, return_type=return_type, recalc=recalc)
        return expr

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
                    self._properties[f'det_{tone:04d}'] = property_dict
                    results_dict[f'{interp_names[0]}_{tone:04d}'] = to_f
                    results_dict[f'{interp_names[1]}_{tone:04d}'] = to_phase
            
            df = pl.DataFrame(results_dict)
            return pl.Series(df.select(pl.struct(df.columns)))

        if len(args) == 5:
            self, phase_low, phase_up, k, max_workers = args

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
        expr, to_calc, calc_col, batches = Detector._batch_calc(_phase_spline, tones, data_col_name, schema, return_col=return_col, return_type=return_type, recalc=recalc, calc_col = calc_col)
        return expr

    @staticmethod
    def calc_phase_to_f(schema, *args, tones: list[int], recalc: bool = False, col_name = ['phase', 'f']):
        def _phase_to_f(df):
            struct = df.struct
            y_to_x, x_to_y = self.properties.select(pl.col([f'{col_name[0]}_to_f_spline', f'f_to_{col_name[0]}_spline']))            
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
        
        if len(args) == 2:
            self, max_workers = args
        else:
            error = 'self, and max_workers are required arguments.'
            rfsoc_io.send_msg('ERROR', error)
            raise ValueError(error) 

        calc_col = [f'{col_name[-1]}_{tone:04d}' for tone in tones]
        return_col = [f'{col_name[-1]}']
        return_type = [pl.Float64]
        expr, to_calc, calc_col, batches = Detector._batch_calc(_phase_to_f, tones, col_name, schema, return_col=return_col, return_type=return_type, recalc=recalc, calc_col = calc_col)
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
    
    #================#
    # Helper Methods #
    #================#
    
    def _get_data_obj(self, data):
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
    def _batch_calc(func, tones, col_name, schema, return_col, return_type, recalc=False, calc_col = None):
        if calc_col is None: calc_col = [f'{col_name[-1]}_{col_name[-2]}_{tone:04d}' for tone in tones]
        to_calc = tones if recalc else [tone for tone, col in zip(tones, calc_col) if col not in schema]
        if not len(to_calc) == 0:
            batches = [[f'{name}_{tone:04d}' for name in col_name[:-1]] for tone in to_calc]
            returns = [[pl.Field(f'{name}_{tone:04d}', dtype) for name, dtype in zip(return_col, return_type)] for tone in to_calc]
            calc_col = f'struct_{col_name[-1]}_{to_calc[0]:04d}'
            
            batches_flat = [col for batch in batches for col in batch]
            returns_flat = [col for ret_col in returns for col in ret_col]
            expr = pl.struct(batches_flat).map_batches(func, return_dtype=pl.Struct(returns_flat)).alias(calc_col)
        else:
            batches = []
            expr = pl.col(calc_col)
        return expr, to_calc, calc_col, batches

    @staticmethod
    def _load_data(data_class, com_to, analysis_cfg, dets, timestamp, data_path, **kwargs):
        data = None
        if data_path is not None or timestamp is not None:
            try:
                data = data_class(com_to = com_to, analysis_cfg = analysis_cfg, tones = dets, timestamp = timestamp, data_path = data_path, **kwargs)
            except Exception as e:
                rfsoc_io.send_msg('ERROR', 'Failed to load %s with exception: %s.', data_class.__name__, e)
                data = None
        return data
