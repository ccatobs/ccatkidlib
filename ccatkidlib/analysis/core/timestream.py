''' Module for analyzing kinetic inductance detector (KID) time ordered (timestream) data

Authors:
    - Darshan Patel <dp649@cornell.edu>

TODO:
    - Multiprocess FFT & PSD calcultions
    
'''

import so3g
import numpy as np
import polars as pl
import time

from scipy.signal import welch
from functools import cached_property
from pathlib import Path
from collections.abc import Iterable
from spt3g import core

import holoviews as hv
import datashader as ds
from holoviews import opts
from holoviews.operation.datashader import rasterize, datashade, dynspread


# Local Imports
import ccatkidlib.rfsoc_io as rfsoc_io
import ccatkidlib.utils as utils
import ccatkidlib.analysis.utils.pair as pair
import ccatkidlib.analysis.utils.multiprocess as ccat_mp
import ccatkidlib.analysis.utils.dataframe as ccat_df

from ccatkidlib.analysis.core.data import Data
from ccatkidlib.utils import method_timer


class Timestream(Data):
    '''Class representing a timestream taken with a radio frequency system on a chip (RFSoC)

    Attributes:
        tones (list[int]): List of tones with loaded timestreams
        start (float): Start time of loaded timestreams in seconds (0 seconds is beginning of timestream) 
        end (float): End time of loaded timestreams in seconds (relative to start time)
        packet_counts (list[int]): List of packet numbers of loaded timestreams 
        data (pl.DataFrame): Polars DataFrame with loaded (and transformed) timestream data
        properties (pl.DataFrame): Polars DataFrame with timestream properties (a 'property' has one value per tone)
        sampling_freq (float): Sampling frequency of timestream data in Hz
    '''

    def __init__(self, com_to, tones: int | list[int] = -1, noise_tones: int | list[int] | None = None, start = 0, end = -1, analysis_cfg = str(Path(__file__).parents[1] / 'analysis_config.yaml'), **kwargs):
        '''
        Constructor for Timestream. 

        Args:
            com_to (str): Which board and drone were used to take the timestreams (in form ``bid.drid``)
            tones (list): List of tones for which to load timestreams
            start (float): Start time in seconds (0 seconds is beginning of timestream) 
            end (float): End time in seconds (relative to start time, -1 for no end time)
            analysis_cfg (str): File path of analysis configuration file. Defaults to analysis configuration file in *ccatkidlib/analysis* directory.
        '''
        kwargs['data_type'] = 'timestream'
        super().__init__(com_to, analysis_cfg, **kwargs)

        # Define list of tones
        # --------------------
        if isinstance(tones, int): 
            if tones >= 0: 
                self.tones = [tones]
            else:
                self.tones = list(range(self.num_tones))
        elif isinstance(tones, Iterable) and all([isinstance(tone, int) for tone in tones]):
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

        # Define timestream start and end times
        # -------------------------------------
        self.start = start
        self.end = np.inf if end < 0 else end

        # End time should be larger than start time
        if not (self.end - self.start > 0): 
            error = "End time must be greater than start time!"
            rfsoc_io.send_msg('ERROR', error)
            raise ValueError(error)

        # Create packet count attribute
        # -----------------------------
        self.packet_counts = []

        self._properties = {f'det_{tone:04d}': {} for tone in self.tones}
        self._properties_df = pl.DataFrame({'det': self.tones})

    #==================#
    # Plotting Methods #
    #==================#

    def plot(self, x_dim, y_dim, x_prefix: str = '', y_prefix: str = '', include: int | list[int] | None = None, exclude: int | list[int] | None = None, unpivot_x=True):
        col_dict = {'sample': 'sample',
                    'x': x_dim,
                    'y': y_dim}

        df, by = self._get_plot_df(col_dict, x_prefix = x_prefix, y_prefix = y_prefix, include = include, exclude = exclude, unpivot_x=unpivot_x)
        col_dict['x'], col_dict['y'] = df.select(pl.exclude('det', 'sample')).columns
        df = df.filter((~pl.col(col_dict['x']).is_nan()) & (~pl.col(col_dict['y']).is_nan()))

        # Create HoloViews plot objects
        line = df.hvplot.line(x=col_dict['x'],
                              y=col_dict['y'],
                              by=by,
                              label='Curve',
                              width=self.viz_cfg['plot']['width'],
                              height=self.viz_cfg['plot']['height'])

        scatter = df.hvplot.scatter(x=col_dict['x'],
                                    y=col_dict['y'],
                                    by=by,
                                    label='Scatter',
                                    width=self.viz_cfg['plot']['width'],
                                    height=self.viz_cfg['plot']['height'])
        
        overlay = hv.Overlay([line, scatter])

        cfg = self.drone_cfg['det_config']
        title = rf"${cfg['detector_type']}\ {cfg['network']}$"

        if not (include is None and exclude is None): overlay.opts(opts.NdOverlay(title=title))
        overlay.opts(opts.Curve(title=title), opts.Scatter(title=title))

        return overlay, df

    def stream_plot(self, col_name: str, prefix: str = '', return_df = False, rasterize=True, include: int | list[int] | None = None, exclude: int | list[int] | None = None):
        ''' Plot the specified data column as a function of time
        
        Args:
            col_name (str): Name of data column (e.g., *I*, *Q*, *mag*, etc.)
            prefix (str, optional): Defaults to ""
            return_df (bool): Whether to return the Polars DataFrame that was used to create the plot. Defaults to *False*
            include (int | list[int] | None, optional): Defaults to *None*
            exclude (int | list[int] | None, optional): Defaults to *None*
        Returns:
            return (hv.NdOverlay | tuple[hv.NdOverlay, pl.DataFrame]): 
        
        '''
        overlay, df = self.plot('t', col_name, y_prefix=prefix, include=include, exclude=exclude, unpivot_x=False)
        xlabel = r'$Time [s]$'
        ylabel = f'{prefix}_{col_name}'

        curve_opts = opts.Curve(xlabel=xlabel,
                                ylabel=ylabel)

        if not (include is None and exclude is None):
            overlay.NdOverlay.Curve.opts(curve_opts)
            overlay = overlay.NdOverlay.Curve
            aggregator = ds.by('det', ds.count())
        else:
            overlay.Curve.Curve.opts(curve_opts)
            overlay = overlay.Curve.Curve
            aggregator = ds.count()
        
        if rasterize: overlay = datashade(overlay, aggregator=aggregator)
        
        if return_df:
            return overlay, df
        else:
            return overlay
    
    def mag_plot(self, x_prefix: str = '', y_prefix: str = '', include: int | list[int] | None = None, exclude: int | list[int] | None = None, return_df = False, rasterize=True):
        overlay, df = self.plot('f', 'mag', x_prefix=x_prefix, y_prefix=y_prefix, include=include, exclude=exclude)
        xlabel = r'$Frequency\ [Hz]$'
        ylabel = r'$|S_{21}|$'

        scatter_opts = opts.Scatter(xlabel=xlabel,
                                    ylabel=ylabel)

        if not (include is None and exclude is None):
            overlay.NdOverlay.Scatter.opts(scatter_opts)
            overlay = overlay.NdOverlay.Scatter
            aggregator = ds.by('det', ds.count())
        else:
            overlay.Scatter.Scatter.opts(scatter_opts)
            overlay = overlay.Scatter.Scatter
            aggregator = ds.count()
        
        if rasterize: overlay = dynspread(datashade(overlay, aggregator=aggregator))

        if return_df:
            return overlay, df
        else:
            return overlay

    def phase_plot(self, x_prefix: str = '', y_prefix: str = '', include: int | list[int] | None = None, exclude: int | list[int] | None = None, return_df = False, rasterize=True):
        overlay, df = self.plot('f', 'phase', x_prefix=x_prefix, y_prefix=y_prefix, include=include, exclude=exclude)
        xlabel = r'$Frequency\ [Hz]$'
        ylabel = r'$Phase\ [rad]$'

        scatter_opts = opts.Scatter(xlabel=xlabel,
                                    ylabel=ylabel)

        if not (include is None and exclude is None):
            overlay.NdOverlay.Scatter.opts(scatter_opts)
            overlay = overlay.NdOverlay.Scatter
            aggregator = ds.by('det', ds.count())
        else:
            overlay.Scatter.Scatter.opts(scatter_opts)
            overlay = overlay.Scatter.Scatter
            aggregator = ds.count()
        
        if rasterize: overlay = dynspread(datashade(overlay, aggregator=aggregator))

        if return_df:
            return overlay, df
        else:
            return overlay

    def IQ_plot(self, prefix: str = '', include: int | list[int] | None = None, exclude: int | list[int] | None = None, return_df = False, rasterize=True):
        overlay, df = self.plot('I', 'Q', x_prefix=prefix, y_prefix=prefix, include=include, exclude=exclude)
        xlabel = r'$I$'
        ylabel = r'$Q$'

        scatter_opts = opts.Scatter(xlabel=xlabel,
                                    ylabel=ylabel)

        if not (include is None and exclude is None):
            overlay.NdOverlay.Scatter.opts(scatter_opts)
            overlay = overlay.NdOverlay.Scatter
            aggregator = ds.by('det', ds.count())
        else:
            overlay.Scatter.Scatter.opts(scatter_opts)
            overlay = overlay.Scatter.Scatter
            aggregator = ds.count()
        
        if rasterize: overlay = dynspread(datashade(overlay, aggregator=aggregator))

        if return_df:
            return overlay, df
        else:
            return overlay

    def psd_plot(self, col_name, prefix: str = '', include: int | list[int] | None = None, exclude: int | list[int] | None = None, return_df = False):
        overlay, df = self.plot('psd_f', col_name, x_prefix=f"psd_{prefix}{'_' if prefix else ''}{col_name}",  y_prefix=f"psd{'_' if prefix else ''}{prefix}", include=include, exclude=exclude, unpivot_x=False)
        xlabel = r'$PSD\ Frequency\ [Hz]$'
        ylabel = f'psd_{prefix}_{col_name}'

        curve_opts = opts.Curve(xlabel=xlabel,
                                ylabel=ylabel)
        scatter_opts = opts.Scatter(xlabel=xlabel,
                                    ylabel=ylabel)

        if not (include is None and exclude is None):
            overlay.NdOverlay.Curve.opts(curve_opts)
            overlay.NdOverlay.Scatter.opts(scatter_opts)
            overlay.NdOverlay.opts(logx=True, logy=True)
        else:
            overlay.Curve.Curve.opts(curve_opts)
            overlay.Scatter.Scatter.opts(scatter_opts)
            overlay.Curve.opts(logx=True, logy=True)
            overlay.Scatter.opts(logx=True, logy=True)

        if return_df:
            return overlay, df
        else:
            return overlay
    
    def fft_plot(self, col_name, prefix: str = '', include: int | list[int] | None = None, exclude: int | list[int] | None = None, return_df = False):
        overlay, df = self.plot('f', col_name, x_prefix=f'fft',  y_prefix=f"fft{'_' if prefix else ''}{prefix}", include=include, exclude=exclude, unpivot_x=False)
        xlabel = r'$FFT\ Frequency\ [Hz]$'
        ylabel = f'fft_{prefix}_{col_name}'

        curve_opts = opts.Curve(xlabel=xlabel,
                                ylabel=ylabel)
        scatter_opts = opts.Scatter(xlabel=xlabel,
                                    ylabel=ylabel)

        if not (include is None and exclude is None):
            overlay.NdOverlay.Curve.opts(curve_opts)
            overlay.NdOverlay.Scatter.opts(scatter_opts)
            overlay.NdOverlay.opts(logy=True)
        else:
            overlay.Curve.Curve.opts(curve_opts)
            overlay.Scatter.Scatter.opts(scatter_opts)
            overlay.Curve.opts(logy=True)
            overlay.Scatter.opts(logy=True)

        if return_df:
            return overlay, df
        else:
            return overlay
    
    #==========================#
    # Lazily Loaded Attributes #
    #==========================#

    @property
    def data(self):
        '''
        Load timestream I, Q data.
        '''

        if self._data is None:
            ftype = Path(self.data_path[0]).suffix
            data = {}

            # Load g3 timestreams
            # -------------------
            if ftype == '.g3' or ftype == '.txt':
                ts, Is, Qs = self._load_g3_timestream(ftype)
                dt, unit = np.array(ts)*1e9, 'ns'

            # Load python timestreams (.npy or .npz)
            # --------------------------------------
            elif ftype == '.npz' or ftype == '.npy':
                ts, Is, Qs = self._load_npy_timestream(ftype)
                dt, unit = np.array(ts)*1e5, 'us'
            else:
                error = f"Invalid timestream file type: '{ftype}'!"
                rfsoc_io.send_msg('ERROR', error)
                raise ValueError(error)
            
            ts, Is, Qs = np.array(ts), np.array(Is, dtype=np.float64), np.array(Qs, dtype=np.float64)
            data['sample'] = range(len(ts))
            data['t'], data['dt'] = ts, dt
            for t, I, Q in zip(self.tones, Is, Qs):
                data[f'I_{t:04d}'] = I
                data[f'Q_{t:04d}'] = Q
            self._data = pl.DataFrame(data)
            self._data = self._data.with_columns(pl.col('dt').cast(pl.Datetime(unit)))
        return self._data

    @data.setter
    def data(self, value: pl.lazyframe.frame.LazyFrame | None): 
        if value is None or isinstance(value, pl.dataframe.frame.DataFrame): 
            self._data = value
        else:
            rfsoc_io.send_msg('ERROR', 'Cannot set data with type %s. Must be a Polars LazyFrame! Convert DataFrame to lazy frame with .lazy() before setting.', type(value))

    @property
    def properties(self):
        # Reshape properties dictionary to have resonator properties as primary keys
        new_dict = {'det': []}

        props_dict = self._properties
        self._properties = {f'det_{tone:04d}': {} for tone in self.tones}

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
    def sampling_freq(self):
        return self.io_cfg['boards'][f'b{self.bid}']['sampling_freq']

    #=====================#
    # Data Getter Methods #
    #=====================#

    def fft(self, prefix: str | list[str] = '', col_name: str = 'phase', sampling_freq=None, include: int | list[int] | None = None, exclude: int | list[int] | None = None, recalc: bool = False) -> pl.dataframe.frame.DataFrame:        
        col_name = [col_name, 'fft']
        f_col = f'{col_name[-1]}_f'
        if sampling_freq is None: sampling_freq = self.sampling_freq
        if recalc or not f_col in self.data.schema: self.data = self.data.with_columns(pl.Series(np.fft.fftshift(np.fft.fftfreq(self.data.height, d=1/sampling_freq))).alias(f_col))

        if isinstance(prefix, str): prefix = [prefix]
        num_prefix = len(prefix)

        col_names = [[]]*num_prefix
        for i, pre in enumerate(prefix):
            data_name = f"{pre}{'_' if pre else ''}{col_name[0]}"
            col_names[i] = ['t', data_name, f'{col_name[-1]}_{data_name}']
        self.transform([Timestream.calc_fft]*num_prefix, include=include, exclude=exclude, recalc=recalc, col_name = col_names)
        return self.get_data(col_name=[f_col] + [col_name[-1] for col_name in col_names], include=include, exclude=exclude)

    def psd(self,
            prefix: str | list[str] = '',
            col_name: str = 'phase',
            include: int | list[int] | None = None,
            exclude: int | list[int] | None = None,
            recalc: bool = False,
            sampling_freq = None,
            window='hann',
            nperseg=None,
            detrend=False,
            average='mean') -> pl.dataframe.frame.DataFrame:
        
        col_name = [col_name, 'psd']

        if isinstance(prefix, str): prefix = [prefix]
        num_prefix = len(prefix)

        col_names = [[]]*num_prefix
        f_cols = ['']*num_prefix
        for i, pre in enumerate(prefix):
            data_name = f"{pre}{'_' if pre else ''}{col_name[0]}"
            f_cols[i] = f'{col_name[-1]}_{data_name}_psd_f'
            col_names[i] = ['t',  data_name, f'{col_name[-1]}_{data_name}']

        if sampling_freq is None: sampling_freq = self.sampling_freq
        height = self.data.height

        psd_f, _ = welch(np.array([0]*height), fs=sampling_freq, window=window, nperseg=nperseg, detrend=detrend, average=average)
        psd_f = psd_f[1:-1]
        psd_f = pl.Series(np.pad(psd_f, (0, height - len(psd_f)), constant_values=None))
        for f_col in f_cols:
            if recalc or not f_col in self.data.schema: self.data = self.data.with_columns(psd_f.alias(f_col))

        args = [[sampling_freq, height, window, nperseg, detrend, average]]*num_prefix # Allow different psd args to be passed for each prefix?
        self.transform([Timestream.calc_psd]*num_prefix, *args, include=include, exclude=exclude, recalc=recalc, col_name = col_names)
        self.data = self._unnest(col_name[-1])
        return self.get_data(col_name =  [col_name[-1] for col_name in col_names], include=include, exclude=exclude)
   
    #==================#
    # Analysis Methods #
    #==================#

    @staticmethod
    def calc_fft(schema, *args, tones: list[int] | None = None, recalc: bool = False, col_name = ['t', 'phase', 'fft_phase']):
        def _fft(data):
            return np.abs(np.fft.fftshift(np.fft.fft(data)))
        
        if tones is not None:
            tone = tones[0]
            col_name = [col_name[0]] + [f'{name}_{tone:04d}' for name in col_name[1:]]

        t_col, data_col, fft_col = col_name

        if recalc or not (fft_col in schema):
            return pl.col(data_col).map_batches(_fft, return_dtype=pl.Float64).alias(fft_col)
        else:
            return pl.col(fft_col)

    @staticmethod
    def calc_psd(schema, *args, tones: list[int] | None = None, recalc: bool = False, col_name = ['t', 'phase', 'psd_phase']):
        def _psd(data):
            try:
                _, psd = welch(data.to_numpy().T, fs=sampling_freq, window=window, nperseg=nperseg, detrend=detrend, average=average)
                psd = np.sqrt(psd[1:-1])
                psd = np.pad(psd, (0, height - len(psd)), constant_values=None)
            except:
                psd = np.zeros(height)
            return psd
        
        if len(args) == 6:
            sampling_freq, height, window, nperseg, detrend, average = args
            if tones is not None: sampling_freq, height, window, nperseg, detrend, average = sampling_freq[0], height[0], window[0], nperseg[0], detrend[0], average[0]
        else:
            rfsoc_io.send_msg('ERROR', 'sampling_freq, window, nperseg, detrend, and average are required arguments')

        if tones is not None:
            tone = tones[0]
            col_name = [col_name[0]] + [f'{name}_{tone:04d}' for name in col_name[1:]]

        t_col, data_col, psd_col = col_name

        if recalc or not (psd_col in schema):
            return pl.col(data_col).map_batches(_psd, return_dtype=pl.Float64).alias(psd_col)
        else:
            return pl.col(psd_col)

    #==========================#
    # Internal Loading Methods #
    #==========================#

    @classmethod
    def load_frame(cls, *args, **kwargs):
        if args and isinstance(args[0], cls):
            return args[0]._load_frame(*args[1:], **kwargs)
    
    def _load_frame(self, frame, start_time, time_precision, mask = None):
        if 'packet_counts' in frame: self.packet_counts += list(frame['packet_counts'])

        ts = []
        g3_data = frame['data']
        data = g3_data.data

        if mask is None:
            ts = np.array(g3_data.times)/time_precision

            t_diff = np.array(ts - start_time)
            mask = np.where((t_diff >= self.start) & (t_diff <= self.end), True, False)

        inds = 2*np.array(self.tones)

        I = data[list(inds)][:, mask]
        Q = data[list(inds+1)][:, mask]

        if not len(ts) == 0:
            return ts[mask], I, Q
        else:
            return ts, I, Q
 
    def _load_g3_timestream(self, ftype):
        '''
        Load g3 timestream data.
        '''           

        time_precision = 1e8
        start_time = -1

        g3_files = []
        for path in sorted(self.data_path):
            if ftype == '.txt':
                g3_root = self.analysis_cfg['file_paths']['g3_root_dir']
                try:
                    original_g3_root = self.io_cfg['file_paths']['g3_root_dir']
                except KeyError:
                    original_g3_root = self.analysis_cfg['file_paths']['original_g3_root_dir']

                if not g3_root[-1] == '/': g3_root += '/'
                if not original_g3_root[-1] == '/': original_g3_root += '/'

                with open(path, 'r') as file:
                    g3_files += file.readlines()
                    g3_files = [pair.replace_root(g3_file, original_g3_root, g3_root) for g3_file in g3_files]
                break # If user specifies multiple txt files only load the first one since there can only be one txt file per timestream
            else:
                g3_files += [path]

        # Do initial pass through of frames in G3 file without fully loading the data to:
        # 1. Aggregate frames from different G3 files into one list
        # 2. Filter out frames that do not contain timestream data
        # 3. Get the number of tones
        # 4. Filter out frames that are not within the specified time range and construct the array of times
        # --------------------------------------------------------------------------------------------------
        frames = []
        masks = []
        ts = []
        for g3_file in g3_files:
            g3_data = core.G3File(g3_file)
            
            for frame in g3_data:
                # Filter out frames that are not G3 Scan frames (those that contain the timestream data)
                if frame.type == core.G3FrameType.Scan:
                    # Determine the number of tones used for timestream
                    if self.tones is None:
                        channel_count = frame['channel_count'] # Get number of tones directly from G3 frame

                        # Send warning if there is a mismatch in the number of tones (but continue execution using the number in the G3 frame)
                        if not self.num_tones == channel_count: 
                            print('WARNING | There is a mismatch between the number of tones in the comb and the number of tones in the timestream packets!')
                            self.num_tones = channel_count
                        self.tones = range(channel_count)

                    # Filter out frames that are not within specified time range
                    times = np.array(frame['data'].times)/time_precision
                    if start_time == -1:
                        start_time = times[0]
                    if times[0] - start_time <= self.end:
                        if times[-1] - start_time >= self.start:
                            t_diff = np.array(times - start_time)
                            mask = np.where((t_diff >= self.start) & (t_diff <= self.end), True, False)
                            frames.append(frame)
                            masks.append(mask)
                            ts = np.append(ts, times[mask])
                    else:
                        break
            else:
                continue
            break

        shape = (len(self.tones), len(ts))

        Is, Qs = np.empty(shape, dtype=np.int32), np.empty(shape, dtype=np.int32)
        curr_ind = 0
        for i in range(len(frames)):
            mask = masks[i]
            t, I, Q = self.load_frame(self, frames[i], start_time, time_precision, mask = mask)

            num_samps = int(np.sum(mask))
            Is[:, curr_ind:num_samps + curr_ind] = I
            Qs[:, curr_ind:num_samps + curr_ind] = Q
            curr_ind += num_samps

        return ts, Is, Qs
 
    def _load_npy_timestream(self, ftype):
        '''
        Load timestream data from .npz or .npy file
        '''

        def _load_npy(data, start_time):
            '''
            Load timestream I, Q data from npy file generated by python timestream
            '''

            start_ind = 0

            # If data has two more rows than the number of tones than both packet counts and timestamps were recorded
            if len(data) % 2 == 0: 
                self.packet_counts += list(data[0])
                start_ind += 1
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
            ts = data[start_ind].real/time_precision
            t_diff = np.array(ts - start_time)
            in_time = np.where((t_diff >= self.start) & (t_diff <= self.end), True, False)

            inds = 2*np.array(self.tones) + start_ind + 1

            I = data[list(inds)][:, in_time]
            Q = data[list(inds+1)][:, in_time]

            return ts[in_time], I, Q

        ts, Is, Qs = [], None, None
        
        time_precision = 1e5
        tstamp_ind = 0
        start_time = -1

        # Do an initial iteration through the timestream data without fully loading into memory to determine which files lie within the specified time span
        # -------------------------------------------------------------------------------------------------------------------------------------------------

        data_arr = []
        for data_path in self.data_path:
            npy_file = np.load(data_path, mmap_mode='r').values() if ftype == '.npz' else [np.load(data_path, mmap_mode='r')]
        
            for npy_data in npy_file:
                if start_time == -1:
                    if len(npy_data) % 2 == 0: tstamp_ind += 1
                    start_time = npy_data[tstamp_ind][0]/time_precision
                if npy_data[tstamp_ind][0]/time_precision - start_time <= self.end:
                    if npy_data[tstamp_ind][-1]/time_precision - start_time >= self.start: data_arr.append(npy_data)
                else:
                    break
            else:
                continue
            break

        # Load the data within the specified timespan
        # -------------------------------------------
        for npy_data in data_arr:
            t, I, Q = _load_npy(npy_data, start_time)
            ts = np.append(ts, t)
            Is = np.append(Is, I, axis=1) if Is is not None else I
            Qs = np.append(Qs, Q, axis=1) if Qs is not None else Q
        
        return ts, Is, Qs

    #=====================#
    # Data Getter Methods #
    #=====================#

    def t(self):
        return self.get_data(col_name = 't')

    def dt(self):
        return self.get_data(col_name = 'dt')
