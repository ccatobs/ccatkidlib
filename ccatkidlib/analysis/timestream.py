import gc
import so3g
import numpy as np
import polars as pl
import multiprocessing as mp
import pickle
import time

from scipy.signal import welch
from multiprocessing import Pool, shared_memory, Lock
from functools import partial, cached_property
from pathlib import Path
from collections.abc import Iterable
from spt3g import core
from numba import guvectorize, float64

# Local Imports
import ccatkidlib.rfsoc_io as rfsoc_io
import ccatkidlib.utils as utils
import ccatkidlib.analysis.pair as pair
import ccatkidlib.analysis.plot_utils as putils

from ccatkidlib.analysis.mp_utils import init_worker, clear_shared_mem, frame_worker
from ccatkidlib.analysis.data import Data
from ccatkidlib.utils import method_timer


class Timestream(Data):
    '''Class representing a timestream taken with a Radio Frequency System on a Chip
    '''

    def __init__(self, com_to, tones = -1, start = 0, end = -1, analysis_cfg = str(Path(__file__).parent / 'analysis_config.yaml'), **kwargs):
        '''
        Constructor for Timestream. 

        Parameters:
            com_to (str): Which board and drone were used to take the timestream (in form 'bid.drid')
            tones (list): List of resonators to use
            start (float): Start time in seconds (0 seconds is beginning of timestream) 
            end (float): End time in seconds (relative to start time, -1 for no end time)
            analysis_cfg (str): File path of analysis config 

        '''
        kwargs['data_type'] = 'timestream'
        super().__init__(com_to, analysis_cfg, **kwargs)
        

        self.mp = False
        self.processes = None
        self.chunk_size = None
        for key, value in kwargs.items():
            if key == 'mp':
                self.mp = value
            elif key == 'processes':
                self.processes = value
            elif key == 'chunk_size':
                self.chunk_size = value


        # Define array of resonator numbers
        # ---------------------------------
        if isinstance(tones, int): 
            if tones >= 0: 
                self.tones = [tones]
            else:
                self.tones = list(range(self.num_tones))
        elif isinstance(tones, Iterable):
            self.tones = tones
        else:
            error = f"Invalid type {type(tones)} for argument 'tones'. Should be int, list[int], or None."
            rfsoc_io.send_msg('CRITICAL', error)
            raise ValueError(error)

        # Define timestream start and end times
        # -------------------------------------
        self.start = start
        self.end = np.inf if end < 0 else end

        # End time should be larger than start time
        if not self.end - self.start > 0: 
            error = "End time must be greater than start time!"
            rfsoc_io.send_msg('ERROR', error)
            raise ValueError(error)

        # Create packet count attribute
        # -----------------------------
        self.packet_counts = []

    #==========================#
    # Lazily Loaded Attributes #
    #==========================#

    @property
    def data(self):
        '''
        Load timestream I, Q data.
        '''

        if self._data is None:
            ftype = self.data_path[0].suffix
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

    @cached_property
    def sampling_freq(self):
        return self.io_cfg['boards'][f'b{self.bid}']['sampling_freq']

    #=====================#
    # Data Getter Methods #
    #=====================#

    def fft(self, prefix: str | list[str] = '', col_name: str = 'phase', include: int | list[int] | None = None, exclude: int | list[int] | None = None, recalc: bool = False) -> pl.dataframe.frame.DataFrame:        
        col_name = [col_name, 'fft']
        f_col = f'{col_name[-1]}_f'
        if not f_col in self.data.schema: self.data = self.data.with_columns(pl.Series(
                                                                             np.fft.fftshift(
                                                                             np.fft.fftfreq(self.data.height, d=1/self.sampling_freq))).alias(f_col))

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
            f_cols[i] = f'{col_name[-1]}_{data_name}_f'
            col_names[i] = ['t',  data_name, f'{col_name[-1]}_{data_name}']

        sampling_freq = self.sampling_freq
        height = self.data.height

        psd_f, _ = welch(np.array([0]*height), fs=sampling_freq, window=window, nperseg=nperseg, detrend=detrend, average=average)
        psd_f = pl.Series(np.pad(psd_f, (0, height - len(psd_f)), constant_values=np.nan))
        for f_col in f_cols:
            if recalc or not f_col in self.data.schema: self.data = self.data.with_columns(psd_f.alias(f_col))

        args = [[sampling_freq, height, window, nperseg, detrend, average]]*num_prefix # Allow different psd args to be passed for each prefix?
        self.transform([Timestream.calc_psd]*num_prefix, *args, include=include, exclude=exclude, recalc=recalc, col_name = col_names)
        self.data = self._unnest(col_name[-1])
        return self.get_data(col_name=f_cols + [col_name[-1] for col_name in col_names], include=include, exclude=exclude)
   
    #==================#
    # Analysis Methods #
    #==================#

    @staticmethod
    def calc_fft(schema, *args, tones: list[int] | None = None, recalc: bool = False, col_name = ['t', 'phase', 'fft_phase']):
        def _fft(data):
            return np.abs(np.fft.fft(data))
        
        if tones is not None:
            tone = tones[0]
            col_name = [col_name[0]] + [f'{name}_{tone:04d}' for name in col_name[1:]]

        t_col, data_col, fft_col = col_name

        if recalc or not (fft_col in schema):
            return pl.col(data_col).map_batches(_fft).alias(fft_col)
        else:
            return pl.col(fft_col)

    @staticmethod
    def calc_psd(schema, *args, tones: list[int] | None = None, recalc: bool = False, col_name = ['t', 'phase', 'psd_phase']):
        def _psd(data):
            _, psd = welch(data.to_numpy().T, fs=sampling_freq, window=window, nperseg=nperseg, detrend=detrend, average=average)
            psd = np.pad(psd, (0, height - len(psd)), constant_values=np.nan)
            return psd
        
        if len(args) == 6:
            sampling_freq, height, window, nperseg, detrend, average = args
        else:
            rfsoc_io.send_msg('ERROR', 'sampling_freq, window, nperseg, detrend, and average are required arguments')

        if tones is not None:
            tone = tones[0]
            col_name = [col_name[0]] + [f'{name}_{tone:04d}' for name in col_name[1:]]

        t_col, data_col, psd_col = col_name

        if recalc or not (psd_col in schema):
            return pl.col(data_col).map_batches(_psd).alias(psd_col)
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
                original_g3_root = self.io_cfg['file_paths']['g3_root_dir']

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

        if self.mp:
            nbytes = np.prod(shape) * np.dtype(np.int32).itemsize

            # Close and unlink the I and Q shared memory blocks if they already exist (e.g. if the program crashed without cleaning them)
            clear_shared_mem('I_mem')
            clear_shared_mem('Q_mem')
            clear_shared_mem('frames')
            clear_shared_mem('masks')
            
            frames_pk = pickle.dumps(frames)
            masks_pk = pickle.dumps(masks)

            # Create shared memory blocks for storing the I and Q data arrays
            I_mem = shared_memory.SharedMemory(name='I_mem', create=True, size=nbytes)
            Q_mem = shared_memory.SharedMemory(name='Q_mem', create=True, size=nbytes)
            frames_mem = shared_memory.SharedMemory(name='frames', create=True, size=len(frames_pk))
            masks_mem = shared_memory.SharedMemory(name='masks', create=True, size=len(masks_pk))
            
            frames_mem.buf[:len(frames_pk)] = frames_pk
            masks_mem.buf[:len(masks_pk)] = masks_pk

            frames_info = (frames_mem.name, len(frames_pk))
            masks_info = (masks_mem.name, len(masks_pk))
            
            I_name = I_mem.name
            Q_name = Q_mem.name

            args = [(i, self, shape, frames_info, masks_info, I_name, Q_name, start_time, time_precision) for i in range(len(frames))]

            try:
                lock = Lock()
                if self.processes is None: self.processes=min([int(0.7*len(frames)) + 1, mp.cpu_count()])
                if self.chunk_size is None: self.chunk_size = 1
                start_time = time.time()
                with Pool(processes=self.processes, initializer=init_worker, initargs=(lock,)) as pool:
                    result = pool.starmap(frame_worker, args, self.chunk_size)          
                Is = np.ndarray(shape, dtype=np.int32, buffer=I_mem.buf)
                Qs = np.ndarray(shape, dtype=np.int32, buffer=Q_mem.buf) 
                Is = np.array(Is[:])
                Qs = np.array(Qs[:])
            except Exception as e:
                print(e)
                Is = []
                Qs = []
            finally:
                I_mem.close(), Q_mem.close(), frames_mem.close(), masks_mem.close()
                I_mem.unlink(), Q_mem.unlink(), frames_mem.unlink(), masks_mem.unlink()
        else:
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
