import numpy as np
import polars as pl

from pathlib import Path
from functools import cached_property
from numba import guvectorize, njit, prange, float64, int64

# Local Imports
import ccatkidlib.log as log
import ccatkidlib.analysis.utils.pair as pair

from ccatkidlib.analysis.core.sweep import Sweep
from ccatkidlib.analysis.fit.fit import linear_fit

class VNA(Sweep):
    '''Class representing a vector network analyzer (VNA) esque sweep taken with a Radio Frequency System on a Chip (RFSoC). 
    
    Subclasses the Sweep class. 

    Attributes:
        cable_delay (float): Cable delay of the RF chain in nanoseconds
    '''

    def __init__(self, com_to: str, cfg_path: str = str(Path(__file__).parents[1] / 'analysis_config.yaml'), **kwargs):
        kwargs['data_type'] = 'vna'
        super().__init__(com_to, cfg_path, **kwargs)

    #=====================#
    # Data Getter Methods #
    #=====================#

    def stitch_phase(self, threshold: float = 1.9*np.pi, stitch_percent: float = 10.0, recalc: bool = False) -> pl.lazyframe.frame.LazyFrame:
        '''Stitch the VNA sweep phase data to remove discontinuites from wrapping and bin edges

        Note:
            Only one of ``include`` and ``exclude`` arguments can be specified. If neither are specified, returns data for all tones.

        Args:
            include (int | list[int], optional): List of tones to include. Defaults to None
            exclude (int | list[int], optional): List of tones to exclude. Defaults to None
        Returns:
            polars.lazyframe.frame.LazyFrame: Polars DataFrame with specified data columns

        '''

        phase_name = self.phase().columns[0] # Ensure that phase data exists, calculate if not

        # Get sweep steps from drone config
        try:
            sweep_steps = self.drone_cfg['tones']['sweep_steps']
        except KeyError:
            sweep_steps = self.drone_cfg['tones']['N_step']

        args = [[sweep_steps, threshold, stitch_percent]]
        col_name = ['f', phase_name, 'stitch_phase']
        self.transform(VNA._calc_stitch_phase, *args, include=None, exclude=None, recalc = recalc, col_name = col_name)
        return self.get_data(col_name=col_name[-1])

    def stitch_mag(self, stitch_percent: float = 10.0, med_win: int = 3, recalc: bool = False) -> pl.lazyframe.frame.LazyFrame:
        '''Stitch the VNA sweep mag data to remove discontinuites at bin edges

        Note:
            Only one of ``include`` and ``exclude`` arguments can be specified. If neither are specified, returns data for all tones.

        Args:
            include (int | list[int], optional): List of tones to include. Defaults to None
            exclude (int | list[int], optional): List of tones to exclude. Defaults to None
        Returns:
            polars.lazyframe.frame.LazyFrame: Polars DataFrame with specified data columns

        '''

        mag_name = self.mag().columns[0] # Ensure that mag data exists, calculate if not

        # Get sweep steps from drone config
        try:
            sweep_steps = self.drone_cfg['tones']['sweep_steps']
        except KeyError:
            sweep_steps = self.drone_cfg['tones']['N_step']

        args = [[sweep_steps, stitch_percent, med_win]]
        col_name = ['f', mag_name, 'stitch_mag']
        self.transform(VNA._calc_stitch_mag, *args, include=None, exclude=None, recalc = recalc, col_name = col_name)
        return self.get_data(col_name=col_name[-1])

    #==================#
    # Analysis Methods #    
    #==================#

    @staticmethod
    def _calc_stitch_phase(schema, *args, tones = None, recalc: bool = False, col_name = ['f', 'phase', 'stitch_phase']):
        if len(args) == 3:
            sweep_steps, threshold, stitch_percent = args
        else:
            log.log('ERROR', 'sweep_steps, threshold, and stitch_percent are required arguments.')
            raise ValueError

        f_col, phase_col, stitch_col = col_name

        if recalc or not (stitch_col in schema):
            return (pl.struct([f_col, phase_col])
                     .map_batches(lambda arrs: stitch_phase(arrs.struct.field(f_col),
                                                            arrs.struct.field(phase_col),
                                                            int(sweep_steps),
                                                            float(threshold),
                                                            float(stitch_percent)
                                                            ), return_dtype=pl.Float64).alias(stitch_col))
        else:
            return pl.col(stitch_col)

    @staticmethod
    def _calc_stitch_mag(schema, *args, tones = None, recalc: bool = False, col_name = ['f', 'mag', 'stitch_mag']):
        if len(args) == 3:
            sweep_steps, stitch_percent, med_win = args
        else:
            log.log('ERROR', 'sweep_steps, stitch_percent, and med_win are required arguments.')
            raise ValueError

        f_col, mag_col, stitch_col = col_name

        if recalc or not (stitch_col in schema):
            return (pl.struct([f_col, mag_col])
                    .map_batches(lambda arrs: stitch_mag(arrs.struct.field(f_col),
                                                         arrs.struct.field(mag_col),
                                                         int(sweep_steps),
                                                         float(stitch_percent),
                                                         int(med_win)
                                                         ), return_dtype=pl.Float64).alias(stitch_col))
        else:
            return pl.col(stitch_col)

    def filter_det_f(self, win: int = 3, stitch_phase: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """ Filters out detectors found by find_resonators that are likely fake (noise or otherwise) based on the slope of the phase around the found frequency

        Original implementation by Ben Keller, modified to work with Numba.
        Args:
            win (int): The window size with which to fit the phase around the found frequency. 
            stitch_phase (bool): Whether to use stitched phase data
        Returns:
            tuple[np.ndarray, np.ndarray]: Frequencies of likely real detectors, frequencies of likely fake detectors.
        """
        
        f, det_f = self.f().to_numpy().T[0], self.det_f
        phase = self.stitch_phase().to_numpy().T[0] if stitch_phase else self.phase().to_numpy().T[0]

        slopes = filter_det_f(f, phase, det_f, int(win))
        real_det = det_f[slopes > 0]
        fake_det = det_f[slopes <= 0]
            
        return real_det, fake_det

    #==========================#
    # Lazily Loaded Attributes #
    #==========================#
    
    @cached_property
    def cable_delay(self) -> float:
        '''Get the cable delay of the RF chain using the phase data

        Returns:
            float: The cable delay in nanoseconds
        '''

        f, phase = self.f(), self.stitch_phase()

        # Get cable delay in units of rad·s
        cable_delay, intercept = linear_fit(f.to_numpy().T[0], phase.to_numpy().T[0]) # Need to convert f and phase to 1D numpy arrays since linear_fit is njitted

        return cable_delay*1e9/(2*np.pi)
    
    #==========================#
    # Internal Loading Methods #
    #==========================#

    def _get_res_s21z(self):
        res_s21z = None
        s21z = self.s21z
        fs = list(self.fs())
        res_freqs = self.res_freqs
        if res_freqs is not None:
            try:
                res_s21z = [s21z[fs.index(freq)] for freq in res_freqs]
            except ValueError:
                res_s21z = super()._get_res_s21z()
        return res_s21z

#=======================#
# Data Analysis GUFuncs #
#=======================# 

@guvectorize([(float64[:], float64[:], int64, float64, float64, float64[:])], '(n),(n),(),(),()->(n)')
def stitch_phase(f, phase, sweep_steps, threshold, stitch_percent, result):
    '''NumPy generalized universal function for stitching vna sweep phase data to remove discontinuities from phase wrapping and bin edges.
    
    Phase wrapping discontinuities are removed by shifting adjacent points that differ by more than the specified threshold.
    
    Bin edge discontinuities are removed by linearly fitting the edges of adjacent bins using the percent specified and using 
    the fits to determine the phase difference between the bins. The bins are then shifted to remove the phase difference.

    Args:
        f (np.array[float64]): VNA sweep frequency data
        phase (np.array[float64]): VNA sweep phase data
        sweep_steps (int): Number of points in a single tone
        threshold (float): Threshold phase difference to be considered phase wrapping 
        stitch_percent (float): Percent of bin edge data to use to determine difference between adjacent bins
    Returns:
        result (np.array[float64]): Stitched VNA sweep phase data 
    '''
    # Stitch phase discontinuities caused by wrapping from -pi to pi
    # --------------------------------------------------------------
    phase = phase.copy() # Need to copy to prevent editing phase data in-place
    curr_shift = 0
    for i in range(len(phase) - 1):
        diff = phase[i+1] - phase[i]
        phase[i] -= curr_shift
        if diff > threshold:
            curr_shift += 2*np.pi 
        elif diff < -1*threshold:
            curr_shift -= 2*np.pi
    phase[-1] -= curr_shift
    
    # Need to ensure that arrays are allocated in a contiguous block of memory for reshaping
    phase_bins = np.ascontiguousarray(phase).reshape((-1, sweep_steps))
    freq_bins  = np.ascontiguousarray(f).reshape((-1, sweep_steps))

    # Stitch phase discontiuities at bin edges
    # ----------------------------------------
    curr_shift = 0
    stitch_ends = int(sweep_steps / stitch_percent)

    slope, intercept = linear_fit(freq_bins[0, -1*stitch_ends:], phase_bins[0, -1*stitch_ends:])

    prev = 0
    next = intercept + slope*freq_bins[0, -1]
    for i in range(len(phase_bins) - 1):
        slope_prev, intercept_prev = linear_fit(freq_bins[i + 1, :stitch_ends], phase_bins[i + 1, :stitch_ends])
        prev = intercept_prev + slope_prev*freq_bins[i + 1, 0]
        diff = prev - next
        slope_next, intercept_next = linear_fit(freq_bins[i + 1, -1*stitch_ends:], phase_bins[i + 1, -1*stitch_ends:])
        next = intercept_next + slope_next*freq_bins[i + 1, -1]
        
        phase_bins[i] -= curr_shift
        curr_shift += diff
    phase_bins[-1] -= curr_shift
    result[:] = phase_bins.flatten()

@guvectorize([(float64[:], float64[:], int64, float64, int64, float64[:])], '(n),(n),(),(),()->(n)')
def stitch_mag(f, mag, sweep_steps, stitch_percent, med_win, result):
    mag_filt = np.ones(mag.size)

    offset = (med_win - 1) // 2
    size = mag.size

    mag_filt[0] = np.median(mag[0:offset+1])
    mag_filt[-1] = np.median(mag[size - offset - 1:])
    for i in np.arange(offset, size - offset, 1):
        mag_filt[i] = np.median(mag[i-offset:i+offset+1])

    mag_bins  = np.ascontiguousarray(mag_filt).reshape(-1, sweep_steps)
    freq_bins = np.ascontiguousarray(f).reshape(-1, sweep_steps)

    # Stitch mag discontiuities at bin edges
    # --------------------------------------
    curr_shift = 0
    stitch_ends = int(sweep_steps / stitch_percent)

    slope, intercept = linear_fit(freq_bins[0, -1*stitch_ends:], mag_bins[0, -1*stitch_ends:])

    prev = 0
    next = intercept + slope*freq_bins[0, -1]
    for i in range(len(mag_bins) - 1):
        slope_prev, intercept_prev = linear_fit(freq_bins[i + 1, :stitch_ends], mag_bins[i + 1, :stitch_ends])
        prev = intercept_prev + slope_prev*freq_bins[i+1, 0]
        diff = prev - next
        slope_next, intercept_next = linear_fit(freq_bins[i + 1, -1*stitch_ends:], mag_bins[i + 1, -1*stitch_ends:])
        next = intercept_next + slope_next*freq_bins[i + 1, -1]
        
        mag_bins[i] -= curr_shift
        curr_shift += diff
    mag_bins[-1] -= curr_shift
    result[:] = mag_bins.flatten()

@njit(parallel=True, cache=True)
def filter_det_f(f, phase, det_f, win):
    offset = (win - 1) // 2
    slopes = np.zeros(det_f.size)
    for i in range(det_f.size):
        peak_idx = np.where(f == det_f[i])[0]

        if peak_idx.size == 0:
            pass
        else:
            peak_idx = peak_idx[0]
            f_win = np.ascontiguousarray(f[peak_idx - offset: peak_idx + offset + 1])
            phase_win = np.ascontiguousarray(phase[peak_idx - offset: peak_idx + offset + 1])
            slope, intercept = linear_fit(f_win, phase_win)
            slopes[i] = slope
    return slopes