import concurrent.futures
import polars as pl
import holoviews as hv
import time

from collections.abc import Iterable
from pathlib import Path
from tqdm import tqdm

import ccatkidlib
import ccatkidlib.rfsoc_io as rfsoc_io
import ccatkidlib.analysis.utils.pair as pair
import ccatkidlib.analysis.utils.multiprocess as ccat_mp
import ccatkidlib.analysis.viz.viz_utils as viz_utils

from ccatkidlib.analysis.core.vna import VNA
from ccatkidlib.analysis.core.target import Target

from ccatkidlib.analysis.core.detector import Detector

class Network:
    '''
    Class representing a single network of a kinetic inductance detector (KID) array.
    '''

    def __init__(self, com_to: str,
                 analysis_cfg: str = str(Path(__file__).parents[1] / 'analysis_config.yaml'),
                 dets: int | list[int] = -1,
                 noise_tones: int | list[int] | None = None,
                 cable_delay: float | None = None,
                 detectors: list[ccatkidlib.analysis.core.detector.Detector] | None = None,
                 sess_ids: str | list[str] | None  = None,
                 include_streams: bool = True,
                 include_targs:   bool = False,
                 **kwargs):
        ''' Initialize Network object by creating Detector objects and loading them into a Polars DataFrame with the specified data columns

        Note:
            Either the full data path must be provided via the ``data_path`` key word argument or
            both the timestamp and data type must be provided via the ``data_type`` and ``timestamp`` key word arguments respectively

        Args:
            com_to (str): Drone that took the data. In form 'Board.Drone'
            analysis_cfg (str, optional): Path to analysis config. Defaults to analysis config in ccatkidlib/ccatkidlib/analysis directory.

            detectors (list[ccatkidlib.analysis.detector.Detector]): List of Detector objects
            sess_ids (list[str]): List of session IDs to load target sweeps/timestreams from

            include_streams (bool): Whether to load timestreams (will also load associated target sweeps)
            include_targs (bool): Whether to load target sweeps (will load all target sweeps, even those without an associated timestream)

            **kwargs: Key word arguments for finding data file. See below:
            root_data_dir (str, optional): Root directory where data is stored. Defaults to that specified in analysis config
            data_dir (str, optional): Directory where data is stored
            dates (str, list[str], optional): Date data was taken
        
        Raises:
            ValueError: If multiple data files are specified with differing file types or timestamps
            FileNotFoundError: If any data files cannot be found 
        '''
        
        bid, drid = com_to.split('.')
        network_dir = f'B{bid}D{drid}'

        self.analysis_cfg, self.viz_cfg = rfsoc_io.load_config(analysis_cfg)

        self.save_fig = self.viz_cfg['save']['save_fig']
        self.overwrite = self.viz_cfg['save']['overwrite']
        self.save_fmt = self.viz_cfg['save']['save_fmt']
        self.figs_per_file = self.viz_cfg['save']['figs_per_file']

        self.root_dir = self.analysis_cfg['file_paths']['root_data_dir']
        if not self.root_dir[-1] == '/': self.root_dir += '/'
        if not isinstance(detectors, Iterable) or len(detectors) == 0:
            if not sess_ids:
                error = 'Must either provide a list of Detector objects or specify session ID(s) of the data to load.'
                rfsoc_io.send_msg('CRITICAL', error)
                raise RuntimeError(error)
            elif not (include_streams or include_targs):
                error = 'Must include target sweeps or timestreams (or both).'
                rfsoc_io.send_msg('CRITICAL', error)
                raise RuntimeError(error)

            data_dir = '**'
            dates = ['**']
            for key, value in kwargs.items():
                if key == 'root_data_dir':
                    self.root_dir = value
                elif key == 'data_dir':
                    data_dir = value
                elif key == 'date':
                    dates = value
            if isinstance(dates, str): dates = [dates]
            if isinstance(sess_ids, str): sess_ids = [sess_ids]

            sess_paths = []
            for sess_id in sess_ids:
                for date in dates:
                    sess_dir = pair.get_sess_dir(sess_id, data_dir = data_dir, root_data_dir = self.root_dir, date = date)
                    if Path(sess_dir).exists():
                        sess_paths.append(Path(sess_dir))
                        break            
            self.vnas  = {str(vna_file): None for sess_path in sess_paths if (vna_dir := (sess_path / 'vna' / network_dir)).exists() for vna_file in vna_dir.iterdir()}
            self.targs = {str(targ_file): None for sess_path in sess_paths if (targ_dir := (sess_path / 'targ' / network_dir)).exists() for targ_file in targ_dir.iterdir()}
            if include_streams: self.streams = {str(stream_file): None for sess_path in sess_paths if (stream_dir := (sess_path / 'timestream' / network_dir)).exists() for stream_file in stream_dir.iterdir()}

            detectors = []
            detector_types = []
            detector_timestamps = []
            if include_targs:
                det_objs, det_types, det_timestamps = self._create_detectors('Target', com_to, self.targs, analysis_cfg, dets, noise_tones, cable_delay)
                detectors += det_objs
                detector_types += det_types
                detector_timestamps += det_timestamps

            if include_streams:
                det_objs, det_types, det_timestamps = self._create_detectors('Timestream', com_to, self.streams, analysis_cfg, dets, noise_tones, cable_delay)
                detectors += det_objs
                detector_types += det_types
                detector_timestamps += det_timestamps
        else:
            # Ensure that all objects in the detectors list are the correct type
            if any([not isinstance(detector, Detector) for detector in detectors]):
                error = 'All detectors must be of type ccatkidlib.analysis.core.detector.Detector.'
                rfsoc_io.send_msg('CRITICAL', error)
                raise ValueError(error)
            
            detector_types = ['Timestream']*len(detectors)
            for i, detector in enumerate(detectors):
                if detector.stream is None: detector_types[i] = 'Target'

        self.det_dict = {str(det) : det for det in detectors}
        self.data = pl.DataFrame({'detector': list(self.det_dict.keys()), 'type': detector_types, 'timestamp': detector_timestamps})

        # Create directory for saving figures
        self.timestamp = '_'.join(sess_ids)
        save_dir = Path(detectors[0].save_dir).parent
        dir_name = f'{'stream' if include_streams else 'targ'}_network_{self.timestamp}'
        self.save_dir = save_dir / dir_name
        rfsoc_io.create_dir(self.save_dir)
        
    def add_columns(self, data_cols: str | list[str], max_workers: int = 1, ex=None) -> pl.dataframe.frame.DataFrame:
        ''' Add columns to the Network.data DataFrame using fields from the ext_cfg or drone_cfg

        Args:
            data_cols (str, list[str]): List of data column names to add. Names must exactly match a field in the ext_cfg or drone_cfg
            max_workers (int, optional): Maximum number of CPU cores to use. Defaults to 1. 
        '''
        detectors = self.data.select(pl.col(['detector', 'type'])).to_numpy()
        detectors = [(self.det_dict[detector], detector_type) for detector, detector_type in detectors]
 
        detector_cfgs = [[detector.stream.drone_cfg, detector.stream.ext_cfg] if detector_type == 'Timestream' else [detector.targ.drone_cfg, detector.targ.ext_cfg] for detector, detector_type in detectors]
        num_detectors = self.data.height

        combined_names = ['_'.join(data_col) if not isinstance(data_col, str) and isinstance(data_col, Iterable) else data_col for data_col in data_cols]
        data_dict = {name: [None]*num_detectors for name in combined_names}
        with ccat_mp.optional_executor(max_workers, ex=ex) as executor:
            future_to_batch = {executor.submit(Network._extract_data,
                                               detector_cfg,
                                               data_cols): i for i, detector_cfg in enumerate(detector_cfgs)}
            for future in concurrent.futures.as_completed(future_to_batch):
                i = future_to_batch[future]
                data = future.result()
                for k, v in zip(combined_names, data):
                    data_dict[k][i] = v
        data_df = pl.DataFrame(data_dict)
        self.data = self.data.drop([name for name in combined_names if name in self.data.schema])
        self.data = pl.concat([self.data, data_df], how='horizontal')
        return self.data

    def match_detectors(self, ):
        ''' Match detectors across different target sweeps using nearest frequency neighbor to 
        '''


        return

    def combine_properties(self, data_cols = []):
        properties_df = None
        for i, (det, *cols) in enumerate(self.data.select(['detector'] + data_cols).iter_rows()):
            df = self.det_dict[det].properties
            df = df.with_columns([pl.lit(data).alias(name) for name, data in zip(data_cols, cols)])
            try:
                properties_df = df if properties_df is None else pl.concat([properties_df, df], how='diagonal')
            except Exception as e:
                rfsoc_io.send_msg('WARNING', 'Failed to combine properties DataFrame with error %s', e)
        return properties_df

    #==================#
    # Plotting Methods #
    #==================#
    @staticmethod
    def _plot(df, plot_opts, *plot_args, **kwargs):
        plot_func, by, overlay_cols, args = plot_args

        kwargs['save_fig'] = False
        
        fig = plot_func(*args, df=df, by=by, **kwargs)
        if overlay_cols is not None: fig = fig.overlay(overlay_cols).opts(show_legend=False)
        fig.opts(*plot_opts)

        return fig
    
    def plot(self, func, data_type, *args, data_cols = [], return_df = True, save_fig: bool | None = None, overwrite: bool | None = None, save_name: str = None, overlay_cols: str | list[str] = None, **kwargs):
        '''
        

        Args:
            func (str): 
            data_type (str):
        '''
        
        # Validate data_type
        if not data_type in ['vna', 'targ', 'stream', 'detector']:
            error = "data_type must be one of: 'vna', 'targ', 'stream', or 'detector'"
            rfsoc_io.send_msg('CRITICAL', error)
            raise ValueError(error)
        
        # Parse func
        func_parts = func.split('_')
        if not func_parts[-1] == 'plot':
            func_parts.append('plot')
            func = '_'.join(func_parts)

        all_cols, by_cols = ['detector'] + data_cols, data_cols
        kwargs['return_df'], kwargs['return_fig'] = True, False
        plot_df, bys = None, [None]*self.data.height
        for i, (det, *cols) in enumerate(self.data.select(all_cols).sort(all_cols).iter_rows()):
            det = self.det_dict[det]
            data_obj = det if data_type == 'detector' else getattr(det, data_type)
            if hasattr(data_obj, func) and callable(plot_func := getattr(data_obj, func)): 
                df, by = plot_func(*args, **kwargs)
                bys[i] = by
                
                df = df.with_columns([pl.lit(data).alias(name) for name, data in zip(data_cols, cols)])
                plot_df = df if plot_df is None else pl.concat([plot_df, df], how='diagonal')
        
        if not len(set(bys)) == 1: 
            error = "Inconsistent by columns specified"
            rfsoc_io.send_msg('CRITICAL', error)
            raise ValueError(error)
        if bys[0] is not None: by_cols += [bys[0]]
        kwargs['return_df'], kwargs['return_fig'] = False, True

        plot_opts = []

        plot_args = [plot_func, by_cols, overlay_cols, args]
        
        # Create plot for immediate visualization
        # ---------------------------------------
        fig = Network._plot(plot_df, plot_opts, *plot_args, **kwargs)

        # Save plot in background
        # -----------------------
        prefix = kwargs['y_prefix'] if 'y_prefix' in kwargs else kwargs.get('prefix', '')
        if save_name is None: save_name = f'network_{data_type}{'_' if prefix else ''}{prefix}_{func}'
        viz_utils.save_fig(self, Network._plot, plot_df, plot_opts, *plot_args, save_fig = save_fig, overwrite=overwrite, save_name=save_name, **kwargs)

        if return_df:
            return fig, plot_df
        else:
            return fig        

    #================#
    # Helper Methods #
    #================#

    def _create_detectors(self, det_type, com_to, path_dict, analysis_cfg, dets, noise_tones, cable_delay):
        '''
        '''
        def _create_sweep(sweep_path, sweep_dict, sweep_class, dets):
            '''
            '''
            if Path(sweep_path).exists():
                sweep = sweep_dict[str(sweep_path)]
                if sweep is None:
                    sweep = sweep_class(com_to = com_to, analysis_cfg = analysis_cfg, data_path = sweep_path, tones=dets, noise_tones=noise_tones)
                    sweep_dict[sweep_path] = sweep
            else:
                sweep = None
            return sweep

        detectors = [None]*len(path_dict)
        detector_types = ['']*len(path_dict)
        detector_timestamps = ['']*len(path_dict)
        for i, data_path in enumerate(tqdm(path_dict, desc='Creating Detectors...')):
            vna_path, targ_path = pair.get_sweep(data_path)

            if det_type == 'Timestream':
                stream_path = data_path
            else:
                stream_path = None
                targ_path = data_path
            

            vna = _create_sweep(vna_path, self.vnas, VNA, None)
            targ = _create_sweep(targ_path, self.targs, Target, dets)
            
            detector = Detector(com_to=com_to, analysis_cfg=analysis_cfg, dets=dets, noise_tones=noise_tones, cable_delay=cable_delay, targ=targ, vna=vna, stream_path=stream_path)
            
            detectors[i] = detector
            detector_types[i] = det_type
            detector_timestamps[i] = rfsoc_io.get_timestamp(data_path)
        return detectors, detector_types, detector_timestamps
    
    @staticmethod
    def _extract_data(cfgs, data_cols):
        data = [None]*len(data_cols) 
        for i, col in enumerate(data_cols):
            for cfg in cfgs:
                cfg_data = ccatkidlib.utils.dict_get(cfg, col)
                if cfg_data is not None:
                    data[i] = cfg_data
                    break
        return data
