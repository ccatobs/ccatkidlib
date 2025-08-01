import concurrent.futures
import polars as pl

from collections.abc import Iterable
from pathlib import Path


import ccatkidlib
import ccatkidlib.rfsoc_io as rfsoc_io

from ccatkidlib.analysis.vna import VNA
from ccatkidlib.analysis.target import Target
from ccatkidlib.analysis.timestream import Timestream

from ccatkidlib.analysis.detector import Detector
from ccatkidlib.analysis import pair

class Network:
    '''
    Class representing a single network of a kinetic inductance detector (KID) array.
    '''

    def __init__(self, com_to: str,
                 analysis_cfg: str = str(Path(__file__).parent / 'analysis_config.yaml'),
                 dets: int | list[int] = -1,
                 cable_delay: float | None = None,
                 detectors: list[ccatkidlib.analysis.detector.Detector] | None = None,
                 sess_ids: str | list[str] | None  = None,
                 include_streams: bool = True,
                 include_targs:   bool = False,
                 **kwargs):
        ''' Initialize Network object by creating detector objects and loading them into a Polars DataFrame with the specified data columns

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

            self.vnas  = {str(vna_file): None for sess_path in sess_paths for vna_file in (sess_path / 'vna' / network_dir).iterdir()}
            self.targs = {str(targ_file): None for sess_path in sess_paths for targ_file in (sess_path / 'targ' / network_dir).iterdir()}
            if include_streams: self.streams = {str(stream_file): None for sess_path in sess_paths for stream_file in (sess_path / 'timestream' / network_dir).iterdir()}

            detectors = []
            detector_types = []
            if include_targs:
                det_objs, det_types = self._create_detectors('Target', com_to, self.targs, analysis_cfg, dets, cable_delay)
                detectors += det_objs
                detector_types += det_types

            if include_streams:
                det_objs, det_types = self._create_detectors('Timestream', com_to, self.streams, analysis_cfg, dets, cable_delay)
                detectors += det_objs
                detector_types += det_types
        else:
            # Ensure that all objects in the detectors list are the correct type
            if any([not isinstance(detector, Detector) for detector in detectors]):
                error = 'All detectors must be of type ccatkidlib.analysis.detector.Detector.'
                rfsoc_io.send_msg('CRITICAL', error)
                raise ValueError(error)
            
            detector_types = ['Timestream']*len(detectors)
            for i, detector in enumerate(detectors):
                if detector.stream is None: detector_types[i] = 'Target'

        self.data = pl.DataFrame({'detector': detectors, 'type': detector_types})

    def add_columns(self, data_cols: str | list[str], max_workers: int = 1) -> pl.dataframe.frame.DataFrame:
        ''' Add columns to the Network.data DataFrame using fields from the ext_cfg or drone_cfg

        Args:
            data_cols (str, list[str]): List of data column names to add. Names must exactly match a field in the ext_cfg or drone_cfg
            max_workers (int, optional): Maximum number of CPU cores to use. Defaults to 1. 
        '''
        detectors = self.data.select(pl.col(['detector', 'type'])).to_numpy()
        num_detectors = self.data.height
        data_dict = {data_col: [None]*num_detectors for data_col in data_cols}
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {executor.submit(Network._extract_data,
                                               detector,
                                               detector_type,
                                               data_cols): i for i, (detector, detector_type) in enumerate(detectors)}
            for future in concurrent.futures.as_completed(future_to_batch):
                i = future_to_batch[future]
                data = future.result()
                for k, v in zip(data_cols, data):
                    data_dict[k][i] = v
        data_df = pl.DataFrame(data_dict)
        self.data = self.data.drop([data_col for data_col in data_cols if data_col in self.data.schema])
        self.data = pl.concat([self.data, data_df], how='horizontal')
        return self.data

    def match_detectors(self, ):
        ''' Match detectors across different target sweeps using nearest frequency neighbor to 
        '''


        return

    #================#
    # Helper Methods #
    #================#

    def _create_detectors(self, det_type, com_to, path_dict, analysis_cfg, dets, cable_delay):
        '''
        '''
        def _create_sweep(sweep_path, sweep_dict, sweep_class, dets):
            '''
            '''
            if Path(sweep_path).exists():
                sweep = sweep_dict[sweep_path]
                if sweep is None:
                    sweep = sweep_class(com_to = com_to, analysis_cfg = analysis_cfg, data_path = sweep_path, tones=dets)
                    sweep_dict[sweep_path] = sweep
            else:
                sweep = None
            return sweep

        detectors = [None]*len(path_dict)
        detector_types = ['']*len(path_dict)
        for i, data_path in enumerate(path_dict):
            vna_path, targ_path = pair.get_sweep(data_path)

            if det_type == 'Timestream':
                stream_path = data_path
            else:
                stream_path = None
                targ_path = data_path

            vna = _create_sweep(vna_path, self.vnas, VNA, None)
            targ = _create_sweep(targ_path, self.targs, Target, dets)
            
            detector = Detector(com_to=com_to, analysis_cfg=analysis_cfg, dets=dets, cable_delay=cable_delay, targ=targ, vna=vna, stream_path=stream_path)
            detectors[i] = detector
            detector_types[i] = det_type
        return detectors, detector_types
    
    @staticmethod
    def _extract_data(detector, detector_type, data_cols):
        cfgs = [detector.stream.drone_cfg, detector.stream.ext_cfg] if detector_type == 'Timestream' else [detector.targ.drone_cfg, detector.targ.ext_cfg]
        data = [None]*len(data_cols)
        for i, col in enumerate(data_cols):
            for cfg in cfgs:
                cfg_data = ccatkidlib.utils.dict_get(cfg, col)
                if cfg_data is not None:
                    data[i] = cfg_data
                    break
        return data