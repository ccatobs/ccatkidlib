import numpy as np
import polars as pl
import pathlib
import sys
import concurrent.futures

from pathlib import Path

# local imports
import ccatkidlib
import ccatkidlib.rfsoc_io as rfsoc_io
import ccatkidlib.analysis.pair as pair

from ccatkidlib.analysis.timestream import Timestream
from ccatkidlib.analysis.vna import VNA
from ccatkidlib.analysis.target import Target


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
                 analysis_cfg: str = str(Path(__file__).parent / 'analysis_config.yaml'),
                 dets: int | list[int] = -1,
                 cable_delay: float | None = None,
                 stream: ccatkidlib.analysis.timestream.Timestream | None = None, stream_path: str | pathlib.PosixPath | list[str] | list[pathlib.PosixPath] | None = None, stream_timestamp: int | str | None = None,
                 targ: ccatkidlib.analysis.target.Target = None, targ_path: str | pathlib.PosixPath | None = None, targ_timestamp: int | str | None = None,
                 vna: ccatkidlib.analysis.vna.VNA = None, vna_path: str | pathlib.PosixPath | None = None, vna_timestamp: int | str | None = None,
                 **kwargs):

        # Create Timestream, Target, and VNA data objects based provided arguments
        # ------------------------------------------------------------------------
        if not isinstance(stream, ccatkidlib.analysis.timestream.Timestream): stream = Detector._load_data(Timestream, com_to, analysis_cfg, dets, stream_timestamp, stream_path, **kwargs)
        if not isinstance(targ, ccatkidlib.analysis.target.Target): targ = Detector._load_data(Target, com_to, analysis_cfg, dets, targ_timestamp, targ_path, **kwargs)
        if not isinstance(vna, ccatkidlib.analysis.vna.VNA): vna = Detector._load_data(VNA, com_to, analysis_cfg, None, vna_timestamp, vna_path, **kwargs)
   
        # Must have a sweep to do meaningful data analysis
        # ------------------------------------------------
        if not isinstance(targ, ccatkidlib.analysis.target.Target):
            if isinstance(stream, ccatkidlib.analysis.timestream.Timestream): # If timestream provided, try to find associated sweep
                self.dets = stream.tones
                vna_path, targ_path = pair.get_sweep(stream.data_path[0], **kwargs)

                # Load found sweep
                if  Path(vna_path).exists(): vna = Detector._load_data(VNA, com_to, analysis_cfg, None, vna_timestamp, vna_path, **kwargs)
                if Path(targ_path).exists(): targ = Detector._load_data(Target, com_to, analysis_cfg, dets, targ_timestamp, targ_path, **kwargs)

                if not isinstance(vna, ccatkidlib.analysis.vna.VNA) and not isinstance(targ, ccatkidlib.analysis.target.Target):
                    error = 'Failed to find target sweep or VNA sweep associated with timestream.'
                    rfsoc_io.send_msg('CRITICAL', error)
                    raise RuntimeError(error)

            else: # Error of no sweep or timestream provided
                error = 'A timestream, target sweep or both need to be specified!'
                rfsoc_io.send_msg('CRITICAL', error)
                raise RuntimeError(error)
        else:
            self.dets = targ.tones
            if vna is None: vna_path, _ = pair.get_sweep(targ.data_path[0], **kwargs)
            if Path(vna_path).exists(): vna = Detector._load_data(VNA, com_to, analysis_cfg, None, vna_timestamp, vna_path, **kwargs)

        self.bid, self.drid = com_to.split('.')
        self.analysis_cfg, self.viz_cfg = rfsoc_io.load_config(analysis_cfg)

        self.stream = stream
        self.targ = targ
        self.vna = vna

        self.cable_delay = self.vna.cable_delay if cable_delay is None and isinstance(self.vna, ccatkidlib.analysis.vna.VNA) else cable_delay

        # Fitting 
        self._properties = {f'det_{det:04d}': {} for det in self.dets}
        self._properties_df = pl.DataFrame({'det': self.dets})
        
        fit_dir = self.analysis_cfg['file_paths']['fit_dir']
        if not fit_dir in sys.path: sys.path.append(fit_dir)
        import resonator_model_v3
        globals()['resonator_model_v3'] = resonator_model_v3

    #==========================#
    # Lazily Loaded Attributes #
    #==========================#

    @property
    def properties(self):

        # Reshape properties dictionary to have resonator properties as primary keys
        new_dict = {'det': []}
        for det, props in self._properties.items():
            for i, (prop, value) in enumerate(props.items()):
                if i == 0: new_dict['det'].append(int(det.split('_')[-1]))
                curr = new_dict.get(prop, [])
                if curr: 
                    curr.append(value)
                else:
                    new_dict[prop] = [value]

        new_df = pl.DataFrame(new_dict)
        shared_cols = set(self._properties_df.columns) & set(new_df.columns) - {'det'}
        self._properties_df = self._properties_df.drop(list(shared_cols))
        self._properties_df = self._properties_df.join(pl.DataFrame(new_dict), on='det', how='full', coalesce=True)
        return self._properties_df

    #=====================#
    # Data Getter Methods #
    #=====================#

    def nonlinear_fit(self, include: int | list[int] | None = None, exclude: int | list[int] | None = None, recalc: bool = False, nonlinear=False, asymm = False, fix_cable = False, fix_thetaQ = False, max_workers=1):
        col_name = ['f', 'I', 'Q', 'nonlinear_fit']
        
        
        args = [[self, nonlinear, asymm, fix_cable, fix_thetaQ, max_workers]]
        self.targ.transform(Detector.calc_nonlinear_fit, *args, include=include, exclude=exclude, recalc = recalc, col_name = col_name, batch_size=len(self.targ.tones))

        struct_cols = []
        schema = self.targ.data.schema
        for name, data in schema.items():
            if isinstance(data, pl.Struct) and col_name[-1] in name:
                self.targ.data = self.targ.data.drop([col for col in dict(data).keys() if col in schema])
                struct_cols.append(name)
        self.targ.data = self.targ.data.unnest(struct_cols)
        return self.targ.get_data(col_name=col_name[-1], include=include, exclude=exclude)

    #=================#
    # Fitting Methods #
    #=================#

    @staticmethod
    def calc_nonlinear_fit(schema, *args, tones: list[int] = 0, recalc: bool = False, col_name = ['f', 'I', 'Q', 'nonlinear_fit']):
        ''' Fit using resonator_model_v3
        '''

        def _nonlinear_fit(df):
            struct = df.struct

            results_dict = {}
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {executor.submit(resonator_model_v3.full_fit,
                                                   struct.field(f_col).to_numpy(),
                                                   struct.field(I_col).to_numpy(),
                                                   struct.field(Q_col).to_numpy(),
                                                   nonlinear=nonlinear,
                                                   asymm=asymm,
                                                   fix_cable=fix_cable,
                                                   fix_thetaQ=fix_thetaQ):  (tone, I_col, Q_col) for tone, (f_col, I_col, Q_col) in zip(to_fit, batches)}
                
                for future in concurrent.futures.as_completed(future_to_batch):
                    tone, I_col, Q_col = future_to_batch[future]
                    try:
                        result = future.result()
                        best_fit = result.best_fit
                        self._properties[f'det_{tone:04d}'] = {f'{col_name[-1]}_{k}': v for k, v in result.best_values.items()}
                    except Exception as e:
                        rfsoc_io.send_msg('WARNING', 'Fit failed for tone %s with exception: %s', tone, e)
                        best_fit = np.zeros(len(struct.field(I_col)))
                    results_dict[f'{col_name[-1]}_{I_col}'] = best_fit.real
                    results_dict[f'{col_name[-1]}_{Q_col}'] = best_fit.imag

            df = pl.DataFrame(results_dict)
            return pl.Series(df.select(pl.struct(df.columns)))
            
        if len(args) == 6:
            self, nonlinear, asymm, fix_cable, fix_thetaQ, max_workers = args
        else:
            error = 'nonlinear, asymm, fix_cable, and fix_thetaQ are required arguments.'
            rfsoc_io.send_msg('ERROR', error)
            raise ValueError(error)

        fit_col = [f'{col_name[-1]}_{col_name[-2]}_{tone:04d}' for tone in tones]
        to_fit = tones if recalc else [tone for tone, col in zip(tones, fit_col) if col not in schema]

        if not len(to_fit) == 0:
            batches = [[f'{name}_{tone:04d}' for name in col_name[:-1]] for tone in to_fit]
            fit_col = f'{col_name[-1]}_{to_fit[0]:04d}'
            
            batches_flat = [col for batch in batches for col in batch]
            return pl.struct(batches_flat).map_batches(_nonlinear_fit).alias(fit_col)
        else:
            return pl.col(fit_col)

    @staticmethod
    def calc_phase_fit(schema, *args, tone: int | None = None, recalc: bool = False):
        return 

    def calc_submm_fit(schema, *args, tone: int | None = None, recalc: bool = False):
        return

    #================#
    # Helper Methods #
    #================#

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
