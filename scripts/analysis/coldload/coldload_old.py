import pickle
import polars as pl
import numpy as np
import time
import sys
import gc

from tqdm import tqdm
from math import ceil
from pathlib import Path
from lmfit import Parameters

from ccatkidlib.analysis.core.network import Network
from ccatkidlib.analysis.core.detector import Detector
import ccatkidlib.analysis.fit.fit as ccat_fit

import panel as pn
import datashader as ds
import holoviews as hv
import hvplot.polars

from holoviews import opts
from collections.abc import Iterable
from holoviews.operation.datashader import rasterize, datashade, dynspread, shade
from bokeh.palettes import Sunset, Plasma256, Inferno256, Cividis256, Magma256, Viridis256

pn.extension('mathjax')
hv.extension('bokeh', enable_mathjax = True)

import warnings
warnings.filterwarnings('ignore')

#====================#
# Analysis Functions #
#====================#

def fit_detectors(detector, nonlinear = True, phase = True, recalc=False):
    if phase:
        detector.nonlinear_fit(nonlinear=False, max_workers=4, recalc=True)
        center_IQ(detector, data='targ', recalc=recalc, delay_col='nonlinear_fit_delay_ns') # Should change to using the nonlinear fit cable_delay
        detector.targ.phase(prefix='mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', recalc=recalc)
        detector.phase_fit(prefix='mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', nonlinear=True, window=6, max_workers=5, recalc=True, method='least_squares') # Smaller window and fit from high to low atten
    if nonlinear: detector.nonlinear_fit(nonlinear=True, max_workers=16, save_model_result=True, recalc=True)
    return detector

def center_IQ(detector, data = 'targ', delay_col='cable_delay', recalc=False):
    detector.IQ_unwind(data=data, recalc=recalc, delay_col=delay_col)
    detector.IQ_circle_fit(prefix='unwind_rotate', max_workers=4, recalc=recalc)
    detector.IQ_circle_real(prefix='unwind_rotate', loc='origin', data=data, recalc=recalc)
    detector.IQ_circle_rotate(prefix='origin_shift_origin_rotate_unwind_rotate', data=data, recalc=recalc, rotation='mismatch')
    return detector

def match_detectors(detectors_df, max_frac_f = 1000e-6, Qc_tol = 5000, recalc=False):
    if not isinstance(to_match, Iterable): to_match = [to_match]
    
    data = detectors_df.to_numpy().T
    detectors = list(data[0])

    over_cols = detectors_df.columns.pop('detector')
    over_data = list(data[1:])
    
    bad_detectors = []
    for detector in detectors:
        try:
            fit_detectors(detector, nonlinear=True, phase=True, recalc=recalc)
        except:
            bad_detectors.append(detector)

    for bad_detector in bad_detectors:
        detectors.pop(bad_detector)

    property_df, property_to_det = combine_properties(detectors, ids=[[True] + [False]*(int(len(detectors) - 1))] + over_data, id_names=['Reference'] + over_cols)

    diff_df = property_df.lazy().with_columns([(pl.col('nonlinear_fit_f_0') - f0).alias(f'{det:04d}') for det, f0 in (property_df.filter(pl.col('Reference'))
                                                                                                                                 .select(['det','nonlinear_fit_f_0'])).iter_rows()])
    # For each reference detector, rank each detector by how close it is to the reference detector (grouped by Detector object)
    diff_df_long = (diff_df.with_columns([(pl.when((np.abs(pl.col(shift_from_ref)) > pl.col('nonlinear_fit_f_0')*max_frac_f))
                                             .then(None)
                                             .otherwise(shift_from_ref)
                                             .alias(shift_from_ref)) for shift_from_ref in diff_df.select(pl.col('^0.*$')).columns])

                            .with_columns([(np.abs(pl.col(ref_det)).rank('dense')
                                                                   .over('det_obj')
                                                                   .alias(ref_det)) for ref_det in diff_df.select(pl.col('^0.*$')).columns])
                            .unpivot(index=['det', 'det_obj', 'coldload_temp', 'nonlinear_fit_f_0', 'nonlinear_fit_Q_i', 'nonlinear_fit_Q_c'],
                                     on=diff_df.select(pl.col('^0.*$')).columns,
                                     variable_name='ref_det',
                                     value_name='rank')
                            .with_columns(pl.col('ref_det').cast(int).alias('ref_det'))).collect()
    
    # Apply further filtering using detector Qc
    diff_Qc_df = (diff_df.select('det')
                         .with_columns([pl.lit(Qc).alias(f'Qc_{det:04d}') for det, Qc in property_df.select(pl.col(['det','nonlinear_fit_Q_c'])).iter_rows()])
                         .unpivot(index='det',
                                  on=[f'Qc_{ref_det}' for ref_det in diff_df.select(pl.col('^0.*$')).columns],
                                  variable_name='ref_det',
                                  value_name='ref_Qc')
                         .drop(['det', 'ref_det'])).collect()

    diff_Q_df = (pl.concat([diff_df_long, diff_Qc_df], how='horizontal')
                   .with_columns(pl.when(np.abs(pl.col('nonlinear_fit_Q_c') -  pl.col('ref_Qc')) < Qc_tol)
                                   .then(pl.col('rank'))
                                   .otherwise(None)
                                   .alias('rank')))

    max_rank = 3
    match_df = (diff_Q_df.with_columns(pl.col('rank').min().over('det_obj', 'ref_det').alias('best_rank')) 
                         .filter((pl.col('rank') == pl.col('best_rank')) & (pl.col('rank') <= max_rank))) # Only Keep Matching Detectors
    return match_df, property_df, property_to_det

def combine_properties(detectors, ids = [], id_names = []):
    if isinstance(id_names, str): id_names = [id_names]
    if not isinstance(ids, Iterable) or not len(ids) == len(detectors): ids = [ids]

    
    property_dfs = [None]*len(detectors)
    property_to_det = {}
    for i, (det, id) in enumerate(zip(detectors, np.array(ids).T)):
        key = str(det)

        add_cols = [pl.lit(val[0]).alias(name) for name, val in zip(id_names, id)] + [pl.lit(key).alias('det_obj')]
        det_property_df = det.properties.lazy().with_columns(add_cols).collect()

        property_dfs[i] = det_property_df
        property_to_det[key] = det
    property_df = pl.concat(property_dfs, how='vertical')
    return property_df, property_to_det

def responsivity_fit(temp, frac_f):
    try:
        responsivity, intercept = ccat_fit.linear_fit(np.array(temp), np.array(frac_f))
    except:
        responsivity, intercept = None, None
    return responsivity, intercept

def calculate_responsivity(match_df):
    resps = np.array([[ref_det] + list(responsivity_fit(temp, frac_f0)) for ref_det, temp, frac_f0 in (match_df.group_by('ref_det')
                                                                                                                .agg(pl.col(['coldload_temp', 'frac_f_0']))
                                                                                                                .iter_rows())])
    resp_df = pl.DataFrame({'ref_det': list(resps[:, 0].astype(int)), 'responsivity': list(resps[:, 1]), 'responsivity_intercept': list(resps[:, 2])}).filter(~(pl.col('responsivity') == 0))

    match_df = match_df.join(resp_df, on='ref_det', how='left', coalesce=True)
    return match_df.filter(~pl.col('responsivity').is_null())

#==========#
# Plotting #
#==========#

def plot_property_vs_coldload(atten_df, prop, det, sense, prop_label):
    def _drive_legend(plot, element):
        plot.state.legend[0].title = 'Drive Attenuation [dB]'
        
    df = (atten_df.with_columns(pl.col('coldload_temp').round().alias('Coldload Temperature'))
                     .sort(pl.col(['Coldload Temperature', 'drive', 'sense']))
                     .filter(pl.col('det') == det)
                     .filter(pl.col('sense') == sense)
                     .filter(~(pl.col('Coldload Temperature').round() == 90))
                     .with_columns(pl.col('drive').alias('Drive Attenuation [dB]')))
                               
    line = (df.hvplot.line(x='Coldload Temperature',
                                  y=prop,
                                  by=['Drive Attenuation [dB]'],
                                  color = hv.Cycle('Sunset'),
                                  xlabel = 'Coldload Temperature [K]', 
                                  ylabel = prop_label,
                                  title=f'{det_type} {det_network} Detector {det} with Sense Attenuation {sense} dB'))

    scatter = (df.hvplot.scatter(x='Coldload Temperature',
                                  y=prop,
                                  by=['Drive Attenuation [dB]'],
                                  color = hv.Cycle('Sunset'),
                                  xlabel = 'Coldload Temperature [K]', 
                                  ylabel = prop_label,
                                  title=f'{det_type} {det_network} Detector {det} with Sense Attenuation {sense} dB',
                                  s=10))
    return line*scatter.opts(hooks=[_drive_legend])

def plot_property_vs_drive(atten_df, prop, det, sense, prop_label):
    def _coldload_legend(plot, element):
        plot.state.legend[0].title = 'Coldload Temperature [K]'
    
    df = (atten_df.with_columns(pl.col('coldload_temp').round())
                     .sort(pl.col(['coldload_temp', 'drive', 'sense']))
                     .filter(pl.col('det') == det)
                     .filter(pl.col('sense') == sense)
                     .filter(~(pl.col('coldload_temp').round() == 90))
                     .with_columns(pl.col('coldload_temp').alias('Coldload Temperature [K]')))
    
    line = (df.hvplot.line(x='drive',
                          y=prop,
                          by=['Coldload Temperature [K]'],
                          color = hv.Cycle('Sunset'),
                          xlabel = 'Drive Attenuation [dB]', 
                          ylabel = prop_label,
                          title=f'{det_type} {det_network} Detector {det} with Sense Attenuation {sense} dB'))

    scatter = (df.hvplot.scatter(x='drive',
                              y=prop,
                              by=['Coldload Temperature [K]'],
                              color = hv.Cycle('Sunset'),
                              xlabel = 'Drive Attenuation [dB]', 
                              ylabel = prop_label,
                              title=f'{det_type} {det_network} Detector {det} with Sense Attenuation {sense} ',
                              s=10))
    return line*scatter.opts(hooks=[_coldload_legend])

def plot_property_drive_hist(atten_df, prop, temp, sense, prop_label):
    df = (atten_df.with_columns(pl.col('coldload_temp').round() == temp)
                     .sort(pl.col(['drive', 'sense']))
                     .filter(pl.col('sense') == sense)
                     .with_columns(pl.col('drive').alias('Drive Attenuation [dB]')))
    
    hist = (df.hvplot.hist(prop,
                          by=['Drive Attenuation [dB]'],
                          color = hv.Cycle('Sunset'),
                          xlabel = prop_label, 
                          ylabel = 'Count',
                          bins=20,
                          alpha=0.8,
                          title=f'{det_type} {det_network} with Sense Attenuation {sense} dB at {temp} K').opts(width=800, height=400))
    return hist

def plot_property_temp_hist(atten_df, prop, drive, sense, prop_label):
    df = (atten_df.with_columns(pl.col('coldload_temp').round(), pl.col('drive') == drive)
                     .sort(pl.col(['coldload_temp', 'sense']))
                     .filter(pl.col('sense') == sense)
                     .with_columns(pl.col('coldload_temp').alias('Coldload Temperature [K]')))
    
    hist = (df.hvplot.hist(prop,
                          by=['Coldload Temperature [K]'],
                          color = hv.Cycle('Sunset'),
                          xlabel = prop_label, 
                          ylabel = 'Count',
                          bins=20,
                          alpha=0.8,
                          title=f'{det_type} {det_network} with Drive Attenuation {drive} dB and Sense Attenuation {sense} dB').opts(width=800, height=400))
    return hist

#====================#
# Pickling Functions #
#====================#

def pickle_network(network, path):
    # Convert the Detector.properties DataFrames to dicts since they may have objects
    for detector in network.data.select('detector').iter_rows():
        detector = detector[0]
        prop_dict = detector.properties.to_dicts()
        setattr(detector, '_data_dict', prop_dict)
        detector._properties_df = None
    
    # Convert network.data DataFrame to dicts since they include Detector objects
    setattr(network, '_data_dict', network.data.to_dicts())
    network.data = None

    with open(path, 'wb') as file:
        pickle.dump(network, file)

def load_network_pickle(path):
    with open(path, 'rb') as file:
        network = pickle.load(file)
    network.data = pl.DataFrame(network._data_dict)
    for detector in network.data.select('detector').iter_rows():
        detector = detector[0]
        detector.properties = pl.DataFrame(detector._data_dict)
    return network

if __name__ == '__main__':
    sys.path.append('/home/pcs/fitting')
    import resonator_model_v3

    sess_ids = ['1754105411', '1755031705', '1755481605']
    com_to =  ['1.2', '1.3', '2.1', '2.3', '3.1', '3.2', '3.3', '5.1']

    dates = ['20250802', '20250812', '20250818']
    data_dir = 'coldload'

    pickle_dir = Path('/home/pcs/scratch/Darshan/coldload/fine/')
    pickle_prefix = 'coldload'

    resave = True

    for com in com_to:
        bid, drid = com.split('.')


        pickle_path = pickle_dir / f'{pickle_prefix}_{bid}_{drid}_{dates[-1]}.pkl'

        # Load pickled Network or create one if one does not exist
        if pickle_path.exists():
            network = load_network_pickle(pickle_path)
        else:
            network = Network(com_to = com,
                              sess_ids = sess_ids,
                              date=dates,
                              data_dir = data_dir,
                              analysis_cfg = str(Path(__file__).parent / 'analysis_config.yaml'),
                              include_targs=False)

            data_cols = ['detector_type', 'network', 'coldload_temp', 'drive', 'sense', 'num_tones', 'bath_temp']
            network.add_columns(data_cols = data_cols, max_workers=20)
        
        
            network.data = (network.data.sort('timestamp')
                                        .filter(pl.col('num_tones') < 40)
                                        .with_columns(pl.col('coldload_temp').cast(float).round().alias('coldload_temp'))
                                        .with_columns((pl.col('bath_temp').list.first().cast(float)*1000).round().alias('bath_temp'))
                                        .filter(pl.col('sense') == 5)
                                        .filter(pl.col('drive') >= 10))
            
            num_streams = network.data.select(pl.col('timestamp').len().over('coldload_temp', 'drive', 'sense', 'bath_temp').alias('num_streams'))
            network.data = pl.concat([network.data, num_streams], how='horizontal')

            sing_streams = network.data.filter(pl.col('num_streams') == 1)
            mult_streams = network.data.filter(~(pl.col('num_streams') == 1))


            min_det = mult_streams.select(pl.col('timestamp').get(1).over('coldload_temp', 'drive', 'sense', 'bath_temp').alias('timestamp_min'))
            mult_streams = pl.concat([mult_streams, min_det], how='horizontal').filter(pl.col('timestamp') == pl.col('timestamp_min'))

            network.data = pl.concat([sing_streams, mult_streams], how='diagonal')
            gc.collect()

        '''
        for detector in tqdm(list(network.data.select(pl.col('detector')).iter_rows())):
            try:
                fit_detectors(detector[0])
            except Exception as e:
                print(e)
        
        pickle_network(network, pickle_path)
        '''
        
        network.data = network.data.with_columns(pl.col('coldload_temp').cast(float).alias('coldload_temp')).sort('coldload_temp')
        
        det_type = network.data.select('detector_type').item(0,0)
        det_network = network.data.select('network').item(0,0)
        
        atten_pairs = network.data.select(pl.col(['drive', 'sense'])).unique()

        atten_df = None
        atten_property_to_det = {}
        for drive, sense in tqdm(atten_pairs.iter_rows(), desc = 'Calculating Responsivities...'):
            try:
                coldload_df = network.data.filter((pl.col('drive') == drive) & (pl.col('sense') == sense)).select(pl.col(['detector', 'coldload_temp', 'drive', 'sense', 'bath_temp']))
                over_cols = coldload_df.columns
                over_cols.remove('detector')
                
                data = coldload_df.to_numpy().T
                detectors = list(data[0])
                over_data = list(data[1:])
                property_df, property_to_det = combine_properties(detectors, ids=[[True] + [False]*(int(len(detectors) - 1))] + over_data, id_names=['Reference'] + over_cols)

                ref_f0s = property_df.filter('Reference').select('nonlinear_fit_f_0').to_numpy().T[0]
                ref_f0s = list(ref_f0s)*int(property_df.height/len(ref_f0s))
                property_df = property_df.with_columns(pl.Series(ref_f0s).alias('nonlinear_fit_ref_f_0'))
                diff_df = property_df.with_columns((1e6*(pl.col('nonlinear_fit_f_0') - pl.col('nonlinear_fit_ref_f_0'))/pl.col('nonlinear_fit_f_0')).alias('nonlinear_fit_frac_f'))

                dx_df = (diff_df.filter(pl.col('coldload_temp') == 115)
                                        .sort('det', 'bath_temp')
                                        .select('det', 'bath_temp', 'nonlinear_fit_frac_f')
                                        .filter((pl.col('bath_temp') == 115) | (pl.col('bath_temp') == 150))
                                        .group_by('bath_temp')
                                        .agg('det', 'nonlinear_fit_frac_f')
                                        .sort('bath_temp'))
                dets = dx_df.select(pl.col('det')).item(0,0)
                x_115, x_150 = dx_df.select('nonlinear_fit_frac_f').to_numpy().T[0]
                dxs = x_150 - x_115

                shift_df = pl.DataFrame({'det': dets, 'dx': dxs})
                diff_df = diff_df.join(shift_df, on='det', how='left', coalesce=True)
                diff_df = diff_df.with_columns(pl.when(pl.col('bath_temp') == 150)
                                                 .then(pl.col('nonlinear_fit_frac_f') - pl.col('dx'))
                                                 .otherwise(pl.col('nonlinear_fit_frac_f'))
                                                 .alias('nonlinear_fit_frac_f'))

                resps = np.array([[det] + list(responsivity_fit(temp, frac_f0)) for det, temp, frac_f0 in (diff_df.group_by('det')
                                                                                                                    .agg(pl.col(['coldload_temp', 'nonlinear_fit_frac_f']))
                                                                                                                    .iter_rows())])
                resp_df = pl.DataFrame({'det': list(resps[:, 0].astype(int)), 'nonlinear_fit_responsivity': list(resps[:, 1]), 'nonlinear_fit_responsivity_intercept': list(resps[:, 2])})
                diff_df = diff_df.join(resp_df, on='det', how='left', coalesce=True)

                ref_f0s = property_df.filter('Reference').select('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_0').to_numpy().T[0]
                ref_f0s = list(ref_f0s)*int(property_df.height/len(ref_f0s))
                diff_df = diff_df.with_columns(pl.Series(ref_f0s).alias('phase_fit_ref_f_0'))
                diff_df = diff_df.with_columns((1e6*(pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_0') - pl.col('phase_fit_ref_f_0'))/pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_0')).alias('phase_fit_frac_f'))
                
                dx_df = (diff_df.filter(pl.col('coldload_temp') == 115)
                                .sort('det', 'bath_temp')
                                .select('det', 'bath_temp', 'phase_fit_frac_f')
                                .filter((pl.col('bath_temp') == 115) | (pl.col('bath_temp') == 150))
                                .group_by('bath_temp')
                                .agg('det', 'phase_fit_frac_f')
                                .sort('bath_temp'))

                dets = dx_df.select(pl.col('det')).item(0,0)
                x_115, x_150 = dx_df.select('phase_fit_frac_f').to_numpy().T[0]
                dxs = x_150 - x_115

                shift_df = pl.DataFrame({'det': dets, 'dx': dxs})
                diff_df = diff_df.join(shift_df, on='det', how='left', coalesce=True)
                diff_df = diff_df.with_columns(pl.when(pl.col('bath_temp') == 150)
                                                 .then(pl.col('phase_fit_frac_f') - pl.col('dx'))
                                                 .otherwise(pl.col('phase_fit_frac_f'))
                                                 .alias('phase_fit_frac_f'))
                
                resps = np.array([[det] + list(responsivity_fit(temp, frac_f0)) for det, temp, frac_f0 in (diff_df.group_by('det')
                                                                                                                            .agg(pl.col(['coldload_temp', 'phase_fit_frac_f']))
                                                                                                                            .iter_rows())])
                resp_df = pl.DataFrame({'det': list(resps[:, 0].astype(int)), 'phase_fit_responsivity': list(resps[:, 1]), 'phase_fit_responsivity_intercept': list(resps[:, 2])})
                diff_df = diff_df.join(resp_df, on='det', how='left', coalesce=True)

                atten_df = diff_df if atten_df is None else pl.concat([atten_df, diff_df], how='vertical')
                atten_property_to_det = atten_property_to_det | property_to_det
            except Exception as e: 
                print(e)
        
        int_dets = atten_df.filter(~((pl.col('bath_temp') == 115) | (pl.col('bath_temp') == 150))).select('det_obj').unique().to_numpy().T[0]
        atten_df = atten_df.filter((pl.col('bath_temp') == 115) | (pl.col('bath_temp') == 150))

        for det in int_dets:
            atten_property_to_det.pop(det)
        
        for sense in tqdm(list(atten_df.select(pl.col('sense')).unique().iter_rows()), desc='Plotting Sense'):
            sense = sense[0]
            for det in tqdm(list(atten_df.select(pl.col('det')).unique().iter_rows()), desc='Plotting Detector'):
                try:
                    det = det[0]
                    path = f"{pickle_dir}/fig/det/detector_props_b{bid}_d{drid}_det{det}_sense{int(sense)}.png"
                    if not Path(path).exists() or resave:
                        df = (atten_df.filter((pl.col('drive') >= 2) & (pl.col('nonlinear_fit_Q_i') < 1e5) & (pl.col('nonlinear_fit_frac_f') <=0))
                                    .with_columns(np.abs(pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a')).alias('phase_fit_a')))
                        Q_i = plot_property_vs_drive(df, 'nonlinear_fit_Q_i', det, sense, r'$$Nonlinear\ Fit\ Q_i$$')
                        Q_c = plot_property_vs_drive(df, 'nonlinear_fit_Q_c', det, sense, r'$$Nonlinear\ Fit\ Q_c$$')
                        Q = plot_property_vs_drive(df, 'nonlinear_fit_Q', det, sense, r'$$Nonlinear\ Fit\ Q$$')
                        a = plot_property_vs_drive(df, 'nonlinear_fit_a', det, sense, r'$$Nonlinear\ Fit\ a$$')
                        frac_f = plot_property_vs_drive(df, 'nonlinear_fit_frac_f', det, sense, r'$$Nonlinear\ Fit\ \delta f_0/f_0\ [ppm]$$')
                        nonlinear_drive = pn.Column(Q_i, Q_c, Q, a, frac_f)

                        Q_i = plot_property_vs_coldload(df, 'nonlinear_fit_Q_i', det, sense, r'$$Nonlinear\ Fit\ Q_i$$')
                        Q_c = plot_property_vs_coldload(df, 'nonlinear_fit_Q_c', det, sense, r'$$Nonlinear\ Fit\ Q_c$$')
                        Q = plot_property_vs_coldload(df, 'nonlinear_fit_Q', det, sense, r'$$Nonlinear\ Fit\ Q$$')
                        a = plot_property_vs_coldload(df, 'nonlinear_fit_a', det, sense, r'$$Nonlinear\ Fit\ a$$')
                        frac_f = plot_property_vs_coldload(df, 'nonlinear_fit_frac_f', det, sense, r'$$Nonlinear\ Fit\ \delta f_0/f_0\ [ppm]$$')
                        nonlinear_coldload = pn.Column(Q_i, Q_c, Q, a, frac_f)

                        df = (atten_df.filter((pl.col('drive') >= 2) & (pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i') < 1e5) & (pl.col('phase_fit_frac_f') <=0))
                                    .with_columns(np.abs(pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a')).alias('phase_fit_a')))

                        Q_i = plot_property_vs_drive(df, 'phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i', det, sense, r'$$Phase\ Fit\ Q_i$$')
                        Q_c = plot_property_vs_drive(df, 'phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_c', det, sense, r'$$Phase\ Fit\ Q_c$$')
                        Q = plot_property_vs_drive(df, 'phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Qr', det, sense, r'$$Phase\ Fit\ Q$$')
                        a = plot_property_vs_drive(df, 'phase_fit_a', det, sense, r'$$Phase\ Fit\ a$$')
                        frac_f = plot_property_vs_drive(df, 'phase_fit_frac_f', det, sense, r'$$Phase\ Fit\ \delta f_0/f_0\ [ppm]$$')
                        phase_drive = pn.Column(Q_i, Q_c, Q, a, frac_f)

                        Q_i = plot_property_vs_coldload(df, 'phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i', det, sense, r'$$Phase\ Fit\ Q_i$$')
                        Q_c = plot_property_vs_coldload(df, 'phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_c', det, sense, r'$$Phase\ Fit\ Q_c$$')
                        Q = plot_property_vs_coldload(df, 'phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Qr', det, sense, r'$$Phase\ Fit\ Q$$')
                        a = plot_property_vs_coldload(df, 'phase_fit_a', det, sense, r'$$Phase\ Fit\ a$$')
                        frac_f = plot_property_vs_coldload(df, 'phase_fit_frac_f', det, sense, r'$$Phase\ Fit\ \delta f_0/f_0\ [ppm]$$')
                        phase_coldload = pn.Column(Q_i, Q_c, Q, a, frac_f)

                        prop_plot = pn.Row(nonlinear_drive, nonlinear_coldload, phase_drive, phase_coldload)
                        prop_plot.save(path, embed=False)
                except Exception as e:
                    print(e)

            '''
            for drive in tqdm(list(atten_df.select(pl.col('drive')).unique().iter_rows()), desc='Plotting Drive Hists'):
                try:
                    drive = drive[0]
                    path = f"{pickle_dir}/fig/temp/temp_hists_b{bid}_d{drid}_drive{int(drive)}_sense{int(sense)}.png"
                    if not Path(path).exists():
                        df = (atten_df.filter((pl.col('nonlinear_fit_Q') < 1e5))
                                        .with_columns(np.abs(pl.col('phase_fit_a')).alias('phase_fit_a')))
                        Q_i = plot_property_temp_hist(df, 'nonlinear_fit_Q_i', drive, sense, r'$$Nonlinear\ Fit\ Q_i$$')
                        Q_c = plot_property_temp_hist(df, 'nonlinear_fit_Q_c', drive, sense, r'$$Nonlinear\ Fit\ Q_c$$')
                        Q = plot_property_temp_hist(df, 'nonlinear_fit_Q', drive, sense, r'$$Nonlinear\ Fit\ Q$$')
                        a = plot_property_temp_hist(df, 'nonlinear_fit_a', drive, sense, r'$$Nonlinear\ Fit\ a$$')
                        nonlinear = pn.Column(Q_i, Q_c, Q, a)

                        df = (atten_df.filter((pl.col('phase_fit_Qr') < 1e5))
                                    .with_columns(np.abs(pl.col('phase_fit_a')).alias('phase_fit_a')))
                        Q_i = plot_property_temp_hist(df, 'phase_fit_Q_i', drive, sense, r'$$Phase\ Fit\ Q_i$$')
                        Q_c = plot_property_temp_hist(df, 'phase_fit_Q_c', drive, sense, r'$$Phase\ Fit\ Q_c$$')
                        Q = plot_property_temp_hist(df, 'phase_fit_Qr', drive, sense, r'$$Phase\ Fit\ Q$$')
                        a = plot_property_temp_hist(df, 'phase_fit_a', drive, sense, r'$$Phase\ Fit\ a$$')
                        phase = pn.Column(Q_i, Q_c, Q, a)

                        prop_plot = pn.Row(nonlinear, phase)
                        prop_plot.save(path, embed=False)
                except Exception as e:
                    print(e)
            for temp in tqdm(list(atten_df.select(pl.col('coldload_temp').round()).unique().iter_rows()), desc='Plotting Temp Hists'):
                try:
                    temp = temp[0]
                    path = f"{pickle_dir}/fig/drive/drive_hists_b{bid}_d{drid}_temp{int(temp)}_sense{int(sense)}.png"
                    if not Path(path).exists():
                        df = (atten_df.filter((pl.col('nonlinear_fit_Q_i') < 1e5))
                                        .with_columns(np.abs(pl.col('phase_fit_a')).alias('phase_fit_a')))
                        Q_i = plot_property_drive_hist(df, 'nonlinear_fit_Q_i', temp, sense, r'$$Nonlinear\ Fit\ Q_i$$')
                        Q_c = plot_property_drive_hist(df, 'nonlinear_fit_Q_c', temp, sense, r'$$Nonlinear\ Fit\ Q_c$$')
                        Q = plot_property_drive_hist(df, 'nonlinear_fit_Q', temp, sense, r'$$Nonlinear\ Fit\ Q$$')
                        a = plot_property_drive_hist(df, 'nonlinear_fit_a', temp, sense, r'$$Nonlinear\ Fit\ a$$')
                        nonlinear = pn.Column(Q_i, Q_c, Q, a)

                        df = (atten_df.filter((pl.col('phase_fit_Q_i') < 1e5))
                                    .with_columns(np.abs(pl.col('phase_fit_a')).alias('phase_fit_a')))
                        Q_i = plot_property_drive_hist(df, 'phase_fit_Q_i', temp, sense, r'$$Phase\ Fit\ Q_i$$')
                        Q_c = plot_property_drive_hist(df, 'phase_fit_Q_c', temp, sense, r'$$Phase\ Fit\ Q_c$$')
                        Q = plot_property_drive_hist(df, 'phase_fit_Qr', temp, sense, r'$$Phase\ Fit\ Q$$')
                        a = plot_property_drive_hist(df, 'phase_fit_a', temp, sense, r'$$Phase\ Fit\ a$$')
                        phase = pn.Column(Q_i, Q_c, Q, a)

                        prop_plot = pn.Row(nonlinear, phase)
                        prop_plot.save(path, embed=False)
                except Exception as e:
                    print(e)
            '''

            try:
                path = f"{pickle_dir}/fig/responsivity_b{bid}_d{drid}_sense{int(sense)}.png"
                if not Path(path).exists():
                    df = (atten_df.filter((pl.col('coldload_temp').round() == 66) & (pl.col('sense') == sense))
                                    .sort(pl.col(['drive']))
                                    .with_columns(pl.col('drive').alias('Drive Attenuation [dB]')))

                    scatter = (df.hvplot.scatter(x='det',
                                                y='nonlinear_fit_responsivity',
                                                by='Drive Attenuation [dB]',
                                                color = hv.Cycle('Sunset'),
                                                xlabel='Detector',
                                                ylabel='Nonlinear Fit Responsivity [ppm/K]',
                                                title=f'{det_type} {det_network} with Sense Attenuation {sense} dB'))

                    hist = df.hvplot.hist('nonlinear_fit_responsivity',
                                        by='Drive Attenuation [dB]',
                                        color=hv.Cycle('Sunset'),
                                        legend=False,
                                        alpha = 0.8,
                                        bins=16,
                                        width=200)

                    nonlinear = (scatter << hist)

                    scatter = (df.hvplot.scatter(x='det',
                                                y='phase_fit_responsivity',
                                                by='Drive Attenuation [dB]',
                                                color = hv.Cycle('Sunset'),
                                                xlabel='Detector',
                                                ylabel='Phase Fit Responsivity [ppm/K]',
                                                title=f'{det_type} {det_network} with Sense Attenuation {sense} dB'))

                    hist = df.hvplot.hist('phase_fit_responsivity',
                                        by='Drive Attenuation [dB]',
                                        color=hv.Cycle('Sunset'),
                                        legend=False,
                                        alpha = 0.8,
                                        bins=16,
                                        width=200)

                    phase = (scatter << hist)

                    prop_plot = pn.Column(nonlinear, phase)
                    prop_plot.save(path, embed=False)
            except Exception as e:
                print(e)
            
            try:
                def _drive_legend(plot, element):
                    plot.state.legend[0].title = 'Drive Attenuation [dB]'
                
                path = f"{pickle_dir}/fig/responsivity_compare_b{bid}_d{drid}_sense{int(sense)}.png"
                if not Path(path).exists():
                    df = (atten_df.filter((pl.col('coldload_temp').round() == 66) & (pl.col('sense') == sense))
                                    .sort(pl.col(['drive']))
                                    .with_columns(pl.col('drive').alias('Drive Attenuation [dB]')))

                    scatter = (df.hvplot.scatter(x='nonlinear_fit_responsivity',
                                                y='phase_fit_responsivity',
                                                by='Drive Attenuation [dB]',
                                                color = hv.Cycle('Sunset'),
                                                xlabel='Nonlinear Fit Responsivity [ppm/K]',
                                                ylabel='Phase Fit Responsivity [ppm/K]',
                                                title=f'{det_type} {det_network} with Sense Attenuation {sense} dB',
                                                width=600,
                                                height=500).opts(legend_position='left', hooks=[_drive_legend]))
                    nonlinear_hist = df.hvplot.hist('nonlinear_fit_responsivity',
                                        by='Drive Attenuation [dB]',
                                        color=hv.Cycle('Sunset'),
                                        legend=False,
                                        alpha = 0.8,
                                        bins=16,
                                        height=200,
                                        width=600)
                    
                    phase_hist = df.hvplot.hist('phase_fit_responsivity',
                                        by='Drive Attenuation [dB]',
                                        color=hv.Cycle('Sunset'),
                                        legend=False,
                                        alpha = 0.8,
                                        bins=16,
                                        width=200,
                                        height=500)
                    with_hists = (scatter << phase_hist << nonlinear_hist)
                    with_hists = pn.Row(hv.Slope(1, 0).opts(color='k')*with_hists)
                    with_hists.save(path, embed=False)
            except Exception as e:
                print(e)

        for det_obj, detector in tqdm(atten_property_to_det.items()):
            try:
                center_IQ(detector, data='both', delay_col = 'nonlinear_fit_delay_ns', recalc=True)
                detector.IQ_norm(prefix='unwind_rotate', data='both')

                detector.targ.phase(prefix='mismatch_rotate_origin_shift_origin_rotate_unwind_rotate')
                detector.stream.phase(prefix='mismatch_rotate_origin_shift_origin_rotate_unwind_rotate')

                detector.phase_to_f(prefix='mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', phase_bounds=0.3, k=3, max_workers=4)

                try:
                    f0s = np.full(detector.stream.comb.height, np.nan)
                    resps = np.ones(detector.stream.comb.height)
                    resp_ints = np.zeros(detector.stream.comb.height)
                    det, ref_f0, resp, resp_int = (atten_df.filter(pl.col('det_obj') == det_obj)
                                                        .select(pl.col(['det', 'nonlinear_fit_ref_f_0', 'nonlinear_fit_responsivity', 'nonlinear_fit_responsivity_intercept']))
                                                        .sort('det')).to_numpy().T
                    det = det.astype(int)
                    f0s[det.astype(int)] = ref_f0
                    resps[det.astype(int)] = resp
                    resp_ints[det.astype(int)] = resp_int
                    
                    detector.frac_f(prefix='mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', f_0=f0s, include=det, recalc=True) # Calculate timestream fractional frequency shifts using same f0 used to calculate target sweep fractional frequency shifts
                    detector.stream.data = detector.stream.data.with_columns([((1e6*pl.col(stream) - resp_ints[i])/resps[i]).alias(f"nonlinear_fit_Temperature_{stream.split('_')[-1]}") for i, stream in enumerate(detector.stream.data.select(pl.col('^frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_.*$')).columns)])
                    detector.stream.psd(prefix='', col_name='nonlinear_fit_Temperature', nperseg=1024, recalc=True)
                except Exception as e:
                    print(1)
                    print(e)

                try:
                    f0s = np.full(detector.stream.comb.height, np.nan)
                    resps = np.ones(detector.stream.comb.height)
                    resp_ints = np.zeros(detector.stream.comb.height)
                    det, ref_f0, resp, resp_int = (atten_df.filter(pl.col('det_obj') == det_obj)
                                                        .select(pl.col(['det', 'phase_fit_ref_f_0', 'phase_fit_responsivity', 'phase_fit_responsivity_intercept']))
                                                        .sort('det')).to_numpy().T
                    det = det.astype(int)
                    f0s[det.astype(int)] = ref_f0
                    resps[det.astype(int)] = resp
                    resp_ints[det.astype(int)] = resp_int
                
                    detector.frac_f(prefix='mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', f_0=f0s, include=det, recalc=True) # Calculate timestream fractional frequency shifts using same f0 used to calculate target sweep fractional frequency shifts
                    detector.stream.data = detector.stream.data.with_columns([((1e6*pl.col(stream) - resp_ints[i])/resps[i]).alias(f"phase_fit_Temperature_{stream.split('_')[-1]}") for i, stream in enumerate(detector.stream.data.select(pl.col('^frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_.*$')).columns)])
                    detector.stream.psd(prefix='', col_name='phase_fit_Temperature', nperseg=1024, recalc=True)
                except Exception as e:
                    print(2)
                    print(e)
            except Exception as e:
                print(3)
                print(e)
        
        #det_df = None
        #for det, temp, drive, sense in network.data.select(pl.col(['detector', 'coldload_temp', 'drive', 'sense'])).iter_rows():
        #    df = det.stream.get_data(['psd']).with_columns(pl.lit(temp).alias('coldload_temp'),
        #                                                   pl.lit(drive).alias('drive'),
        #                                                   pl.lit(sense).alias('sense'))
        #    try:
        #        det_df = df if det_df is None else pl.concat([det_df, df], how='vertical')
        #    except:
        #        pass



        freq_threshold = 100
        med_psd_df_nonlin = None
        for det, temp, drive, sense in network.data.select(pl.col(['detector', 'coldload_temp', 'drive', 'sense'])).iter_rows():
            try:
                df = det.stream.get_data(['psd_nonlinear_fit_Temperature'])
                unpivot_cols = df.select(pl.all().exclude('psd_nonlinear_fit_Temperature_f')).columns
                psd_meds = ((df.unpivot(index='psd_nonlinear_fit_Temperature_f',
                                            on=unpivot_cols,
                                            variable_name='det',
                                            value_name='psd')
                                                    .filter((pl.col('psd_nonlinear_fit_Temperature_f') > freq_threshold) & ~(pl.col('psd').is_nan()))
                                    .select(pl.col('det'), np.sqrt(pl.col('psd').median().over('det'))))
                                    .unique()
                                    .filter(pl.col('psd') < 0.02)
                                    .with_columns(pl.lit(temp).alias('coldload_temp'),
                                                  pl.lit(drive).alias('drive'),
                                                  pl.lit(sense).alias('sense'),
                                                  pl.lit('nonlinear').alias('fit')))
                med_psd_df_nonlin = psd_meds if med_psd_df_nonlin is None else pl.concat([med_psd_df_nonlin, psd_meds], how='vertical')
            except:
                pass
        
        med_psd_df_phase = None
        for det, temp, drive, sense in network.data.select(pl.col(['detector', 'coldload_temp', 'drive', 'sense'])).iter_rows():
            try:
                df = det.stream.get_data(['psd_phase_fit_Temperature'])
                unpivot_cols = df.select(pl.all().exclude('psd_phase_fit_Temperature_f')).columns
                psd_meds = ((df.unpivot(index='psd_phase_fit_Temperature_f',
                                            on=unpivot_cols,
                                            variable_name='det',
                                            value_name='psd')
                                                    .filter((pl.col('psd_phase_fit_Temperature_f') > freq_threshold) & ~(pl.col('psd').is_nan()))
                                    .select(pl.col('det'), np.sqrt(pl.col('psd').median().over('det'))))
                                    .unique()
                                    .filter(pl.col('psd') < 0.02)
                                    .with_columns(pl.lit(temp).alias('coldload_temp'),
                                                  pl.lit(drive).alias('drive'),
                                                  pl.lit(sense).alias('sense'),
                                                  pl.lit('phase').alias('fit')))
                med_psd_df_phase = psd_meds if med_psd_df_phase is None else pl.concat([med_psd_df_phase, psd_meds], how='vertical')
            except:
                pass
            
        med_psd_df = pl.concat([med_psd_df_nonlin, med_psd_df_phase], how = 'vertical')

        for sense in tqdm(list(med_psd_df.select(pl.col('sense')).unique().iter_rows()), desc='Plotting Sense'):
            sense = sense[0]
            for det in tqdm(list(med_psd_df.select(pl.col('det')).unique().iter_rows()), desc='Plotting Detector'):
                try:
                    det = int(det[0].split('_')[-1])
                    path = f"{pickle_dir}/fig/det/med_white_noise_b{bid}_d{drid}_det{det}_sense{int(sense)}.png"
                    if not Path(path).exists() or resave:
                        def _drive_legend(plot, element):
                            plot.state.legend[0].title = 'Drive Attenuation [dB]'
                        line = (med_psd_df.filter(pl.col('det') == f'psd_nonlinear_fit_Temperature_{det:04d}')
                                   .sort(pl.col(['drive', 'coldload_temp']))
                                   .filter((pl.col('sense') == sense) & (pl.col('drive') >= 2))
                                   .with_columns(pl.col('drive').alias('Drive Attenuation [dB]'))
                                   .hvplot.line(x='coldload_temp',
                                                y='psd',
                                                by='Drive Attenuation [dB]',
                                                color=hv.Cycle('Sunset'),
                                                loglog=False,
                                                xlabel='Coldload Temperature [K]',
                                                ylabel=r'$$Nonlinear\ Fit\ \sqrt{S_{TT}}\ White\ Noise\ [K/\sqrt{Hz}]$$',
                                                title=f'{det_type} {det_network} Detector {det} with Sense Attenuation {sense} dB',
                                                width=1000,
                                                height=650))
                        scatter = (med_psd_df.filter(pl.col('det') == f'psd_nonlinear_fit_Temperature_{det:04d}')
                                   .sort(pl.col(['drive', 'coldload_temp']))
                                   .filter((pl.col('sense') == sense) & (pl.col('drive') >= 2))
                                   .with_columns(pl.col('drive').alias('Drive Attenuation [dB]'))
                                   .hvplot.scatter(x='coldload_temp',
                                                y='psd',
                                                by='Drive Attenuation [dB]',
                                                color=hv.Cycle('Sunset'),
                                                loglog=False,
                                                xlabel='Coldload Temperature [K]',
                                                ylabel=r'$$Nonlinear\ Fit\ \sqrt{S_{TT}}\ White\ Noise\ [K/\sqrt{Hz}]$$',
                                                title=f'{det_type} {det_network} Detector {det} with Sense Attenuation {sense} dB',
                                                s=10,
                                                width=1000,
                                                height=650))
                        nonlinear = line*scatter.opts(hooks=[_drive_legend])
                        line = (med_psd_df.filter(pl.col('det') == f'psd_phase_fit_Temperature_{det:04d}')
                                            .sort(pl.col(['drive', 'coldload_temp']))
                                            .filter((pl.col('sense') == sense) & (pl.col('drive') >= 2))
                                            .with_columns(pl.col('drive').alias('Drive Attenuation [dB]'))
                                            .hvplot.line(x='coldload_temp',
                                                            y='psd',
                                                            by='Drive Attenuation [dB]',
                                                            color=hv.Cycle('Sunset'),
                                                            loglog=False,
                                                            xlabel='Coldload Temperature [K]',
                                                            ylabel=r'$$Phase\ Fit\ \sqrt{S_{TT}}\ White\ Noise\ [K/\sqrt{Hz}]$$',
                                                            title=f'{det_type} {det_network} Detector {det} with Sense Attenuation {sense} dB',
                                                            width=1000,
                                                            height=650))
                        scatter = (med_psd_df.filter(pl.col('det') == f'psd_phase_fit_Temperature_{det:04d}')
                                            .sort(pl.col(['drive', 'coldload_temp']))
                                            .filter((pl.col('sense') == sense) & (pl.col('drive') >= 2))
                                            .with_columns(pl.col('drive').alias('Drive Attenuation [dB]'))
                                            .hvplot.scatter(x='coldload_temp',
                                            y='psd',
                                            by='Drive Attenuation [dB]',
                                            color=hv.Cycle('Sunset'),
                                            loglog=False,
                                            xlabel='Coldload Temperature [K]',
                                            ylabel=r'$$Phase\ Fit\ \sqrt{S_{TT}}\ White\ Noise\ [K/\sqrt{Hz}]$$',
                                            title=f'{det_type} {det_network} Detector {det} with Sense Attenuation {sense} dB',
                                            s=10,
                                            width=1000,
                                            height=650))
                        phase = line*scatter.opts(hooks=[_drive_legend])

                        prop_plot = pn.Row(nonlinear, phase)
                        prop_plot.save(path, embed=False)
                except Exception as e:
                    print(e)

            try:
                path = f"{pickle_dir}/fig/med_white_noise_b{bid}_d{drid}_sense{int(sense)}.png"
                if not Path(path).exists() or resave:
                    def _drive_legend(plot, element):
                            plot.state.legend[0].title = 'Drive Attenuation [dB]'
                    line = (med_psd_df.sort(pl.col(['drive', 'coldload_temp']))
                                            .filter((pl.col('sense') == sense) & (pl.col('fit') == 'nonlinear') & (pl.col('drive') >= 2))
                                            .with_columns(pl.col('psd').median().over(['coldload_temp', 'drive']).alias('med_psd'))
                                            .with_columns(pl.col('drive').alias('Drive Attenuation [dB]'))
                                            .hvplot.line(x='coldload_temp',
                                                            y='med_psd',
                                                            by='Drive Attenuation [dB]',
                                                            loglog=False,
                                                            color=hv.Cycle('Sunset'),
                                                            xlabel='Coldload Temperature [K]',
                                                            ylabel=r'$$Nonlinear\ Fit\ \sqrt{S_{TT}}\ Median\ White\ Noise\ [K/\sqrt{Hz}]$$',
                                                            title=f'{det_type} {det_network} with Sense Attenuation {sense} dB',
                                                            width=1000,
                                                            height=650))
                    scatter = (med_psd_df.sort(pl.col(['drive', 'coldload_temp']))
                                            .filter((pl.col('sense') == sense) & (pl.col('fit') == 'nonlinear') & (pl.col('drive') >= 2))
                                            .with_columns(pl.col('psd').median().over(['coldload_temp', 'drive']).alias('med_psd'))
                                            .with_columns(pl.col('drive').alias('Drive Attenuation [dB]'))
                                            .hvplot.scatter(x='coldload_temp',
                                                            y='med_psd',
                                                            by='Drive Attenuation [dB]',
                                                            loglog=False,
                                                            color=hv.Cycle('Sunset'),
                                                            xlabel='Coldload Temperature [K]',
                                                            ylabel=r'$$Nonlinear\ Fit\ \sqrt{S_{TT}}\ Median\ White\ Noise\ [K/\sqrt{Hz}]$$',
                                                            title=f'{det_type} {det_network} with Sense Attenuation {sense} dB',
                                                            width=1000,
                                                            height=650))
                    nonlinear = line*scatter.opts(hooks=[_drive_legend])
                    line = (med_psd_df.sort(pl.col(['drive', 'coldload_temp']))
                                            .filter((pl.col('sense') == sense) & (pl.col('fit') == 'phase') & (pl.col('drive') >= 2))
                                            .with_columns(pl.col('psd').median().over(['coldload_temp', 'drive']).alias('med_psd'))
                                            .with_columns(pl.col('drive').alias('Drive Attenuation [dB]'))
                                            .hvplot.line(x='coldload_temp',
                                                            y='med_psd',
                                                            by='Drive Attenuation [dB]',
                                                            loglog=False,
                                                            color=hv.Cycle('Sunset'),
                                                            xlabel='Coldload Temperature [K]',
                                                            ylabel=r'$$Phase\ Fit\ \sqrt{S_{TT}}\ Median\ White\ Noise\ [K/\sqrt{Hz}]$$',
                                                            title=f'{det_type} {det_network} with Sense Attenuation {sense} dB',
                                                            width=1000,
                                                            height=650))
                    scatter = (med_psd_df.sort(pl.col(['drive', 'coldload_temp']))
                                            .filter((pl.col('sense') == sense) & (pl.col('fit') == 'phase') & (pl.col('drive') >= 2))
                                            .with_columns(pl.col('psd').median().over(['coldload_temp', 'drive']).alias('med_psd'))
                                            .with_columns(pl.col('drive').alias('Drive Attenuation [dB]'))
                                            .hvplot.scatter(x='coldload_temp',
                                                            y='med_psd',
                                                            by='Drive Attenuation [dB]',
                                                            loglog=False,
                                                            color=hv.Cycle('Sunset'),
                                                            xlabel='Coldload Temperature [K]',
                                                            ylabel=r'$$Phase\ Fit\ \sqrt{S_{TT}}\ Median\ White\ Noise\ [K/\sqrt{Hz}]$$',
                                                            title=f'{det_type} {det_network} with Sense Attenuation {sense} dB',
                                                            width=1000,
                                                            height=650))
                    phase = line*scatter.opts(hooks=[_drive_legend])
                    prop_plot=pn.Row(nonlinear, phase)
                    prop_plot.save(path, embed=False)
            except Exception as e:
                print(e)

        #pickle_network(network, pickle_path)
        #match_df, property_df, property_to_det = match_detectors(coldload_df)

            #match_df = (match_df.sort('coldload_temp') # Sort by temperature so that fractional frequency is calculated relative to lowest temperature
            #                  .with_columns((1e6*(pl.col('nonlinear_fit_f_0') - pl.col('nonlinear_fit_f_0').first())/pl.col('nonlinear_fit_f_0').first()).over('ref_det').alias('frac_f_0'))) # Calculate Fractional Frequency Shift

            #match_df = calculate_responsivity(match_df)

            #atten_df = match_df if atten_df is None else pl.concat([atten_df, match_df], how='vertical')        
            #atten_property_to_det = atten_property_to_det | property_to_det    

        # Pickle network for future use to circumvent expensive load times and analysis operations
        #pickle_network(network, pickle_path)

        #ref_f0_df = (atten_df.group_by('det', 'drive', 'sense')
        #                     .agg(pl.col('f0').first().alias('ref_f0')))
        #atten_df = atten_df.join(ref_f0_df, on=['ref_det', 'drive', 'sense'], how='left', coalesce=True)
        
        # Calculate timestream fractional frequency shifts
        
