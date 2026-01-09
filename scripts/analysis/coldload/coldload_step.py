# General imports
import pickle
import argparse
import polars as pl
import numpy as np
import time
import sys
import os
import concurrent.futures
import pymupdf
import gc
import multiprocessing as mp


from tqdm import tqdm
from pathlib import Path

# ccatkidlib imports
from ccatkidlib.analysis.core.network import Network
from ccatkidlib.analysis.core.detector import Detector

import ccatkidlib.analysis.pair as pair
import ccatkidlib.analysis.fit.fit as ccat_fit
import ccatkidlib.rfsoc_io as rfsoc_io
import ccatkidlib.utils as ccat_utils

# Plotting
import panel as pn
import datashader as ds
import holoviews as hv
import hvplot.polars
import matplotlib.pyplot as plt
import matplotlib as mpl

from holoviews import opts
from collections.abc import Iterable
from holoviews.operation.datashader import datashade, spread
from matplotlib.backends.backend_pdf import PdfPages
from bokeh.palettes import Sunset, Plasma256, Inferno256, Cividis256, Magma256, Viridis256

pn.extension('mathjax')
hv.extension('matplotlib', enable_mathjax = True, webgl=False)

import warnings
warnings.filterwarnings('ignore')

def main():
    args = eval_args() # Parse command line arguments
    
    # Load configs
    # ============
    analysis_cfg, viz_cfg = rfsoc_io.load_config(args.config)
    # Add directory with fitting code to system path
    # ==============================================
    fit_dir = analysis_cfg['file_paths']['fit_dir']
    if not fit_dir in sys.path: sys.path.append(fit_dir)
    import resonator_model_v3
    globals()['resonator_model_v3'] = resonator_model_v3 # Make resonator_model_v3 globally accessible (for loading pickled Networks)
    
    # Create Network objects
    # ======================
    sess_dir = get_sess_dir(args, analysis_cfg)
    sess_dir = sess_dir.replace('md0', 'ext')
    pickle_files = get_pickle_files(args, sess_dir)
    networks, repickle = init_networks(args, pickle_files) # Initialize ccatkidlib network objects
   
    # Fit Detectors
    # =============
    data_cols = ['com_to', 'detector_type', 'network', 'drive', 'sense', 'num_tones', 'tone_placement_method']
    network_iterator = tqdm(zip(networks, pickle_files), desc = 'Fitting Networks', total=len(networks), colour='blue')
    for i, (network, pickle_file) in enumerate(network_iterator):
        network.add_columns(data_cols = data_cols, max_workers=20)
        filter_network(network)
 
        det_type, network_num = network.data.select('detector_type', 'network').to_numpy()[0]
        network_iterator.set_postfix_str(f'Network: {det_type} {network_num}')
        for det, *_ in tqdm(network.data.select('detector').iter_rows(), desc = 'Fitting Detectors', total=network.data.height, colour='green'):
            try:
                fit_detectors(det, nonlinear=True, phase=True, recalc=False)
                phase_to_f(det, recalc=False)
                normalize_detectors(det, dB = True, recalc=False)
                center_timestream(det, recalc=False)
            except Exception as e:
                tqdm.write(f'Caught Exception {e}')

        if args.pickle and repickle[i]: 
            pickle_network(network, pickle_file)
            network = load_network_pickle(pickle_file)  
        networks[i] = network

    # Normalize Detectors & Convert Timestream from phase to frequency
    # ================================================================
    network_iterator = tqdm(networks, desc = 'Normalizing Networks', colour='red')   
    for network in network_iterator:     
        det_type, network_num = network.data.select('detector_type', 'network').to_numpy()[0]
        network_iterator.set_postfix_str(f'Network: {det_type} {network_num}')
        for det, *_ in tqdm(network.data.select(['detector']).iter_rows(), desc = 'Normalizing Detectors', total=network.data.height, colour='green'):
            pass

    # Plot Detector Dashboards
    # ========================
    dashboard_data = ['detector_type', 'network', 'drive', 'sense']
    network_iterator = tqdm(networks, desc = 'Plotting Networks', colour='red')   
    for network in network_iterator:     
        det_type, network_num = network.data.select('detector_type', 'network').to_numpy()[0]
        network_iterator.set_postfix_str(f'Network: {det_type} {network_num}')

        #plot_heatmap('drive', 'sense', 'frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_white_noise', network, sess_dir, logz=True, include=args.tones, xlabel='Drive [dB]', ylabel='Sense [dB]', clabel=r'$\sqrt{S_{xx}}\ \left[Hz^{-1/2}\right]$', over=['coldload_temp', 'bath_temp'])#, over=['LNA_bias'])
        #plot_heatmap('drive', 'LNA_bias', 'frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_white_noise', network, sess_dir, logz=True, include=args.tones, over=['sense'], xlabel='Drive [dB]', ylabel='LNA Bias Current [mA]', clabel=r'$\sqrt{S_{xx}}\ \left[Hz^{-1/2}\right]$')
        #plot_heatmap('LNA_bias', 'sense', 'frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_white_noise', network, sess_dir, logz=True, include=args.tones, over=['drive'], xlabel='LNA Bias Current [mA]', ylabel='Sense [dB]', clabel=r'$\sqrt{S_{xx}}\ \left[Hz^{-1/2}\right]$')
        #plot_heatmap('drive', 'sense', 'freq_diss_ratio', network, sess_dir, include=args.tones, xlabel='Drive [dB]', ylabel='Sense [dB]', clabel='Frequency/Dissipation Ratio', over=['coldload_temp', 'bath_temp'])#, over=['LNA_bias'])
        #plot_heatmap('drive', 'LNA_bias', 'freq_diss_ratio', network, sess_dir, include=args.tones, over=['sense'], xlabel='Drive [dB]', ylabel='LNA Bias Current [mA]', clabel='Frequency/Dissipation Ratio')
        #plot_heatmap('LNA_bias', 'sense', 'freq_diss_ratio', network, sess_dir, include=args.tones, over=['drive'], xlabel='LNA Bias Current [mA]', ylabel='Sense [dB]', clabel='Frequency/Dissipation Ratio')
        
        for det, *data in tqdm(network.data.select(['detector'] + dashboard_data).iter_rows(), desc = 'Plotting Detectors', total=network.data.height, colour='orange'):
            det.stream.mag()
            det.stream.phase()
            plot_det_dashboard(det, data, sess_dir, viz_cfg, include=args.tones)     
    #plot_step(network)
                        
def init_networks(args, pickle_files):
    ''' Initialize ccatkidlib Network objects
    '''    
    com_to = args.com_to
    sess_ids = args.sess_ids
    dates = args.dates
    data_dir = args.data_dir
    
    networks = [None]*len(com_to)
    repickle = [True]*len(com_to)
    if args.pickle:
        for i, pickle_file in enumerate(pickle_files):
            if pickle_file.exists(): 
                networks[i] = load_network_pickle(pickle_file)
                repickle[i] = False
                                                        
    for i, com in enumerate(com_to):
        if networks[i] is None: networks[i] = Network(com_to = com, sess_ids = sess_ids, date = dates, data_dir = data_dir, analysis_cfg = args.config)
    return networks, repickle

def filter_network(network):
    network.data = network.data.drop(pl.col(pl.Null))
    network.data = network.data.filter(pl.col('num_tones') > 400)
    for det, *_ in network.data.select(pl.col('detector')).iter_rows():
        det.stream.data = det.stream.data.with_columns((pl.col('t') - pl.col('t').first()).alias('time')).filter((pl.col('time') > 175) & (pl.col('time') < 180)) 
    #network.data = (network.data.sort('timestamp')
    #                            .with_columns(pl.col('coldload_temp').cast(float).round().alias('coldload_temp'))
    #                            .with_columns((pl.col('bath_temp').list.first().cast(float)*1000).round().alias('bath_temp')))

    #network.data = network.data.filter(pl.col('coldload_temp') == pl.col('coldload_temp').min())

    #bias_data = []
    #for det, *_ in network.data.select('detector').iter_rows():
    #    cfg = det.stream.drone_cfg
    #    cfg_data = ccat_utils.dict_get(cfg, 'LNA_bias')
    #    if isinstance(cfg_data, str): cfg_data = float(cfg_data[:-1])
    #    if cfg_data is None: cfg_data = 7.51
    #    bias_data.append(cfg_data)
    #network.data = network.data.with_columns(pl.Series('LNA_bias', bias_data))
    #network.data = network.data.filter(pl.col('LNA_bias') == 7.5)
    return network

#====================#
# Plotting Functions #
#====================#

def plot_step(network):
    network.data = network.data.filter(pl.col('drive') == 18)
    return

def plot_heatmap(x_dim, y_dim, z_dim, network, sess_dir, logz=False, include=None, over=[], xlabel='', ylabel='', clabel=''):
    def _plot_heatmap(tone):
        try:
            over_str = ' '.join([f'{name} {val}' for val, name in zip(pair, over)])
            heatmap = (df.filter(pl.col('det') == tone)
                                    .hvplot.heatmap(x=x_dim,
                                                    y=y_dim,
                                                    C=z_dim,
                                                    title=f'{det_type} {network_num} Tone {tone}: {over_str}',
                                                    logz=logz,
                                                    cmap='plasma',
                                                    xlabel=xlabel,
                                                    ylabel=ylabel,
                                                    clabel=clabel))
            
            mpl_fig = hv.render(heatmap.opts(show_values=False, colorbar_opts={"label": clabel}), backend='matplotlib')
            save_file = f'{fig_file}_{tone:04d}.pdf'
            mpl_fig.savefig(save_file, dpi=100, bbox_inches='tight')
            plt.close(mpl_fig)
            return heatmap, save_file
        except:
            return None, ''

    bid, drid = network.data.select('com_to').item(0,0).split('.')
    if not xlabel: xlabel = x_dim
    if not ylabel: ylabel = y_dim
    if not clabel: clabel = z_dim

    data_cols = ['detector_type', 'network']
    for dim in [x_dim, y_dim, z_dim] + over: 
        if dim in network.data.columns: data_cols.append(dim)
    properties_df = network.combine_properties(data_cols=data_cols)
    det_type, network_num = network.data.select('detector_type', 'network').to_numpy()[0]
    properties_df = properties_df.select(['det', x_dim, y_dim, z_dim] + over)
    tones = include if include is not None else properties_df.select('det').unique().to_numpy().T.flatten()

    fig_dir = Path(sess_dir) / 'fig' / f'B{bid}D{drid}' / 'heatmap' / f'{x_dim}_v_{y_dim}'
    if not fig_dir.exists(): rfsoc_io.create_dir(fig_dir)

    over_pairs = properties_df.select(over).unique().to_numpy()
    iterator = tqdm(over_pairs)
    for pair in iterator:
        over_str = '_'.join([f'{name}_{val}' for name, val in zip(over, pair)])
        fig_file = fig_dir / f'{x_dim}_v_{y_dim}_{z_dim}_heatmap_{over_str}'

        filt = pl.all_horizontal([pl.col(name) == val for name, val in zip(over, pair)])
        df = properties_df.filter(filt)
        files = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            for result in tqdm(executor.map(_plot_heatmap, tones), total = len(tones), desc='Plotting Heatmaps'):
                _, file = result
                files.append(file)

        merged_pdf = pymupdf.open()
        for file in tqdm(sorted(files), desc='Merging PDFs'):
            try:
                with pymupdf.open(file) as pdf:
                    merged_pdf.insert_pdf(pdf)
                os.remove(file)
            except:
                continue
        try:
            merged_pdf.save(f'{fig_file}.pdf')
        except:
            pass

def plot_det_dashboard(det, data_cols, sess_dir, viz_cfg, include=None, cmap='tab10', f_threshold=150):
    '''
    '''
    bid, drid = det.stream.bid, det.stream.drid
    det_type, network, drive, sense = data_cols
    tones = include if include is not None else det.stream.tones
    if isinstance(tones, int): tones = [tones]
    fig_dir = Path(sess_dir) / 'fig' / f'B{bid}D{drid}' / 'dashboard'
    if not fig_dir.exists(): rfsoc_io.create_dir(fig_dir)
    fig_file = fig_dir / f'det_dashboard_{det.stream.timestamp}'
    
    spawn_context = mp.get_context('spawn')
    det._properties_df = det._properties_df.drop(pl.col(pl.Object))
    det.targ._properties_df = det.targ._properties_df.drop(pl.col(pl.Object))
    det.stream._properties_df = det.stream._properties_df.drop(pl.col(pl.Object))

    with concurrent.futures.ProcessPoolExecutor(max_workers=min(48, int(len(tones)/2)), mp_context=spawn_context) as executor:
        futures = [executor.submit(_plot_det_dashboard, tone, det, cmap, viz_cfg, fig_file, det_type, network, drive, sense) for tone in tones[::5]]#, coldload_temp, bath_temp) for tone in tones]
        results = [future.result() for future in tqdm(futures, total = len(futures), desc='Plotting Dashboards')]
    files=[]
    for result in results:
        fig, file = result
        files.append(file)

    hists = plot_property_hists(det.properties, viz_cfg, data_cols = ['nonlinear_fit_Q_i',
                                                                      'nonlinear_fit_Q_c',
                                                                      'nonlinear_fit_Q',
                                                                      'frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_white_noise',
                                                                      'phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i',
                                                                      'phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_c',
                                                                      'phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Qr',
                                                                      'freq_diss_ratio'], 
                                                        labels=[r'$Complex\ Fit\ Q_i$',
                                                                r'$Complex\ Fit\ Q_c$',
                                                                r'$Complex\ Fit\ Q$',
                                                                r'$\sqrt{S_{xx}}\ \left[Hz^{-1/2}\right]$',
                                                                r'$Phase\ Fit\ Q_i$',
                                                                r'$Phase\ Fit\ Q_c$',
                                                                r'$Phase\ Fit\ Q$',
                                                                'Frequency/Dissipation Ratio'],
                                                        bins=50)
    
    mpl_hist_fig = hv.render(hists, backend='matplotlib')
    mpl_hist_fig.suptitle(f'Histograms {det_type} {network}: Drive {drive} dB, Sense {sense} dB')
    hist_file = f'{fig_file}_hists.pdf'
    mpl_hist_fig.savefig(hist_file, dpi=100, bbox_inches="tight", pad_inches=0.1)
    plt.close(mpl_hist_fig)

    files = [hist_file] + sorted(files)

    merged_pdf = pymupdf.open()
    for file in tqdm(files, desc='Merging PDFs'):
        try:
            with pymupdf.open(file) as pdf:
                merged_pdf.insert_pdf(pdf)
            os.remove(file)
        except Exception as e:
            print(e)
            continue
    try:
        merged_pdf.save(f'{fig_file}.pdf')
    except:
        pass

def plot_property_hists(property_df, viz_cfg, data_cols, labels, bins = 100):
    hists = [None]*len(data_cols)
    for i, (data_col, label) in enumerate(zip(data_cols, labels)):
        try:
            df = (property_df.select(data_col)
                             .filter(~pl.col(data_col).is_nan())
                             .with_columns(pl.col(data_col).median().alias('median'))
                             .with_columns(np.abs(pl.col(data_col) - pl.col('median')).median().alias('MAD')))

            data_med, data_mad = df.select(pl.col('median'), pl.col('MAD'))[0].to_numpy().T.flatten()            
            tqdm.write(f'{data_col}: {data_med:0.2e} +- {data_mad:0.2e}')

            
            df = df.filter((pl.col(data_col) > (pl.col('median') - 10*pl.col('MAD'))) & (pl.col(data_col) < (pl.col('median') + 10*pl.col('MAD'))))
            hist = df.hvplot.hist(data_col, bins = bins, width=viz_cfg['plot']['width'], height=viz_cfg['plot']['height'])

            hist.opts(xlabel=label, ylabel='Count') 
            hists[i] = hist
        except Exception as e:
            pass
        if hists[i] is None: hists[i] = hv.Scatter([])
    return hv.Layout(hists).opts(shared_axes=False, sublabel_format="").cols(4)

def _plot_det_dashboard(tone, det, cmap, viz_cfg, fig_file, det_type, network, drive, sense):
    try:
        # Plot |S21|
        # ==========
        # Plot |S21| of target sweep
        targ_mag, targ_mag_df = det.targ.mag_plot(prefix='norm_scale_dB', include=tone, return_df = True)
        targ_mag_fit, targ_mag_fit_df = det.targ.mag_plot(prefix='norm_scale_nonlinear_fit_dB', include=tone, return_df = True)
        targ_mag_fit_fig = targ_mag_fit.NdOverlay.Curve[tone].opts(opts.Curve(color='k', linewidth=2)).relabel(label='Fit', group='Sweep')
        targ_mag_fig = targ_mag.NdOverlay.Scatter[tone].opts(opts.Scatter(color=mpl.colormaps[cmap](1))).relabel(label='Sweep Data', group='Sweep')
        # Plot |S21| timestream
        stream_mag, stream_mag_df = det.stream.mag_plot(x_prefix='mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', y_prefix='norm_scale_dB', include=tone, return_df = True, rasterize=False)
        stream_mag_fig = spread(datashade(stream_mag, dynamic=False, cnorm='eq_hist', cmap='blues', width=viz_cfg['plot']['width'], height=viz_cfg['plot']['height']), px=5, shape='circle').relabel(label='Data', group='Timestream')
        stream_mag_legend = stream_mag_df[0].hvplot.scatter('f', 'mag', label='Stream Data', size=300)
        mag_fig = targ_mag_fit_fig*stream_mag_legend*targ_mag_fig*stream_mag_fig
        mag_fig.opts(ylabel=r'$|S_{21}|\ [dB]$', title='Target Sweep Magnitude')

        # Plot phase
        # ==========
        # Plot phase of target sweep
        stream_timestamp = str(det.stream.timestamp)
        targ_phase, targ_phase_df = det.targ.phase_plot(prefix='mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', include=tone, return_df = True)
        targ_phase_fit, targ_phase_fit_df = det.targ.phase_plot(prefix='phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', include=tone, return_df = True)
        
        targ_phase_to_f_spline, targ_phase_to_f_spline_df = det.targ.plot(stream_timestamp, 'phase', x_prefix=f'mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_phase_to_f_spline', y_prefix='mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', include=tone)
        targ_f_to_phase_spline, targ_f_to_phase_spline_df = det.targ.plot('f', stream_timestamp, x_prefix='', y_prefix=f'f_to_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_phase_spline', include=tone)
        
        if not (targ_phase_to_f_spline_df[stream_timestamp] == 0).all():
            targ_phase_spline_fig = targ_phase_to_f_spline.NdOverlay.Curve[tone].opts(opts.Curve(color='green', linewidth=2)).relabel(label='Spline', group='Sweep')
        elif not (targ_f_to_phase_spline_df[stream_timestamp] == 0).all():
            targ_phase_spline_fig = targ_f_to_phase_spline.NdOverlay.Curve[tone].opts(opts.Curve(color='green', linewidth=2)).relabel(label='Spline', group='Sweep')
        else:
            targ_phase_spline_fig = hv.Scatter([])

        targ_phase_fit_fig = targ_phase_fit.NdOverlay.Curve[tone].opts(opts.Curve(color='k', linewidth=2)).relabel(label='Phase Fit', group='Sweep') 
        targ_phase_fig = targ_phase.NdOverlay.Scatter[tone].opts(opts.Scatter(color=mpl.colormaps[cmap](1))).relabel(label='Sweep Data', group='Sweep')
        # Plot phase timestream
        stream_phase, stream_phase_df = det.stream.phase_plot(x_prefix='mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', y_prefix='mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', include=tone, return_df = True, rasterize=False)
        stream_phase_fig = spread(datashade(stream_phase, dynamic=False, cnorm='eq_hist', cmap='blues', width=viz_cfg['plot']['width'], height=viz_cfg['plot']['height']), px=5, shape='circle').relabel(label='Data', group='Timestream')
        stream_phase_legend = stream_phase_df[0].hvplot.scatter('f', 'phase', label='Stream Data', size=300)
        phase_fig = targ_phase_fit_fig*stream_phase_legend*targ_phase_fig*targ_phase_spline_fig*stream_phase_fig
        phase_fig.opts(title='Target Sweep Phase')

        # Plot IQ
        # =======
        # Plot IQ circle of target sweep
        targ_IQ, targ_IQ_df = det.targ.IQ_plot(prefix='origin_shift_origin_rotate_unwind_rotate', include=tone, return_df = True)
        circle_fit_IQ, circle_fit_IQ_df = det.targ.IQ_plot(prefix='origin_shift_origin_rotate_circle_fit_unwind_rotate', include=tone, return_df = True)
        circle_fit_fig = circle_fit_IQ.NdOverlay.Curve[tone].opts(opts.Curve(color='k', linewidth=2)).relabel(label='Circle Fit', group='Sweep')
        targ_IQ_fig = targ_IQ.NdOverlay.Scatter[tone].opts(opts.Scatter(color=mpl.colormaps[cmap](1))).relabel(label='Sweep Data', group='Sweep')
        # Plot IQ timestream
        stream_IQ, stream_IQ_df = det.stream.IQ_plot(prefix='origin_shift_origin_rotate_unwind_rotate', include=tone, return_df = True, rasterize=False)
        stream_IQ_fig = spread(datashade(stream_IQ, dynamic=False, cnorm='eq_hist', cmap='blues', width=viz_cfg['plot']['height'], height=viz_cfg['plot']['height']), px=2, shape='circle').relabel(label='Stream Data', group='Timestream')
        stream_IQ_legend = stream_IQ_df[0].hvplot.scatter('I', 'Q', label='Stream Data', size=300)
        IQ_fig = circle_fit_fig*hv.HLine(0).opts(color='k', linewidth=0.5)*hv.VLine(0).opts(color='k', linewidth=0.5)*stream_IQ_legend*targ_IQ_fig*stream_IQ_fig
    
        targ_I_min, targ_I_max = targ_IQ_df['I'].min(), targ_IQ_df['I'].max()
        targ_Q_min, targ_Q_max = targ_IQ_df['Q'].min(), targ_IQ_df['Q'].max()

        targ_full_min, targ_full_max = min(targ_I_min, targ_Q_min), max(targ_I_max, targ_Q_max)
        targ_abs_max = max(np.abs(targ_full_min), np.abs(targ_full_max))
        targ_abs_max *= 1.1

        IQ_fig.opts(xlim=(-targ_abs_max, targ_abs_max), ylim=(-targ_abs_max, targ_abs_max), title='Target Sweep IQ')

        # PLot Tables
        # ===========
        det_properties = det.properties.filter(pl.col('det') == tone)
        properties = det_properties.select(pl.col(pl.Float64)).select([pl.col('tone_freqs').alias('Tone Frequency'),
                                                                       pl.col('tone_powers').alias('Tone Power'),
                                                                       pl.col('nonlinear_fit_Q_i').alias(r'$Complex\ Fit\ Q_i$'),
                                                                       pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i').alias(r'$Phase\ Fit\ Q_i$'),
                                                                       pl.col('nonlinear_fit_Q_c').alias(r'$Complex\ Fit\ Q_c$'),
                                                                       pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_c').alias(r'$Phase\ Fit\ Q_c$'),
                                                                       pl.col('nonlinear_fit_Q').alias(r'$Complex\ Fit\ Q$'),
                                                                       pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Qr').alias(r'$Phase\ Fit\ Q$')])
        properties = properties.unpivot(on=properties.columns,
                                        variable_name='Property',
                                        value_name='Value')
        
        table = properties.hvplot.table(columns=['Property', 'Value'], width=viz_cfg['plot']['width'], height=viz_cfg['plot']['height'])
        row_1 = mag_fig + phase_fig + IQ_fig + table

        # Plot timestreams
        # ================
        _, timestream_mag_df = det.stream.stream_plot('mag', prefix='', include=tone, return_df = True, rasterize=False)
        timestream_mag_df = timestream_mag_df.with_columns([(pl.col('t') - pl.col('t').first()).alias('t'), (pl.col('mag') - pl.col('mag').mean()).alias('mag')])
        timestream_mag = timestream_mag_df.hvplot.line('t', 'mag')
        timestream_mag_fig = datashade(timestream_mag, dynamic=False, line_width=1, cnorm='linear', cmap='blues', width=viz_cfg['plot']['width'], height=viz_cfg['plot']['height'])
       
        _, timestream_phase_df = det.stream.stream_plot('phase', prefix='', include=tone, return_df = True, rasterize=False)
        timestream_phase_df = timestream_phase_df.with_columns([(pl.col('t') - pl.col('t').first()).alias('t'), (pl.col('phase') - pl.col('phase').mean()).alias('phase')])
        timestream_phase = timestream_phase_df.hvplot.line('t', 'phase')
        timestream_phase_fig = datashade(timestream_phase, dynamic=False, line_width=1.25, pixel_ratio=2, cnorm='linear', cmap='blues', width=viz_cfg['plot']['width'], height=viz_cfg['plot']['height'])
        
        _, timestream_freq_df = det.stream.stream_plot('f', prefix='ppm_frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', include=tone, return_df = True, rasterize=False)
        timestream_freq_df = timestream_freq_df.with_columns([(pl.col('t') - pl.col('t').first()).alias('t'), (pl.col('f') - pl.col('f').mean()).alias('f')])
        timestream_freq = timestream_freq_df.hvplot.line('t', 'f')
        timestream_freq_fig = datashade(timestream_freq, dynamic=False, line_width=1.5, pixel_ratio=2, cnorm='linear', cmap='blues', width=viz_cfg['plot']['width'], height=viz_cfg['plot']['height'])

        _, timestream_I_df = det.stream.stream_plot('I', prefix='timestream_rotate_origin_shift_origin_rotate_unwind_rotate', include=tone, return_df = True, rasterize=False)
        timestream_I_df = timestream_I_df.with_columns([(pl.col('t') - pl.col('t').first()).alias('t'), (pl.col('I') - pl.col('I').mean()).alias('I')])

        _, timestream_Q_df = det.stream.stream_plot('Q', prefix='timestream_rotate_origin_shift_origin_rotate_unwind_rotate', include=tone, return_df = True, rasterize=False)
        timestream_Q_df = timestream_Q_df.with_columns([(pl.col('t') - pl.col('t').first()).alias('t'), (pl.col('Q') - pl.col('Q').mean()).alias('Q')])

        timestream_IQ_df = timestream_I_df.join(timestream_Q_df, on=['t'], how='left', coalesce=True).with_columns((pl.col('Q')/pl.col('I')).alias('Q/I'))
        IQ_std = timestream_IQ_df.select(pl.col('Q/I').std()).item(0, 0)
        timestream_IQ_df = timestream_IQ_df.filter((pl.col('Q/I') > -2.5*IQ_std) & (pl.col('Q/I') < 2.5*IQ_std))
        timestream_IQ = timestream_IQ_df.hvplot.line('t', 'Q/I')
        timestream_IQ_fig = datashade(timestream_IQ, dynamic=False, line_width=1.25, pixel_ratio=2, cnorm='linear', cmap='blues', width=viz_cfg['plot']['width'], height=viz_cfg['plot']['height'])
        
        timestream_IQ_fig.opts(xlabel=r'$Time\ [s]$', ylabel=r'$Q/I$', title='Frequency/Dissipation Timestream', aspect=1)
        timestream_mag_fig.opts(xlabel=r'$Time\ [s]$', ylabel=r'$|S_{21}|$', title='Raw Magnitude Timestream', aspect=1)
        timestream_phase_fig.opts(xlabel=r'$Time\ [s]$', ylabel=r'$Phase\ [rad]$', title='Raw Phase Timestream', aspect=1)
        timestream_freq_fig.opts(xlabel=r'$Time\ [s]$', ylabel=r'$Fractional\ Frequency\ Shift\ [ppm]$', title='Fractional Frequency Timestream', aspect=1)
        row_2 = timestream_IQ_fig + timestream_mag_fig + timestream_phase_fig + timestream_freq_fig

        # Plot centerd IQ timestream
        # ==========================
        stream_centered_IQ, stream_centered_IQ_df = det.stream.plot('I', 'Q', x_prefix='centered_timestream_rotate_origin_shift_origin_rotate_unwind_rotate', y_prefix='timestream_rotate_origin_shift_origin_rotate_unwind_rotate', include=tone)
        stream_centered_IQ = stream_centered_IQ.NdOverlay.Scatter
        stream_centered_IQ = datashade(stream_centered_IQ, dynamic=False, cnorm='linear', cmap='blues', pixel_ratio=0.075, width=viz_cfg['plot']['height'], height=viz_cfg['plot']['height']).opts(aspect=1)
        stream_I_min, stream_I_max = stream_centered_IQ_df['I'].min(), stream_centered_IQ_df['I'].max()
        stream_Q_min, stream_Q_max = stream_centered_IQ_df['Q'].min(), stream_centered_IQ_df['Q'].max()

        stream_full_min, stream_full_max = min(stream_I_min, stream_Q_min), max(stream_I_max, stream_Q_max)
        stream_abs_max = max(np.abs(stream_full_min), np.abs(stream_full_max))
        stream_abs_max *= 1.1

        stream_centered_IQ.opts(xlim=(-stream_abs_max, stream_abs_max), ylim=(-stream_abs_max, stream_abs_max), xlabel=r'$I\ [arb]$', ylabel=r'$Q\ [arb]$', title='Centered IQ Timestream', aspect=1)

        # PLot PSDs
        # =========
        psd_I, psd_I_df = det.stream.psd_plot('I', prefix='timestream_rotate_origin_shift_origin_rotate_unwind_rotate', include=tone, return_df = True)
        psd_Q, psd_Q_df = det.stream.psd_plot('Q', prefix='timestream_rotate_origin_shift_origin_rotate_unwind_rotate', include=tone, return_df = True)
        psd_I_fig = psd_I.NdOverlay.Curve[tone].opts(linewidth=2).relabel('Dissipation Quadrature (I)')
        psd_Q_fig = psd_Q.NdOverlay.Curve[tone].opts(linewidth=2).relabel('Frequency Quadrature (Q)')
        psd_IQ_fig_legend = psd_I_df[0].hvplot.scatter('psd_f', 'I', label=f"Frequency/Dissipation Ratio: {det_properties.select('freq_diss_ratio').item():0.2f}").opts(alpha=0)
        psd_IQ_fig = psd_I_fig*psd_Q_fig*psd_IQ_fig_legend

        psd_x, psd_x_df = det.stream.psd_plot('f', prefix='frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', include=tone, return_df = True)
        psd_x_wn = det_properties.select('frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_white_noise').item()
        psd_x_fig = psd_x.NdOverlay.Curve[tone].opts(linewidth=2).relabel(label = rf"$White\ Noise:\ {psd_x_wn:0.2e}\ Hz^{{-1/2}} $")
        
        psd_IQ_fig.opts(ylabel=r'$PSD\ \left[\sqrt{arb/Hz}\right]$', title='I vs. Q PSDs', logx=True, logy=True)
        psd_x_fig.opts(ylabel=r'$\sqrt{S_{xx}}\ \left[Hz^{-1/2}\right]$', title='Fractional Frequency PSD', logx=True, logy=True)
        row_3 = stream_centered_IQ*hv.HLine(0).opts(color='k', linewidth=0.5)*hv.VLine(0).opts(color='k', linewidth=0.5) + psd_IQ_fig + psd_x_fig*hv.Scatter([])

        fig = (row_1 + row_2 + row_3).cols(4).opts(sublabel_format="", shared_axes=False)
        mpl_fig = hv.render(fig, backend='matplotlib')
        mpl_fig.suptitle(f'{det_type} {network} Tone {tone}: Drive {drive} dB, Sense {sense} dB')
        save_file = f'{fig_file}_{tone:04d}.pdf'
        mpl_fig.savefig(save_file, dpi=100, bbox_inches="tight", pad_inches=0.1)
        plt.close(mpl_fig)
        return fig, save_file
    except Exception as e:
        print(e)
        return None, ''

#====================#
# Analysis Functions #
#====================#

def center_IQ(detector, data = 'both', delay_col='cable_delay', recalc=False):
    ''' Center KID IQ circles at the origin
    '''
    detector.IQ_unwind(data=data, delay_col=delay_col, recalc=recalc)
    detector.IQ_circle_fit(prefix='unwind_rotate', max_workers=min(48, int(detector.targ.num_tones/8)), recalc=recalc)
    detector.IQ_circle_real(prefix='unwind_rotate', loc='origin', data=data, recalc=recalc)
    detector.IQ_circle_real(prefix='circle_fit_unwind_rotate', loc='origin', data='targ', recalc=recalc)
    return detector
  
def fit_detectors(detector, nonlinear = True, phase = True, recalc=False):
    ''' Fit KIDs
    '''
    if phase:
        detector.nonlinear_fit(nonlinear=False, max_workers=min(48, int(detector.targ.num_tones/8)), recalc=recalc)
        center_IQ(detector, data='targ', recalc=recalc, delay_col='nonlinear_fit_delay_ns')
        detector.IQ_circle_rotate(prefix='origin_shift_origin_rotate_unwind_rotate', data='targ', recalc=recalc, rotation='mismatch')
        detector.targ.phase(prefix='mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', recalc=recalc)
        detector.phase_fit(prefix='mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', nonlinear=True, window=6, max_workers=min(48, int(detector.targ.num_tones/8)), recalc=recalc, method='least_squares')
    if nonlinear: detector.nonlinear_fit(nonlinear=True, max_workers=min(48, int(detector.targ.num_tones/2)), save_model_result=True, recalc=recalc)
    return detector

def phase_to_f(detector, recalc = False, phase_bounds=0.2, k=2, f_threshold=150):
    ''' Convert timestreams from phase to frequency
    '''
    fit_detectors(detector, phase=True, nonlinear=False, recalc=recalc)
    f_0s = detector.properties.select('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_0').to_numpy().T[0]
    center_IQ(detector, data='both', delay_col='nonlinear_fit_delay_ns', recalc=recalc)
    detector.IQ_circle_rotate(prefix='origin_shift_origin_rotate_unwind_rotate', data='both', rotation='mismatch', recalc=recalc)
    detector.targ.phase(prefix='mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', recalc=recalc)
    detector.stream.phase(prefix='mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', recalc=recalc)
    detector.stream.mag(prefix='mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', recalc=recalc)
    detector.phase_to_f(prefix='mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', phase_bounds = phase_bounds, k = k, max_workers = min(48, int(detector.targ.num_tones/2)), recalc=recalc)
    detector.frac_f(prefix='mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', f_0 = f_0s, recalc=recalc)
    frac_f_cols = [pl.col(f'frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_{tone:04d}') for tone in detector.stream.tones]
    detector.stream.data = detector.stream.data.with_columns([(frac_f_col*1e6).name.prefix('ppm_') for frac_f_col in frac_f_cols])

    nperseg = 2**round(np.log2(detector.stream.data.height / 5))
    detector.stream.psd(prefix='frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', col_name='f', recalc=recalc, nperseg=nperseg, detrend='linear')
    
    psd_df = (detector.stream.get_data(col_name=['psd_frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_psd_f',
                                                 'psd_frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f'])
                             .filter((pl.col('psd_frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_psd_f') > f_threshold) & ~(pl.col('psd_frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_psd_f').is_nan()))
                             .drop('psd_frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_psd_f'))
    avgs = psd_df.select([pl.col(col).mean().name.prefix('mean_') for col in psd_df.columns])
    avg_frac_f = pl.Series('frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_white_noise', avgs.to_numpy().T.flatten()) 
    detector.properties = detector.properties.with_columns(avg_frac_f)
    return detector

def normalize_detectors(detector, dB = True, recalc=False):
    ''' Normalize detectors using cable profile from nonlinear fit
    '''
    # Normalize IQ data
    detector.nonlinear_fit(nonlinear=False, max_workers=min(48, int(detector.targ.num_tones/8)), recalc=recalc)
    detector.IQ_norm(prefix='', norm_col='cable_nonlinear_fit', data='both', recalc=recalc) # Normalize sweep and timestream data
    detector.IQ_norm(prefix='nonlinear_fit', norm_col='cable_nonlinear_fit', data='targ', recalc=recalc) # Normalize fit
    
    # Calculate magnitude of normalize data
    detector.targ.mag('norm_scale', dB=dB, recalc=recalc)
    detector.targ.mag('norm_scale_nonlinear_fit', dB=dB, recalc=recalc)
    detector.stream.mag('norm_scale', dB=dB, recalc=recalc)
    return detector

def center_timestream(detector, recalc=False, f_threshold=150):
    detector.IQ_circle_rotate(prefix='origin_shift_origin_rotate_unwind_rotate', data='both', rotation='timestream', recalc=recalc)
    I_cols = [pl.col(f'timestream_rotate_origin_shift_origin_rotate_unwind_rotate_I_{tone:04d}') for tone in detector.stream.tones]
    detector.stream.data = detector.stream.data.with_columns([(I_col - I_col.median()).name.prefix('centered_') for I_col in I_cols])
    nperseg = 2**round(np.log2(detector.stream.data.height / 5))

    detector.stream.psd(prefix='timestream_rotate_origin_shift_origin_rotate_unwind_rotate', col_name='I', recalc=recalc, nperseg=nperseg, detrend='linear')
    detector.stream.psd(prefix='timestream_rotate_origin_shift_origin_rotate_unwind_rotate', col_name='Q', recalc=recalc, nperseg=nperseg, detrend='linear')
    psd_df = (detector.stream.get_data(col_name=['psd_timestream_rotate_origin_shift_origin_rotate_unwind_rotate_I_psd_f',
                                                 'psd_timestream_rotate_origin_shift_origin_rotate_unwind_rotate_I',
                                                 'psd_timestream_rotate_origin_shift_origin_rotate_unwind_rotate_Q'])
                             .filter((pl.col('psd_timestream_rotate_origin_shift_origin_rotate_unwind_rotate_I_psd_f') > f_threshold) & ~(pl.col('psd_timestream_rotate_origin_shift_origin_rotate_unwind_rotate_I_psd_f').is_nan()))
                             .drop('psd_timestream_rotate_origin_shift_origin_rotate_unwind_rotate_I_psd_f'))
    avgs = psd_df.select([pl.col(col).mean().name.prefix('mean_') for col in psd_df.columns])
    avg_I = pl.Series('timestream_rotate_origin_shift_origin_rotate_unwind_rotate_I_white_noise', avgs.select(pl.col('^.*_I_.*$')).to_numpy().T.flatten()) 
    avg_Q = pl.Series('timestream_rotate_origin_shift_origin_rotate_unwind_rotate_Q_white_noise', avgs.select(pl.col('^.*_Q_.*$')).to_numpy().T.flatten())
    detector.properties = detector.properties.with_columns(avg_I, avg_Q)
    detector.properties = detector.properties.with_columns((pl.col('timestream_rotate_origin_shift_origin_rotate_unwind_rotate_Q_white_noise')/pl.col('timestream_rotate_origin_shift_origin_rotate_unwind_rotate_I_white_noise')).alias('freq_diss_ratio'))
    return detector

#====================#
# Pickling Functions #
#====================#

def get_pickle_files(args, sess_dir):
    sess_ids = sorted(args.sess_ids)
    pickle_name = args.name
    com_to = args.com_to
    
    pickle_files = [None]*len(com_to)
    for i, com in enumerate(com_to):
        bid, drid = com.split('.')
        pickle_dir = Path(sess_dir) / 'pickle' / f'B{bid}D{drid}'
        if not pickle_dir.exists(): rfsoc_io.create_dir(pickle_dir)
        pickle_files[i] = pickle_dir / (f'{pickle_name}_' + '_'.join(sess_ids) + '.pkl')
    return pickle_files

def pickle_network(network, path):
    ''' Pickle ccatkidlib Network object 
    '''
    # Convert the Detector.properties DataFrames to dicts since they may have objects (e.g., BSpline objects)
    # =======================================================================================================
    for det, *_ in network.data.select('detector').iter_rows():
        prop_dict = det.properties.to_dicts()
        setattr(det, '_data_dict', prop_dict)
        det._properties_df = None

        if det.targ._properties_df is not None:
            prop_dict = det.targ.properties.to_dicts()
            setattr(det.targ, '_data_dict', prop_dict)
            det.targ._properties_df = None
        
        if det.stream._properties_df is not None:
            prop_dict = det.stream.properties.to_dicts()
            setattr(det.stream, '_data_dict', prop_dict)
            det.stream._properties_df = None
    
    # Convert network.data DataFrame to dicts since they include Detector objects
    # ===========================================================================
    setattr(network, '_data_dict', network.data.to_dicts())
    network.data = None
    
    # Pickle Network object
    # =====================
    with open(path, 'wb') as file:
        pickle.dump(network, file)
    gc.collect()

def load_network_pickle(path):
    ''' Load pickled ccatkidlib network object
    '''
    
    # Load pickled Network object
    # ===========================
    with open(path, 'rb') as file:
        network = pickle.load(file)
    
    # Convert dictionary to polars DataFrame
    network.data = pl.DataFrame(network._data_dict)
    
    # Convert detector properties dictionaries to DataFrames
    for det, *_ in network.data.select('detector').iter_rows():
        det.properties = pl.DataFrame(det._data_dict)
        det.targ.properties = pl.DataFrame(det.targ._data_dict)
        det.stream.properties = pl.DataFrame(det.stream._data_dict)

    gc.collect()
    return network

#==================#
# Helper Functions # 
#==================#

def twinx(plot, element):
    ax = plot.handles['axis'] 
    twinax = ax.twinx()
    twinax.set_ylabel(str(element.last.get_dimension(1))) 
    plot.handles['axis'] = twinax

def get_sess_dir(args, analysis_cfg):
    sess_ids = sorted(args.sess_ids)
    dates = args.dates
    data_dir = args.data_dir
    root_dir = analysis_cfg['file_paths']['root_data_dir']
    
    last_sess = sess_ids[-1]
    for date in dates:
        sess_dir = pair.get_sess_dir(last_sess, data_dir = data_dir, root_data_dir = root_dir, date = date)
        if Path(sess_dir).exists(): return sess_dir
    else:
        tqdm.write(f'Failed to find session {last_sess} for any of the provided dates: {dates}')
    return Path('invalid/path')

def eval_args():
    # Initialize arg parser
    parser = argparse.ArgumentParser(prog='Coldload Noise Analysis', description='''Analyze kinetic inductance detector (KID) noise data at various coldload temperatures''')
    
    # Add arguments
    parser.add_argument('--com_to', nargs = '+',  type = str, help = 'Which drones to run analysis')
    parser.add_argument('-t', '--tones', nargs = '?', type = int, help = 'Which tones to plot')
    parser.add_argument('-d', '--data_dir', type = str, help = 'Directory of coldload data')
    parser.add_argument('--sess_ids', nargs = '+', type = str, help = 'Session IDs of coldload data')
    parser.add_argument('--dates', nargs = '+',  type = str, help = 'Dates of coldload data')
    parser.add_argument('-p', '--pickle', action = 'store_true', help = 'Whether to pickle/load processed data')
    parser.add_argument('-n', '--name', type= str, help = 'Name of pickled data file')
    parser.add_argument('-c', '--config', type = str, default = './analysis_config.yaml', help = 'Path to analysis configuration file')
    
    return parser.parse_args()

if __name__ == '__main__':
    #mp.set_start_method('spawn')
    main()
