import marimo

__generated_with = "0.23.10"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""
    ### Overview

    This notebook is a template for loading RFSoC data taken using *ccatkidlib* and should be useable as the basis for analysis notebooks of any type of data set.
    """)
    return


@app.cell(hide_code=True)
def _(analysis_cfg_browser, cfg_editor, data_browser, data_desc, mo):
    mo.md(rf"""
    ### Select Data to Load

    All *ccatkidlib* data analysis objects (**Detector**, **Network**, etc.) require an `analysis_config.yaml`, which can be selected below. If you do not already have a config file setup, an example can be found at `ccatkidlib/analysis/example_analysis_config.yaml`. After a config file is chosen, the directory containing the data to load can be selected. The default behavior is to only allow loading data corresponding to a single **sess_id** (measurement) but can be modified as necessary.


    {
        mo.vstack(
            [
                mo.hstack(
                    [mo.vstack([analysis_cfg_browser, data_browser]), cfg_editor],
                    widths=[1, 1],
                ),
                mo.callout(data_desc),
            ]
        )
    }
    """)
    return


@app.cell(hide_code=True)
def _(
    cable_fit_workers_selector,
    circle_fit_workers_selector,
    com_to_selector,
    interp_workers_selector,
    mismatch_mean_points_selector,
    mo,
    num_pickle_files_selector,
    phase_fit_window_selector,
    phase_fit_workers_selector,
    pickle_name_selector,
    psd_trim_slider,
    psd_workers_selector,
    savgol_order_selector,
    savgol_window_selector,
    savgol_workers_selector,
    spline_bounds_selector,
    spline_k_selector,
    spline_workers_selector,
    transform_button,
    trim_mean_points_selector,
    trim_window_selector,
):
    mo.md(rf"""
    ### Transform Data

    The typical data analysis procedure involves loading the data into **Network** objects, transforming/analyzing the data using the underlying **Detector** objects, and pickling the processed data. One often wants to only analyze the data for a subset of RFSoC drones which can be selected below. Many of the built-in data analysis methods can be multi-processed and the maximum number of CPU cores to use can be specified below. Finally, the name of the pickle file storing the processed data can be chosen below.

    #### Data Selection and Output

    {mo.vstack([mo.hstack([com_to_selector, num_pickle_files_selector, pickle_name_selector])])}

    #### Multiprocessing Workers

    {mo.vstack([mo.hstack([savgol_workers_selector, circle_fit_workers_selector, cable_fit_workers_selector, phase_fit_workers_selector, spline_workers_selector, interp_workers_selector, psd_workers_selector])])}

    #### Signal Processing Parameters

    ##### Savgol Filter

    {mo.hstack([savgol_window_selector, savgol_order_selector])}

    ##### Data Trimming

    {mo.hstack([trim_window_selector, trim_mean_points_selector, mismatch_mean_points_selector])}

    ##### Phase Fitting

    {mo.hstack([phase_fit_window_selector])}

    ##### Spline Interpolation

    {mo.hstack([spline_bounds_selector, spline_k_selector])}

    ##### Power Spectral Density

    {mo.hstack([psd_trim_slider])}

    #### Execute Transformation

    {mo.hstack([transform_button])}
    """)
    return


@app.cell(hide_code=True)
def _(load_pickle_button, mo, pickle_tabs):
    mo.md(rf"""
    ### Load Pickled Data

    If the processed data was saved as pickle files, it can be loaded here without having to re-run any transformation/analysis methods. The RFSoC drones to load the data of can be selected in the **Transform Data** cell. If valid pickle files are found, they can be selected below.

    {mo.vstack([pickle_tabs, load_pickle_button])}
    """)
    return


@app.cell(hide_code=True)
def _(bin_selector, hist_col_selector, mad_selector, mo, property_histogram):
    mo.md(rf"""
    ### Property Histogram

    {mo.vstack([mo.hstack([hist_col_selector, bin_selector, mad_selector], widths=[1,1,1]), property_histogram])}
    """)
    return


@app.cell(disabled=True)
def _(bivariate_kde, combined_properties, hv, pl):
    _low_col, _high_col = ('low_max_diff_FWHM_Q_scale_tail_trim_tail_shift_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_lw',
                          'high_max_diff_FWHM_Q_scale_tail_trim_tail_shift_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_lw')



    _filt_properties = (combined_properties.with_columns(pl.when(pl.col(_low_col).abs() > pl.col(_high_col).abs())
                                        .then(_low_col)
                                        .otherwise(_high_col)                           .alias('max_lw')).filter(pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a') > -1,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a') < 0,
                               pl.col('freq/diss') > 1,
                               pl.col('freq/diss') < 50,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q') > 5000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q') < 40000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_c') < 100000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_c') > 10000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i') > 2000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i') < 75000,
                               pl.col('freq_white_noise_ff') > 2e-9,
                               pl.col('freq_white_noise_ff') < 5e-8,

    ).with_columns(pl.col('freq/diss').max().over('det').alias('max_freq/diss'))
     .with_columns((pl.col('freq/diss')/pl.col('max_freq/diss')).alias('norm_freq/diss')))
     #.filter(pl.col('norm_freq/diss') == 1 ))


    _dist = hv.Bivariate(_filt_properties.select(['phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a', 'freq/diss']).to_numpy())
    kde = bivariate_kde(_dist, x_range=(-100, 100), y_range=(0, 100), bw_method='silverman', n_samples=20, bandwidth=1000)
    kde.opts(colorbar=True)
    return


@app.cell
def _():
    from holoviews.operation.stats import bivariate_kde

    return (bivariate_kde,)


@app.cell
def _(mo):
    det_selector = mo.ui.slider(start=0, stop=500, step=1, full_width=True)
    return (det_selector,)


@app.cell
def _(det_selector, hv, mo, networks, np, pl, viz_utils):
    _network = list(networks.values())[0]

    _det = int(det_selector.value)

    _plot, _df = _network.plot('phase', 'targ', prefix='phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', data_cols=['drive'], overlay_cols=['drive'], filter_exprs=[pl.col('drive').mod(1) == 0, pl.col('drive') < 18], cmap='cividis', save_fig=False, dynamic=False, include=_det, return_df=True)

    _cs = viz_utils.cycle_cmap('cividis', num_colors=18).values
    _plots = []
    for _c, _drive in zip(_cs, np.arange(0, 18)):
        _plot = _df.filter(pl.col('drive') == _drive).hvplot.line(x='f', y='phase', label=f'Drive Attenuation: {_drive} dB').opts(show_legend=True, aspect=1, color = _c, xlabel='Frequency [Hz]', ylabel='Phase [rad]', marker='o', ms=1.5, linewidth=1)
        _plots.append(_plot)

    mo.vstack([det_selector, 
    hv.Layout(_plots).cols(3).opts(sublabel_format='', shared_axes=True, fig_size=75)])
    return


@app.cell
def _(det_selector, hv, mo, networks, np, pl, viz_utils):
    _network = list(networks.values())[0]

    _det = int(det_selector.value)

    _plot, _df = _network.plot('phase', 'targ', prefix='phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', data_cols=['drive'], overlay_cols=['drive'], filter_exprs=[pl.col('drive').mod(1) == 0, pl.col('drive') < 18], cmap='cividis', save_fig=False, dynamic=False, include=_det, return_df=True)

    _cs = viz_utils.cycle_cmap('cividis', num_colors=18).values
    _plots = []
    for _c, _drive in zip(_cs, np.arange(0, 18)):
        _plot = _df.filter(pl.col('drive') == _drive).hvplot.line(x='f', y='phase', label=f'Drive Attenuation: {_drive} dB').opts(show_legend=True, aspect=1, color = _c, xlabel='Frequency [Hz]', ylabel='Phase [rad]', marker='o', ms=1.5, linewidth=1)
        _plots.append(_plot)

    mo.vstack([det_selector, 
    hv.Layout(_plots).cols(3).opts(sublabel_format='', shared_axes=True, fig_size=75)])
    return


@app.cell(disabled=True)
def _(
    BoundaryNorm,
    ListedColormap,
    det_selector,
    hv,
    networks,
    np,
    opts,
    pl,
    plt,
    viz_utils,
):
    _network = list(networks.values())[0]

    _det = int(det_selector.value)

    _layout = (_network.plot('phase', 'targ', prefix='mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', data_cols=['drive'], overlay_cols=['drive'], filter_exprs=[pl.col('drive').mod(5) == 0], cmap='cividis', save_fig=False, include=_det).opts(opts.Curve(linewidth=0, ms=4, alpha=0.1))
    *
    _network.plot('phase', 'targ', prefix='phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', data_cols=['drive'], overlay_cols=['drive'], filter_exprs=[pl.col('drive').mod(5) == 0], cmap='cividis', save_fig=False, include=_det)[_det].opts(opts.Curve(linewidth=2.5, ms=0)) )

    _layout.opts(opts.Overlay(fontsize={'xlabel':12, 'ylabel':12, 'legend':11, 'xticks':12, 'yticks':12})).opts(xlabel='Frequency [Hz]', ylabel='Phase [rad]', title='Al', fig_size=150)
    _fig = hv.render(_layout, backend='matplotlib')
    _axs = _fig.get_axes()

    for _ax in _axs:
        _ax.grid(True, which='both', linewidth=0.5)

    _drives = _network.data.select('drive').sort('drive').filter(pl.col('drive').mod(5) == 0).to_numpy().T[0]
    _cmap = ListedColormap(viz_utils.cycle_cmap('cividis', num_colors=len(_drives)).values)
    _norm = BoundaryNorm(boundaries=np.arange(_drives[0]-5/2, _drives[-1]+5/2+1, 5), ncolors=_cmap.N)

    _sm = plt.cm.ScalarMappable(cmap=_cmap, norm=_norm)
    _cbar = plt.colorbar(_sm, ax=_axs, aspect=60, fraction=0.25)
    _cbar.set_ticks(_drives, labels=map(str, _drives))
    _cbar.ax.tick_params(labelsize=12)
    _cbar.set_label(label='Drive Attenuation [dB]', fontsize=12)

    _fig.set_layout_engine('constrained')
    targ_plot = _fig
    targ_plot
    return


@app.cell
def _(networks):
    _network = list(networks.values())[0]
    return


@app.cell
def _():
    from matplotlib.colors import ListedColormap, BoundaryNorm


    return BoundaryNorm, ListedColormap


@app.cell
def _(combined_properties, hv, pl):
    _low_col, _high_col = ('low_max_diff_FWHM_Q_scale_tail_trim_tail_shift_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_lw',
                          'high_max_diff_FWHM_Q_scale_tail_trim_tail_shift_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_lw')



    _filt_properties = (combined_properties.with_columns(pl.when(pl.col(_low_col).abs() > pl.col(_high_col).abs())
                                        .then(_low_col)
                                        .otherwise(_high_col).alias('max_lw'),
                                        (-1*pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_b')).alias('fit_a')).filter(
                               pl.col('fit_a') > -3,
                               pl.col('fit_a') < 3,
                               pl.col('freq/diss') > 1,
                               pl.col('freq/diss') < 50,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q') > 5000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q') < 40000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_c') < 100000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_c') > 10000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i') > 2000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i') < 75000,
                               pl.col('freq_white_noise_ff') > 2e-9,
                               pl.col('freq_white_noise_ff') < 5e-8,

    ).with_columns(pl.col('freq/diss').max().over('det').alias('max_freq/diss'))
     .with_columns((pl.col('freq/diss')/pl.col('max_freq/diss')).alias('norm_freq/diss')))

    (_filt_properties.hvplot.line(x='drive', y='fit_a', groupby='det', marker='o', ms=5).opts(xlabel='Drive Attenuation [dB]', ylabel='Nonlinearity Parameter [a]', fontsize={'xlabel':12, 'ylabel':12, 'legend':11, 'xticks':12, 'yticks':12}, show_grid=True, aspect=1.5)*hv.HLine(0.77).opts(c='k', linestyle='--') + 
    _filt_properties.hvplot.line(x='drive', y='freq/diss', groupby='det', marker='o', ms=5).opts(xlabel='Drive Attenuation [dB]', ylabel='Reactive/Dissipative Noise', fontsize={'xlabel':12, 'ylabel':12, 'legend':11, 'xticks':12, 'yticks':12}, aspect=1.5, show_grid=True)).cols(1).opts(sublabel_format='', shared_axes=False)
    return


@app.cell
def _(combined_properties, pl):
    _low_col, _high_col = ('low_max_diff_FWHM_Q_scale_tail_trim_tail_shift_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_lw',
                          'high_max_diff_FWHM_Q_scale_tail_trim_tail_shift_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_lw')



    _filt_properties = (combined_properties.with_columns(pl.when(pl.col(_low_col).abs() > pl.col(_high_col).abs())
                                        .then(_low_col)
                                        .otherwise(_high_col).alias('max_lw'),
                                        (-1*pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a')).alias('fit_a')).filter(
                               pl.col('fit_a') > -1,
                               pl.col('fit_a') < 1,
                               pl.col('freq/diss') > 1,
                               pl.col('freq/diss') < 50,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q') > 5000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q') < 40000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_c') < 100000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_c') > 10000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i') > 2000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i') < 75000,
                               pl.col('freq_white_noise_ff') > 2e-9,
                               pl.col('freq_white_noise_ff') < 5e-8,

    ).with_columns(pl.col('freq/diss').max().over('det').alias('max_freq/diss'))
     .with_columns((pl.col('freq/diss')/pl.col('max_freq/diss')).alias('norm_freq/diss'))
                       .filter(pl.col('norm_freq/diss') == 1))

    _filt_properties.hvplot.scatter(x='fit_a', y='freq/diss', marker='o', ms=5).opts(xlabel='Drive Attenuation [dB]', ylabel='Nonlinearity Parameter [a]', fontsize={'xlabel':12, 'ylabel':12, 'legend':11, 'xticks':12, 'yticks':12}, show_grid=True)
    return


@app.cell
def _(combined_properties, hv, pl):
    _low_col, _high_col = ('low_max_diff_FWHM_Q_scale_tail_trim_tail_shift_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_lw',
                          'high_max_diff_FWHM_Q_scale_tail_trim_tail_shift_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_lw')



    _filt_properties = (combined_properties.with_columns(pl.when(pl.col(_low_col).abs() > pl.col(_high_col).abs())
                                        .then(_low_col)
                                        .otherwise(_high_col).alias('max_lw'),
                                        (-1*pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a')).alias('fit_a')).filter(
                               pl.col('fit_a') > -1,
                               pl.col('fit_a') < 1,
                               pl.col('freq/diss') > 1,
                               pl.col('freq/diss') < 50,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q') > 5000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q') < 40000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_c') < 100000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_c') > 10000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i') > 2000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i') < 75000,
                               pl.col('freq_white_noise_ff') > 2e-9,
                               pl.col('freq_white_noise_ff') < 5e-8,

    ).with_columns(pl.col('freq/diss').max().over('det').alias('max_freq/diss'))
     .with_columns((pl.col('freq/diss')/pl.col('max_freq/diss')).alias('norm_freq/diss'))
     .filter(pl.col('norm_freq/diss') == 1))

    hv.Bivariate(_filt_properties.select(['fit_a', 'freq/diss']).to_numpy()).opts(bandwidth=0.2)
    return


@app.cell
def _():
    return


@app.cell(column=1)
def _(analysis_cfg):
    # Define data transformation prefixes
    # -----------------------------------

    NAME, PREFIX = analysis_cfg['convention']['name'], analysis_cfg['convention']['prefix']

    cable_prefix = '_'.join([PREFIX['remove_cable'],
                             PREFIX['rotate']])

    mismatch_prefix = '_'.join([PREFIX['remove_impedance_mismatch'],
                                PREFIX['rotate'],
                                PREFIX['center_origin'],
                                PREFIX['translate'],
                                PREFIX['center_origin'],
                                PREFIX['rotate'],
                                cable_prefix])

    circle_fit_prefix = '_'.join([PREFIX['IQ_circle_fit'],
                                  PREFIX['trim_tail'],
                                  PREFIX['trim'],
                                  cable_prefix])

    norm_prefix = '_'.join([PREFIX['normalize'],
                            PREFIX['scale'],
                            mismatch_prefix])

    readout_noise_prefix = "_".join([PREFIX['isolate_dissipation_noise'],
                                     PREFIX['rotate'],
                                     norm_prefix])

    trim_prefix = "_".join([PREFIX['trim_tail'],
                            PREFIX['trim'],
                            norm_prefix])

    proj_prefix = '_'.join([PREFIX['dissipation_correction'],
                            norm_prefix])

    mag_prefix = f"{PREFIX['savgol_filter']}0"
    off_res_prefix = 'off_res'
    off_res_shift_prefix = '_'.join([off_res_prefix,
                                     PREFIX['translate']])
    off_res_trim_prefix = '_'.join([PREFIX['trim_tail'],
                             PREFIX['trim'],
                             off_res_shift_prefix])

    trim_norm, trim_readout =  (f"white_noise_{PREFIX['trim']}_{PREFIX['power_spectral_density']}_{norm_prefix}", 
                                      f"white_noise_{PREFIX['trim']}_{PREFIX['power_spectral_density']}_{readout_noise_prefix}")

    freq_wn, diss_wn = (f"freq_white_noise_{NAME['fractional_frequency']}",
                          f"diss_white_noise_{NAME['fractional_frequency']}")

    # Define property names
    # ---------------------
    scale_name = '_'.join([PREFIX['remove_impedance_mismatch'], 
                           cable_prefix,
                           NAME['magnitude']])

    f_0_name = '_'.join([PREFIX['middle_frequency_point'],
                        f"{PREFIX['savgol_filter']}0", 
                        NAME['full_width_half_max'],
                        NAME['frequency']])

    norm_f_0 = '_'.join([PREFIX['middle_frequency_point'],
                        'tail_shift',
                         norm_prefix,
                         NAME['full_width_half_max'],
                         NAME['frequency']])


    low_FWHM_name = '_'.join([PREFIX['low_frequency_side'],
                        f"{PREFIX['savgol_filter']}0", 
                        NAME['full_width_half_max'],
                        NAME['frequency']])
    high_FWHM_name = '_'.join([PREFIX['high_frequency_side'],
                        f"{PREFIX['savgol_filter']}0", 
                        NAME['full_width_half_max'],
                        NAME['frequency']])

    R_name = '_'.join([circle_fit_prefix,
                       NAME['IQ_circle_radius']])

    circle_center_name = '_'.join([circle_fit_prefix,
                                   NAME['IQ_circle_center'],
                                   NAME['magnitude']])
    return (
        NAME,
        PREFIX,
        R_name,
        circle_center_name,
        circle_fit_prefix,
        diss_wn,
        f_0_name,
        freq_wn,
        mag_prefix,
        norm_f_0,
        norm_prefix,
        proj_prefix,
        readout_noise_prefix,
        scale_name,
        trim_norm,
        trim_prefix,
        trim_readout,
    )


@app.cell
def transform_data(
    Network,
    analysis_cfg,
    ccat_pickle,
    com_to_selector,
    dump_transform,
    mo,
    num_pickle_files_selector,
    pickle_name_selector,
    pickle_selector,
    root_data_dir,
    transform_button,
    viz_cfg,
):
    mo.stop(not transform_button.value)

    _data_dir, _date, _sess_id = pickle_selector.value[com_to_selector.value[0]][
        0
    ].parts[-8:-5]

    for _com_to in com_to_selector.value:
        _network = Network(
            com_to=_com_to,
            sess_ids=_sess_id,
            date=_date,
            data_dir=_data_dir,
            root_data_dir=root_data_dir,
            analysis_cfg=analysis_cfg,
            viz_cfg=viz_cfg,
        )

        _network.add_columns(
            data_cols=[
                "com_to",
                "drive",
                "sense",
                "detector_type",
                "network",
                "coldload_temp"
            ],
            max_workers=10,
        )
        mo.output.append(_network.data)

        ccat_pickle.multi_dump(
            _network,
            pickle_name_selector.value,
            num_segments=num_pickle_files_selector.value,
            transform=dump_transform,
        )
    return


@app.cell
def load_pickle(
    ccat_pickle,
    com_to_selector,
    load_pickle_button,
    load_transform,
    mo,
    pickle_selector,
    root_data_dir,
):
    mo.stop(not load_pickle_button.value)

    _data_dir, _date, _sess_id = pickle_selector.value[com_to_selector.value[0]][
        0
    ].parts[-8:-5]

    networks = {
        _com_to: ccat_pickle.multi_load(
            _com_to,
            pickle_selector.value[_com_to][0].stem,
            _sess_id,
            data_dir=_data_dir,
            date=_date,
            root_data_dir=root_data_dir,
            transform=load_transform,
        )
        for _com_to in com_to_selector.value
    }
    return (networks,)


@app.cell
def _(networks):
    network = networks['1.1']
    #network.data = network.data.filter(pl.col('coldload_temp') == 63.25)
    return (network,)


@app.cell
def _(network, pl):
    list(network.det_dict.values())[0].frac_f(data='targ', prefix='')
    x_diff = list(network.det_dict.values())[0].targ.get_data('ff', include=0).select(pl.col('ff_000000').diff())[-1].item()*1e5
    return


@app.cell
def _(mo, network, pl):
    combined_properties = network.combine_properties(data_cols=['drive']).sort('drive')

    _m_col = 'linear_fit_tail_trim_tail_shift_norm_scale_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_lw_ff_m'
    _y_col_low = "low_max_diff_tail_trim_tail_shift_norm_scale_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_lw"
    _y_col_high = "high_max_diff_tail_trim_tail_shift_norm_scale_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_lw"

    mo.output.append(combined_properties.columns)

    #_low_power_slopes = combined_properties.select('det', _m_col, norm_f_0, 'drive').filter(pl.col('drive') == 30).drop('drive').rename({_m_col:'low_power_m', norm_f_0: 'low_power_f'})
    _low_power_slopes = combined_properties.select('det', _m_col, 'drive').filter(pl.col('drive') == 30).drop('drive').rename({_m_col:'low_power_m'})


    combined_properties = combined_properties.join(_low_power_slopes, on='det').with_columns((1e4*pl.col(_y_col_low)/pl.col(_m_col)).alias('low_norm_diff_lw'),
                                                                                             (1e4*pl.col(_y_col_high)/pl.col(_m_col)).alias('high_norm_diff_lw'))
    return (combined_properties,)


@app.cell
def _(combined_properties, network, pl):
    for _drive, _det in network.data.select('drive', 'detector').iter_rows():
        _det = network.det_dict[_det]
        _props = combined_properties.filter(pl.col('drive') == _drive)
        _shifts = _props.sort('det')['f_diff'].to_numpy()
        _det.targ.shift(col_name='f', prefix='', shift=-1*_shifts, name='low_power', recalc=True)
    return


@app.cell
def _(combined_properties, pl):
    combined_properties.filter(pl.col('drive') != 6).hvplot.scatter(x='drive', y='f_diff', groupby='det').opts(show_grid=True)
    return


@app.cell
def _(networks, pl):
    all_network_props = None
    for _network in networks.values():
        _props = list(networks.values())[0].combine_properties(data_cols=['drive']).sort('drive')
        all_network_props = _props if all_network_props is None else pl.concat([all_network_props, _props], how='vertical')
    return (all_network_props,)


@app.cell
def _(combined_properties):
    combined_properties['norm_circle_fit_tail_trim_unwind_rotate_R'].median()
    return


@app.cell
def _(network, norm_prefix, pl, proj_prefix):
    (network.plot('IQ', 'targ', prefix=proj_prefix, include=2, data_cols=['drive'], overlay_cols=['drive'], save_fig=False, return_df=False, cmap='red', ms = 0.5, filter_exprs=[pl.col('drive') == 20]).opts(xlim=(-0.25, 0.25), ylim=(-0.25,0.25), data_aspect=1)*
    network.plot('IQ', 'targ', prefix=norm_prefix, include=2, data_cols=['drive'], overlay_cols=['drive'], save_fig=False, return_df=False, cmap='blue', ms = 0.5, filter_exprs=[pl.col('drive') == 20]).opts(xlim=(-0.25, 0.25), ylim=(-0.30,0.25), data_aspect=1))
    return


@app.cell
def _(network, pl, proj_prefix):
    network.plot('IQ', 'targ', prefix=f"tail_shift_{proj_prefix}", include=1, data_cols=['drive'], overlay_cols=['drive'], save_fig=False, return_df=False, cmap='blue', ms = 1, filter_exprs=[pl.col('drive') == 20]).opts(xlim=(-0.12, 0.12), ylim=(-0.12,0.12), data_aspect=1, show_legend=False, legend_position='right')
    return


@app.cell
def _(network, norm_prefix, pl, trim_prefix):
    _drive = 15
    _det = 38
    (network.plot('phase', 'targ', prefix=norm_prefix, include=_det, data_cols=['drive'], overlay_cols=['drive'], save_fig=False, return_df=False, cmap='viridis', ms = 1, filter_exprs=[pl.col('drive')==_drive])
    *network.plot('phase', 'targ', prefix=f"phase_fit_{trim_prefix}", include=_det, data_cols=['drive'], overlay_cols=['drive'], save_fig=False, return_df=False, color='red', ms = 1, filter_exprs=[pl.col('drive')==_drive]))
    return


@app.cell
def _(mo):
    drive_selector = mo.ui.slider(start=0, stop=25, step=1)
    return (drive_selector,)


@app.cell
def _(combined_properties, pl, trim_prefix):
    _det = 2

    combined_properties.filter(pl.col('det') == _det).with_columns((-1*pl.col(f'phase_fit_{trim_prefix}_a')).alias(f'phase_fit_{trim_prefix}_a')).hvplot(x='drive', y=f'phase_fit_{trim_prefix}_a')#*combined_properties.filter(pl.col('det') == _det).with_columns((-1*pl.col(f'phase_fit_{trim_prefix}_b')).alias(f'phase_fit_{trim_prefix}_b')).hvplot(x='drive', y=f'phase_fit_{trim_prefix}_b')
    return


@app.cell
def _(drive_selector):
    drive_selector
    return


@app.cell
def _():
    0.1/8e-5
    return


@app.cell
def _(network, norm_prefix, opts, pl):
    _drive = 0
    _det = 20
    network.plot('plot', 'targ', x_dim='ff', y_dim='lw', y_prefix=f"tail_trim_tail_shift_{norm_prefix}", include=_det, data_cols=['drive'], overlay_cols=['drive'], save_fig=False, return_df=False, cmap='viridis', ms = 1, filter_exprs=[pl.col('drive') == _drive]).opts(ylim=(-50, 50), show_grid=True).opts(opts.Curve(show_grid=True)).opts(xlabel='Frequency [Hz]', ylabel=r'$y \propto Q_c\left(\frac{f - f_0}{f_0}\right)$')*network.plot('plot', 'targ', x_dim='ff', y_dim='lw', y_prefix=f"linear_fit_tail_trim_tail_shift_{norm_prefix}", include=_det, data_cols=['drive'], overlay_cols=['drive'], save_fig=False, return_df=False, cmap='reds', ms = 1, filter_exprs=[pl.col('drive') == _drive]).opts(ylim=(-50, 50), show_grid=True).opts(opts.Curve(show_grid=True)).opts(xlabel='Frequency [Hz]', ylabel=r'$y \propto Q_c\left(\frac{f - f_0}{f_0}\right)$')
    return


@app.cell
def _(network, norm_prefix):
    _det = list(network.det_dict.values())[5]
    _det.targ.mag(prefix=f"tail_shift_{norm_prefix}")
    _det.targ.mag_plot(prefix=f"tail_shift_{norm_prefix}", include=38)
    return


@app.cell
def _(combined_properties, f_0_name, pl):
    _norm_f_0 = "mid_tail_shift_norm_scale_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_FWHM_f"

    combined_properties.select(f_0_name, _norm_f_0).select(pl.col(f_0_name) - pl.col(_norm_f_0))
    return


@app.cell
def _(network, norm_prefix, opts, pl):
    _drive = 0
    _det = 154
    network.plot('plot', 'targ', x_dim='f', y_dim='lw', y_prefix=f"tail_trim_tail_shift_{norm_prefix}", include=_det, data_cols=['drive'], overlay_cols=['drive'], save_fig=False, return_df=False, cmap='viridis', ms = 1, tone_ms=100,  filter_exprs=[pl.col('drive') != 6, pl.col('drive')%2 == 0]).opts(ylim=(-15, 25), show_grid=True, ).opts(opts.Curve(show_grid=True)).opts(xlabel='Frequency [Hz]', ylabel=r'$y \propto Q_c\left(\frac{f - f_0}{f_0}\right)$')
    return


@app.cell
def _(combined_properties, norm_f_0, pl):

    _df = combined_properties.filter(pl.col('det') == 38).select(norm_f_0, 'drive')
    _low_f = _df.filter(pl.col('drive') == 30)[norm_f_0].item()

    _df.with_columns(pl.col(norm_f_0) - _low_f).hvplot.scatter(x='drive', y=norm_f_0)
    return


@app.cell
def _(network, norm_prefix, opts, pl):
    _drive = 0
    _det = 38
    network.plot('plot', 'targ', x_dim='f', y_dim='lw', y_prefix=f"tail_trim_tail_shift_{norm_prefix}", include=_det, data_cols=['drive'], overlay_cols=['drive'], save_fig=False, return_df=False, cmap='viridis', ms = 1, tone_ms=100,  filter_exprs=[pl.col('drive') != 6]).opts(ylim=(-15, 15), xlim=(-10e-5, 10e-5), show_grid=True).opts(opts.Curve(show_grid=True)).opts(xlabel='Frequency [Hz]', ylabel=r'$y \propto Q_c\left(\frac{f - f_0}{f_0}\right)$')
    return


@app.cell
def _(combined_properties):
    _y_col = "linear_fit_tail_trim_tail_shift_proj_diss_norm_scale_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_lw_ff_m"

    combined_properties.hvplot.line(x='drive', y=_y_col, groupby='det')
    return


@app.cell
def _(combined_properties, pl):
    _y_col_low = "low_max_diff_tail_trim_tail_shift_proj_diss_norm_scale_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_lw"
    _y_col_high = "high_max_diff_tail_trim_tail_shift_proj_diss_norm_scale_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_lw"


    combined_properties.filter(pl.col(_y_col_low).abs() < 4).hvplot.line(x='drive', y=_y_col_low, groupby='det')*combined_properties.filter(pl.col(_y_col_high).abs() < 1).hvplot.line(x='drive', y=_y_col_high, groupby='det')
    return


@app.cell
def _(combined_properties, hv, opts, pl, trim_prefix):
    _x_col = f"phase_fit_{trim_prefix}_a"
    _y_col = 'low_norm_diff_lw'
    _z_col = 'tone_freq'#f"phase_fit_{trim_prefix}_Q_c"


    (combined_properties.filter(pl.col(_x_col) < -0.1, pl.col(_x_col) > -6, pl.col(_y_col) < 0, pl.col(_y_col) > -10, pl.col('freq/diss') < 50, pl.col('freq/diss') > 1, pl.col('drive') < 20, pl.col( f"phase_fit_{trim_prefix}_Q")<7e4, pl.col( f"phase_fit_{trim_prefix}_Q")>0.1e4).with_columns((-1*pl.col(_x_col)).alias(_x_col), (-1*pl.col(_y_col)).alias(_y_col)).hvplot.scatter(x=f"phase_fit_{trim_prefix}_a", y='low_norm_diff_lw', c=_z_col, cmap='viridis', s=10)*hv.VLine(0.77).opts(c='k', linestyle='--')).opts(logx=True, logy=True, show_grid=True, xlabel='Nonlinearity Parameter', ylabel=r'Max $\frac{1}{Q_c}\frac{dy}{dx}$').opts(opts.Scatter(logz=True, clabel='Resonant Frequency [Hz]' ))
    return


@app.cell
def _(combined_properties, hv, opts, pl, trim_prefix):
    _x_col = f"phase_fit_{trim_prefix}_a"
    _y_col = 'low_norm_diff_lw'

    (combined_properties.filter(pl.col(_x_col) < 1, pl.col(_x_col) > -2, pl.col(_y_col) < 0.1, pl.col(_y_col) > -1, pl.col('freq/diss') < 30, pl.col('freq/diss') > 1, pl.col('drive') < 20).with_columns((-1*pl.col(_x_col)).alias(_x_col), (-1*pl.col(_y_col)).alias(_y_col)).hvplot.scatter(x=f"phase_fit_{trim_prefix}_a", y='low_norm_diff_lw', c='freq/diss', cmap='viridis', s=10)*hv.VLine(0.77)).opts(logx=False, logy=False, show_grid=True, xlabel='Nonlinearity Parameter', ylabel=r'Max $\frac{1}{Q_c}\frac{dy}{dx}$').opts(opts.Scatter(logz=True, clabel='Drive Attenuation [dB]' ))
    return


@app.cell
def _(combined_properties, hv, opts, pl, trim_prefix):
    _x_col = f"phase_fit_{trim_prefix}_a"
    _y_col = 'high_norm_diff_lw'

    (combined_properties.filter(pl.col(_x_col) < 1, pl.col(_x_col) > -2, pl.col(_y_col)>0, pl.col(_y_col) < 0.15, pl.col('freq/diss') < 30, pl.col('freq/diss') > 1, pl.col('drive') < 20).with_columns((-1*pl.col(_x_col)).alias(_x_col)).hvplot.scatter(x=f"phase_fit_{trim_prefix}_a", y=_y_col, c='tone_freq', cmap='viridis', s=10)*hv.VLine(0.77)).opts(logx=False, logy=False, show_grid=True, xlabel='Nonlinearity Parameter', ylabel=r'Max $\frac{1}{Q_c}\frac{dy}{dx}$').opts(opts.Scatter(logz=True, clabel='Drive Attenuation [dB]' ))
    return


@app.cell
def _(combined_properties, hv, opts, pl, trim_prefix):
    _x_col = f"phase_fit_{trim_prefix}_a"
    _y_col = 'high_norm_diff_lw'

    (combined_properties.filter(pl.col(_x_col) > 0.05, pl.col(_x_col) < 1, pl.col(_y_col) > -0.2, pl.col(_y_col) < 0.1, pl.col('freq/diss') < 30, pl.col('freq/diss') > 0, pl.col('drive') < 20).hvplot.scatter(x=f"phase_fit_{trim_prefix}_a", y=_y_col, c='tone_freq', cmap='viridis', s=10)*hv.VLine(0.77)).opts(logx=True, logy=True, show_grid=True, xlabel='Nonlinearity Parameter', ylabel=r'Max $\frac{1}{Q_c}\frac{dy}{dx}$').opts(opts.Scatter(logz=True, clabel='Drive Attenuation [dB]' ))
    return


@app.cell
def _(combined_properties, hv, pl, trim_prefix):
    _y_col_low = "low_norm_diff_lw"
    _y_col_high = f"phase_fit_{trim_prefix}_a"


    (combined_properties.filter(pl.col(_y_col_low).abs() < 4).hvplot.line(x='drive', y=_y_col_low, groupby='det', label=r'Max $\frac{1}{Q_c}\frac{dy}{dx}$')*combined_properties.filter(pl.col(_y_col_high).abs() < 4).hvplot.line(x='drive', y=_y_col_high, groupby='det', label='Nonlinearity Parameter')*hv.HLine(-0.77).opts(c='k', linestyle='--')).opts(xlabel='Drive Attenuation [dB]', ylabel='')
    return


@app.cell
def _(combined_properties, pl):
    combined_properties.filter(pl.col('det') == 38)
    return


@app.cell
def _(network):
    list(network.det_dict.values())[0].properties
    return


@app.cell
def _():
    return


@app.cell
def _(network, pl, proj_prefix):
    _drive = 20
    _det = 38
    (network.plot('plot', 'targ', x_dim='ff', y_dim='lw', y_prefix=f"linear_fit_tail_trim_tail_shift_{proj_prefix}", include=_det, data_cols=['drive'], overlay_cols=['drive'], save_fig=False, return_df=False, color='red', ms = 0, filter_exprs=[pl.col('drive')==_drive])*network.plot('plot', 'targ', x_dim='ff', y_dim='lw', y_prefix=f"tail_trim_tail_shift_{proj_prefix}", include=_det, data_cols=['drive'], overlay_cols=['drive'], save_fig=False, return_df=False, cmap='viridis', ms = 1, filter_exprs=[pl.col('drive')==_drive])
    ).opts(xlim=(-10e-5, 15e-5), ylim=(-1, 1))
    return


@app.cell
def _(network, pl, proj_prefix):
    _drive = 20
    _det = 38
    (network.plot('plot', 'targ', x_dim='ff', y_dim='lw', y_prefix=f"diff_tail_trim_tail_shift_{proj_prefix}", include=_det, data_cols=['drive'], overlay_cols=['drive'], save_fig=False, return_df=False, color='red', ms = 0, filter_exprs=[pl.col('drive')==_drive]))
    return


@app.cell
def _(f_0_name, network, pl):
    _det = network.det_dict[network.data.filter(pl.col('drive') == 25)['detector'].item()]

    _det.frac_f(prefix='', 
                       data='targ', 
                       ref_f= f_0_name)

    _det.targ.properties

    #_det.targ.plot(x_dim='ff', y_dim='lw', y_prefix=f"FWHM_Q_scale_tail_trim_tail_shift_{proj_prefix}", include=38).opts(xlim=(-7.5e-5, 7.5e-5))
    return


@app.cell
def _(all_network_props, pl):
    _low_col, _high_col = ('low_max_diff_FWHM_Q_scale_tail_trim_tail_shift_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_lw',
                          'high_max_diff_FWHM_Q_scale_tail_trim_tail_shift_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_lw')


    _filt_props = (all_network_props.with_columns(pl.when(pl.col(_low_col).abs() > pl.col(_high_col).abs())
                                        .then(_low_col)
                            .otherwise(_high_col)                           .alias('max_lw')
        ,(-1*pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a')).alias('fit_a'))
                        .filter(
                               pl.col('fit_a') > -3,
                               pl.col('fit_a') < 3,
                               pl.col('freq/diss') > 1,
                               pl.col('freq/diss') < 50,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q') > 2000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q') < 80000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_c') < 100000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_c') > 5000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i') > 2000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i') < 75000,
                               pl.col('freq_white_noise_ff') > 1e-10,
                               pl.col('freq_white_noise_ff') < 1e-7,

    ).with_columns(pl.col('freq/diss').max().over(['com_to', 'det']).alias('max_freq/diss'))
     .with_columns((pl.col('freq/diss')/pl.col('max_freq/diss')).alias('norm_freq/diss')))


    _df = _filt_props.filter(pl.col('norm_freq/diss') == 1).with_columns((pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q')*((pl.col('tone_freq') - pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_0')))/pl.col('tone_freq')).alias('lws')).filter(pl.col('lws').abs() < 1) 

    _df.hvplot.hist('lws', bins=75).opts(xlabel=r'Linewidths $\left[Q \left(\frac{f_* - f_0}{f_0}\right)\right]$', title=f'{_df.height} 280 GHz TiN Detectors', fontsize={'title':18, 'labels':18, 'ticks':18, 'legend':18})
    return


@app.cell
def _(all_network_props, hv, pl):
    _low_col, _high_col = ('low_max_diff_FWHM_Q_scale_tail_trim_tail_shift_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_lw',
                          'high_max_diff_FWHM_Q_scale_tail_trim_tail_shift_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_lw')


    _filt_props = (all_network_props.with_columns(pl.when(pl.col(_low_col).abs() > pl.col(_high_col).abs())
                                        .then(_low_col)
                            .otherwise(_high_col)                           .alias('max_lw')
        ,(-1*pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a')).alias('fit_a'))
                        .filter(
                               pl.col('fit_a') > -3,
                               pl.col('fit_a') < 3,
                               pl.col('freq/diss') > 1,
                               pl.col('freq/diss') < 100,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q') > 2000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q') < 80000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_c') < 100000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_c') > 5000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i') > 2000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i') < 75000,
                               pl.col('freq_white_noise_ff') > 2e-9,
                               pl.col('freq_white_noise_ff') < 5e-8,

    ).with_columns(pl.col('freq/diss').max().over(['com_to', 'det']).alias('max_freq/diss'))
     .with_columns((pl.col('freq/diss')/pl.col('max_freq/diss')).alias('norm_freq/diss')))

    _best_hist = _filt_props.filter(pl.col('norm_freq/diss') == 1).with_columns((pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q')*((pl.col('tone_freq') - pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_0')))/pl.col('tone_freq')).alias('lws')).filter(pl.col('lws').abs() < 1).hvplot.hist('lws', bins=75, alpha=0.6)
    _med = _filt_props.with_columns((pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q')*((pl.col('tone_freq') - pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_0')))/pl.col('tone_freq')).alias('lws')).filter(pl.col('lws').abs() < 1).filter(pl.col('norm_freq/diss') == 1)['lws'].median()
    _best_hist *= hv.VLine(_med).opts(linestyle='-', linewidth=3)*hv.Curve(([1, 1],[-1, -1]), label=f'Highest Freq/Diss').opts(linestyle='-', linewidth=3)
    _title =f'{_filt_props.filter(pl.col('norm_freq/diss') == 1).height} 280 GHz Al Detectors'


    _threshes = [-1, 0.2, 0.4, 0.6]

    _plots = [_best_hist]
    for _thresh in _threshes:
        #_min_df = _filt_props.with_columns((pl.col('fit_a') - _thresh).abs().alias('abs_a')).filter(pl.col('abs_a') == #pl.col('abs_a').min().over('com_to', 'det'))
        _min_df =_filt_props.with_columns((pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q')*((pl.col('tone_freq') - pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_0')))/pl.col('tone_freq')).alias('lws')).filter(pl.col('lws').abs() < 1) 
        _hist = _min_df.hvplot.hist('lws', bins=75, label=_thresh, alpha=0.1).opts(show_legend=True)
        _med = _min_df.select('lws').median()
        _hist *= hv.VLine(_med.item()).opts(linestyle='--', linewidth=3)*hv.Curve(([1, 1],[-1, -1]), label=f'a = {_thresh if not _thresh == -1 else -0.5}').opts(linestyle='--', linewidth=3)
        _plots.append(_hist)
    hv.Overlay(_plots).opts(show_legend=True, xlabel='Drive Attenuation [dB]', ylim=(0, None), title=_title, fontsize={'title':18, 'labels':18, 'ticks':18, 'legend':18}, aspect=1)
    return


@app.cell
def _(bin_selector, combined_properties, hist_col_selector, mad_selector, pl):
    a_col = hist_col_selector.value
    _mads = mad_selector.value

    _filt_properties = (combined_properties.with_columns(pl.col(_col).median().alias("median"))
                                           .with_columns((pl.col(_col) - pl.col("median"))
                                                        .abs()
                                                        .median()
                                                        .alias("MAD")
                                            )
                                            .filter(
                                                (
                                                    pl.col(_col)
                                                    > (pl.col("median") - _mads * pl.col("MAD"))
                                                )
                                                & (
                                                    pl.col(_col)
                                                    < (pl.col("median") + _mads * pl.col("MAD"))
                                                )
                                            ))


    property_histogram = _filt_properties.hvplot.hist(_col, bins=bin_selector.value)
    return (property_histogram,)


@app.cell(disabled=True)
def _(combined_properties, hv, np, pl, plt):
    hv.extension('matplotlib')
    _data_col = 'norm_freq/diss'
    _low_a, _high_a, _step = -1, 1, 0.1

    _breaks = np.arange(_low_a, _high_a + _step, _step)
    _labels = list(map(str, np.round(np.convolve(_breaks, [0.5, 0.5], "valid"), 2)))
    _labels += [f'{_low_a}', f'{_high_a}']

    _filt_properties = (combined_properties.filter(pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a') > _low_a - _step,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a') < _high_a + _step,
                               pl.col('freq/diss') > 1,
                               pl.col('freq/diss') < 25,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q') > 5000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q') < 40000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_c') < 100000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_c') > 10000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i') > 2000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i') < 75000,
                               pl.col('freq_white_noise_ff') > 2e-9,
                               pl.col('freq_white_noise_ff') < 5e-8,

    ).with_columns(pl.col('freq/diss').max().over('det').alias('max_freq/diss'))
    .with_columns((pl.col('freq/diss')/pl.col('max_freq/diss')).alias('norm_freq/diss'),
                   pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a').cut(breaks=_breaks, labels=_labels).alias('discrete_a').cast(pl.String).cast(pl.Float64))
    .sort('discrete_a')
    .with_columns(pl.col('discrete_a').cast(pl.String)))

    _weights = _filt_properties.filter(pl.col('norm_freq/diss') == 1).select(['discrete_a', pl.col('det').count().over('discrete_a')]).unique(maintain_order=True).select(pl.col('det')/pl.col('det').max()).to_numpy().T[0]/10

    _violins = [_list.to_numpy() for _list in _filt_properties.group_by(['discrete_a', 'det'], maintain_order=True).agg(pl.max(_data_col)).group_by('discrete_a', maintain_order=True).agg(_data_col)[_data_col]][1:-1]
    _positions = _filt_properties['discrete_a'].unique(maintain_order=True)[1:-1]
    _fig, _ax = plt.subplots(figsize=(20,10))
    _ax.violinplot(_violins, positions=np.array(list(map(float, _positions))), widths=_weights[1:-1], showmedians=True)
    _ax.set_aspect(0.4)

    _fig
    return


@app.cell(disabled=True)
def _(combined_properties, hv, np, pl, plt):
    hv.extension('matplotlib')
    _data_col = 'norm_freq/diss'

    _filt_properties = (combined_properties.filter(pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a') > -2,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a') <  2,
                               pl.col('freq/diss') > 1,
                               pl.col('freq/diss') < 25,
                               pl.col('33') > 5000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q') < 40000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_c') < 100000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_c') > 10000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i') > 2000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i') < 75000,
                               pl.col('freq_white_noise_ff') > 2e-9,
                               pl.col('freq_white_noise_ff') < 5e-8,

    ).with_columns(pl.col('freq/diss').max().over('det').alias('max_freq/diss'))
    .with_columns((pl.col('freq/diss')/pl.col('max_freq/diss')).alias('norm_freq/diss')))

    _weights = _filt_properties.filter(pl.col('norm_freq/diss') == 1).select(['drive', pl.col('det').count().over('drive')]).unique(maintain_order=True).select(pl.col('det')/pl.col('det').max()).to_numpy().T[0]/1.5

    _violins = [_list.to_numpy() for _list in _filt_properties.group_by(['drive', 'det'], maintain_order=True).agg(pl.max(_data_col)).group_by('drive', maintain_order=True).agg(_data_col)[_data_col]][1:-1]
    _positions = _filt_properties['drive'].unique(maintain_order=True)[1:-1]
    _fig, _ax = plt.subplots(figsize=(20,10))
    _ax.violinplot(_violins, positions=np.array(list(map(float, _positions))), showmedians=True)
    _ax.set_aspect(3)

    _fig
    return


@app.cell
def _(combined_properties, pl):
    _filt_properties = (combined_properties.filter(pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a') > -2,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a') < 2,
                               pl.col('freq/diss') > 1,
                               pl.col('freq/diss') < 50,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q') > 5000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q') < 40000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_c') < 100000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_c') > 10000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i') > 2000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i') < 75000,
                               pl.col('freq_white_noise_ff') > 2e-9,
                               pl.col('freq_white_noise_ff') < 5e-8,

    ).with_columns(pl.col('freq/diss').max().over('det').alias('max_freq/diss'))
     .with_columns((pl.col('freq/diss')/pl.col('max_freq/diss')).alias('norm_freq/diss'))
     .filter(pl.col('norm_freq/diss') == 1 ))

    _filt_properties.hvplot.hist('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a', bins=40)
    return


@app.cell
def _(combined_properties, det_selector, mo, networks, opts, pl):
    _det = det_selector.value
    _network = list(networks.values())[0]

    _plot = _network.plot('phase','targ', prefix='tail_trim_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', include=_det, data_cols=['drive'], cmap='viridis', save_fig=False).opts(opts.Curve(linewidth=0, ms=0.1))
    _fit = _network.plot('phase','targ', prefix='phase_fit_tail_trim_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', include=_det, data_cols=['drive'], cmap='viridis', save_fig=False).opts(opts.Curve(ms=0, linewidth=1))

    _nonlin_a = combined_properties.filter(pl.col('det') == _det).hvplot.scatter(x='drive', y='phase_fit_tail_trim_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a').opts(xlabel='Drive', ylabel='Nonlinearity Parameter', aspect=1)
    _nonlin_b = combined_properties.filter(pl.col('det') == _det).hvplot.scatter(x='drive', y='phase_fit_tail_trim_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_b').opts(xlabel='Drive', ylabel='Nonlinearity Parameter', aspect=1)


    mo.output.append(det_selector)
    mo.output.append((_plot*_fit + _nonlin_a*_nonlin_b).cols(1))
    return


@app.cell
def _(networks):
    _network = list(networks.values())[0]
    _det = 302
    _plot, _df1 = _network.plot('plot','targ', 'f', 'lw', y_prefix='fit_Q_scale_tail_trim_tail_shift_mismatch_rotate_origin_shift_origin_rotate_norm_scale_unwind_rotate', include=_det, data_cols=['drive'], overlay_cols=['drive'], cmap='viridis', save_fig=False, return_df=True)#.opts(ylim=(-5, 5))
    _plot, _df2 = _network.plot('plot','targ', 'f', 'lw', y_prefix='FWHM_Q_scale_tail_trim_tail_shift_mismatch_rotate_origin_shift_origin_rotate_norm_scale_unwind_rotate', include=_det, data_cols=['drive'], overlay_cols=['drive'], cmap='viridis', save_fig=False, return_df=True)#.opts(ylim=(-5, 5))



    (_network.plot('IQ','targ', prefix='tail_trim_tail_shift_mismatch_rotate_origin_shift_origin_rotate_norm_scale_unwind_rotate', include=_det, data_cols=['drive'], cmap='viridis', save_fig=False)+
    _df1.hvplot.line(x='f', y='lw', groupby=['det', 'drive'], ms=1, marker='o' , linewidth=0.5).opts(aspect=1, fig_size=100, show_legend=False)
    *_df2.hvplot.line(x='f', y='lw', groupby=['det', 'drive'], ms=1, marker='o' , linewidth=0.5).opts(aspect=1, fig_size=100, show_legend=False)).cols(1)#*hv.HLine(0).opts(c='r', linewidth=0.1)
    return


@app.cell
def _(combined_properties, hv, pl):
    (combined_properties.with_columns((pl.col('mid_savgol0_FWHM_f')/ (pl.col('high_savgol0_FWHM_f') - pl.col('low_savgol0_FWHM_f'))).alias('FWHM_Q'))
                        .filter(pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q' ) < 40000)
        .hvplot.scatter(x='FWHM_Q', y='phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q', alpha=1, data_aspect=1, c='phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a', cmap='viridis', s=10)*hv.Slope(1, 0)).opts(fig_size=150)
    return


@app.cell
def _(combined_properties, pl):
    (combined_properties.with_columns((pl.col('mid_savgol0_FWHM_f')/ (pl.col('high_savgol0_FWHM_f') - pl.col('low_savgol0_FWHM_f'))).alias('FWHM_Q')).with_columns((pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q') - pl.col('FWHM_Q')).alias('Q_diff')).filter(pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q' ) < 50000).hvplot.hist('Q_diff', bins=200))
    #*combined_properties.filter(pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q' ) < 100000).hvplot.hist('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q', alpha=0.5))
    return


@app.cell
def _(combined_properties, pl):
    _low_col, _high_col = ('low_max_diff_fit_Q_scale_tail_trim_tail_shift_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_lw',
                          'high_max_diff_fit_Q_scale_tail_trim_tail_shift_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_lw')

    (combined_properties.with_columns(pl.when(pl.col(_low_col).abs() > pl.col(_high_col).abs())
                                        .then(_low_col)
                                        .otherwise(_high_col)
                                        .alias('max_lw'))
    .filter(pl.col('max_lw').abs() < 100, pl.col('det') == 10).hvplot.line(x="drive", y=[_low_col, _high_col, 'max_lw'], s =10))
    return


@app.cell
def _(combined_properties, pl):
    _low_col, _high_col = ('low_max_diff_FWHM_Q_scale_tail_trim_tail_shift_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_lw',
                          'high_max_diff_FWHM_Q_scale_tail_trim_tail_shift_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_lw')


    _filt_properties = (combined_properties.with_columns(pl.when(pl.col(_low_col).abs() > pl.col(_high_col).abs())
                                        .then(_low_col)
                                        .otherwise(_high_col)
                                        .alias('max_lw'))
        .filter(pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a') > -1,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a') < 0,
                               pl.col('freq/diss') > 1,
                               pl.col('freq/diss') < 50,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q') > 5000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q') < 40000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_c') < 100000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_c') > 10000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i') > 2000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i') < 75000,
                               pl.col('freq_white_noise_ff') > 2e-9,
                               pl.col('freq_white_noise_ff') < 5e-8,
                               pl.col('max_lw') > -30,
                               pl.col('max_lw') < 30

    ).with_columns(pl.col('freq/diss').max().over('det').alias('max_freq/diss'))
     .with_columns((pl.col('freq/diss')/pl.col('max_freq/diss')).alias('norm_freq/diss')))
     #.filter(pl.col('norm_freq/diss') == 1 ))

    _filt_properties.with_columns(pl.col('max_lw').abs()).hvplot.scatter(y='max_lw', x='phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a', s = 10, c='phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_0', cmap='viridis')
    #(_filt_properties.hvplot.hist('max_lw', bins=100)+
    #_filt_properties.hvplot.hist('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a', bins=100))
    return


@app.cell
def _(combined_properties, pl):
    _low_col, _high_col = ('low_max_diff_fit_Q_scale_tail_trim_tail_shift_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_lw',
                          'high_max_diff_fit_Q_scale_tail_trim_tail_shift_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_lw')


    _filt_properties = (combined_properties.with_columns(pl.when(pl.col(_low_col).abs() > pl.col(_high_col).abs())
                                        .then(_low_col)
                                        .otherwise(_high_col)
                                        .alias('max_lw'))
        .filter(pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a') > -1,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a') < 0,
                               pl.col('freq/diss') > 1,
                               pl.col('freq/diss') < 50,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q') > 1000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q') < 40000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_c') < 500000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_c') > 5000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i') > 1000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i') < 75000,
                               #pl.col('freq_white_noise_ff') > 2e-9,
                               #pl.col('freq_white_noise_ff') < 5e-8,
                               pl.col('max_lw') > -40,
                               pl.col('max_lw') < 40

    ).with_columns(pl.col('freq/diss').max().over('det').alias('max_freq/diss'))
     .with_columns((pl.col('freq/diss')/pl.col('max_freq/diss')).alias('norm_freq/diss')))
     #.filter(pl.col('norm_freq/diss') == 1 ))

    _filt_properties.with_columns(pl.col('max_lw').abs()).hvplot.scatter(y='max_lw', x='phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a', s = 10, c ='phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_0', cmap='viridis')
    #(_filt_properties.hvplot.hist('max_lw', bins=100)+
    #_filt_properties.hvplot.hist('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a', bins=100))
    return


@app.cell
def _(combined_properties, pl):
    _low_col, _high_col = ('low_max_diff_tail_trim_tail_shift_mismatch_rotate_origin_shift_origin_rotate_norm_scale_unwind_rotate_lw',
                          'high_max_diff_tail_trim_tail_shift_mismatch_rotate_origin_shift_origin_rotate_norm_scale_unwind_rotate_lw')


    _filt_properties = (combined_properties.with_columns(pl.when(pl.col(_low_col).abs() > pl.col(_high_col).abs())
                                        .then(_low_col)
                                        .otherwise(_high_col)
                                        .alias('max_lw'))
        .filter(pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_norm_scale_unwind_rotate_a') > -3,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_norm_scale_unwind_rotate_a') < 3,
                          #     pl.col('freq/diss') > 2,
                          #     pl.col('freq/diss') < 50,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_norm_scale_unwind_rotate_Q') > 5000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_norm_scale_unwind_rotate_Q') < 40000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_norm_scale_unwind_rotate_Q_c') < 100000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_norm_scale_unwind_rotate_Q_c') > 10000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_norm_scale_unwind_rotate_Q_i') > 2000,
                               pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_norm_scale_unwind_rotate_Q_i') < 75000,
                             #  pl.col('freq_white_noise_ff') > 2e-9,
                             #  pl.col('freq_white_noise_ff') < 5e-8,
                               pl.col('max_lw') > -1,
                               pl.col('max_lw') < 1

    ).with_columns(pl.col('freq/diss').max().over('det').alias('max_freq/diss'))
     .with_columns((pl.col('freq/diss')/pl.col('max_freq/diss')).alias('norm_freq/diss')))
     #.filter(pl.col('norm_freq/diss') == 1 ))

    #_filt_properties.with_columns(pl.col('max_lw').abs()).hvplot.scatter(y='max_lw', x='phase_fit_mismatch_rotate_origin_shift_origin_rotate_norm_scale_unwind_rotate_a', s = 10, c ='phase_fit_mismatch_rotate_origin_shift_origin_rotate_norm_scale_unwind_rotate_Q', cmap='viridis')
    (_filt_properties.hvplot.hist('max_lw', bins=100)+
    _filt_properties.hvplot.hist('phase_fit_mismatch_rotate_origin_shift_origin_rotate_norm_scale_unwind_rotate_a', bins=100))
    return


@app.cell
def _():
    return


@app.cell(column=2)
def _():
    # General
    import marimo as mo

    from tqdm import tqdm
    from functools import partial
    import time

    # IO
    import os
    import ast
    import json
    import pickle
    from pathlib import Path

    return Path, json, mo, os, tqdm


@app.cell
def _():
    # Data Analysis
    import numpy as np
    import polars as pl

    return np, pl


@app.cell
def _():
    # ccatkidlib
    import ccatkidlib.io as ccat_io
    import ccatkidlib.log as ccat_log
    import ccatkidlib.analysis.utils.pickle as ccat_pickle
    import ccatkidlib.analysis.utils.pair as ccat_pair
    import ccatkidlib.analysis.utils.dataframe as ccat_df
    import ccatkidlib.analysis.viz.viz_utils as viz_utils

    from ccatkidlib.rfsoc.rfsoc_daq import R
    from ccatkidlib.analysis.core.detector import Detector
    from ccatkidlib.analysis.core.network import Network

    return Network, ccat_io, ccat_pickle, viz_utils


@app.cell
def _():
    # Multiprocessing
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    mp.set_start_method("spawn", force=True)
    return (ProcessPoolExecutor,)


@app.cell
def _():
    # Plotting
    import matplotlib.pyplot as plt
    import holoviews as hv
    import hvplot.polars
    import panel as pn

    from holoviews import opts

    hv.extension("matplotlib")
    return hv, opts, plt


@app.cell
def _():
    import seaborn as sns

    return


@app.cell
def _(Path, mo, os):
    # Create file browser for selecting analysis config file
    HOME_DIR = Path(os.environ["HOME"])
    analysis_cfg_browser = mo.ui.file_browser(
        initial_path=HOME_DIR,
        filetypes=[".yaml"],
        multiple=False,
        ignore_empty_dirs=True,
        label="Select analysis configuration file...",
    )
    return HOME_DIR, analysis_cfg_browser


@app.cell
def _(analysis_cfg_browser, ccat_io, json, mo):
    # Load selected analysis config
    _editor_height = 750
    analysis_cfg, viz_cfg = {}, {}
    if _browser_value := analysis_cfg_browser.value:
        analysis_cfg_path = _browser_value[0].path
        analysis_cfg, viz_cfg = ccat_io.load_config(cfg_path=analysis_cfg_path)

    # Display config contents in code_editor UI element
    cfg_editor = mo.ui.code_editor(
        value=json.dumps(analysis_cfg, indent=4) if analysis_cfg else "",
        disabled=True,
        min_height=_editor_height,
        max_height=_editor_height,
        placeholder="Configuration file contents will display here once a valid file is selected!",
    )
    return analysis_cfg, cfg_editor, viz_cfg


@app.cell
def _(HOME_DIR, analysis_cfg, mo):
    # Extract root data directory from analysis config file
    root_data_dir = (
        analysis_cfg["file_paths"]["root_data_dir"] if analysis_cfg else HOME_DIR
    )

    # Create file browser for selecting data sess_id directory
    data_browser = mo.ui.file_browser(
        initial_path=root_data_dir,
        selection_mode="directory",
        multiple=False,
        restrict_navigation=True,
        ignore_empty_dirs=True,
        label="Select data directory(ies)...",
    )
    return data_browser, root_data_dir


@app.cell
def _(ccat_io, data_browser):
    # Extract data description for selected data directory
    # ----------------------------------------------------
    data_dirs = []
    try:
        if not int(data_browser.value[0].name) > 1.7e9:
            raise ValueError
        data_dirs = [_file.path for _file in data_browser.value]
        _cfgs = [
            ccat_io.load_config(
                list((_path / "config").glob("init_config_io*.yaml"))[0]
            )
            for _path in data_dirs
        ]
        data_desc = "<br>".join(
            [f"{_cfg['sess_id']}: {_cfg['desc']}" for _cfg in _cfgs]
        )
    except ValueError:
        data_desc = (
            "Selected data directories must be session IDs (e.g. '1773339248')"
        )
    except IndexError:
        data_desc = "No data directory selected."
    return data_desc, data_dirs


@app.cell
def _(data_browser, data_dirs, mo):
    mo.stop(not data_browser.value)

    # Determine which drones took data
    _com_tos = [set()] * len(data_dirs)
    for _i, _data_dir in enumerate(data_dirs):
        _drones = [
            _dir.name.split("D")
            for _dir in (_data_dir / "config").iterdir()
            if _dir.is_dir()
        ]
        _com_tos[_i] = set([f"{_drone[0][1:]}.{_drone[1]}" for _drone in _drones])
    _all_com_tos = sorted(
        list(set.intersection(*_com_tos))
    )  # Only allow drones that took data for all measurements

    # Create selector for drones that took data
    com_to_selector = mo.ui.multiselect(
        _all_com_tos,
        value=_all_com_tos[0:1],
        label="Select Drone(s)...",
        full_width=True,
    )
    return (com_to_selector,)


@app.cell
def _(mo, os):
    # Create UI elements for transforming data
    # ----------------------------------------

    # Create selector for max number of CPU cores to use
    savgol_workers_selector = mo.ui.number(
        start=1,
        stop=os.cpu_count(),
        step=1,
        label="Savgol Max Workers",
        value=5,
        full_width=True,
    )

    circle_fit_workers_selector = mo.ui.number(
        start=1,
        stop=os.cpu_count(),
        step=1,
        label="Circle Fit Max Workers",
        value=15,
        full_width=True,
    )

    cable_fit_workers_selector = mo.ui.number(
        start=1,
        stop=os.cpu_count(),
        step=1,
        label="Cable Fit Max Workers",
        value=20,
        full_width=True,
    )

    phase_fit_workers_selector = mo.ui.number(
        start=1,
        stop=os.cpu_count(),
        step=1,
        label="Phase Fit Max Workers",
        value=20,
        full_width=True,
    )

    spline_workers_selector = mo.ui.number(
        start=1,
        stop=os.cpu_count(),
        step=1,
        label="Target Sweep Spline Workers",
        value=20,
        full_width=True,
    )

    interp_workers_selector = mo.ui.number(
        start=1,
        stop=os.cpu_count(),
        step=1,
        label="Stream Interpolation Workers",
        value=25,
        full_width=True,
    )

    psd_workers_selector = mo.ui.number(
        start=1,
        stop=os.cpu_count(),
        step=1,
        label="PSD Workers",
        value=5,
        full_width=True,
    )
    # Create text element for choosing pickle file save name
    pickle_name_selector = mo.ui.text(
        value="network_pickle",
        debounce=True,
        label="Enter Custom Pickle File Name",
        full_width=True,
    )

    # Create selector for the number of pickle files to split Network objects into
    num_pickle_files_selector = mo.ui.number(
        start=1,
        step=1,
        label="Number of Pickle Files",
        value=1,
        full_width=True,
    )
    return (
        cable_fit_workers_selector,
        circle_fit_workers_selector,
        interp_workers_selector,
        num_pickle_files_selector,
        phase_fit_workers_selector,
        pickle_name_selector,
        psd_workers_selector,
        savgol_workers_selector,
        spline_workers_selector,
    )


@app.cell
def _(com_to_selector, mo):
    # Create run button for transforming data
    transform_button = mo.ui.run_button(
        kind="success",
        label="Run Data Transformation",
        tooltip="Click to transform data for the selected drones",
        full_width=True,
        disabled=not com_to_selector.value,
    )
    return (transform_button,)


@app.cell
def _(mo):
    # Create UI elements for dump_transform parameters
    # ---------------------------------------------------

    # Savgol filter parameters
    savgol_window_selector = mo.ui.number(
        start=1,
        stop=30,
        step=1,
        label="Savgol Window",
        value=9,
        full_width=True,
    )

    savgol_order_selector = mo.ui.number(
        start=0,
        stop=10,
        step=1,
        label="Savgol Order",
        value=1,
        full_width=True,
    )

    # Trim parameters
    trim_window_selector = mo.ui.number(
        start=1,
        stop=5,
        step=1,
        label="Trim Window",
        value=2,
        full_width=True,
    )

    trim_mean_points_selector = mo.ui.number(
        start=1,
        stop=50,
        step=1,
        label="Trim Mean Points",
        value=15,
        full_width=True,
    )

    mismatch_mean_points_selector = mo.ui.number(
        start=1,
        stop=50,
        step=1,
        label="Mismatch Mean Points",
        value=15,
        full_width=True,
    )

    # Phase fit parameters
    phase_fit_window_selector = mo.ui.number(
        start=1,
        stop=5,
        step=0.1,
        label="Phase Fit Window",
        value=1.5,
        full_width=True,
    )

    # Spline parameters
    spline_bounds_selector = mo.ui.number(
        start=0,
        stop=3,
        value=1,
        step=0.01,
        label="Spline Bounds",
        full_width=True,
    )

    spline_k_selector = mo.ui.number(
        start=1,
        stop=5,
        step=1,
        label="Spline K (degree)",
        value=2,
        full_width=True,
    )

    # PSD parameters
    psd_trim_slider = mo.ui.range_slider(
        start=0,
        stop=300,
        value=[135, 300],
        step=5,
        label="PSD Trim Frequency Range (Hz)",
        full_width=True,
    )
    return (
        mismatch_mean_points_selector,
        phase_fit_window_selector,
        psd_trim_slider,
        savgol_order_selector,
        savgol_window_selector,
        spline_bounds_selector,
        spline_k_selector,
        trim_mean_points_selector,
        trim_window_selector,
    )


@app.cell
def _(com_to_selector, data_dirs, mo):
    # Search for pickle files
    # -----------------------

    _pickle_dir = data_dirs[0] / "pickle"
    _com_tos = com_to_selector.value

    pickle_dict = {}
    for _com_to in sorted(_com_tos):
        _bid, _drid = _com_to.split(".")
        if (
            _com_to_pickle := (_pickle_dir / f"B{_bid}D{_drid}" / "network")
        ).exists():
            _avail_com_to_pickle = []
            for _sess_id in _com_to_pickle.iterdir():
                for _pickle_file in _sess_id.iterdir():
                    _avail_com_to_pickle.append(_pickle_file)
            _avail_com_to_pickle = sorted(_avail_com_to_pickle)
            pickle_dict[_com_to] = mo.ui.multiselect(
                _avail_com_to_pickle,
                value=[_avail_com_to_pickle[0]],
                full_width=True,
                label="Select Pickle Files...",
                max_selections=1,
            )
        else:
            pickle_dict[_com_to] = mo.ui.text(
                "No pickle files to select", disabled=True, full_width=True
            )

    pickle_selector = mo.ui.dictionary(pickle_dict, label="Pickle File Selector")
    return (pickle_selector,)


@app.cell
def _(com_to_selector, mo, pickle_selector):
    # Create UI elements for selecting and loading pickle files
    # ---------------------------------------------------------
    pickle_tabs = mo.ui.tabs(
        {_drone: _selector for _drone, _selector in pickle_selector.items()}
    )

    load_pickle_button = mo.ui.run_button(
        kind="success",
        label="Load Pickle Files",
        disabled=not com_to_selector.value,
        full_width=True,
    )
    return load_pickle_button, pickle_tabs


@app.cell
def _(NAME, combined_properties, mo):
    hist_col_selector = mo.ui.dropdown(options=combined_properties.columns, value=NAME['tone_frequency'], searchable=True, full_width=True, label='Property Column')

    mad_selector = mo.ui.number(start=1, stop=50, step=0.5, value=10, debounce=True, full_width=True, label='Number of MADs')

    bin_selector = mo.ui.number(start=1, stop=500, step=1, value=100, debounce=True, full_width=True, label='Number of Histogram Bins')
    return bin_selector, hist_col_selector, mad_selector


@app.cell
def _():
    return


@app.cell(column=3)
def dump_transform(
    NAME,
    Network,
    PREFIX,
    ProcessPoolExecutor,
    R_name,
    cable_fit_workers_selector,
    circle_center_name,
    circle_fit_prefix,
    circle_fit_workers_selector,
    diss_wn,
    f_0_name,
    freq_wn,
    interp_workers_selector,
    mag_prefix,
    mismatch_mean_points_selector,
    norm_f_0,
    norm_prefix,
    phase_fit_window_selector,
    phase_fit_workers_selector,
    pl,
    psd_trim_slider,
    psd_workers_selector,
    readout_noise_prefix,
    savgol_order_selector,
    savgol_window_selector,
    savgol_workers_selector,
    scale_name,
    spline_bounds_selector,
    spline_k_selector,
    spline_workers_selector,
    tqdm,
    trim_mean_points_selector,
    trim_norm,
    trim_prefix,
    trim_readout,
    trim_window_selector,
):
    def dump_transform(network: Network) -> Network:
        """
        Transformations to run on Network objects before dumping to pickle files
        """
        from ccatkidlib.analysis.routines.common import IQ_circle_center, IQ_noise, phase_to_ff
        import ccatkidlib.analysis.routines.properties as property_routines
        import ccatkidlib.analysis.routines.tune as tune_routines
        import ccatkidlib.analysis.fit as ccat_fit

        _max_workers = max(
            savgol_workers_selector.value,
            circle_fit_workers_selector.value,
            phase_fit_workers_selector.value,
            spline_workers_selector.value,
            interp_workers_selector.value,
            psd_workers_selector.value,
            cable_fit_workers_selector.value,
        )

        _pipeline_steps = ["IQ_Circle_Center",
                           "IQ_Noise",
                           "Phase to FF",
                           "Phase Fit",
                           "PSD Calculation",
                           "PSD Trim",
                           "Linewidth Shift"]

        params = None
        with ProcessPoolExecutor(max_workers=_max_workers) as _ex:
            for det_idx, (det) in enumerate(tqdm(
                network.data.sort('drive', descending=True)['detector'],
                desc="Processing Detectors",
                total=len(network.det_dict),
                unit="Detector",
            )):
                try:
                    det = network.det_dict[det]

                    # IQ Circle Center
                    # ----------------
                    tqdm.write(f"  [Detector {det_idx}] Running: {_pipeline_steps[0]}")
                    IQ_circle_center(
                        det,
                        normalize=True,
                        data="both",
                        savgol_window=savgol_window_selector.value,
                        savgol_order=savgol_order_selector.value,
                        trim_window=trim_window_selector.value,
                        trim_mean_points=trim_mean_points_selector.value,
                        mismatch_mean_points=mismatch_mean_points_selector.value,
                        savgol_workers=savgol_workers_selector.value,
                        circle_fit_workers=circle_fit_workers_selector.value,
                        cable_fit_workers=cable_fit_workers_selector.value,
                        ex=_ex,
                    )

                    # IQ Noise
                    # --------
                    tqdm.write(f"  [Detector {det_idx}] Running: {_pipeline_steps[1]}")
                    IQ_noise(det,
                             prefix=norm_prefix)

                    # Phase to FF
                    # -----------
                    tqdm.write(f"  [Detector {det_idx}] Running: {_pipeline_steps[2]}")
                    phase_to_ff(det,
                                prefix=[norm_prefix, readout_noise_prefix],
                                spline_prefix=norm_prefix,
                                spline_bounds=spline_bounds_selector.value,
                                spline_k=spline_k_selector.value,
                                spline_workers=spline_workers_selector.value,
                                interp_workers=interp_workers_selector.value,
                                ref_f=f_0_name,
                                ex=_ex)

                    # Phase Fit
                    # ---------
                    tqdm.write(f"  [Detector {det_idx}] Running: {_pipeline_steps[3]}")
                    det.IQ_trim(
                        prefix=norm_prefix,
                        window=phase_fit_window_selector.value,
                        mean_points=trim_mean_points_selector.value,
                        use_fit=False,
                        mag_prefix=mag_prefix,
                    )

                    det.targ._properties_df = det.targ.properties.with_columns((pl.col(R_name)/pl.col(scale_name)).alias(f"{PREFIX['normalize']}_{R_name}"), (pl.col(circle_center_name)/pl.col(scale_name)).alias(f"{PREFIX['normalize']}_{circle_center_name}"), pl.lit(0.05).alias(f"const_{R_name}"))

                    det.targ.phase(prefix=trim_prefix)

                    det.fit_result = {}
                    det.phase_fit(
                        prefix=trim_prefix,
                        circle_fit_prefix=f"{PREFIX['normalize']}_{circle_fit_prefix}",
                        nonlinear=True,
                        params=params,
                        max_workers=phase_fit_workers_selector.value,
                        ex=_ex,
                    )

                    result = det.fit_result[f"{PREFIX['phase_fit']}_{trim_prefix}_{NAME['fit_result']}"]
                    params = [res.params if res is not None else None for res in result]
                    for param in params: 
                        if param is not None: param.pop('R')


                    # PSD Calculation
                    # ---------------
                    tqdm.write(f"  [Detector {det_idx}] Running: {_pipeline_steps[4]}")
                    det.stream.psd(col_name=NAME['fractional_frequency'],
                                   prefix=[norm_prefix, readout_noise_prefix],
                                   nperseg=512,
                                   max_workers=psd_workers_selector.value,
                                   ex=_ex)

                    # PSD Trim
                    # --------
                    tqdm.write(f"  [Detector {det_idx}] Running: {_pipeline_steps[5]}")
                    det.stream.psd_trim(col_name=NAME['fractional_frequency'],
                                        prefix=[norm_prefix, readout_noise_prefix],
                                        low_f=psd_trim_slider.value[0],
                                        high_f=psd_trim_slider.value[1],
                                        name='white_noise')

                    # Calc freq/diss ratio
                    # --------------------
                    property_routines.agg(det.stream, "median", col_name = NAME['fractional_frequency'], prefix=trim_norm)
                    property_routines.agg(det.stream, "median", col_name = NAME['fractional_frequency'], prefix=trim_readout)
                    det.stream._properties_df = (det.stream.properties.rename({
                        f"median_stream_{trim_norm}_{NAME['fractional_frequency']}": freq_wn,
                        f"median_stream_{trim_readout}_{NAME['fractional_frequency']}": diss_wn})
                                                    .with_columns((pl.col(freq_wn)/pl.col(diss_wn)).alias('freq/diss'))
                                                )  

                    # Project data
                    # ------------
                    #det.targ.mag(prefix=norm_prefix, dB=False)
                    #proj_df = det.IQ_circle_diss_corr(prefix=norm_prefix,
                    #                                  circle_fit_prefix=f"const_{circle_fit_prefix}",
                    #                                  max_workers=2,
                    #                                  ex=_ex)

                    # Calc y = Qx
                    # -----------
                    tqdm.write(f"  [Detector {det_idx}] Running: {_pipeline_steps[6]}")
                    #det.targ._properties_df = det.targ.properties.with_columns((pl.col(f_0_name)/ (pl.col(high_FWHM_name)-pl.col(low_FWHM_name))).alias('FWHM_Q'),
                    #pl.col(f"{PREFIX['phase_fit']}_{trim_prefix}_{NAME['total_quality_factor']}").alias('fit_Q'))

                    #tune_routines.max_linewidth_dist_f(det, prefix=proj_prefix, Q_col='FWHM_Q', trim_window=1.5, 
                    #                                   trim_mag_prefix=mag_prefix, savgol_window=1, savgol_k=2, 
                    #                                   max_workers=savgol_workers_selector.value)
                    #tune_routines.max_linewidth_dist_f(det, prefix=proj_prefix, Q_col='fit_Q', trim_window=1.5, 
                    #                                  trim_mag_prefix=mag_prefix, savgol_window=1, savgol_k=2, 
                    #                                   max_workers=savgol_workers_selector.value)
                    #tune_routines.max_linewidth_dist_f(det, prefix=proj_prefix, Q_col='', trim_window=5, 
                    #                                   trim_mag_prefix=mag_prefix, savgol_window=1, savgol_k=2, 
                    #                                   max_workers=savgol_workers_selector.value, ex=_ex)  
                    #det.targ.linear_fit(x_col_name=NAME['fractional_frequency'], 
                    ##                    y_col_name=NAME['linewidth_shift'], 
                    #                    y_prefix=f"tail_trim_tail_shift_{proj_prefix}",
                    #                    max_workers=15,
                    #                    ex=_ex)

                    tune_routines.max_linewidth_dist_f(det, prefix=norm_prefix, Q_col='', trim_window=2.5, 
                                       trim_mag_prefix=mag_prefix, savgol_window=1, savgol_k=2, 
                                       max_workers=savgol_workers_selector.value, ex=_ex) 
                    det.targ.mag(prefix=f"tail_shift_{norm_prefix}", dB=False)
                    property_routines.fwhm(det.targ, mag_prefix=f"tail_shift_{norm_prefix}", peak=True)
                    det.frac_f(prefix='', 
                               data='targ', 
                               ref_f= norm_f_0)

                    det.targ.linear_fit(x_col_name=NAME['fractional_frequency'], 
                                        y_col_name=NAME['linewidth_shift'], 
                                        y_prefix=f"tail_trim_tail_shift_{norm_prefix}",
                                        max_workers=15,
                                        ex=_ex)
                except Exception as e:
                    print(e)
        return network

    return (dump_transform,)


@app.cell
def _(Network, analysis_cfg, pl, savgol_workers_selector, tqdm):
    def load_transform(network: Network) -> Network:
        """
        Transformations to run on Network objects loaded from pickle files
        """
        return network

        import ccatkidlib.analysis.routines.properties as property_routines
        import ccatkidlib.analysis.routines.tune as tune_routines

        _name, _prefix = network.analysis_cfg["convention"]["name"], network.analysis_cfg["convention"]["prefix"]

        _mismatch_prefix = "_".join([
                                _prefix["remove_impedance_mismatch"],
                                _prefix["rotate"],
                                _prefix["center_origin"],
                                _prefix["translate"],
                                _prefix["center_origin"],
                                _prefix["rotate"],
                                _prefix["remove_cable"],
                                _prefix["rotate"],
                                ])

        _readout_noise_prefix = "_".join([_prefix['isolate_readout_noise'],
                                          _prefix['rotate'],
                                          _mismatch_prefix])

        _trim_mismatch, _trim_readout =  (f"white_noise_{_prefix['trim']}_{_prefix['power_spectral_density']}_{_mismatch_prefix}", 
                                          f"white_noise_{_prefix['trim']}_{_prefix['power_spectral_density']}_{_readout_noise_prefix}")

        _freq_wn, _diss_wn = (f"freq_white_noise_{_name['fractional_frequency']}",
                              f"diss_white_noise_{_name['fractional_frequency']}")

        for det_idx, (det) in enumerate(tqdm(
        network.det_dict.values(),
        desc="Processing Detectors",
        total=len(network.det_dict),
        unit="Detector",)):
            property_routines.agg(det.stream, "median", col_name = _name['fractional_frequency'], prefix=_trim_mismatch)
            property_routines.agg(det.stream, "median", col_name = _name['fractional_frequency'], prefix=_trim_readout)
            det.stream._properties_df = (det.stream.properties.rename({
                f"median_stream_{_trim_mismatch}_{_name['fractional_frequency']}": _freq_wn,
                f"median_stream_{_trim_readout}_{_name['fractional_frequency']}": _diss_wn})
                                            .with_columns((pl.col(_freq_wn)/pl.col(_diss_wn)).alias('freq/diss'))
                                        )        
            det.targ.mag(prefix=_mismatch_prefix)
            det.targ.analysis_cfg = analysis_cfg

            det.targ._properties_df = det.targ.properties.with_columns((pl.col('mid_savgol0_FWHM_f')/ (pl.col('high_savgol0_FWHM_f') - pl.col('low_savgol0_FWHM_f'))).alias('FWHM_Q'),
            pl.col('phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q').alias('fit_Q'))

            tune_routines.max_linewidth_dist_f(det, prefix=_mismatch_prefix, Q_col='FWHM_Q', trim_window=1.5, trim_mag_prefix=f"{_prefix['savgol_filter']}0", savgol_window=1, savgol_k=2, max_workers=savgol_workers_selector.value)
            tune_routines.max_linewidth_dist_f(det, prefix=_mismatch_prefix, Q_col='fit_Q', trim_window=1.5, trim_mag_prefix=f"{_prefix['savgol_filter']}0", savgol_window=1, savgol_k=2, max_workers=savgol_workers_selector.value)
            tune_routines.max_linewidth_dist_f(det, prefix=_mismatch_prefix, Q_col='', trim_window=1.5, trim_mag_prefix=f"{_prefix['savgol_filter']}0", savgol_window=1, savgol_k=2, max_workers=savgol_workers_selector.value)  

        return network

    return (load_transform,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
