import marimo

__generated_with = "0.25.1"
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


@app.cell(column=1)
def _(analysis_cfg):
    # Define data transformation prefixes
    # -----------------------------------

    NAME, PREFIX = analysis_cfg['convention']['name'], analysis_cfg['convention']['prefix']

    cable_prefix = "_".join([PREFIX['remove_cable'], PREFIX['rotate']])
    trim_prefix = "_".join([PREFIX['trim_tail'], PREFIX['trim'], cable_prefix])
    circle_prefix = "_".join([PREFIX['IQ_circle_fit'], trim_prefix])
    center_prefix = "_".join([PREFIX['center_origin'], PREFIX['translate'], PREFIX['center_origin'], PREFIX['rotate'], cable_prefix])
    mismatch_prefix = "_".join([PREFIX['remove_impedance_mismatch'], PREFIX['rotate'], center_prefix])
    trim_mismatch_prefix  = "_".join(['tail', 'trim', mismatch_prefix])
    norm_prefix = "_".join([PREFIX['normalize'], PREFIX['scale'], mismatch_prefix])
    tail_prefix = "_".join([PREFIX['center_tail'], PREFIX['translate'], norm_prefix])
    trim_tail_prefix = "_".join([PREFIX['trim_tail'],PREFIX['trim'], tail_prefix])
    linear_prefix = "_".join(['linear', PREFIX['translate'], tail_prefix])
    trim_linear_prefix = "_".join(['tail', 'trim', linear_prefix])
    #trim_res_prefix = "_".join([PREFIX['trim_resonance'], PREFIX['trim'], linear_prefix])
    trim_res_prefix = "_".join(['res', 'trim', trim_tail_prefix])
    diameter_linear_prefix = "_".join(['diameter', 'scale', linear_prefix])
    trim_diameter_linear_prefix = "_".join(['tail', 'trim', diameter_linear_prefix])
    diameter_tail_prefix = "_".join(['diameter', 'scale', tail_prefix])
    trim_diameter_tail_prefix = "_".join(['tail', 'trim', diameter_tail_prefix])
    return (
        PREFIX,
        circle_prefix,
        diameter_linear_prefix,
        diameter_tail_prefix,
        linear_prefix,
        mismatch_prefix,
        norm_prefix,
        tail_prefix,
        trim_diameter_linear_prefix,
        trim_diameter_tail_prefix,
        trim_linear_prefix,
        trim_mismatch_prefix,
        trim_res_prefix,
        trim_tail_prefix,
    )


@app.cell
def transform_data(
    Network,
    analysis_cfg,
    com_to_selector,
    data_dirs,
    mo,
    root_data_dir,
    transform_button,
    viz_cfg,
):
    mo.stop(not transform_button.value)

    _data_dir, _date, _sess_id = data_dirs[0].parts[-3:]


    for _com_to in com_to_selector.value:
        network = Network(
            com_to=_com_to,
            sess_ids=_sess_id,
            date=_date,
            data_dir=_data_dir,
            root_data_dir=root_data_dir,
            analysis_cfg=analysis_cfg,
            viz_cfg=viz_cfg,
        )

        network.add_columns(
            data_cols=[
                "com_to",
                "drive",
                "sense",
                "detector_type",
                "network",
            ],
            max_workers=5,
        )
        mo.output.append(network.data)

        #ccat_pickle.multi_dump(
        #    _network,
        #    pickle_name_selector.value,
        #    num_segments=num_pickle_files_selector.value,
        #    transform=dump_transform,
        #)
    return (network,)


@app.cell
def _(
    IQ_circle_origin,
    PREFIX,
    ProcessPoolExecutor,
    circle_fit_workers_selector,
    circle_prefix,
    coupling_linewidth,
    linewidth,
    mismatch_prefix,
    network,
    norm_prefix,
    savgol_workers_selector,
    tqdm,
    trim_mismatch_prefix,
):
    _max_workers = max(
        savgol_workers_selector.value,
        circle_fit_workers_selector.value
    )

    _cm, _cb = None, None
    with ProcessPoolExecutor(max_workers=_max_workers) as _ex:
        for _det, _drive in tqdm(network.data.sort('drive', descending=True).select(['detector', 'drive']).iter_rows()):
            _det = network.det_dict[_det]
            _window = 2.5#np.linspace(2.5, 1.5, _det.targ.num_tones)

            # Center circle
            # -------------
            IQ_circle_origin(
                _det,
                normalize=True,
                normalization='linear',
                cable_delay='vna',
                data="targ",
                savgol_window=9,
                savgol_order=1,
                cable_trim_window=2.5,
                fit_trim_window=8,
                mean_points=10, 
                savgol_workers=savgol_workers_selector.value,
                circle_fit_workers=circle_fit_workers_selector.value,
                recalc=False,
                ex=_ex
                )
            _det.targ.res_IQ_trim(prefix=mismatch_prefix, mean_points=10, mag_prefix=f"{PREFIX['savgol_filter']}0", window=3, recalc=False)
            _det.targ.phase(prefix=trim_mismatch_prefix, recalc=False)
            _det.phase_fit(prefix=trim_mismatch_prefix, circle_fit_prefix=circle_prefix, nonlinear=True, normalized=False, max_workers=10, ex=_ex, recalc=False)

            _det.IQ_circle_tail(prefix=norm_prefix, data='targ', mean_points=10)
            linewidth(_det, tail_window = 2.5, res_window = 2, ex = _ex, recalc=False)
            _cm, _cb = coupling_linewidth(_det, tail_window = 2.5, res_window = 2, m=_cm, b=_cb, ex = _ex, recalc=True)
            #coupling_linewidth(_det, tail_window = 3, res_window = 2.5, m=_cm, b=_cb, ex = _ex, recalc=True)
    return


@app.cell
def _(network, opts, trim_mismatch_prefix):
    network.plot("phase", "targ", prefix="_".join(['phase_fit', trim_mismatch_prefix]), data_cols=['drive'], overlay_cols=['drive'], cmap='viridis', include=31).opts(opts.Curve(ms=1))
    return


@app.cell
def _(network, opts, tail_prefix):
    network.plot("IQ", "targ", prefix=tail_prefix, data_cols=['drive'], overlay_cols=['drive'], cmap='viridis', include=331).opts(opts.Curve(ms=1))
    return


@app.cell
def _(network, opts, trim_diameter_linear_prefix, trim_linear_prefix):
    _tone = 331
    network.plot("plot", "targ", x_dim="f", y_dim="lw", y_prefix=trim_linear_prefix, data_cols=['drive'], overlay_cols=['drive'], cmap='viridis', include=_tone).opts(opts.Curve(ms=1)) + network.plot("plot", "targ", x_dim="f", y_dim="clw", y_prefix=trim_diameter_linear_prefix, data_cols=['drive'], overlay_cols=['drive'], cmap='viridis', include=_tone).opts(opts.Curve(ms=1))
    return


@app.cell
def _(network, opts, trim_res_prefix):
    _tone = 331
    network.plot("plot", "targ", x_dim="f", y_dim="lw", y_prefix=trim_res_prefix, data_cols=['drive'], overlay_cols=['drive'], cmap='viridis', include=_tone).opts(opts.Curve(ms=1)) + network.plot("plot", "targ", x_dim="f", y_dim="clw", y_prefix=trim_res_prefix, data_cols=['drive'], overlay_cols=['drive'], cmap='viridis', include=_tone).opts(opts.Curve(ms=1))
    return


@app.cell
def _(
    network,
    opts,
    trim_diameter_linear_prefix,
    trim_diameter_tail_prefix,
    trim_linear_prefix,
    trim_tail_prefix,
):
    _tone = 27
    network.plot("plot", "targ", x_dim="lw", y_dim="lw", x_prefix=trim_tail_prefix, y_prefix=trim_linear_prefix, data_cols=['drive'], overlay_cols=['drive'], cmap='viridis', include=_tone).opts(opts.Curve(ms=1, show_grid=True)).opts(xlim=(-2, 2)) + network.plot("plot", "targ", x_dim="clw", y_dim="clw", x_prefix=trim_diameter_tail_prefix, y_prefix=trim_diameter_linear_prefix, data_cols=['drive'], overlay_cols=['drive'], cmap='viridis', include=_tone).opts(opts.Curve(ms=1, show_grid=True)).opts(xlim=(-4, 4), ylim=(-10, 2))
    return


@app.cell
def _(network, opts, trim_linear_prefix, trim_tail_prefix):
    _tone = 29

    (network.plot("plot", "targ", x_dim="lw", y_dim="lw", x_prefix=trim_tail_prefix, y_prefix=trim_linear_prefix, data_cols=['drive'], overlay_cols=['drive'], cmap='viridis', include=_tone).opts(opts.Curve(ms=1, show_grid=True)).opts(xlim=(-2, 2), ylim=(-1, 4)).opts(xlabel='y', ylabel=r'$y-y_0 = \frac{a}{1+y^2}$').opts(opts.Curve(linewidth=0)))
    return


@app.cell
def _(network, opts, trim_diameter_linear_prefix, trim_diameter_tail_prefix):
    _tone = 29

    (network.plot("plot", "targ", x_dim="clw", y_dim="clw", x_prefix=trim_diameter_tail_prefix, y_prefix=trim_diameter_linear_prefix, data_cols=['drive'], overlay_cols=['drive'], cmap='viridis', include=_tone).opts(opts.Curve(ms=1, show_grid=True)).opts(xlim=(-2, 2), ylim=(-3, 1)).opts(xlabel='y', ylabel=r'$y-y_0 = \frac{a}{1+y^2}$').opts(opts.Curve(linewidth=0)))
    return


@app.cell
def _(
    diameter_prefix,
    linear_prefix,
    network,
    opts,
    trim_diameter_prefix,
    trim_tail_prefix,
):
    _tone = 331
    network.plot("plot", "targ", x_dim="lw", y_dim="lw", x_prefix=trim_tail_prefix, y_prefix="_".join(['tail', 'trim', linear_prefix]), data_cols=['drive'], overlay_cols=['drive'], cmap='viridis', include=_tone).opts(opts.Curve(ms=1, show_grid=True)) + network.plot("plot", "targ", x_dim="clw", y_dim="clw", x_prefix=trim_diameter_prefix, y_prefix="_".join(['tail', 'trim', diameter_prefix]), data_cols=['drive'], overlay_cols=['drive'], cmap='viridis', include=_tone).opts(opts.Curve(ms=1, show_grid=True))
    return


@app.cell
def _(network):
    network.det_dict[network.data['detector'][0]]._properties_df
    return


@app.cell
def _(network):
    combined_properties = network.combine_properties(data_cols=['drive'])
    return (combined_properties,)


@app.cell
def _(combined_properties, pl):
    combined_properties.filter(pl.col('det') == 31).sort('drive')
    return


@app.cell
def _(
    combined_properties,
    pl,
    tone_selector,
    trim_diameter_linear_prefix,
    trim_linear_prefix,
    trim_mismatch_prefix,
):
    _tone = tone_selector.value
    print(_tone)
    (combined_properties.filter(pl.col('det') == _tone).with_columns((-1*pl.col("_".join(['phase_fit', trim_mismatch_prefix, 'a']))).alias('a')).hvplot.scatter(x='drive', y="a")*
    combined_properties.filter(pl.col('det') == _tone).hvplot.scatter(x='drive', y="_".join(['max', 'targ', trim_linear_prefix, 'lw']))*
     combined_properties.filter(pl.col('det') == _tone).hvplot.scatter(x='drive', y="_".join(['max', 'targ', trim_diameter_linear_prefix, 'clw']))
    ).opts(ylim=(-5, 4))
    return


@app.cell
def _(
    combined_properties,
    diameter_linear_prefix,
    pl,
    tone_selector,
    trim_diameter_linear_prefix,
    trim_mismatch_prefix,
):
    _tone = tone_selector.value
    print(_tone)
    (combined_properties.filter(pl.col('det') == _tone).with_columns((-1*pl.col("_".join(['phase_fit', trim_mismatch_prefix, 'a']))).alias('a')).hvplot.scatter(x='drive', y="a", label='fit')*
    # combined_properties.filter(pl.col('det') == _tone).hvplot.scatter(x='drive', y="_".join(['max', 'targ', 'tail', 'trim', diameter_prefix, 'clw']), label='max')*
    # combined_properties.filter(pl.col('det') == _tone).hvplot.scatter(x='drive', y="_".join(['min', 'targ', 'tail', 'trim', diameter_prefix, 'clw']), label='min')*
     combined_properties.filter(pl.col('det') == _tone).hvplot.scatter(x='drive', y="_".join(['zero', trim_diameter_linear_prefix, 'clw']), label='zero')*
      combined_properties.filter(pl.col('det') == _tone).hvplot.scatter(x='drive', y="lorentz_clw", label='lorentz')*
     combined_properties.filter(pl.col('det') == _tone).with_columns((pl.col("_".join(['zero', trim_diameter_linear_prefix, 'clw'])) - pl.col("_".join(['min', 'targ', 'tail', 'trim', diameter_linear_prefix, 'clw']))).alias('min_diff')).hvplot.scatter(x='drive', y='min_diff', label='min_diff')*
     combined_properties.filter(pl.col('det') == _tone).with_columns((pl.col("_".join(['zero', trim_diameter_linear_prefix, 'clw'])) - pl.col("_".join(['max', 'targ', 'tail', 'trim', diameter_linear_prefix, 'clw']))).alias('max_diff')).hvplot.scatter(x='drive', y='max_diff', label='max_diff')
    ).opts(ylim=(-5, 5))
    return


@app.cell
def _(
    combined_properties,
    diameter_linear_prefix,
    hv,
    pl,
    tone_selector,
    trim_diameter_linear_prefix,
    trim_mismatch_prefix,
):
    _tone = tone_selector.value
    _props = (combined_properties
    .filter(pl.col('det') == _tone)
    .with_columns((-1*pl.col("_".join(['phase_fit', trim_mismatch_prefix, 'a']))).alias('a'))
    .with_columns((pl.col("_".join(['zero', trim_diameter_linear_prefix, 'clw'])) - pl.col("_".join(['min', 'targ', 'tail', 'trim', diameter_linear_prefix, 'clw']))).alias('min_diff'),
    (pl.col("_".join(['zero', trim_diameter_linear_prefix, 'clw'])) - pl.col("_".join(['max', 'targ', 'tail', 'trim', diameter_linear_prefix, 'clw']))).alias('max_diff'))
    .with_columns(pl.when(pl.col('max_diff').abs() > pl.col('min_diff').abs())
                    .then(pl.col('max_diff'))
                    .otherwise(pl.col('min_diff'))
                    .alias('a_estimate')))

    print(_tone)
    (_props.hvplot.scatter(x='drive', y=['a', 'a_estimate', "_".join(['zero', trim_diameter_linear_prefix, 'clw']), "_".join(['max', 'targ',trim_diameter_linear_prefix, 'clw'])])*hv.HLine(0.77)
    ).opts(ylim=(-2, 3)).opts(legend_position='right')
    return


@app.cell
def _(
    combined_properties,
    diameter_linear_prefix,
    pl,
    tone_selector,
    trim_diameter_linear_prefix,
    trim_mismatch_prefix,
):
    _tone = tone_selector.value
    _props = (combined_properties
    .filter(pl.col("_".join(['phase_fit', trim_mismatch_prefix, 'Q'])) > 5000,
            pl.col("_".join(['phase_fit', trim_mismatch_prefix, 'Q'])) < 40000,
            pl.col("_".join(['phase_fit', trim_mismatch_prefix, 'Q_c'])) < 100000,                   pl.col("_".join(['phase_fit', trim_mismatch_prefix, 'Q_c'])) > 10000,                    pl.col("_".join(['phase_fit', trim_mismatch_prefix, 'Q_i'])) > 2000,                     pl.col("_".join(['phase_fit', trim_mismatch_prefix, 'Q_i'])) < 75000)
    .with_columns((-1*pl.col("_".join(['phase_fit', trim_mismatch_prefix, 'a']))).alias('a'))
    .with_columns((pl.col("_".join(['zero', trim_diameter_linear_prefix, 'clw'])) - pl.col("_".join(['min', 'targ', 'tail', 'trim', diameter_linear_prefix, 'clw']))).alias('min_diff'),
    (pl.col("_".join(['zero', trim_diameter_linear_prefix, 'clw'])) - pl.col("_".join(['max', 'targ', 'tail', 'trim', diameter_linear_prefix, 'clw']))).alias('max_diff'))
    .with_columns(pl.when(pl.col('max_diff').abs() > pl.col('min_diff').abs())
                    .then(pl.col('max_diff'))
                    .otherwise(pl.col('min_diff'))
                    .alias('a_estimate')))

    _props.hvplot.scatter(x='a',  y="_".join(['zero', trim_diameter_linear_prefix, 'clw']), c='drive', cmap='viridis').opts(data_aspect=1, alpha=0.7, ylim=(-5, 5), xlim=(-3, 3), fig_size=100, s=2)#*hv.Slope(1, 0)
    return


@app.cell
def _(
    combined_properties,
    diameter_linear_prefix,
    hv,
    pl,
    tone_selector,
    trim_diameter_linear_prefix,
    trim_mismatch_prefix,
):
    _tone = tone_selector.value
    _props = (combined_properties
    #.filter(pl.col("_".join(['phase_fit', trim_mismatch_prefix, 'Q'])) > 1000,
    #        pl.col("_".join(['phase_fit', trim_mismatch_prefix, 'Q'])) < 40000,
    #        pl.col("_".join(['phase_fit', trim_mismatch_prefix, 'Q_c'])) < 100000,          #         pl.col("_".join(['phase_fit', trim_mismatch_prefix, 'Q_c'])) > 10000,          #          pl.col("_".join(['phase_fit', trim_mismatch_prefix, 'Q_i'])) > 1000,          #           pl.col("_".join(['phase_fit', trim_mismatch_prefix, 'Q_i'])) < 75000,
    #        pl.col('drive') < 10)
    .with_columns((-1*pl.col("_".join(['phase_fit', trim_mismatch_prefix, 'a']))).alias('a'))
    .with_columns((pl.col("_".join(['zero', trim_diameter_linear_prefix, 'clw'])) - pl.col("_".join(['min', 'targ', 'tail', 'trim', diameter_linear_prefix, 'clw']))).alias('min_diff'),
    (pl.col("_".join(['zero', trim_diameter_linear_prefix, 'clw'])) - pl.col("_".join(['max', 'targ', 'tail', 'trim', diameter_linear_prefix, 'clw']))).alias('max_diff'))
    .with_columns(pl.when(pl.col('max_diff').abs() > pl.col('min_diff').abs())
                    .then(pl.col('max_diff'))
                    .otherwise(pl.col('min_diff'))
                    .alias('a_estimate'))
    .with_columns((pl.col("_".join(['zero', trim_diameter_linear_prefix, 'clw'])) - pl.col('a')).alias("a_diff")))

    (_props.hvplot.scatter(x='a',  y="lorentz_clw", c='drive', cmap='viridis').opts(alpha=0.25, xlim=(-3, 3), ylim=(-3, 4), fig_size=100, s=2)*hv.VLine(0.77)*hv.Slope(1, 0)).opts(xlabel='Phase Fit a', ylabel=r'$y-y_0$ a')
    return


@app.cell
def _(network, tail_prefix):
    _tone = 31
    network.plot("plot", "targ", x_dim='f', y_dim="diameter", y_prefix=tail_prefix, data_cols=['drive'], overlay_cols=['drive'], cmap='viridis', include=_tone).opts(ylim=(0, 0.5))
    return


@app.cell
def _(mo, network):
    tone_selector = mo.ui.slider(start=0, stop=network.det_dict[network.data['detector'][0]].targ.num_tones, step=1, label='Tone')
    return (tone_selector,)


@app.cell
def _(tone_selector, wigglystuff):
    wigglystuff.FloatingPanel(tone_selector, corner='bottom-left')
    return


@app.cell
def _(network, pl, tone_selector, trim_res_prefix, trim_tail_prefix):
    _drive=30
    (network.det_dict[network.data.filter(pl.col('drive')==_drive).item(0,0)].targ.plot(x_dim="f", y_dim="lw", y_prefix=trim_tail_prefix, include=tone_selector.value)*
    network.det_dict[network.data.filter(pl.col('drive')==_drive).item(0,0)].targ.plot(x_dim="f", y_dim="lw", y_prefix=trim_res_prefix, include=tone_selector.value)
    *network.det_dict[network.data.filter(pl.col('drive')==_drive).item(0,0)].targ.plot(x_dim="f", y_dim="lw", y_prefix="_".join(['linear_fit', trim_tail_prefix]), include=tone_selector.value))
    return


@app.cell
def nonlin(diameter_prefix, network, pl, tone_selector, trim_diameter_prefix):
    _drive=4
    (network.det_dict[network.data.filter(pl.col('drive')==30).item(0,0)].targ.plot(x_dim="clw", y_dim="clw", x_prefix=trim_diameter_prefix, y_prefix=diameter_prefix, include=tone_selector.value)*
    network.det_dict[network.data.filter(pl.col('drive')==_drive).item(0,0)].targ.plot(x_dim="clw", y_dim="clw", x_prefix=trim_diameter_prefix, y_prefix=diameter_prefix, include=tone_selector.value)).opts(xlim=(-5, 5))
    return


@app.cell
def _(ccat_df, diameter_prefix, network, pl, trim_diameter_prefix):
    _det = network.det_dict[network.data.filter(pl.col('drive') == 15)['detector'].item()]

    _y_col = "_".join([trim_diameter_prefix, "clw"])
    _y_sub_col = "_".join([diameter_prefix, "clw"])
    _y_z_col = "_".join(["zero", diameter_prefix, "clw"])

    _y_df = (_det.targ.get_data(col_name=_y_col, strict=True)
                .unpivot(
                    variable_name="det",
                    value_name=_y_col,
                        )
                    .with_columns(pl.col("det").str.strip_prefix(f"{_y_col}_").cast(pl.Int32))
    )

    _y_sub_df = (_det.targ.get_data(col_name=_y_sub_col, strict=True)
                .unpivot(variable_name='tmp', value_name=_y_sub_col)
                .drop('tmp'))

    _zero_dat = (pl.concat([_y_df, _y_sub_df], how='horizontal').lazy()
                                                            .filter(pl.col(_y_col) > 0)
                                                            .filter(pl.col(_y_col) == pl.col(_y_col).min().over('det'))
                                                            .collect()).select('det', _y_sub_col).rename({_y_sub_col:_y_z_col})
    _shared_cols = [_y_z_col] if _y_z_col in _det.targ._properties_df.schema else []
    _det.targ._properties_df = ccat_df.coalesce_join(
        _det.targ._properties_df,
        _zero_dat,
        "det",
        _shared_cols,
    )

    _det.properties
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

    return (pl,)


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
    from ccatkidlib.analysis.routines.common import IQ_circle_origin

    return IQ_circle_origin, Network, ccat_df, ccat_io


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
    return hv, opts


@app.cell
def _():
    import wigglystuff

    return (wigglystuff,)


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


@app.cell(column=3)
def dump_transform(
    PREFIX,
    diameter_linear_prefix,
    diameter_tail_prefix,
    linear_prefix,
    tail_prefix,
    trim_diameter_linear_prefix,
    trim_diameter_tail_prefix,
    zero_value,
):
    def linewidth(det, tail_window = 1.75, res_window = 1.5, ex = None, recalc=False):
        det.targ.linewidth(prefix=tail_prefix, recalc=recalc)
        det.targ.linewidth_shift(prefix=tail_prefix, 
                                 coupling_linewidth=False,
                                 res_trim_window=res_window, 
                                 tail_trim_window=tail_window,
                                 mean_points=10,
                                 mag_prefix = f"{PREFIX['savgol_filter']}0", 
                                 max_workers=5,
                                 ex=ex,
                                 recalc=recalc)
        det.targ.res_trim(col_name='lw', prefix=[linear_prefix, tail_prefix], mean_points=10, window=2, invert=False, mag_prefix=f"{PREFIX['savgol_filter']}0", recalc=True)
        det.targ.agg(operation='max', col_name='lw', prefix="_".join(['tail', 'trim', linear_prefix]), recalc=recalc)
        det.targ.agg(operation='min', col_name='lw', prefix="_".join(['tail', 'trim', linear_prefix]), recalc=recalc)
        zero_value(det, tail_prefix, linear_prefix, 'lw')


    def coupling_linewidth(det,tail_window = 2.5, res_window = 1.75, m = None, b = None, ex = None, recalc=False):
        det.targ.IQ_circle_diameter(prefix=tail_prefix, recalc=recalc)
        det.targ.coupling_linewidth(prefix=tail_prefix, recalc=recalc)
        _, m, b = det.targ.linewidth_shift(prefix=tail_prefix, 
                                           coupling_linewidth=True,
                                           res_trim_window=res_window, 
                                           tail_trim_window=tail_window,
                                           m=m,
                                           b=b,
                                           mean_points=10,
                                           mag_prefix = f"{PREFIX['savgol_filter']}0", 
                                           max_workers=5,
                                           ex=ex,
                                           return_fit_params=True,
                                           recalc=recalc)
        diameter = det.targ.get_data("_".join([tail_prefix, 'diameter'])).to_numpy().T
        det.targ.scale(col_name='clw', prefix=[linear_prefix, tail_prefix], scale=diameter, name='diameter', recalc=recalc)
        det.targ.res_trim(col_name='clw', prefix=[diameter_linear_prefix, diameter_tail_prefix], mean_points=10, window=2, invert=False, mag_prefix=f"{PREFIX['savgol_filter']}0", recalc=True)
        y = det.targ.get_data(["_".join([trim_diameter_tail_prefix, 'clw'])]).to_numpy().T
        det.targ.scale(col_name='clw', prefix=trim_diameter_linear_prefix, scale=1+y**2, name='lorentz', recalc=recalc)
        det.targ.agg(operation='max', col_name='clw', prefix="_".join(['tail', 'trim', diameter_linear_prefix]), recalc=recalc)
        det.targ.agg(operation='min', col_name='clw', prefix="_".join(['tail', 'trim', diameter_linear_prefix]), recalc=recalc)
        zero_value(det, trim_diameter_tail_prefix, trim_diameter_linear_prefix, 'clw')
        return m, b

    return coupling_linewidth, linewidth


@app.cell
def _(ccat_df, pl):
    def zero_value(det, y_prefix, linear_y_prefix, lw):
        y_col = "_".join([y_prefix, lw])
        y_sub_col = "_".join([linear_y_prefix, lw])
        y_z_col = "_".join(["zero", linear_y_prefix, lw])
        lorentz_col = "_".join(["lorentz", lw])

        y_df = (det.targ.get_data(col_name=y_col, strict=True)
                    .unpivot(
                        variable_name="det",
                        value_name=y_col,
                            )
                        .with_columns(pl.col("det").str.strip_prefix(f"{y_col}_").cast(pl.Int32))
        )

        y_sub_df = (det.targ.get_data(col_name=y_sub_col, strict=True)
                    .unpivot(variable_name='tmp', value_name=y_sub_col)
                    .drop('tmp'))

        zero_dat = (pl.concat([y_df, y_sub_df], how='horizontal').lazy()
                                                                .filter(pl.col(y_col) > 0)
                                                                .filter(pl.col(y_col) == pl.col(y_col).min().over('det'))
                                                                .with_columns((pl.col(y_sub_col)*(1 + pl.col(y_col)**2)).alias(lorentz_col))
                                                                .collect()).select('det', y_sub_col, lorentz_col).rename({y_sub_col:y_z_col})
        shared_cols = [lorentz_col, y_z_col] if lorentz_col in det.targ._properties_df.schema else []
        det.targ._properties_df = ccat_df.coalesce_join(
            det.targ._properties_df,
            zero_dat,
            on="det",
            shared_cols=shared_cols,
        )

    return (zero_value,)


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

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
