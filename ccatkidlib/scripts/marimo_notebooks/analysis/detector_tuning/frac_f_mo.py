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
    circle_fit_workers_selector,
    com_to_selector,
    mo,
    num_pickle_files_selector,
    pickle_name_selector,
    savgol_workers_selector,
    transform_button,
):
    mo.md(rf"""
    ### Transform Data

    The typical data analysis procedure involves loading the data into **Network** objects, transforming/analyzing the data using the underlying **Detector** objects, and pickling the processed data. One often wants to only analyze the data for a subset of RFSoC drones which can be selected below. Many of the built-in data analysis methods can be multi-processed and the maximum number of CPU cores to use can be specified below. Finally, the name of the pickle file storing the processed data can be chosen below. 


    {
        mo.vstack(
            [
                mo.hstack(
                    [
                        com_to_selector,
                        savgol_workers_selector,
                        circle_fit_workers_selector,
                        num_pickle_files_selector,
                        pickle_name_selector,
                    ],
                    widths=[2, 2, 2 ,2,  4],
                ),
                mo.hstack([transform_button]),
            ]
        )
    }
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

    proj_prefix = '_'.join([PREFIX['dissipation_correction'],
                            norm_prefix])

    mag_prefix = f"{PREFIX['savgol_filter']}0"
    off_res_prefix = 'off_res'
    off_res_shift_prefix = '_'.join([off_res_prefix,
                                     PREFIX['translate']])
    off_res_trim_prefix = '_'.join([PREFIX['trim_tail'],
                             PREFIX['trim'],
                             off_res_shift_prefix])

    # Define property names
    # ---------------------
    scale_name = '_'.join([PREFIX['remove_impedance_mismatch'], 
                           cable_prefix,
                           NAME['magnitude']])

    f_0_name = '_'.join([PREFIX['middle_frequency_point'],
                        f"{PREFIX['savgol_filter']}0", 
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
    return (
        NAME,
        PREFIX,
        R_name,
        cable_prefix,
        circle_fit_prefix,
        f_0_name,
        high_FWHM_name,
        low_FWHM_name,
        mag_prefix,
        mismatch_prefix,
        norm_prefix,
        off_res_prefix,
        off_res_shift_prefix,
        off_res_trim_prefix,
        proj_prefix,
        scale_name,
    )


@app.cell
def transform_data(
    Network,
    analysis_cfg,
    ccat_pickle,
    com_to_selector,
    data_dirs,
    dump_transform,
    mo,
    num_pickle_files_selector,
    pickle_name_selector,
    root_data_dir,
    transform_button,
    viz_cfg,
):
    mo.stop(not transform_button.value)

    _data_dir, _date, _sess_id = data_dirs[-1].parts[-3:]

    for _com_to in com_to_selector.value:
        _network = Network(
            com_to=_com_to,
            sess_ids=_sess_id,
            date=_date,
            data_dir=_data_dir,
            root_data_dir=root_data_dir,
            analysis_cfg=analysis_cfg,
            viz_cfg=viz_cfg,
            include_streams=False,
            include_targs=True
        )

        _network.add_columns(
            data_cols=[
                "com_to",
                "drive",
                "sense",
                "detector_type",
                "network",
            ],
            max_workers=1,
        )

        _network.data = _network.data.sort('drive', 'timestamp', descending=True).gather_every(2)

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
    network = networks['2.1']
    return (network,)


@app.cell
def _(network):
    network.data
    return


@app.cell
def _(network):
    list(network.det_dict.values())[0].properties
    return


@app.cell
def _(
    NAME,
    PREFIX,
    R_name,
    f_0_name,
    high_FWHM_name,
    low_FWHM_name,
    network,
    norm_prefix,
    off_res_trim_prefix,
    pl,
    proj_prefix,
    tone_selector,
):
    _mismatch_mag = f"{off_res_trim_prefix}_{norm_prefix}_{NAME['magnitude']}"
    _proj_mag = f"{off_res_trim_prefix}_{proj_prefix}_{NAME['magnitude']}"
    _mismatch_x = f"{PREFIX['linear']}_{norm_prefix}_{NAME['fractional_frequency']}"
    _proj_x = f"{PREFIX['linear']}_{proj_prefix}_{NAME['fractional_frequency']}"

    _names = ['f', 'x', 'mismatch_x', 'proj_x', 'mismatch_mag', 'proj_mag']

    combined_df = None
    for _drive, _det in network.data['drive', 'detector'].group_by('drive', maintain_order=True).agg(pl.col('detector').last()).iter_rows():
        _det = network.det_dict[_det]
        _R, _f_0, _f_low, _f_high = _det.get_properties([R_name, f_0_name, low_FWHM_name, high_FWHM_name],
                                                        include=tone_selector.value, strict=True).to_numpy()[0][1:]
        _fwhm = _f_high - _f_low

        _df = _det.targ.get_data(['f', 
                                  'ff',
                                  _mismatch_x,
                                  _proj_x,
                                  _mismatch_mag,
                                  _proj_mag], 
                                  include=tone_selector.value,
                                  strict=True)
        _df = (_df.rename({_col:_name for _col, _name in zip(_df.columns, _names)})
                   .filter(pl.col('mismatch_mag').is_not_null())
                   .with_columns(pl.lit(_drive).alias('drive'))
                   .with_columns( (pl.col('mismatch_mag')**2).alias('corr_mismatch_mag')       )
                   .with_columns( (pl.col('proj_mag')**2).alias('corr_proj_mag')       )
                   #.with_columns(((pl.col('x')-pl.col('proj_x'))*1e6).alias('diff'))
                   .with_columns(((pl.lit(_f_0)- pl.col('f')/(pl.col('proj_x') + 1))/1e6).alias('diff'))
              )
        _markers=['o']*_df.height
        _markers[0], _markers[-1] = '>','<'
        _df = (_df.with_columns(pl.Series('markers', _markers))
                  .with_columns(pl.when(pl.col('f').is_in([_f_low, _f_0, _f_high]))
                                  .then(pl.lit('d'))
                                  .otherwise('markers').alias('markers')))

        combined_df = _df if combined_df is None else pl.concat([combined_df, _df], how='vertical')
    combined_df = combined_df.sort('diff')
    return (combined_df,)


@app.cell
def _(network, norm_prefix, proj_prefix, tone_selector):
    _, IQ_df = network.plot('IQ', 
                            'targ', 
                            prefix=norm_prefix,
                            include=tone_selector.value, 
                            data_cols=['drive'], 
                            save_fig=False, 
                            return_df=True,
                            grouping='groupby')

    _, proj_IQ_df = network.plot('IQ', 
                            'targ', 
                            prefix=proj_prefix, 
                            include=tone_selector.value, 
                            data_cols=['drive'], 
                            save_fig=False, 
                            return_df=True,
                            grouping='groupby')
    return IQ_df, proj_IQ_df


@app.cell(disabled=True)
def _(combined_df, mismatch_prefix, mo, network, pl, tone_selector, viz_utils):
    _cmap = viz_utils.cycle_cmap(cmap='viridis', num_colors=network.data.height)

    _plot = combined_df.with_columns(((pl.col('x') - pl.col('proj_x'))*1e6).alias('diff')).hvplot.scatter(x='corr_mismatch_mag', y='diff', by='drive', s=5, color=_cmap).opts(aspect=1, fig_size=100, show_grid=True, ylim=(-200, 200), ylabel='$x_{measured} - x_{linear}$ [ppm]', xlabel=r'$\frac{Q_{r}}{R^2}\cdot |z-z_{off}|^2 \propto I^2$', show_legend=False)

    _IQ_plot = network.plot('IQ', 'targ', prefix=f'{mismatch_prefix}', include=tone_selector.value, data_cols=['drive'], overlay_cols=['drive'], save_fig=False, cmap='viridis')

    mo.vstack([tone_selector, mo.hstack([_plot, _IQ_plot])])
    return


@app.cell
def _(combined_df):
    combined_df
    return


@app.cell
def _(
    IQ_df,
    combined_df,
    drive_selector,
    hv,
    mo,
    network,
    np,
    opts,
    pl,
    proj_IQ_df,
    tone_selector,
    viz_utils,
):
    hv.extension('matplotlib')
    _drives = network.data['drive']
    _cmap = viz_utils.cycle_cmap(cmap='viridis', num_colors=len(_drives))
    _colors = np.array(_cmap.values)

    _plot_proj_data = combined_df.filter(pl.col('drive') == drive_selector.value)

    # Plot each marker type separately
    _plot_start = _plot_proj_data.filter(pl.col('markers') == '>').hvplot.scatter(
        x='corr_proj_mag', y='diff', s=50**2, marker='>',
        color=_colors[_drives.to_numpy() == drive_selector.value][0]
    )

    _plot_middle = _plot_proj_data.filter(pl.col('markers') == 'o').hvplot.scatter(
        x='corr_proj_mag', y='diff', s=10, marker='o',
        color=_colors[_drives.to_numpy() == drive_selector.value][0]
    )

    _plot_fwhm = _plot_proj_data.filter(pl.col('markers') == 'd').hvplot.scatter(
        x='corr_proj_mag', y='diff', s=50**2, marker='d',
        color=_colors[_drives.to_numpy() == drive_selector.value][0]
    )

    _plot_end = _plot_proj_data.filter(pl.col('markers') == '<').hvplot.scatter(
        x='corr_proj_mag', y='diff', s=50**2, marker='<',
        color=_colors[_drives.to_numpy() == drive_selector.value][0]
    )

    # Overlay all three
    _plot_proj = (_plot_start * _plot_middle * _plot_fwhm * _plot_end).opts(
        aspect=1, 
        fig_size=200, 
        show_grid=True, 
        ylim=(-0.03, 0.03), 
        ylabel='$f_{measured} - f_{linear}$ [MHz]', 
        xlabel=r'$ |z-z_{off}|^2 \propto I^2$',
        show_legend=False,
        logy=False,
        xlim=(0, combined_df.select(pl.col('corr_proj_mag').max()).item())
    ).opts(opts.Scatter(show_grid=True))

    _IQ_plot = (proj_IQ_df.filter(pl.col('drive') == drive_selector.value)
                     .hvplot.scatter(x='I', 
                                     y='Q', 
                                     s=20, 
                                     color='k')
                    .opts(data_aspect=1, 
                          fig_size=200, 
                          show_grid=True, 
                          ylabel='I [ADU]', 
                          xlabel='Q [ADU]',
                          show_legend=False,
                         )
               *IQ_df.filter(pl.col('drive') == drive_selector.value)
                     .hvplot.scatter(x='I', 
                                     y='Q', 
                                     s=20, 
                                     color=_colors[_drives.to_numpy() == drive_selector.value][0])
                    .opts(data_aspect=1, 
                          fig_size=200, 
                          show_grid=True, 
                          ylabel='I [ADU]', 
                          xlabel='Q [ADU]',
                          show_legend=False,
                         )
               )

    mo.vstack([tone_selector, drive_selector, mo.hstack([_plot_proj, _IQ_plot])])
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

    return Path, json, mo, os, time, tqdm


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
    import ccatkidlib.analysis.viz.viz_utils as viz_utils
    import ccatkidlib.analysis.utils.dataframe as ccat_df
    import ccatkidlib.analysis.routines.common as routines
    import ccatkidlib.analysis.routines.properties as property_routines

    from ccatkidlib.rfsoc.rfsoc_daq import R
    from ccatkidlib.analysis.core.detector import Detector
    from ccatkidlib.analysis.core.network import Network

    return (
        Network,
        ccat_io,
        ccat_pickle,
        property_routines,
        routines,
        viz_utils,
    )


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
    # Create UI elements for transforming data
    # ----------------------------------------

    # Create selector for max number of CPU cores to use
    savgol_workers_selector = mo.ui.number(
        start=1,
        stop=os.cpu_count(),
        step=1,
        label="Savgol Filter Max Workers",
        value=os.cpu_count() // 6,
        full_width=True,
    )

    circle_fit_workers_selector = mo.ui.number(
        start=1,
        stop=os.cpu_count(),
        step=1,
        label="Circle Fit Max Workers",
        value=os.cpu_count() // 3,
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
        circle_fit_workers_selector,
        num_pickle_files_selector,
        pickle_name_selector,
        savgol_workers_selector,
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
def _(mo, network):
    tone_selector = mo.ui.slider(start=0, stop=list(network.det_dict.values())[0].targ.properties.height, step=1, value=0, label='Select tone', full_width=True, debounce=True)

    _drives = network.data['drive'].sort()
    drive_selector = mo.ui.slider(start=_drives.min(), stop=_drives.max(), step=1, value=_drives[0], label='Select Drive Attenuation [dB]', full_width=True, debounce=True)
    return drive_selector, tone_selector


@app.cell
def _():
    return


@app.cell(column=3)
def _(
    Network,
    PREFIX,
    ProcessPoolExecutor,
    R_name,
    cable_prefix,
    circle_fit_prefix,
    circle_fit_workers_selector,
    f_0_name,
    mag_prefix,
    mismatch_prefix,
    norm_prefix,
    off_res_prefix,
    off_res_shift_prefix,
    pl,
    proj_prefix,
    property_routines,
    routines,
    savgol_workers_selector,
    scale_name,
    time,
    tqdm,
):
    def dump_transform(network: Network) -> Network:
        """
        Transformations to run on Network objects before dumping to pickle files
        """

        # Get list of Detector objects and max_workers for multiprocessing
        # ----------------------------------------------------------------
        max_workers = max([savgol_workers_selector.value, circle_fit_workers_selector.value])
        dets = list(network.det_dict.values())

        # Transform data
        # --------------
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            with tqdm(range(len(dets)), desc = f'Transforming Data') as pbar:
                for i in pbar:
                    det = dets[i]

                    # Center IQ circle at origin
                    # --------------------------
                    pbar.set_postfix_str('Centering IQ Circle...')
                    start_time = time.perf_counter_ns()

                    routines.IQ_circle_center(
                        det,
                        data="targ",
                        savgol_window=9,
                        savgol_order=1,
                        trim_window=4,
                        trim_mean_points=10,
                        mismatch_mean_points=10,
                        savgol_workers=savgol_workers_selector.value,
                        circle_fit_workers=circle_fit_workers_selector.value,
                        normalize=False,
                        ex=ex
                    )

                    ex_time = time.perf_counter_ns() - start_time
                    tqdm.write(f'IQ Circle Center Execution Time: {ex_time/1e9}')

                    # Remove gain
                    # -----------
                    pbar.set_postfix_str('Removing Cable Gain...')
                    start_time = time.perf_counter_ns()

                    _scale = property_routines.mismatch_dist(det.targ, prefix=cable_prefix).to_numpy().T[1]
                    det.targ.IQ_scale(prefix=mismatch_prefix, name=PREFIX['normalize'], scale=1/_scale)

                    ex_time = time.perf_counter_ns() - start_time
                    tqdm.write(f'Cable Gain Removal Execution Time: {ex_time/1e9}')

                    # Project data onto IQ circle with no dissipation shifts
                    # ------------------------------------------------------
                    pbar.set_postfix_str('Correcting Dissipative Shifts...')
                    start_time = time.perf_counter_ns()

                    det._properties_df = det.properties.with_columns((pl.col(R_name)/pl.col(scale_name)).alias(f"{PREFIX['normalize']}_{R_name}"))
                    det.targ.mag(prefix=norm_prefix, dB=False)
                    proj_df = det.IQ_circle_diss_corr(prefix=norm_prefix,
                                                      circle_fit_prefix=f"{PREFIX['normalize']}_{circle_fit_prefix}",
                                                      max_workers=2,
                                                      ex=ex)

                    ex_time = time.perf_counter_ns() - start_time
                    tqdm.write(f'Dissipation Correction Execution Time: {ex_time/1e9}')


                    # Calculate linear and nonlinear fractional frequency shifts
                    # ----------------------------------------------------------
                    pbar.set_postfix_str('Calculating Fractional Frequency Shifts...')
                    start_time = time.perf_counter_ns()

                    det.frac_f(prefix='', 
                               data='targ', 
                               ref_f= f_0_name)

                    det.linear_frac_f(prefix=[norm_prefix, proj_prefix], 
                                      circle_fit_prefix=f"{PREFIX['normalize']}_{circle_fit_prefix}", 
                                      mag_prefix=mag_prefix,
                                      max_workers=2,
                                      ex=ex)

                    ex_time = time.perf_counter_ns() - start_time
                    tqdm.write(f'Fractional Frequency Shift Calculation Execution Time: {ex_time/1e9}')


                    # Calculate off resonance distance
                    # --------------------------------
                    pbar.set_postfix_str('Calculating Off Resonance Distances...')
                    start_time = time.perf_counter_ns()

                    shift_I = det.get_properties(col_name=f"{PREFIX['normalize']}_{R_name}").to_numpy().T[1]
                    det.targ.IQ_shift(prefix=[norm_prefix, proj_prefix], 
                                      shift_I = shift_I, 
                                      name=off_res_prefix)
                    det.IQ_trim(prefix=[f"{off_res_shift_prefix}_{norm_prefix}",
                                        f"{off_res_shift_prefix}_{proj_prefix}"], 
                                window=4, 
                                mag_prefix=mag_prefix)
                    det.targ.mag(prefix=[f"{PREFIX['trim_tail']}_{PREFIX['trim']}_{off_res_shift_prefix}_{norm_prefix}",
                                         f"{PREFIX['trim_tail']}_{PREFIX['trim']}_{off_res_shift_prefix}_{proj_prefix}"], 
                                 dB=False)

                    ex_time = time.perf_counter_ns() - start_time
                    tqdm.write(f'Off Resonance Distance Calculation Execution Time: {ex_time/1e9}')
        return network

    return (dump_transform,)


@app.cell
def _(Network):
    def load_transform(network: Network) -> Network:
        """
        Transformations to run on Network objects loaded from pickle files
        """
        return network

    return (load_transform,)


if __name__ == "__main__":
    app.run()
