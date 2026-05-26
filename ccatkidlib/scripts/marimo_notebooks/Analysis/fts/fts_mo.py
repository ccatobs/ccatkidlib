# /// script
# dependencies = [
#     "holoviews==1.22.1",
#     "hvplot==0.12.2",
#     "marimo>=0.22.0",
#     "numpy==2.0.2",
#     "panel==1.8.10",
#     "polars==1.39.3",
#     "tqdm==4.67.1",
# ]
# [tool.marimo.venv]
# path = "/opt/conda"
# ///

import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns", layout_file="layouts/fts_mo.slides.json")


@app.cell(column=0, hide_code=True)
def _(analysis_cfg_browser, cfg_editor, data_browser, data_desc, mo):
    mo.md(rf"""
    ### Select Data to Load

    {mo.hstack([mo.vstack([analysis_cfg_browser, data_browser, data_desc]), cfg_editor])}
    """)
    return


@app.cell(hide_code=True)
def _(
    col_name_selector,
    com_to_selector,
    load_pickle_button,
    mo,
    optical_freq_range,
    pickle_name_selector,
    pickle_tabs,
    prefix_selector,
):
    mo.md(rf"""
    ### Load Pickled Data

    {mo.vstack([mo.hstack([com_to_selector, pickle_name_selector], widths=[0.5, 1]), mo.hstack([col_name_selector, prefix_selector], widths=[0.5, 1]), optical_freq_range, pickle_tabs, load_pickle_button])}
    """)
    return


@app.cell(hide_code=True)
def _(detector_map_browser, load_det_map_button, mo):
    mo.md(rf"""
    ### Load Detector Map

    {mo.vstack([detector_map_browser, load_det_map_button])}
    """)
    return


@app.cell(hide_code=True)
def _(
    SN_slider,
    band_edge_threshold,
    center_freq_selector,
    color_bar_range,
    color_col_selector,
    default_color_column,
    det_map,
    det_map_cb_title,
    det_map_fig_size,
    det_map_title,
    log_cb,
    mo,
    plot_network_selector,
    timestamp_selector,
):
    mo.md(rf"""
    ### Plot FTS Spatial Response

    {
        mo.vstack(
            [
                mo.hstack(
                    [
                        timestamp_selector,
                        plot_network_selector,
                        center_freq_selector,
                    ],
                    widths=[1, 1, 1],
                ),
                mo.hstack(
                    [default_color_column, color_col_selector], widths=[1, 1]
                ),
                mo.hstack([log_cb, color_bar_range], widths=[1, 4]),
                mo.hstack([band_edge_threshold, SN_slider]),
                mo.hstack(
                    [det_map_title, det_map_cb_title, det_map_fig_size],
                    widths=[1, 1, 1],
                ),
                det_map,
            ]
        )
    }
    """)
    return


@app.cell
def _(spec_df):
    spec_df.write_parquet('/home/jovyan/notebooks/Darshan/Al1_response_new.parquet')
    return


@app.cell
def _():
    return


@app.cell(column=1)
def _(
    com_to_col_name_selector,
    com_to_optical_freq_range,
    com_to_prefix_selector,
    com_to_selector,
    load_pickle_button,
    load_pickle_transforms,
    mo,
    pickle,
    pickle_selector,
    pl,
    tqdm,
):
    mo.stop(not load_pickle_button.value)

    networks = {}
    for _i, _com_to in tqdm(enumerate(com_to_selector.value)):
        _network = None
        for _file in tqdm(pickle_selector.value[_com_to]):
            with open(_file, "rb") as _f:
                _sub_network = pickle.load(_f)
                if _network is None:
                    _network = _sub_network
                else:
                    # Need to combine DataFrames and dictionaries with mappings to Detector objects
                    _network.data = pl.concat(
                        [_network.data, _sub_network.data], how="vertical"
                    )
                    _network.det_dict = _network.det_dict | _sub_network.det_dict
        load_pickle_transforms(
            _network,
            com_to_col_name_selector[_com_to].value[0],
            prefix=com_to_prefix_selector.value[_com_to],
            freq_thresholds=[
                _freq * 1e9 for _freq in com_to_optical_freq_range[_com_to].value
            ],
        )
        networks[_com_to] = _network
    return (networks,)


@app.cell
def _(det_map_df):
    det_map_df
    return


@app.cell
def _(
    color_bar_range,
    color_col_selector,
    colors,
    det_map_cb_title,
    det_map_df,
    det_map_fig_size,
    det_map_title,
    log_cb,
    mo,
    plt,
):
    _norm = colors.LogNorm if log_cb.value else colors.Normalize

    plt.figure(
        clear=True, figsize=(det_map_fig_size.value, det_map_fig_size.value)
    )
    plt.scatter(
        x=det_map_df["x_0"],
        y=det_map_df["y_0"],
        c=det_map_df[color_col_selector.value],
        s=20,
        cmap="viridis",
        norm=_norm(
            vmin=color_bar_range.value[0],
            vmax=color_bar_range.value[1],
        ),
    )
    plt.xlabel("Beam Mapper X Position [mm]")
    plt.ylabel("Beam Mapper Y Position [mm]")
    plt.title(det_map_title.value)

    _ax = plt.gca()
    _ax.set_aspect("equal")
    _ax.invert_xaxis()
    plt.colorbar(ax=_ax, label=det_map_cb_title.value)


    det_map = mo.ui.matplotlib(_ax, debounce=True)
    return (det_map,)


@app.cell
def _(avg_spec_plot):
    avg_spec_plot
    return


@app.cell(hide_code=True)
def _(fts_com_to_selector, fts_det_selector, indiv_spec_plot, mo):
    mo.md(rf"""
    ### Individual Detector Spectral Response

    {mo.vstack([mo.hstack([fts_com_to_selector, fts_det_selector]), indiv_spec_plot])}
    """)
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

    return Path, json, mo, os, pickle, tqdm


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
    import ccatkidlib.analysis.utils.pair as ccat_pair
    import ccatkidlib.analysis.utils.dataframe as ccat_df

    from ccatkidlib.rfsoc.rfsoc_daq import R
    from ccatkidlib.analysis.core.detector import Detector
    from ccatkidlib.analysis.core.network import Network

    return ccat_df, ccat_io


@app.cell
def _():
    # Multiprocessing
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    mp.set_start_method("spawn", force=True)
    return


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
    import matplotlib.colors as colors

    return (colors,)


@app.cell(hide_code=True)
def _(Path, mo, os):
    HOME_DIR = Path(os.environ["HOME"])
    analysis_cfg_browser = mo.ui.file_browser(
        initial_path=HOME_DIR,
        filetypes=[".yaml"],
        multiple=False,
        ignore_empty_dirs=True,
        label="Select analysis configuration file...",
    )
    return HOME_DIR, analysis_cfg_browser


@app.cell(hide_code=True)
def _(analysis_cfg_browser, ccat_io, json, mo):
    _editor_height = 750
    analysis_cfg, viz_cfg = {}, {}
    if _browser_value := analysis_cfg_browser.value:
        analysis_cfg_path = _browser_value[0].path
        analysis_cfg, viz_cfg = ccat_io.load_config(cfg_path=analysis_cfg_path)

    cfg_editor = mo.ui.code_editor(
        value=json.dumps(analysis_cfg, indent=4) if analysis_cfg else "",
        disabled=True,
        min_height=_editor_height,
        max_height=_editor_height,
        placeholder="Configuration file contents will display here once a valid file is selected!",
    )
    return analysis_cfg, cfg_editor


@app.cell(hide_code=True)
def _(HOME_DIR, analysis_cfg, mo):
    root_data_dir = (
        analysis_cfg["file_paths"]["root_data_dir"] if analysis_cfg else HOME_DIR
    )
    data_browser = mo.ui.file_browser(
        initial_path=root_data_dir,
        selection_mode="directory",
        multiple=False,
        restrict_navigation=True,
        ignore_empty_dirs=True,
        label="Select data directory(ies)...",
    )
    return data_browser, root_data_dir


@app.cell(hide_code=True)
def _(ccat_io, data_browser):
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


@app.cell(hide_code=True)
def _(data_browser, data_dirs, mo):
    mo.stop(not data_browser.value)

    _com_tos = [set()] * len(data_dirs)
    for _i, _data_dir in enumerate(data_dirs):
        _drones = [
            _dir.name.split("D")
            for _dir in (_data_dir / "config").iterdir()
            if _dir.is_dir()
        ]
        _com_tos[_i] = set([f"{_drone[0][1:]}.{_drone[1]}" for _drone in _drones])
    _all_com_tos = sorted(list(set.intersection(*_com_tos)))
    com_to_selector = mo.ui.multiselect(
        _all_com_tos,
        value=_all_com_tos[0:1],
        label="Select Drone(s)...",
    )
    return (com_to_selector,)


@app.cell(hide_code=True)
def _(mo):
    pickle_name_selector = mo.ui.text(
        value="fts",
        debounce=True,
        label="Pickle File Name",
        full_width=True,
    )
    return (pickle_name_selector,)


@app.cell(hide_code=True)
def _(mo):
    col_name_selector = mo.ui.multiselect(
        options=["f", "phase"],
        full_width=True,
        value="f",
        max_selections=1,
        label="Data Column Name",
    )

    optical_freq_range = mo.ui.range_slider(
        start=1,
        stop=1000,
        step=1,
        debounce=True,
        show_value=True,
        label="Optical Frequency Range [GHz]",
        full_width=True,
    )
    return col_name_selector, optical_freq_range


@app.cell(hide_code=True)
def _(col_name_selector, mo):
    _name = "mismatch_rotate_origin_shift_origin_rotate_unwind_rotate"
    if col_name_selector.value[0] == "f":
        _name = f"frac_{_name}"

    prefix_selector = mo.ui.text(
        value=_name,
        debounce=True,
        label="Data Prefix",
        full_width=True,
    )
    return (prefix_selector,)


@app.cell(hide_code=True)
def _(
    col_name_selector,
    com_to_selector,
    data_dirs,
    mo,
    optical_freq_range,
    pickle_name_selector,
    prefix_selector,
):
    _pickle_dir = data_dirs[0] / "pickle"
    _com_tos = com_to_selector.value

    pickle_dict, col_name_dict, prefix_dict, threshold_dict = {}, {}, {}, {}
    for _com_to in sorted(_com_tos):
        _bid, _drid = _com_to.split(".")
        if (
            _com_to_pickle := (_pickle_dir / f"B{_bid}D{_drid}" / "network")
        ).exists():
            _avail_com_to_pickle = []
            for _sess_id in _com_to_pickle.iterdir():
                _avail_com_to_pickle += list(
                    _sess_id.glob(f"*{pickle_name_selector.value}*")
                )

            _avail_com_to_pickle = sorted(_avail_com_to_pickle)
            pickle_dict[_com_to] = mo.ui.multiselect(
                _avail_com_to_pickle,
                full_width=True,
                value=_avail_com_to_pickle,
                label="Select Pickle Files...",
            )
        else:
            pickle_dict[_com_to] = mo.ui.text(
                "No pickle files to select", disabled=True, full_width=True
            )
        col_name_dict[_com_to] = mo.ui.multiselect(
            options=["f", "phase"],
            full_width=True,
            value=col_name_selector.value,
            max_selections=1,
            label="Data Column Name",
        )

        prefix_dict[_com_to] = mo.ui.text(
            value=prefix_selector.value,
            debounce=True,
            label="Data Prefix",
            full_width=True,
        )

        threshold_dict[_com_to] = mo.ui.range_slider(
            start=1,
            stop=1000,
            step=1,
            value=optical_freq_range.value,
            debounce=True,
            show_value=True,
            label="Optical Frequency Range [GHz]",
            full_width=True,
        )

    (
        pickle_selector,
        com_to_col_name_selector,
        com_to_prefix_selector,
        com_to_optical_freq_range,
    ) = (
        mo.ui.dictionary(pickle_dict),
        mo.ui.dictionary(col_name_dict),
        mo.ui.dictionary(prefix_dict),
        mo.ui.dictionary(threshold_dict),
    )
    return (
        com_to_col_name_selector,
        com_to_optical_freq_range,
        com_to_prefix_selector,
        pickle_selector,
    )


@app.cell(hide_code=True)
def _(
    com_to_col_name_selector,
    com_to_optical_freq_range,
    com_to_prefix_selector,
    com_to_selector,
    mo,
    pickle_selector,
):
    _tabs_dict = {}
    for _drone in com_to_selector.value:
        _tabs_dict[_drone] = mo.vstack(
            [
                pickle_selector[_drone],
                mo.hstack(
                    [
                        com_to_col_name_selector[_drone],
                        com_to_prefix_selector[_drone],
                    ],
                    widths=[0.5, 1],
                ),
                com_to_optical_freq_range[_drone],
            ]
        )

    pickle_tabs = mo.ui.tabs(_tabs_dict)

    load_pickle_button = mo.ui.run_button(
        kind="success",
        label="Load Pickle Files",
        disabled=not com_to_selector.value,
        full_width=True,
    )
    return load_pickle_button, pickle_tabs


@app.cell(hide_code=True)
def _(mo, root_data_dir):
    detector_map_browser = mo.ui.file_browser(
        initial_path=root_data_dir,
        filetypes=[".parquet"],
        multiple=True,
        ignore_empty_dirs=True,
        label="Select parquet file(s) with detector map data...",
    )
    return (detector_map_browser,)


@app.cell(hide_code=True)
def _(detector_map_browser, mo):
    load_det_map_button = mo.ui.run_button(
        kind="success",
        label="Load Detector Map",
        disabled=not detector_map_browser.value,
        full_width=True,
    )
    return (load_det_map_button,)


@app.cell(hide_code=True)
def _(detector_map_browser, load_det_map_button, mo, pl):
    mo.stop(not load_det_map_button.value)

    detector_map_data = None
    for _file in detector_map_browser.value:
        _sub_detector_map = pl.scan_parquet(_file.path)
        detector_map_data = (
            _sub_detector_map
            if detector_map_data is None
            else pl.concat([detector_map_data, _sub_detector_map], how="diagonal")
        )
    detector_map_data.collect()
    detector_map_data = det_map_transforms(detector_map_data)
    return (detector_map_data,)


@app.cell(hide_code=True)
def _(mo):
    center_freq_selector = mo.ui.number(
        start=1,
        stop=1000,
        value=350,
        step=1,
        debounce=True,
        label="Bandpass Center Frequency [GHz]",
        full_width=True,
    )

    band_edge_threshold = mo.ui.slider(
        start=0,
        stop=1,
        step=0.01,
        value=0.2,
        debounce=True,
        label="FTS Band Edge Threshold",
        show_value=True,
        include_input=True,
        full_width=True,
    )

    SN_slider = mo.ui.slider(
        start=0,
        stop=500,
        step=0.1,
        value=20,
        debounce=True,
        label="S/N Threshold",
        show_value=True,
        include_input=True,
        full_width=True,
    )

    default_color_column = mo.ui.text(
        value="tone_powers",
        full_width=True,
        label="Default Color Column",
        debounce=True,
    )
    return (
        SN_slider,
        band_edge_threshold,
        center_freq_selector,
        default_color_column,
    )


@app.cell(hide_code=True)
def _(
    band_edge_threshold,
    calc_band_center,
    center_freq_selector,
    com_to_col_name_selector,
    com_to_prefix_selector,
    networks,
    pl,
):
    _data_cols = [
        "timestamp",
        "com_to",
        "detector_type",
        "network",
        "drive",
        "fts_start",
        "fts_end",
        "source_temperature",
    ]

    fts_data = None
    for _com_to, _network in networks.items():
        _col_name = com_to_col_name_selector[_com_to].value[0]
        _prefix = com_to_prefix_selector[_com_to].value

        calc_band_center(
            _network,
            col_name=_col_name,
            prefix=_prefix,
            threshold=band_edge_threshold.value,
            center_freq=center_freq_selector.value * 1e9,
        )

        _network_data = _network.combine_properties(data_cols=_data_cols)
        _network_data = fts_transforms(_network_data, _col_name, prefix=_prefix)
        fts_data = (
            _network_data
            if fts_data is None
            else pl.concat([fts_data, _network_data], how="diagonal")
        )
    return (fts_data,)


@app.cell(hide_code=True)
def _(fts_data, mo):
    _timestamps = fts_data["timestamp"].unique()
    timestamp_selector = mo.ui.dropdown(
        options=_timestamps,
        value=_timestamps[0],
        searchable=True,
        label="Select Timestamp",
        full_width=True,
    )
    return (timestamp_selector,)


@app.cell(hide_code=True)
def _(detector_map_data, fts_data, mo, pl):
    mo.stop(detector_map_data is None)

    fts_detector_map = fts_data.join(
        detector_map_data.collect(), on=["com_to", "det"]
    ).with_columns(
        (pl.col("detector_type") + " " + pl.col("network")).alias("typed_network")
    )
    return (fts_detector_map,)


@app.cell(hide_code=True)
def _(fts_detector_map, mo):
    _networks = fts_detector_map["typed_network"].unique().sort().to_list()
    plot_network_selector = mo.ui.multiselect(
        _networks,
        value=_networks,
        label="Select Network(s) to plot...",
        full_width=True,
    )
    return (plot_network_selector,)


@app.cell(hide_code=True)
def _(
    SN_slider,
    fts_detector_map,
    pl,
    plot_network_selector,
    timestamp_selector,
):
    det_map_df = fts_detector_map.filter(
        pl.col("timestamp") == timestamp_selector.value,
        pl.col("spec_S/N") > SN_slider.value,
        ~pl.col("spec_S/N").is_nan(),
        pl.col("typed_network").is_in(plot_network_selector.value),
    ).unique()
    return (det_map_df,)


@app.cell(hide_code=True)
def _(default_color_column, det_map_df, mo, pl):
    _color_cols = (
        det_map_df.select(pl.exclude("det")).select(pl.selectors.numeric()).columns
    )
    color_col_selector = mo.ui.dropdown(
        options=_color_cols,
        value=default_color_column.value,
        searchable=True,
        label="Select Color Column",
        full_width=True,
    )
    return (color_col_selector,)


@app.cell(hide_code=True)
def _(color_col_selector, det_map_df, mo, pl):
    _color_data = (
        det_map_df.select(color_col_selector.value)
        if det_map_df.height > 0
        else pl.DataFrame({"tmp": [0, 1]})
    )
    color_bar_range = mo.ui.range_slider(
        start=_color_data.min().item(),
        stop=_color_data.max().item(),
        debounce=True,
        label=f"{color_col_selector.value} Color Bar Limits",
        full_width=True,
        show_value=True,
    )
    return (color_bar_range,)


@app.cell
def _(mo):
    log_cb = mo.ui.switch(value=False, label="Colorbar Log Scale?")
    return (log_cb,)


@app.cell(hide_code=True)
def _(color_col_selector, mo):
    det_map_title = mo.ui.text(
        value="Detector Map",
        debounce=True,
        label="Detector Map Title",
        full_width=True,
    )

    det_map_cb_title = mo.ui.text(
        value=color_col_selector.value,
        debounce=True,
        label="Detector Map Colorbar Title",
        full_width=True,
    )

    det_map_fig_size = mo.ui.number(
        start=1,
        stop=50,
        value=12,
        label="Figure Size",
        debounce=True,
        step=1,
        full_width=True,
    )
    return det_map_cb_title, det_map_fig_size, det_map_title


@app.cell(hide_code=True)
def _(
    col_name_selector,
    det_map,
    det_map_df,
    get_spectra_df,
    mo,
    networks,
    pl,
    prefix_selector,
    timestamp_selector,
):
    _mask = det_map.value.get_mask(det_map_df["x_0"], det_map_df["y_0"])

    mo.stop(not any(_mask))

    masked_df = det_map_df.filter(_mask)

    _col_name = col_name_selector.value
    _prefix = prefix_selector.value

    full_spec_df = get_spectra_df(
        masked_df.select("com_to", "det"),
        networks,
        _col_name[0],
        prefix=_prefix,
        timestamp=timestamp_selector.value,
    )

    # Plot Average Spectra
    spec_df = (
        full_spec_df.select(pl.exclude("com_to", "det", "norm_f_y"))
        .unique()
        .sort("sample")
    )
    return full_spec_df, masked_df, spec_df


@app.cell
def _(masked_df, mo):
    _com_to_cols = masked_df["com_to"].to_list()
    fts_com_to_selector = mo.ui.dropdown(
        options=_com_to_cols,
        value=_com_to_cols[0],
        searchable=True,
        label="Select Drone",
        full_width=True,
    )
    return (fts_com_to_selector,)


@app.cell
def _(fts_com_to_selector, masked_df, mo, pl):
    _det_cols = (
        masked_df.filter(pl.col("com_to") == fts_com_to_selector.value)["det"]
        .unique()
        .sort()
        .to_list()
    )
    fts_det_selector = mo.ui.slider(
        steps=_det_cols,
        show_value=True,
        debounce=True,
        label="Select Detector",
        full_width=True,
    )
    return (fts_det_selector,)


@app.cell
def _(hv, mo, opts, spec_df):
    _avg_spec_plot = spec_df.hvplot.line(
        "f_x",
        "norm_mean_norm_f_y",
        marker="o",
        ms=3,
        linewidth=0.75,
        xlabel="Optical Frequency [GHz]",
        ylabel="Average Normalized Spectral Response",
        title=f"350 GHz Al Array 2 Spectral Response \nAverage of {spec_df['num_dets'][0]} Detectors",
    )
    _err_plot = hv.Spread(
        (
            spec_df["f_x"],
            spec_df["norm_mean_norm_f_y"],
            spec_df["norm_std_norm_f_y"],
        ),
        label=r"$\pm 1 \sigma$",
    ).relabel(label=r"$\pm 1 \sigma$")

    _opts = [
        opts.Spread(alpha=0.3),
        opts.Curve(show_grid=True),
        opts.Overlay(aspect=2, show_legend=True, fig_size=250),
    ]

    avg_spec_plot = mo.mpl.interactive(
        hv.render(
            (_avg_spec_plot * _err_plot).opts(*_opts), backend="matplotlib"
        ).gca()
    )
    return (avg_spec_plot,)


@app.cell
def _(
    band_edge_threshold,
    fts_com_to_selector,
    fts_det_selector,
    full_spec_df,
    hv,
    masked_df,
    opts,
    pl,
):
    # Individual Spectra

    _indiv_spec_plot = full_spec_df.filter(
        pl.col("com_to") == fts_com_to_selector.value,
        pl.col("det") == fts_det_selector.value,
    ).hvplot.line(
        "f_x",
        "norm_f_y",
        marker="o",
        ms=3,
        linewidth=0.75,
        xlabel="Optical Frequency [GHz]",
        ylabel="Normalized Spectral Response",
    )

    _band_fs = (
        masked_df.filter(
            pl.col("com_to") == fts_com_to_selector.value,
            pl.col("det") == fts_det_selector.value,
        )
        .select("low_edge_f", "high_edge_f", "band_center")
        .to_numpy()
        .T
    )

    _vlines = hv.VLines(_band_fs)
    _hline = hv.HLine(band_edge_threshold.value)

    _opts = [
        opts.Curve(show_grid=True, fig_size=220, aspect=2),
        opts.VLines(c="red", linestyle="dotted", linewidth=0.75),
        opts.HLine(c="green", linestyle="dotted", linewidth=0.75),
    ]

    indiv_spec_plot = (_indiv_spec_plot * _vlines * _hline).opts(*_opts)
    return (indiv_spec_plot,)


@app.cell
def _():
    return


@app.cell(column=3)
def _(pl):
    def load_pickle_transforms(
        network, col_name, prefix="", freq_thresholds=[250e9, 350e9]
    ):
        low_freq, high_freq = freq_thresholds

        for det in network.data["detector"]:
            det = network.det_dict[det]
            stream = det.stream
            tone = stream.tones[0]

            stream._data = stream._data.filter(
                pl.col(
                    f"fft_f_{prefix}{'_' if prefix else ''}{col_name}_{tone:0{stream.padding}d}"
                )
                >= low_freq,
                pl.col(
                    f"fft_f_{prefix}{'_' if prefix else ''}{col_name}_{tone:0{stream.padding}d}"
                )
                <= high_freq,
            )

        return network

    return (load_pickle_transforms,)


@app.cell
def _(ccat_df, pl):
    def calc_band_center(
        network, col_name, prefix="", threshold=0.5, center_freq=280e9
    ):
        f_col, data_col = (
            f"fft_f_{prefix}{'_' if prefix else ''}{col_name}",
            f"fft_{prefix}{'_' if prefix else ''}{col_name}",
        )

        for det in network.data["detector"]:
            det = network.det_dict[det]
            fft_f_df = (
                det.stream.get_data(col_name=f_col, strict=True)
                .unpivot(variable_name="det", value_name="fft_f")
                .with_columns(
                    pl.col("det").str.strip_prefix(f"{f_col}_").cast(pl.Int32)
                )
            )
            fft_data_df = (
                det.stream.get_data(col_name=data_col)
                .unpivot(variable_name="tmp", value_name="fft")
                .drop("tmp")
            )
            fft_df = pl.concat([fft_f_df, fft_data_df], how="horizontal")
            fft_df = fft_df.join(
                det.properties.select("det", f"{data_col}_max"), on="det"
            ).rename({f"{data_col}_max": "fft_max"})

            low_edge = (
                fft_df.lazy()
                .filter(pl.col("fft_f") < center_freq)
                .with_columns(
                    (pl.col("fft") - pl.lit(threshold) * pl.col("fft_max"))
                    .abs()
                    .alias("diff")
                )
                .group_by("diff", "det")
                .agg(pl.all().first())
                .filter((pl.col("diff") == pl.col("diff").min()).over("det"))
                .rename({"fft_f": "low_edge_f"})
                .select("det", "low_edge_f")
                .with_columns((pl.col("low_edge_f") / 1e9).alias("low_edge_f"))
                .unique()
                .sort("det")
                .collect()
            )

            high_edge = (
                fft_df.lazy()
                .filter(pl.col("fft_f") > center_freq)
                .with_columns(
                    (pl.col("fft") - pl.lit(threshold) * pl.col("fft_max"))
                    .abs()
                    .alias("diff")
                )
                .group_by("diff", "det")
                .agg(pl.all().first())
                .filter((pl.col("diff") == pl.col("diff").min()).over("det"))
                .rename({"fft_f": "high_edge_f"})
                .select("det", "high_edge_f")
                .with_columns((pl.col("high_edge_f") / 1e9).alias("high_edge_f"))
                .unique()
                .sort("det")
                .collect()
            )
            edge_df = low_edge.join(high_edge, on="det", how="left").with_columns(
                ((pl.col("low_edge_f") + pl.col("high_edge_f")) / 2).alias(
                    "band_center"
                ),
                (pl.col("high_edge_f") - pl.col("low_edge_f")).alias("bandwidth"),
            )
            shared_cols = (
                ["low_edge_f", "high_edge_f", "band_center", "bandwidth"]
                if "band_center" in det.properties.schema
                else []
            )
            det._properties_df = ccat_df.coalesce_join(
                det._properties_df, edge_df, "det", shared_cols
            )
        return network

    return (calc_band_center,)


@app.function
def fts_transforms(df, col_name, prefix=""):
    data_col = f"fft_{prefix}{'_' if prefix else ''}{col_name}"

    df = (
        df.lazy()
        .select(
            "com_to",
            "det",
            "tone_freqs",
            "tone_powers",
            "timestamp",
            "drive",
            "fts_start",
            "fts_end",
            f"{data_col}_white_noise",
            f"{data_col}_white_noise_rms",
            f"{data_col}_S/N",
            f"{data_col}_max",
            "low_edge_f",
            "high_edge_f",
            "band_center",
            "bandwidth",
        )
        .rename(
            {
                f"{data_col}_white_noise": "spec_white_noise",
                f"{data_col}_white_noise_rms": "spec_white_noise_rms",
                f"{data_col}_S/N": "spec_S/N",
                f"{data_col}_max": "spec_max",
            }
        )
        .unique()
        .sort("com_to", "det")
        .collect()
    )
    return df


@app.function
def det_map_transforms(df):

    df = (
        df.select(
            "com_to",
            "detector_type",
            "network",
            "detector_array",
            "det",
            "x_0",
            "x_0_err",
            "y_0",
            "y_0_err",
            "amp",
            "amp_err",
            "sigma",
            "sigma_err",
        )
        .rename(
            {
                "x_0_err": "x_err",
                "y_0_err": "y_err",
                "amp": "A",
                "amp_err": "A_err",
            }
        )
        .unique()
        .sort("com_to", "det")
    )
    return df


@app.cell
def _(hv, pl):
    def get_spectra_df(det_df, networks, col_name, prefix="", timestamp=None):
        num_dets = det_df.height
        det_df = det_df.group_by("com_to").agg(pl.col("det"))
        det_dict = {pair[0]: pair[1] for pair in det_df.to_numpy()}

        hv.extension("matplotlib")
        spec_df = None
        for com_to, network in networks.items():
            if com_to in det_dict:
                dets = det_dict[com_to]
                _, df = network.plot(
                    "plot",
                    "stream",
                    col_name,
                    col_name,
                    x_prefix=f"fft_f{'_' if prefix else ''}{prefix}",
                    y_prefix=f"savgol0_wn_shift_fft{'_' if prefix else ''}{prefix}",
                    include=dets,
                    data_cols=["com_to", "detector_type", "network", "timestamp"],
                    return_df=True,
                    datashade=False,
                    save_fig=False,
                )
                if timestamp is not None:
                    df = df.filter(pl.col("timestamp") == timestamp).with_columns(
                        pl.col("sample") - pl.col("sample").min()
                    )
                spec_df = (
                    df
                    if spec_df is None
                    else pl.concat([spec_df, df], how="diagonal")
                )

        aligned_fs = (
            spec_df["sample", "f_x"]
            .lazy()
            .unique()
            .group_by("sample")
            .agg(pl.col("f_x").first())
            .sort("sample")
            .with_columns(pl.col("f_x") / 1e9)
            .collect()
        )

        plot_df = (
            spec_df.lazy()
            .with_columns(
                pl.col("f_y").mean().over("sample").alias("mean_f_y"),
                (pl.col("f_y") / pl.col("f_y").max().over("com_to", "det")).alias(
                    "norm_f_y"
                ),
            )
            .with_columns(
                pl.col("norm_f_y").mean().over("sample").alias("mean_norm_f_y"),
                pl.col("norm_f_y").std().over("sample").alias("std_norm_f_y"),
                (pl.col("mean_f_y") / pl.col("mean_f_y").max()).alias(
                    "norm_mean_f_y"
                ),
            )
            .with_columns(
                (pl.col("mean_norm_f_y") / pl.col("mean_norm_f_y").max()).alias(
                    "norm_mean_norm_f_y"
                ),
                (pl.col("std_norm_f_y") / pl.col("mean_norm_f_y").max()).alias(
                    "norm_std_norm_f_y"
                ),
            )
            .with_columns(
                (2 * pl.col("norm_std_norm_f_y") / pl.lit(num_dets).sqrt()).alias(
                    "norm_err_norm_f_y"
                )
            )
            .select(
                "com_to",
                "detector_type",
                "network",
                "det",
                "sample",
                "norm_f_y",
                "mean_f_y",
                "norm_mean_f_y",
                "mean_norm_f_y",
                "norm_mean_norm_f_y",
                "norm_std_norm_f_y",
                "norm_err_norm_f_y",
            )
            .unique()
            .with_columns(pl.lit(num_dets).alias("num_dets"))
            .sort("sample")
            .collect()
            .join(aligned_fs, on="sample", how="left")
        )
        return plot_df

    return (get_spectra_df,)


if __name__ == "__main__":
    app.run()
