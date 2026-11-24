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
def _(analysis_cfg_browser, cfg_editor, data_browser, mo):
    mo.md(rf"""
    ### Select Data to Load

    All *ccatkidlib* data analysis objects (**Detector**, **Network**, etc.) require an `analysis_config.yaml`, which can be selected below. If you do not already have a config file setup, an example can be found at `ccatkidlib/analysis/example_analysis_config.yaml`. After a config file is chosen, the directory containing the data to load can be selected. The default behavior is to only allow loading data corresponding to a single **sess_id** (measurement) but can be modified as necessary.


    {mo.hstack(
                    [mo.vstack([analysis_cfg_browser, data_browser]), cfg_editor],
                    widths=[1, 1],
                )       
    }
    """)
    return


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(
    aspect_selector,
    cmap_range_selector,
    cmap_selector,
    fig_height,
    fig_width,
    font_scale_selector,
    mo,
    optical_freq_range,
    plot_error_selector,
    plot_tabs,
):
    mo.md(rf"""
    ### Plot Individual Bandpasses

    {mo.vstack([optical_freq_range, mo.hstack([cmap_selector, cmap_range_selector], widths=[1,4]), mo.hstack([font_scale_selector, fig_width, fig_height, aspect_selector, plot_error_selector], widths=[2, 2, 2, 2, 1]), plot_tabs])}
    """)
    return


@app.cell(hide_code=True)
def _(
    aspect_selector,
    cmap_range_selector,
    cmap_selector,
    fig_height,
    fig_width,
    font_scale_selector,
    mo,
    optical_freq_range,
    overlay_plot,
    plot_error_selector,
    plot_selector,
    subtitle_selector,
    title_selector,
):
    mo.md(rf"""
    ### Plot Overlaid Bandpasses

    {mo.vstack([plot_selector, mo.hstack([cmap_selector, cmap_range_selector], widths=[1,4]), mo.hstack([title_selector, subtitle_selector, font_scale_selector, plot_error_selector], widths=[2, 2, 1, 1]), mo.hstack([fig_width, fig_height, aspect_selector], widths=[1,1,1]) ,optical_freq_range, overlay_plot])}
    """)
    return


@app.cell(column=2)
def _():
    # General
    import marimo as mo

    from tqdm import tqdm
    from functools import partial
    from itertools import cycle as itercycle
    import time

    # IO
    import os
    import ast
    import json
    import pickle
    from pathlib import Path

    return Path, itercycle, json, mo, os, pickle


@app.cell
def _():
    # Data Analysis
    import numpy as np
    import polars as pl
    from scipy.signal import savgol_filter

    return pl, savgol_filter


@app.cell
def _():
    # ccatkidlib
    import ccatkidlib.io as ccat_io
    import ccatkidlib.log as ccat_log
    import ccatkidlib.utils as ccat_utils
    import ccatkidlib.analysis.viz.viz_utils as viz_utils
    import ccatkidlib.analysis.utils.pickle as ccat_pickle
    import ccatkidlib.analysis.utils.pair as ccat_pair
    import ccatkidlib.analysis.utils.dataframe as ccat_df

    from ccatkidlib.rfsoc.rfsoc_daq import R
    from ccatkidlib.analysis.core.detector import Detector
    from ccatkidlib.analysis.core.network import Network

    return ccat_io, viz_utils


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
    return hv, opts


@app.cell
def _():
    import matplotlib

    return (matplotlib,)


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
    return analysis_cfg, cfg_editor


@app.cell
def _(HOME_DIR, Path, analysis_cfg, mo):
    # Extract root data directory from analysis config file
    root_data_dir = (
        analysis_cfg["file_paths"]["root_data_dir"] if analysis_cfg else HOME_DIR
    )

    # Create file browser for selecting data sess_id directory
    data_browser = mo.ui.file_browser(
        initial_path=root_data_dir if Path(root_data_dir).exists() else HOME_DIR,
        selection_mode="file",
        filetypes=['.parquet', '.p', '.txt'],
        multiple=True,
        restrict_navigation=False,
        ignore_empty_dirs=True,
        label="Select bandpass data files...",
    )

    optical_freq_range = mo.ui.range_slider(
        start=1,
        stop=1000,
        step=1,
        debounce=True,
        show_value=True,
        label="Optical Frequency Range [GHz] (All Bandpasses)",
        full_width=True,
    )
    return data_browser, optical_freq_range


@app.cell
def _(data_browser, mo, optical_freq_range):
    mo.stop(not data_browser.value, mo.md('**Select bandpass data file(s) to continue.**'))

    _data_files = data_browser.value
    _freq_range = {}

    for _i, _file in enumerate(_data_files):
        _stem = str(_file.path.stem)

        _freq_range[_stem] = mo.ui.range_slider(
            start=1,
            stop=1000,
            step=1,
            value=optical_freq_range.value,
            debounce=True,
            show_value=True,
            label="Optical Frequency Range [GHz]",
            full_width=True,
        )

    freq_range = mo.ui.dictionary(_freq_range)
    return (freq_range,)


@app.cell
def load_files(data_browser, mo, pickle, pl):
    mo.stop(not data_browser.value, mo.md('**Select bandpass data file(s) to continue.**'))

    _data_files = data_browser.value
    _plot_labels, _plot_titles = {}, {}
    bandpass_dfs = {'nist': {'predicted': {}, 'measured': {}}, 'cornell': {'measured': {}}}

    for _i, _file in enumerate(_data_files):
        _file = _file.path
        _ext, _stem = _file.suffix, str(_file.stem)

        _plot_labels[_stem] = mo.ui.text(value='', placeholder='Enter Bandpass Label Here', debounce=True, label='Bandpass Label', full_width=True)
        _plot_titles[_stem] = mo.ui.text(value='', placeholder='Enter Bandpass Title Here', debounce=True, label='Bandpass Title', full_width=True)

        if _ext == '.parquet':
            bandpass_dfs['cornell']['measured'][_stem] = pl.read_parquet(_file)
        elif _ext == '.p': 
            with open(_file, 'rb') as _f:
                _pickle_data = pickle.load(_f)
                bandpass_dfs['nist']['measured'][_stem] = pl.DataFrame(_pickle_data).explode(pl.col(pl.Array))
        elif _ext == '.txt': 
            bandpass_dfs['nist']['predicted'][_stem] = pl.read_csv(_file, separator='\t')

    plot_labels, plot_titles = mo.ui.dictionary(_plot_labels), mo.ui.dictionary(_plot_titles)
    return bandpass_dfs, plot_labels, plot_titles


@app.cell
def _(data_browser, matplotlib, mo):
    _file_names = [str(_f.path.stem) for _f in data_browser.value]
    plot_selector = mo.ui.multiselect(options = _file_names, value=_file_names, full_width=True, label='Select Bandpasses to Overlay')

    cmap_selector = mo.ui.dropdown(options=list(matplotlib.colormaps), value='Set1', allow_select_none=False, searchable=True, label='Select Colormap', full_width=True)

    cmap_range_selector = mo.ui.range_slider(start=0, stop=1, step=0.1, label='Select Colormap Range', full_width=True, value=[0, 1], debounce=True, show_value=True)

    title_selector = mo.ui.text(placeholder='Enter Title Here', debounce=True, full_width=True, label='Title')
    subtitle_selector = mo.ui.text(placeholder='Enter Sub-title Here', debounce=True, full_width=True, label='Sub-Title')

    plot_error_selector = mo.ui.checkbox(value=True, label='Plot Error Bars')

    font_scale_selector = mo.ui.number(start=0.1, stop=10, step=0.1, value=1, label='Font Scale', debounce=True, full_width=True)
    fig_width = mo.ui.number(start=1, stop=20, step=0.5, value=4, label='Figure Width (in.)', debounce=True, full_width=True)
    fig_height = mo.ui.number(start=1, stop=20, step=0.5, value=2, label='Figure Height (in.)', debounce=True, full_width=True)
    return (
        cmap_range_selector,
        cmap_selector,
        fig_height,
        fig_width,
        font_scale_selector,
        plot_error_selector,
        plot_selector,
        subtitle_selector,
        title_selector,
    )


@app.cell
def _(fig_height, fig_width, mo):
    aspect_selector = mo.ui.number(start=1, stop=20, step=1, value=max(1, int(fig_width.value/fig_height.value)), label='Aspect Ratio', debounce=True, full_width=True)
    return (aspect_selector,)


@app.cell
def _(
    aspect_selector,
    bandpass_dfs,
    cmap_range_selector,
    cmap_selector,
    fig_height,
    fig_width,
    font_scale_selector,
    freq_range,
    hv,
    itercycle,
    mo,
    opts,
    plot_cornell_bandpass,
    plot_error_selector,
    plot_labels,
    plot_nist_bandpass,
    plot_selector,
    plot_titles,
    subtitle_selector,
    title_selector,
    viz_utils,
):
    _plot_tabs = {}
    _plot_list = []

    _cmap_cycle = itercycle(viz_utils.cycle_cmap(cmap_selector.value, num_colors = max(1, len(plot_selector.value)), cmap_range=cmap_range_selector.value).values)

    _func_dict = {'cornell': {'measured': plot_cornell_bandpass}, 'nist': {'measured': plot_nist_bandpass, 'predicted': plot_nist_predicted}}

    _curr_color = next(_cmap_cycle)
    for _institute, _v in bandpass_dfs.items():
        for _type, _vv in _v.items():
            for _f, _df in _vv.items():
                _label, _title, _freq_range = plot_labels[_f], plot_titles[_f], freq_range[_f]
                _plot = _func_dict[_institute][_type](_df, _freq_range.value, label=_label.value, title=_title.value, c=_curr_color, plot_error = plot_error_selector.value)
                _fig = hv.render(_plot, backend='matplotlib')
                _fig.tight_layout()
                _plot_tabs[_f] = mo.lazy(mo.vstack([mo.hstack([_label, _title], widths=[1,1]), _freq_range, mo.mpl.interactive(_fig.gca())]))
                if _f in plot_selector.value: 
                    _plot_list.append(_plot)
                    _curr_color = next(_cmap_cycle)
    plot_tabs = mo.ui.tabs(_plot_tabs)

    _overlay_opts = [opts.Overlay(title=subtitle_selector.value, fig_inches=(fig_width.value, fig_height.value), fontscale=font_scale_selector.value, aspect=aspect_selector.value)]

    if _plot_list:
        _overlay_plot = hv.render(hv.Overlay(_plot_list).opts(*_overlay_opts), backend='matplotlib')
        _suptitle = _overlay_plot.suptitle(title_selector.value, y=0.91)
        _title_size = _suptitle.get_fontsize()
        _suptitle.set_fontsize(_title_size*font_scale_selector.value)
        overlay_plot = mo.mpl.interactive(_overlay_plot.gca())
    else:
        overlay_plot = mo.callout('At least one bandpass must be selected.')
    return overlay_plot, plot_tabs


@app.cell
def _():
    return


@app.cell(column=3)
def _(
    aspect_selector,
    fig_height,
    fig_width,
    font_scale_selector,
    hv,
    opts,
    pl,
):
    def plot_cornell_bandpass(df, freq_range, label='', title='', c=None, plot_error = True):
        df = df.filter(pl.col('f_x')  >= freq_range[0], pl.col('f_x') <= freq_range[1])

        plot = df.hvplot.line(
            "f_x",
            "norm_mean_norm_f_y",
            marker="o",
            ms=3,
            linewidth=0.75,
            xlabel="Optical Frequency [GHz]",
            ylabel="Normalized Spectral Response",
            label=label,
            title=title if title else f"Average Spectral Response of {df['num_dets'][0]} Detectors",
            color = c
        )
        err_plot = hv.Spread(
            (
                df["f_x"],
                df["norm_mean_norm_f_y"],
                df["norm_std_norm_f_y"],
            ),
            label=r"$\pm 1 \sigma$",

        )

        plot_opts = [
            opts.Spread(alpha=0.2, facecolor=c, fig_inches=(fig_width.value, fig_height.value),aspect=aspect_selector.value, fontscale=font_scale_selector.value),
            opts.Curve(show_grid=True, fig_inches=(fig_width.value, fig_height.value), aspect=aspect_selector.value, fontscale=font_scale_selector.value),
            opts.Overlay(show_legend=True, fig_inches=(fig_width.value, fig_height.value), aspect=aspect_selector.value, fontscale=font_scale_selector.value)
        ]

        if plot_error: plot *= err_plot

        return plot.opts(*plot_opts) 

    return (plot_cornell_bandpass,)


@app.cell
def _(
    aspect_selector,
    fig_height,
    fig_width,
    font_scale_selector,
    opts,
    pl,
    savgol_filter,
):
    def plot_nist_bandpass(df, freq_range, label='', title='', c = None, **kwargs):
        # Get average spectral response across detectors
        df = (df.rename({'frequency': 'f_x', 'spectral_responsivity': 'f_y'})
                .with_columns(pl.col('f_x').cum_count().over("res_idx").alias('sample'))
                .with_columns(pl.col("f_y").mean().over("sample").alias("mean_f_y"),))
        num_dets = df.select('res_idx').unique().height

        aligned_fs = (
            df["sample", "f_x"]
            .lazy()
            .unique()
            .group_by("sample")
            .agg(pl.col("f_x").first())
            .sort("f_x")
            .collect()
        )

        # Savgol filter spectral response
        savgol_filtered = savgol_filter(df['mean_f_y'].to_numpy(), 10, 1)
        df = df.with_columns(pl.Series(savgol_filtered).alias('savgol0_mean_f_y'))

        # Normalize Spectral Response
        df = (
            df.lazy()
            .with_columns(
                (pl.col("savgol0_mean_f_y") / pl.col("savgol0_mean_f_y").max().over('res_idx')).alias(
                    "norm_savgol0_mean_f_y"
                ),
            )
            .unique()
            .with_columns(pl.lit(num_dets).alias("num_dets"))
            .sort("sample")
            .collect()
            .join(aligned_fs, on="sample", how="left")
            .filter(pl.col('f_x')  >= freq_range[0], pl.col('f_x') <= freq_range[1])
        )

        plot = df.hvplot.line(
            "f_x",
            "norm_savgol0_mean_f_y",
            marker="^",
            ms=3,
            linestyle='-',
            linewidth=0.75,
            xlabel="Optical Frequency [GHz]",
            ylabel="Normalized Spectral Response",
            label=label,
            title=title if title else f"Average Spectral Response of {df['num_dets'][0]} Detectors",
            color=c

        )

        plot_opts = [opts.Curve(show_grid=True, show_legend=True, fig_inches=(fig_width.value, fig_height.value), aspect=aspect_selector.value, fontscale=font_scale_selector.value)]

        return plot.opts(*plot_opts)

    return (plot_nist_bandpass,)


@app.function
def plot_nist_predicted(df, freq_range, label='', **kwargs):
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
