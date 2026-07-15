import marimo

__generated_with = "0.23.10"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""
    ### Overview

    This notebook provides an overview of converting time-ordered kinetic inductance detector (KID) data to fractional frequency shift units.
    """)
    return


@app.cell(hide_code=True)
def _(analysis_cfg_browser, cfg_editor, data_browser, data_desc, mo):
    mo.md(rf"""
    ### Select Data to Load

    All *ccatkidlib* data analysis objects (**Detector**, **Network**, etc.) require an `analysis_config.yaml`, which can be selected below. If you do not already have a config file setup, an example can be found at `ccatkidlib/analysis/example_analysis_config.yaml`. After a config file is chosen, the directory with data can be chosen below. 


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
    phase_spline_workers_selector,
    savgol_workers_selector,
    stream_selector,
    transform_button,
):
    mo.md(rf"""
    ### Select Drone & Sweep

    The Radio Frequency System on a Chip (RFSoC) drone (readout chain) with data to load can be selected below. After a drone is selected, a specific target sweep data file can be chosen.

    {mo.vstack([mo.hstack([com_to_selector, stream_selector], widths=[1, 1]), mo.hstack([savgol_workers_selector, circle_fit_workers_selector, phase_spline_workers_selector], widths=[1, 1, 1]), transform_button])}
    """)
    return


@app.cell(column=1)
def _(
    Detector,
    analysis_cfg,
    com_to_selector,
    mo,
    stream_selector,
    transform_button,
):
    mo.stop(not transform_button.value)

    det = Detector(
        com_to=com_to_selector.value[0],
        stream_path=stream_selector.value[0],
        analysis_cfg=analysis_cfg,
    )
    return (det,)


@app.cell
def _(det):
    LABELS = {
        "mag": {"xlabel": "Frequency [Hz]", "ylabel": r"$|S_{21}|$"},
        "phase": {"xlabel": "Frequency [Hz]", "ylabel": "Phase [rad]"},
        "IQ": {"xlabel": "I", "ylabel": "Q"},
    }

    NAMES, PREFIX = (
        det.analysis_cfg["convention"]["name"],
        det.analysis_cfg["convention"]["prefix"],
    )
    return LABELS, PREFIX


@app.cell
def _(circle_fit_workers_selector, det, routines, savgol_workers_selector):
    routines.IQ_circle_center(
        det,
        data="both",
        savgol_window=9,
        savgol_order=1,
        trim_window=10,
        trim_mean_points=10,
        mismatch_mean_points=10,
        savgol_workers=savgol_workers_selector.value,
        fit_workers=circle_fit_workers_selector.value,
        recalc=True,
    )

    get_centered_dfs = True
    return (get_centered_dfs,)


@app.cell
def _(PREFIX, det, get_centered_dfs, mo):
    mo.stop(not get_centered_dfs)

    mismatch_prefix = f"{PREFIX['remove_impedance_mismatch']}_{PREFIX['rotate']}_\
    {PREFIX['center_origin']}_{PREFIX['translate']}_\
    {PREFIX['center_origin']}_{PREFIX['rotate']}_\
    {PREFIX['remove_cable']}_{PREFIX['rotate']}"

    mismatch_plot_dfs = get_plot_dfs(det, mismatch_prefix)
    return mismatch_plot_dfs, mismatch_prefix


@app.cell
def _(create_dashboard, mismatch_plot_dfs, plot_sweep):
    mismatch_plots, _mag_phase_opts, _IQ_opts = plot_sweep(
        mismatch_plot_dfs, include_tones=True
    )
    mismatch_plots_dashboard = create_dashboard(
        mismatch_plots, _mag_phase_opts, _IQ_opts
    )
    return (mismatch_plots_dashboard,)


@app.cell(hide_code=True)
def _(mismatch_plots_dashboard, mo):
    mo.md(rf"""
    ### Centered IQ Circle (*Best Viewed as Fullscreen*)

    {mismatch_plots_dashboard}
    """)
    return


@app.cell
def _(
    det,
    mismatch_prefix,
    phase_spline_workers_selector,
    properties_routines,
):
    det.stream.phase(
        prefix=mismatch_prefix,
    )

    _min_phase, _max_phase = (
        properties_routines.agg(det.stream, "min", "phase", prefix=mismatch_prefix)
        .to_numpy()
        .T[1],
        properties_routines.agg(det.stream, "max", "phase", prefix=mismatch_prefix)
        .to_numpy()
        .T[1],
    )

    _bounds = 1
    det.phase_spline(
        prefix=mismatch_prefix,
        phase_low=_min_phase - _bounds,
        phase_up=_max_phase + _bounds,
        k=2,
        max_workers=phase_spline_workers_selector.value,
        recalc=True,
    )

    spline_calculated = True
    return (spline_calculated,)


@app.cell
def _(PREFIX, det, mismatch_prefix, mo, routines, spline_calculated):
    mo.stop(not spline_calculated)

    routines.IQ_noise(det, prefix='mismatch_rotate_origin_shift_origin_rotate_unwind_rotate')
    noise_prefix = f"{PREFIX['isolate_readout_noise']}_{PREFIX['rotate']}_{mismatch_prefix}"
    det.stream.phase(prefix=noise_prefix)
    noise_rotated = True
    return noise_prefix, noise_rotated


@app.cell
def _(
    det,
    mismatch_prefix,
    mo,
    noise_prefix,
    noise_rotated,
    phase_spline_workers_selector,
):
    mo.stop(not noise_rotated)

    det.phase_to_f(
        prefix=[mismatch_prefix, noise_prefix],
        spline_prefix=mismatch_prefix,
        max_workers=phase_spline_workers_selector.value,
        recalc=True,
    )

    det.frac_f(
        prefix=[mismatch_prefix, noise_prefix],
        recalc=True,
    )
    return


@app.cell
def _(det):
    _tone = 10
    (det.stream.stream_plot(col_name='ff', prefix='mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', time_col='zt', include=_tone, datashade=False)
    *det.stream.stream_plot(col_name='ff', prefix='readout_noise_rotate_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', time_col='zt', include=_tone, datashade=False))
    return


@app.cell
def _(det):
    _tone = 10
    (det.stream.IQ_plot(prefix='mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', include=_tone, datashade=False)
    *det.stream.IQ_plot(prefix='readout_noise_rotate_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', include=_tone, datashade=False))
    return


@app.cell
def _(det):
    det.stream.psd(col_name='ff', prefix=['mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', 'readout_noise_rotate_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate'], nperseg=512, recalc=True)
    return


@app.cell
def _(det, hv, np):
    _f = np.linspace(0.5, 200, 1000)
    _l = 1e-8*_f**-1/2
    _tone = 162
    (det.stream.psd_plot(col_name='ff', prefix='mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', include=_tone)*
     det.stream.plot(x_dim='ff', y_dim='ff', x_prefix='psd_f_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', y_prefix='white_noise_trim_psd_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', include=_tone, datashade=False, logx=True, logy=True)*
    det.stream.psd_plot(col_name='ff', prefix='readout_noise_rotate_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', include=_tone)*hv.Curve((_f, _l))).opts(aspect=2, fig_size=250)
    return


@app.cell
def _(det, properties_routines):
    properties_routines.fwhm(det.targ, mag_prefix='savgol0', recalc=True)
    return


@app.cell
def _(det):
    det.frac_f(prefix='', data='targ', ref_f='mid_savgol0_FWHM_f')[0]
    return


@app.cell
def _(det):
    det.targ.plot(x_dim='ff', y_dim='mag', ms=1, linewidth=0)
    return


@app.cell
def _(det):
    det.targ.scale('f', scale=1/1e6, name='MHz', recalc=True)
    return


@app.cell
def _(det, pl):
    det.stream.get_data(['sample', 'psd_f_mismatch.*_ff']).unpivot(index='sample', value_name='frequency', variable_name='det').with_columns(pl.col('det').str.strip_prefix(f'psd_f_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_ff_').cast(pl.Int32)).filter(pl.col('frequency').is_between(200, 250)).with_columns(pl.col('sample').get(pl.col('frequency').arg_min().over('det')).alias('low_index'),
                       pl.col('sample').get(pl.col('frequency').arg_max()).alias('upper_index')).select('det', 'low_index', 'upper_index').unique().sort('det')
    return


@app.cell
def _(det):
    det.stream.psd_trim(col_name='ff', prefix=['mismatch_rotate_origin_shift_origin_rotate_unwind_rotate', 'readout_noise_rotate_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate'], low_f=100, name='white_noise', recalc=True)
    return


app._unparsable_cell(
    r"""
    properties_routines.agg(det.stream, 'median', col_name='ff', prefix='white_noise_trim_psd_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate'])
    """,
    name="_"
)


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

    return Path, json, mo, os


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
    import ccatkidlib.analysis.utils.pair as ccat_pair
    import ccatkidlib.analysis.utils.dataframe as ccat_df
    import ccatkidlib.analysis.routines.common as routines
    import ccatkidlib.analysis.routines.properties as properties_routines

    from ccatkidlib.analysis.core.detector import Detector

    return Detector, ccat_io, properties_routines, routines


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
def _(HOME_DIR, analysis_cfg, mo):
    # Extract root data directory from analysis config file
    root_data_dir = (
        analysis_cfg["file_paths"]["root_data_dir"] if analysis_cfg else HOME_DIR
    )

    # Create file browser for selecting target sweep file
    data_browser = mo.ui.file_browser(
        initial_path=root_data_dir,
        selection_mode="directory",
        multiple=False,
        restrict_navigation=True,
        ignore_empty_dirs=True,
        label="Select data directory(ies)...",
    )
    return (data_browser,)


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
def _(mo, stream_selector):
    # Create run button for transforming data
    transform_button = mo.ui.run_button(
        kind="success",
        label="Convert to Fractional Frequency",
        tooltip="Click to convert timestream data to fractional frequency",
        full_width=True,
        disabled=not stream_selector.value,
    )
    return (transform_button,)


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
        label="Select Drone...",
        full_width=True,
        max_selections=1,
    )
    return (com_to_selector,)


@app.cell
def _(com_to_selector, data_dirs, mo):
    if _selected_drone := com_to_selector.value:
        _bid, _drid = _selected_drone[0].split(".")
        _stream_files = sorted(
            list(
                map(
                    str,
                    (data_dirs[0] / "timestream" / f"B{_bid}D{_drid}").iterdir(),
                )
            )
        )
    else:
        _stream_files = []

    stream_selector = mo.ui.multiselect(
        _stream_files,
        value=_stream_files[0:1] if _stream_files else None,
        label="Select Timestream Sweep File...",
        full_width=True,
        max_selections=1,
    )
    return (stream_selector,)


@app.cell
def _(mo, os):
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

    phase_spline_workers_selector = mo.ui.number(
        start=1,
        stop=os.cpu_count(),
        step=1,
        label="Phase Spline Max Workers",
        value=os.cpu_count() // 3,
        full_width=True,
    )
    return (
        circle_fit_workers_selector,
        phase_spline_workers_selector,
        savgol_workers_selector,
    )


@app.cell
def _(det, mo):
    mo.stop(not det)

    tone_selector = mo.ui.slider(
        steps=det.targ.tones,
        debounce=True,
        show_value=True,
        full_width=True,
        label="Tone",
    )
    return (tone_selector,)


@app.function(column=3)
def get_plot_dfs(det, prefix):
    # Calculate target sweep magnitude and phase
    det.targ.mag(prefix=prefix, dB=True)
    det.targ.phase(prefix=prefix)

    tones = det.targ.tones

    dB_prefix = f"{det.analysis_cfg['convention']['prefix']['decible']}{'_' if prefix else ''}{prefix}"
    mag_plot = det.targ.mag_plot(
        prefix=dB_prefix,
        include=tones,
        return_df=True,
        return_fig=False,
        save_fig=False,
    )[0].rename({"f": "x", "mag": "y"})
    phase_plot = det.targ.phase_plot(
        prefix=prefix,
        include=tones,
        return_df=True,
        return_fig=False,
        save_fig=False,
    )[0].rename({"f": "x", "phase": "y"})
    IQ_plot = det.targ.IQ_plot(
        prefix=prefix,
        include=tones,
        return_df=True,
        return_fig=False,
        save_fig=False,
    )[0].rename({"I": "x", "Q": "y"})

    return {"mag": mag_plot, "phase": phase_plot, "IQ": IQ_plot}


@app.cell
def _(LABELS, opts, pl, tone_selector):
    def plot_sweep(plot_dfs, include_tones=True):
        tone = tone_selector.value

        mag_phase_opts = [
            opts.Layout(shared_axes=False, sublabel_format="", fig_size=150),
            opts.Scatter("Scatter.Data", aspect=1.7, s=10),
            opts.Scatter("Scatter.Tone", c="red", s=100, marker="*"),
        ]

        IQ_opts = [
            opts.Scatter("Scatter.Data", data_aspect=1, fig_size=240, s=10),
            opts.Scatter("Scatter.Tone", c="red", s=100, marker="*"),
        ]

        plots = {}
        for type, df in plot_dfs.items():
            line_opts = opts.Scatter(
                xlabel=LABELS[type]["xlabel"], ylabel=LABELS[type]["ylabel"]
            )

            df = df.filter(pl.col("det") == tone)
            data_plot = df.hvplot.scatter(x="x", y="y", label="Data").opts(
                line_opts
            )
            if include_tones:
                data_plot *= df.filter(pl.col("tone")).hvplot.scatter(
                    x="x", y="y", label="Tone"
                )
            plots[type] = data_plot
        return plots, mag_phase_opts, IQ_opts

    return (plot_sweep,)


@app.cell
def _(hv, mo, tone_selector):
    def create_dashboard(plots, mag_phase_opts, IQ_opts):
        mag_phase_layout = hv.Layout([plots["mag"], plots["phase"]]).cols(1)
        plot = mo.hstack(
            [
                plots["IQ"].opts(IQ_opts),
                mag_phase_layout.opts(*mag_phase_opts),
            ],
            widths=[1, 1],
        )

        return mo.vstack([tone_selector, plot])  # Best viewed in fullscreen mode

    return (create_dashboard,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
