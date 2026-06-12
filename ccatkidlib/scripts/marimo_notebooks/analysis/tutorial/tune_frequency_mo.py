import marimo

__generated_with = "0.23.2"
app = marimo.App(
    width="columns",
    layout_file="layouts/tune_frequency_mo.slides.json",
)


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
    com_to_selector,
    mo,
    savgol_workers_selector,
    targ_selector,
    transform_button,
):
    mo.md(rf"""
    ### Select Drone & Sweep

    The Radio Frequency System on a Chip (RFSoC) drone (readout chain) with data to load can be selected below. After a drone is selected, a specific target sweep data file can be chosen.

    {mo.vstack([mo.hstack([com_to_selector, targ_selector], widths=[1, 1]), mo.hstack([savgol_workers_selector], widths=[1]), transform_button])}
    """)
    return


@app.cell
def _():
    return


@app.cell(column=1)
def _(
    Detector,
    analysis_cfg,
    com_to_selector,
    mo,
    targ_selector,
    transform_button,
):
    mo.stop(not transform_button.value)

    det = Detector(
        com_to=com_to_selector.value[0],
        targ_path=targ_selector.value[0],
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
def _(det, savgol_workers_selector, tune_routines):
    tuned_fs = tune_routines.max_IQ_dist_f(
        det,
        trim_window=2,
        trim_savgol_window=9,
        diff_savgol_window=21,
        trim_savgol_k=1,
        diff_savgol_k=1,
        max_workers=savgol_workers_selector.value,
        recalc=True,
    ).to_numpy().T[1]

    get_max_dist_f = True
    return get_max_dist_f, tuned_fs


@app.cell
def _(PREFIX, det, get_max_dist_f, mo):
    mo.stop(not get_max_dist_f)
    cable_prefix = f"{PREFIX['remove_cable']}_{PREFIX['rotate']}"
    unwind_plot_dfs = get_plot_dfs(det, cable_prefix)
    return (unwind_plot_dfs,)


@app.cell
def _(
    create_dashboard,
    hv,
    plot_sweep,
    tone_selector,
    tuned_fs,
    unwind_plot_dfs,
):
    unwind_plots, _mag_phase_opts, _IQ_opts = plot_sweep(unwind_plot_dfs)
    unwind_plots['mag'] *= hv.VLine(tuned_fs[tone_selector.value])
    unwind_plots_dashboard = create_dashboard(
        unwind_plots, _mag_phase_opts, _IQ_opts
    )
    return (unwind_plots_dashboard,)


@app.cell
def _(unwind_plots_dashboard):
    unwind_plots_dashboard
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

    return Path, json, mo, os


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
    import ccatkidlib.analysis.routines.tune as tune_routines
    import ccatkidlib.analysis.routines.properties as properties_routines

    from ccatkidlib.analysis.core.detector import Detector

    return Detector, ccat_io, tune_routines


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
def _(mo, targ_selector):
    # Create run button for transforming data
    transform_button = mo.ui.run_button(
        kind="success",
        label="Calculate Max IQ Distance",
        tooltip="Click to max IQ distance",
        full_width=True,
        disabled=not targ_selector.value,
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
        _targ_files = sorted(
            list(
                map(
                    str,
                    (data_dirs[0] / "targ" / f"B{_bid}D{_drid}").iterdir(),
                )
            )
        )
    else:
        _targ_files = []

    targ_selector = mo.ui.multiselect(
        _targ_files,
        value=_targ_files[0:1] if _targ_files else None,
        label="Select Target Sweep File...",
        full_width=True,
        max_selections=1,
    )
    return (targ_selector,)


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
    return (savgol_workers_selector,)


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


@app.cell
def _():
    return


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
