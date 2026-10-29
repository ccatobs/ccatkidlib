import marimo

__generated_with = "0.25.1"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""
    ### Overview

    This notebook provides an overview of fitting kinetic inductance detectors (KIDs).
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
    fit_det_button,
    mo,
    phase_fit_workers_selector,
    savgol_workers_selector,
    targ_selector,
):
    mo.md(rf"""
    ### Select Drone & Sweep

    The Radio Frequency System on a Chip (RFSoC) drone (readout chain) with data to load can be selected below. After a drone is selected, a specific target sweep data file can be chosen.

    {mo.vstack([mo.hstack([com_to_selector, targ_selector], widths=[1, 1]), mo.hstack([savgol_workers_selector, circle_fit_workers_selector, phase_fit_workers_selector], widths=[1, 1, 1]), fit_det_button])}
    """)
    return


@app.cell(disabled=True)
def _(
    Network,
    analysis_cfg,
    com_to_selector,
    data_dirs,
    root_data_dir,
    viz_cfg,
):
    _data_dir, _date, _sess_id = data_dirs[0].parts[-3:]

    network = Network(
            com_to=com_to_selector.value[0],
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
            max_workers=10,
    )
    return (network,)


@app.cell(disabled=True)
def _(dump_transform, network):
    dump_transform(network)
    return


@app.cell
def _(network):
    combined_properties = network.combine_properties(data_cols=['drive']).sort('drive')
    return (combined_properties,)


@app.cell
def _(combined_properties):
    combined_properties
    return


@app.cell
def _(tone_selector, wigglystuff):
    wigglystuff.FloatingPanel(tone_selector)
    return


@app.cell
def _(
    NAMES,
    PREFIX,
    combined_properties,
    pl,
    tone_selector,
    trim_mismatch_prefix,
):
    combined_properties.filter(pl.col('det') == tone_selector.value).hvplot.line(x='drive', y=f"{PREFIX['phase_fit']}_{trim_mismatch_prefix}_{NAMES['ki_nonlinearity_parameter']}")
    return


@app.cell
def _(PREFIX, network, tone_selector, trim_mismatch_prefix):
    (network.plot('phase', 'targ', prefix=f"{PREFIX['phase_fit']}_{trim_mismatch_prefix}", include=tone_selector.value, data_cols=['drive'], cmap='viridis', grouping='groupby')
    *network.plot('phase', 'targ', prefix=trim_mismatch_prefix, include=tone_selector.value, data_cols=['drive'], cmap='viridis', grouping='groupby'))
    return


@app.cell
def _():
    return


@app.cell(column=1)
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
    return LABELS, NAMES, PREFIX


@app.cell
def _(
    Detector,
    analysis_cfg,
    com_to_selector,
    fit_det_button,
    mo,
    targ_selector,
):
    mo.stop(not fit_det_button.value)

    det = Detector(
        com_to=com_to_selector.value[0],
        targ_path=targ_selector.value[0],
        analysis_cfg=analysis_cfg,
    )
    return (det,)


@app.cell
def _(circle_fit_workers_selector, det, routines, savgol_workers_selector):
    routines.IQ_circle_origin(
        det,
        data="targ",
        savgol_window=9,
        savgol_order=1,
        fit_trim_window=10,
        mean_points=10,
        savgol_workers=savgol_workers_selector.value,
        circle_fit_workers=circle_fit_workers_selector.value,
        normalize=True,
        recalc=True,
    )

    get_centered_dfs = True
    return (get_centered_dfs,)


@app.cell
def _(PREFIX, det, get_centered_dfs, mo):
    mo.stop(not get_centered_dfs)

    mismatch_prefix = f"{PREFIX['normalize']}_{PREFIX['scale']}_\
    {PREFIX['remove_impedance_mismatch']}_{PREFIX['rotate']}_\
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
def _(PREFIX, det, mismatch_prefix):
    det.IQ_circle_tail(prefix=mismatch_prefix, data='targ', mean_points=10)
    tail_prefix = "_".join([PREFIX['center_tail'], PREFIX['translate'], mismatch_prefix])
    return (tail_prefix,)


@app.cell
def _(PREFIX, det, tail_prefix):
    det.targ.IQ_circle_diameter(prefix=tail_prefix)
    det.targ.linewidth(prefix=tail_prefix)
    det.targ.linewidth_shift(prefix=tail_prefix, 
                             coupling_linewidth=False,
                             res_trim_window=2, 
                             tail_trim_window=3,
                             mean_points=10,
                             mag_prefix = f"{PREFIX['savgol_filter']}0", 
                             max_workers=5,
                             return_fit_params=True,
                             recalc=False)
    diameter = det.targ.get_data("_".join([tail_prefix, 'diameter'])).to_numpy().T
    linear_prefix = "_".join(['linear', PREFIX['translate'], tail_prefix])
    return (linear_prefix,)


@app.cell
def _(PREFIX, det, linear_prefix, tail_prefix):
    det.targ.res_trim(col_name='y', prefix=[linear_prefix, tail_prefix], mean_points=10, window=3, invert=False, mag_prefix=f"{PREFIX['savgol_filter']}0", recalc=True)

    trim_tail_prefix = "_".join(['tail', 'trim', tail_prefix])
    trim_linear_prefix = "_".join(['tail', 'trim', linear_prefix])
    return trim_linear_prefix, trim_tail_prefix


@app.cell
def _(det, mo, tone_selector, trim_linear_prefix, trim_tail_prefix):
    mo.output.append(tone_selector)
    mo.output.append(det.targ.plot(x_dim='y', y_dim='y', x_prefix=trim_tail_prefix, y_prefix=trim_linear_prefix, include=tone_selector.value))#.opts(xlim=(-3, 3), ylim=(-1, 3)))
    return


@app.cell
def _(Parameters):
    _params = [("a", 0, True, -10, 10),
               ("b", 0, True, -10, 10),
               ("y_ki", 0, True, -1, 1),
               ("y_qp", 0, True, -1, 1),
               ("y_exp", 4, False, 0, 10),
               ("y_off", 0, True, -1, 1)]

    params = Parameters()
    for _param in _params: params.add(*_param)
    return (params,)


@app.cell
def _(det, params, trim_linear_prefix, trim_tail_prefix):
    det.targ.linewidth_fit(lw_prefix=trim_tail_prefix,
                           lw_shift_prefix=trim_linear_prefix,
                           coupling_linewidth=False,
                           params=[params]*det.targ.num_tones,
                           max_workers=5, 
                           recalc=True)
    fit_linear_prefix = "_".join(["lw_fit", trim_linear_prefix])
    return (fit_linear_prefix,)


@app.cell
def _(
    det,
    fit_linear_prefix,
    mo,
    tone_selector,
    trim_linear_prefix,
    trim_tail_prefix,
):
    mo.output.append(tone_selector)
    mo.output.append((det.targ.plot(x_dim='y', y_dim='y', x_prefix=trim_tail_prefix, y_prefix=trim_linear_prefix, include=tone_selector.value)
                   * det.targ.plot(x_dim='y', y_dim='y', x_prefix=trim_tail_prefix, y_prefix=fit_linear_prefix, include=tone_selector.value)).opts(xlim=(-3, 3), ylim=(-1, 3)))
    return


@app.cell
def _(det, pl, tone_selector):
    det.properties.filter(pl.col('det') == tone_selector.value).select(pl.col("^lw_fit_.*$"), 'a_eff')
    return


@app.cell
def _(np):
    _a = np.ones(10)
    np.logical_and(~np.isnan(_a), _a > 2)
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
    import ccatkidlib.analysis.fit.fit as ccat_fit
    import ccatkidlib.analysis.utils.pair as ccat_pair
    import ccatkidlib.analysis.utils.dataframe as ccat_df
    import ccatkidlib.analysis.routines.common as routines
    from lmfit import Model, Parameters

    from ccatkidlib.analysis.core.detector import Detector
    from ccatkidlib.analysis.core.network import Network

    return Detector, Network, Parameters, ccat_io, routines


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

    # Create file browser for selecting target sweep file
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
def _(mo, targ_selector):
    # Create run button for transforming data
    fit_det_button = mo.ui.run_button(
        kind="success",
        label="Fit Detectors",
        tooltip="Click to fit detectors",
        full_width=True,
        disabled=not targ_selector.value,
    )
    return (fit_det_button,)


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
            list(map(str, (data_dirs[0] / "targ" / f"B{_bid}D{_drid}").iterdir()))
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

    circle_fit_workers_selector = mo.ui.number(
        start=1,
        stop=os.cpu_count(),
        step=1,
        label="Circle Fit Max Workers",
        value=os.cpu_count() // 3,
        full_width=True,
    )

    phase_fit_workers_selector = mo.ui.number(
        start=1,
        stop=os.cpu_count(),
        step=1,
        label="Phase Fit Max Workers",
        value=os.cpu_count() // 4,
        full_width=True,
    )
    return (
        circle_fit_workers_selector,
        phase_fit_workers_selector,
        savgol_workers_selector,
    )


@app.cell
def _(det, mo):
    mo.stop(not det)

    tone_selector = mo.ui.slider(
        steps=det.targ.tones,
        debounce=True,
        show_value=True,
        full_width=False,
        label="Tone",
    )
    return (tone_selector,)


@app.cell
def _(mo):
    phase_fit_window = mo.ui.number(start=0, stop=200, step=0.1, value=2)
    return (phase_fit_window,)


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
def _(
    Network,
    ProcessPoolExecutor,
    circle_fit_workers_selector,
    phase_fit_window,
    phase_fit_workers_selector,
    pl,
    routines,
    savgol_workers_selector,
    tqdm,
):
    def dump_transform(network: Network) -> Network:
        """
        Transformations to run on Network objects before dumping to pickle files
        """
        from ccatkidlib.analysis.routines.common import (
            IQ_circle_origin,
        )

        NAMES, PREFIX = (
            network.analysis_cfg["convention"]["name"],
            network.analysis_cfg["convention"]["prefix"],
        )

        max_workers = max(
            savgol_workers_selector.value,
            circle_fit_workers_selector.value,
            phase_fit_workers_selector.value,
        )

        mismatch_prefix = f"{PREFIX['normalize']}_{PREFIX['scale']}_{PREFIX['remove_impedance_mismatch']}_{PREFIX['rotate']}_{PREFIX['center_origin']}_{PREFIX['translate']}_{PREFIX['center_origin']}_{PREFIX['rotate']}_{PREFIX['remove_cable']}_{PREFIX['rotate']}"

        trim_mismatch_prefix = f"{PREFIX['trim_tail']}_{PREFIX['trim']}_{mismatch_prefix}"

        circle_fit_prefix = f"{PREFIX['IQ_circle_fit']}_{PREFIX['trim_tail']}_{PREFIX['trim']}_{PREFIX['remove_cable']}_{PREFIX['rotate']}"
        norm_scale = f"{PREFIX['remove_impedance_mismatch']}_{PREFIX['remove_cable']}_{PREFIX['rotate']}_{NAMES['magnitude']}"
        norm_circle_fit_prefix = f"{PREFIX['normalize']}_{PREFIX['scale']}_{circle_fit_prefix}"

        params = None
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for det in tqdm(network.data.sort("drive", descending=True)["detector"]):
                det = network.det_dict[det]

                # IQ Circle Center
                # ----------------
                routines.IQ_circle_origin(det,
                                          data="targ",
                                          savgol_window=9,
                                          savgol_order=1,
                                          trim_window=10,
                                          trim_mean_points=10,
                                          mismatch_mean_points=10,
                                          savgol_workers=savgol_workers_selector.value,
                                          circle_fit_workers=circle_fit_workers_selector.value,
                                          normalize=True,
                                          ex=ex
                                    )

                # Phase Fit
                # ---------
                det.targ.res_IQ_trim(prefix=mismatch_prefix,
                                    window=phase_fit_window.value,
                                    mean_points=15,
                                    use_fit=False,
                                    mag_prefix=f"{PREFIX['savgol_filter']}0",
                                    recalc=True
                )
                det.targ.phase(prefix=trim_mismatch_prefix, recalc=True)

                det.targ._properties_df = det.targ.properties.with_columns(
                    (
                        pl.col(
                            [
                                f"{circle_fit_prefix}_{NAMES['IQ_circle_radius']}",
                                f"{circle_fit_prefix}_{NAMES['IQ_circle_center']}_{NAMES['magnitude']}",
                            ]
                        )
                        / pl.col(norm_scale)
                    ).name.prefix(f"{PREFIX['normalize']}_{PREFIX['scale']}_")
                )

                det.fit_result = {}
                det.phase_fit(
                    prefix=trim_mismatch_prefix,
                    circle_fit_prefix=norm_circle_fit_prefix,
                    nonlinear=True,
                    params=params,
                    max_workers=phase_fit_workers_selector.value,
                    recalc=True,
                    ex=ex
                )

                result = det.fit_result[f"{PREFIX['phase_fit']}_{trim_mismatch_prefix}_{NAMES['fit_result']}"]
                params = [res.params if res is not None else None for res in result]
                for param in params: 
                    if param is not None: param.pop('R')

        return network

    return (dump_transform,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
