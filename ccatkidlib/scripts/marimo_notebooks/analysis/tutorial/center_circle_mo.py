import marimo

__generated_with = "0.23.10"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""
    ### Overview

    Kinetic inductance detectors (KIDs) are superconducting microwave resonators that detect photons by exploiting the change in kinetic inductance caused by quasiparticle excitation. In a standard microwave readout scheme, a probe tone is swept across the KID's resonance frequency and the complex transmission $S_{21}$ is recorded in the IQ (In-phase/Quadrature) plane. As the readout frequency sweeps through resonance, the transmitted signal traces a **circle in IQ space**.

    Before the IQ circle can be used for calibration or analysis — for example, to extract resonance frequency shifts, responsivity, or noise spectra — it must be properly **centered at the origin** and **oriented**. This notebook walks through the `IQ_circle_center` pipeline step by step:

    1. **Remove cable delay** — derotate the IQ data to undo the frequency-dependent phase accumulated along the readout chain
    2. **Trim tails** — discard samples far from the resonance using a FWHM-based window
    3. **Fit IQ circle** — fit a circle to the trimmed IQ data to determine its center and radius
    4. **Center at origin** — translate and rotate the circle so that its center lies at the origin
    5. **Remove impedance mismatch** — rotate so that the off-resonance point lies on the positive real axis
    """)
    return


@app.cell(hide_code=True)
def _(analysis_cfg_browser, cfg_editor, data_browser, data_desc, mo):
    mo.md(rf"""
    ### Select Data to Load

    All *ccatkidlib* data analysis objects (**Detector**, **Network**, etc.) require an `analysis_config.yaml` that specifies file paths, naming conventions, and analysis parameters. Select your config file using the browser on the left; its contents will be displayed on the right for reference. Once a config is loaded, select a data directory below — directories are identified by their Unix timestamp session ID (e.g. `1773339248`). A description of the selected session will appear in the callout box.

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
def _(com_to_selector, mo, run_pipeline_button, targ_selector):
    mo.md(rf"""
    ### Select Drone & Sweep

    The Radio Frequency System on a Chip (RFSoC) drone is the readout chain used to acquire the data.  Select the drone below, then choose a **target sweep** — a fine-grained frequency sweep taken near each resonance tone to trace out the IQ circle. Only drones that recorded data across all selected sessions are shown. Once a file is selected, click **Center Circle** to run the full pipeline.

    {mo.vstack((mo.hstack([com_to_selector, targ_selector], widths=[1, 1]), run_pipeline_button))}
    """)
    return


@app.cell(column=1)
def _(
    Detector,
    analysis_cfg,
    com_to_selector,
    mo,
    run_pipeline_button,
    targ_selector,
):
    mo.stop(not run_pipeline_button.value)

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
    return LABELS, NAMES, PREFIX


@app.cell
def _(det):
    raw_plot_dfs = get_plot_dfs(det, prefix="")
    return (raw_plot_dfs,)


@app.cell
def _(create_dashboard, plot_sweep, raw_plot_dfs):
    raw_plots, _mag_phase_opts, _IQ_opts = plot_sweep(raw_plot_dfs)
    raw_plots_dashboard = create_dashboard(raw_plots, _mag_phase_opts, _IQ_opts)
    return (raw_plots_dashboard,)


@app.cell(hide_code=True)
def _(mo, raw_plots_dashboard):
    mo.md(rf"""
    ### Raw Target Sweep Data (*Best Viewed As Fullscreen*)

    Below we plot the unprocessed KID data: the **IQ circle** in the complex plane, the **transmission magnitude** in dB, and the **phase** response. At this stage the IQ circle is offset from the origin due to the background transmission level and appears wound or rotated due to cable delay accumulated along the readout chain.  The **red star** marks the readout tone frequency, which is placed near the resonance center during the measurement.

    {raw_plots_dashboard}
    """)
    return


@app.cell
def _(NAMES, det):
    det.vna.phase()  # Use VNA sweep phase data to calculate cable delay
    det.cable_delay
    det.IQ_unwind(
        delay_col=f"vna_{NAMES['cable_delay']}", data="targ"
    )  # Remove cable delay from target sweeps

    get_cable_dfs = True
    return (get_cable_dfs,)


@app.cell
def _(PREFIX, det, get_cable_dfs, mo):
    mo.stop(not get_cable_dfs)
    cable_prefix = f"{PREFIX['remove_cable']}_{PREFIX['rotate']}"
    unwind_plot_dfs = get_plot_dfs(det, cable_prefix)
    return cable_prefix, unwind_plot_dfs


@app.cell
def _(create_dashboard, plot_sweep, unwind_plot_dfs):
    unwind_plots, _mag_phase_opts, _IQ_opts = plot_sweep(unwind_plot_dfs)
    unwind_plots_dashboard = create_dashboard(
        unwind_plots, _mag_phase_opts, _IQ_opts
    )
    return unwind_plots, unwind_plots_dashboard


@app.cell(hide_code=True)
def _(mo, unwind_plots_dashboard):
    mo.md(rf"""
    ### Remove Cable Delay

    The readout signal travels through coaxial cables and other transmission line components before reaching the detector, accumulating a frequency-dependent phase $\phi(\nu) = 2\pi\tau\nu$ where $\tau$ is the cable delay time. This phase winding rotates the IQ data as a function of frequency, causing the IQ circle to appear twisted. The cable delay $\tau$ is estimated from the VNA calibration sweep by fitting the linear phase slope across frequency, then removed from the target sweep by derotating each sample by $-2\pi\tau\nu$. After this correction the IQ circle should appear more circular.

    {mo.lazy(unwind_plots_dashboard)}
    """)
    return


@app.cell
def _(
    NAMES,
    cable_prefix,
    det,
    savgol_k_selector,
    savgol_max_workers_selector,
    savgol_window_selector,
    trim_mean_points_selector,
    trim_window_selector,
):
    det.targ.savgol(
        col_name=NAMES["magnitude"],
        prefix=det.analysis_cfg["convention"]["prefix"]["decible"],
        deriv=0,
        window=savgol_window_selector.value,
        k=savgol_k_selector.value,
        max_workers=savgol_max_workers_selector.value,
        recalc=True,
    )

    savgol_prefix = f"{det.analysis_cfg['convention']['prefix']['savgol_filter']}0_{det.analysis_cfg['convention']['prefix']['decible']}"
    det.IQ_trim(
        prefix=cable_prefix,
        window=trim_window_selector.value,
        mean_points=trim_mean_points_selector.value,
        use_fit=False,
        mag_prefix=savgol_prefix,
        recalc=True,
    )

    get_trim_dfs = True
    return get_trim_dfs, savgol_prefix


@app.cell
def _(PREFIX, cable_prefix, det, get_trim_dfs, mo):
    mo.stop(not get_trim_dfs)

    trim_prefix = f"{PREFIX['trim_tail']}_{PREFIX['trim']}_{cable_prefix}"
    trim_plot_dfs = get_plot_dfs(det, trim_prefix)
    return trim_plot_dfs, trim_prefix


@app.cell
def _(NAMES, PREFIX, det, pl, savgol_prefix, tone_selector):
    _suffix = f"{savgol_prefix}_{NAMES['full_width_half_max']}_{NAMES['sample']}"
    _low_prefix, _mid_prefix, _high_prefix = (
        f"{PREFIX['low_frequency_side']}_{_suffix}",
        f"{PREFIX['middle_frequency_point']}_{_suffix}",
        f"{PREFIX['high_frequency_side']}_{_suffix}",
    )

    _tone = tone_selector.value
    fwhm_samples = (
        det.properties.filter(pl.col("det") == _tone)
        .select(_low_prefix, _mid_prefix, _high_prefix)
        .to_numpy()[0]
    )
    _f_df = det.targ.get_data(["sample", NAMES["frequency"]], include=_tone)
    fwhm_freqs = (
        _f_df.filter(pl.col("sample").is_in(fwhm_samples))
        .select(pl.exclude("sample"))
        .to_numpy()
        .T[0]
    )
    return (fwhm_samples,)


@app.cell
def _(
    create_dashboard,
    fwhm_samples,
    opts,
    pl,
    plot_sweep,
    tone_selector,
    trim_plot_dfs,
    unwind_plot_dfs,
    unwind_plots,
):
    _tone = tone_selector.value
    _fwhm_scatter_opts = opts.Scatter("Scatter.FWHM", c="green", s=50, marker="D")

    trim_plots, _mag_phase_opts, _IQ_opts = plot_sweep(trim_plot_dfs)
    trim_unwind_plots = {
        _k: unwind_plots[_k] * _v for _k, _v in trim_plots.items()
    }

    # Overlay green diamond markers at FWHM sample points on all plots
    for _type, _df in unwind_plot_dfs.items():
        _fwhm_df = _df.filter(
            (pl.col("det") == _tone) & (pl.col("sample").is_in(fwhm_samples.tolist()))
        )
        trim_unwind_plots[_type] *= _fwhm_df.hvplot.scatter(
            x="x", y="y", label="FWHM"
        ).opts(_fwhm_scatter_opts)

    _mag_phase_opts += [opts.VLines(linewidth=1, linestyle="dotted", c="k")]
    trim_plots_dashboard = create_dashboard(
        trim_unwind_plots, _mag_phase_opts, _IQ_opts
    )
    return trim_plots, trim_plots_dashboard


@app.cell(hide_code=True)
def _(
    mo,
    savgol_k_selector,
    savgol_max_workers_selector,
    savgol_window_selector,
    trim_mean_points_selector,
    trim_plots_dashboard,
    trim_window_selector,
):
    mo.md(rf"""
    ### Trim Tails

    The target sweep spans a wide frequency range, but only samples close to the resonance carry useful information — samples in the far "tails" are dominated by background transmission and degrade the subsequent circle fit. The trim window is expressed as a multiple of the **FWHM** (Full-Width at Half-Maximum) of the resonance dip, determined from a **Savitzky-Golay smoothed** magnitude spectrum. The Savitzky-Golay filter fits a polynomial of order **k** within a sliding window of fixed size to smooth the noisy raw magnitude before the FWHM is computed. The **green diamond** markers indicate the identified low, mid, and high FWHM sample points.

    {mo.hstack([savgol_window_selector, savgol_k_selector, savgol_max_workers_selector], widths=[1, 1, 1])}

    {mo.hstack([trim_window_selector, trim_mean_points_selector], widths=[1, 1])}

    {mo.lazy(trim_plots_dashboard)}
    """)
    return


@app.cell
def _(circle_fit_max_workers_selector, det, trim_prefix):
    det.IQ_circle_fit(
        prefix=trim_prefix, max_workers=circle_fit_max_workers_selector.value, recalc=True
    )  # Fit target sweep IQ circles

    get_circle_dfs = True
    return (get_circle_dfs,)


@app.cell
def _(PREFIX, det, get_circle_dfs, mo, trim_prefix):
    mo.stop(not get_circle_dfs)

    circle_prefix = f"{PREFIX['IQ_circle_fit']}_{trim_prefix}"
    circle_fit_plot_dfs = get_plot_dfs(det, circle_prefix)
    return circle_fit_plot_dfs, circle_prefix


@app.cell
def _(circle_fit_plot_dfs, create_dashboard, plot_sweep, trim_plots):
    circle_fit_plots, _mag_phase_opts, _IQ_opts = plot_sweep(
        circle_fit_plot_dfs, include_tones=False
    )
    circle_fit_plots = {
        _k: circle_fit_plots[_k] * _v if _k == "IQ" else _v
        for _k, _v in trim_plots.items()
    }
    circle_fit_plots_dashboard = create_dashboard(
        circle_fit_plots, _mag_phase_opts, _IQ_opts
    )
    return (circle_fit_plots_dashboard,)


@app.cell(hide_code=True)
def _(circle_fit_max_workers_selector, circle_fit_plots_dashboard, mo):
    mo.md(rf"""
    ### Fit IQ Circle

    With the tails trimmed, a circle is fit to the IQ data near the resonance. The fit returns the center coordinates $(I_0, Q_0)$ and radius $R$ of the resonance circle in the complex plane. These parameters capture both the resonance response and any residual background offset remaining after cable delay removal, and are stored in the detector properties DataFrame for use in the centering step.

    {circle_fit_max_workers_selector}

    {circle_fit_plots_dashboard}
    """)
    return


@app.cell
def _(cable_prefix, circle_prefix, det):
    det.IQ_circle_origin(
        prefix=cable_prefix,
        circle_fit_prefix=circle_prefix,
        data="targ",
        recalc=False,
    )  # Rotate and translate circle to the origin

    get_center_dfs = True
    return (get_center_dfs,)


@app.cell
def _(PREFIX, cable_prefix, det, get_center_dfs, mo):
    mo.stop(not get_center_dfs)

    center_prefix = f"{PREFIX['center_origin']}_{PREFIX['translate']}_{PREFIX['center_origin']}_{PREFIX['rotate']}_{cable_prefix}"
    center_plot_dfs = get_plot_dfs(det, center_prefix)
    return center_plot_dfs, center_prefix


@app.cell
def _(center_plot_dfs, create_dashboard, hv, opts, plot_sweep):
    _origin_scatter = hv.Scatter([(0, 0)], label="Origin").opts(
        opts.Scatter("Scatter.Origin", c="black", s=100, marker="+")
    )
    center_plots, _mag_phase_opts, _IQ_opts = plot_sweep(
        center_plot_dfs, include_tones=True
    )
    center_plots["IQ"] = center_plots["IQ"] * _origin_scatter
    center_plots_dashboard = create_dashboard(
        center_plots, _mag_phase_opts, _IQ_opts
    )
    return (center_plots_dashboard,)


@app.cell(hide_code=True)
def _(center_plots_dashboard, mo):
    mo.md(rf"""
    ### Center IQ Circle

    Using the fitted circle center $(I_0, Q_0)$, the IQ data is first **rotated** onto the real axis then **translated** to the origin. The black **+** marker indicates the origin. At this stage the circle is centered but may still carry a rotational offset due to an impedance mismatch between the feedline and the resonator — this is corrected in the final step.

    {mo.lazy(center_plots_dashboard)}
    """)
    return


@app.cell
def _(center_prefix, det):
    det.IQ_circle_rotate(
        prefix=center_prefix,
        data="targ",
        rotation="mismatch",
        recalc=False,
    )

    get_mismatch_dfs = True
    return (get_mismatch_dfs,)


@app.cell
def _(PREFIX, center_prefix, det, get_mismatch_dfs, mo):
    mo.stop(not get_mismatch_dfs)

    mismatch_prefix = (
        f"{PREFIX['remove_impedance_mismatch']}_{PREFIX['rotate']}_{center_prefix}"
    )
    mismatch_plot_dfs = get_plot_dfs(det, mismatch_prefix)
    return (mismatch_plot_dfs,)


@app.cell
def _(create_dashboard, hv, mismatch_plot_dfs, opts, plot_sweep):
    _origin_scatter = hv.Scatter([(0, 0)], label="Origin").opts(
        opts.Scatter("Scatter.Origin", c="black", s=100, marker="+")
    )
    mismatch_plots, _mag_phase_opts, _IQ_opts = plot_sweep(
        mismatch_plot_dfs, include_tones=True
    )
    mismatch_plots["IQ"] = mismatch_plots["IQ"] * _origin_scatter
    mismatch_plots_dashboard = create_dashboard(
        mismatch_plots, _mag_phase_opts, _IQ_opts
    )
    return (mismatch_plots_dashboard,)


@app.cell(hide_code=True)
def _(
    circle_fit_max_workers_selector,
    mismatch_plots_dashboard,
    mo,
    savgol_k_selector,
    savgol_max_workers_selector,
    savgol_window_selector,
    trim_mean_points_selector,
    trim_window_selector,
):
    mo.md(rf"""
    ### Remove Impedance Mismatch

    An impedance mismatch between the transmission feedline and the resonator introduces an additional rotation of the IQ circle, shifting the off-resonance point away from the positive real axis by a mismatch angle $\theta_m$. This angle is estimated from the data and removed, placing the off-resonance point on the negative real axis. After this final correction the IQ circle is fully centered at the origin and correctly oriented — the standard form used for responsivity calibration and noise analysis. The black **+** marker indicates the origin. All upstream pipeline parameters can be adjusted below.

    {mo.hstack([savgol_window_selector, savgol_k_selector, savgol_max_workers_selector], widths=[1, 1, 1])}

    {mo.hstack([trim_window_selector, trim_mean_points_selector, circle_fit_max_workers_selector], widths=[1, 1, 1])}

    {mo.lazy(mismatch_plots_dashboard)}
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

    from ccatkidlib.analysis.core.detector import Detector

    return Detector, ccat_io


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
def _(HOME_DIR, Path, analysis_cfg, mo):
    # Extract root data directory from analysis config file
    root_data_dir = (
        analysis_cfg["file_paths"]["root_data_dir"] if analysis_cfg else HOME_DIR
    )

    if not Path(root_data_dir).exists(): root_data_dir = HOME_DIR
    
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
def _(mo, targ_selector):
    run_pipeline_button = mo.ui.run_button(
        kind="success",
        disabled=not targ_selector.value,
        tooltip="Click to run circle centering pipeline",
        label="Center Circle",
        full_width=True,
    )
    return (run_pipeline_button,)


@app.cell
def _(mo, os):
    # Create UI elements for transforming data
    # ----------------------------------------

    # Create selectors for Savitzky-Golay filter parameters
    savgol_window_selector = mo.ui.number(
        start=3,
        stop=99,
        step=1,
        label="Savgol Window",
        value=9,
        full_width=True,
    )

    savgol_k_selector = mo.ui.number(
        start=0,
        stop=10,
        step=1,
        label="Savgol k (Polynomial Order)",
        value=1,
        full_width=True,
    )

    savgol_max_workers_selector = mo.ui.number(
        start=1,
        stop=os.cpu_count(),
        step=1,
        label="Savgol Max Workers",
        value=2,
        full_width=True,
    )

    # Create selectors for IQ trim parameters
    trim_window_selector = mo.ui.number(
        start=1,
        stop=5,
        step=0.1,
        label="Trim Window",
        value=3,
        full_width=True,
    )

    trim_mean_points_selector = mo.ui.number(
        start=1,
        stop=100,
        step=1,
        label="Trim Mean Points",
        value=10,
        full_width=True,
    )

    # Create selector for IQ circle fit max workers
    circle_fit_max_workers_selector = mo.ui.number(
        start=1,
        stop=os.cpu_count(),
        step=1,
        label="Circle Fit Max Workers",
        value=8,
        full_width=True,
    )
    return (
        circle_fit_max_workers_selector,
        savgol_k_selector,
        savgol_max_workers_selector,
        savgol_window_selector,
        trim_mean_points_selector,
        trim_window_selector,
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
    det.targ.mag(prefix=prefix, dB=True, recalc=True)
    det.targ.phase(prefix=prefix, recalc=True)

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
