import marimo

__generated_with = "0.23.10"
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Theoretical KID
    """)
    return


@app.cell
def _(mo):
    f_0_slider = mo.ui.slider(start=1, stop=1000, step=1, value=500, full_width=True, label='Resonant Frequency $f_0$ [MHz]', show_value=True, debounce=True)
    Q_1_slider =  mo.ui.slider(start=1000, stop=1e6, step=100, value=30000, full_width=True, label='Total Quality Factor $Q_r$', show_value=True, debounce=True)
    Q_2_slider =  mo.ui.slider(start=1000, stop=1e6, step=100, value=30000, full_width=True, label='Total Quality Factor $Q_r$', show_value=True, debounce=True)
    Q_c_slider =  mo.ui.slider(start=1000, stop=1e6, step=100, value=60000, full_width=True, label='Coupling Quality Factor $Q_c$', show_value=True, debounce=True)
    return Q_1_slider, Q_2_slider, Q_c_slider, f_0_slider


@app.cell
def _(Q_1_slider, Q_2_slider, Q_c_slider, f_0_slider, mo):
    mo.vstack((f_0_slider, Q_1_slider, Q_2_slider, Q_c_slider))
    return


@app.cell
def s21(
    Q_1_slider,
    Q_2_slider,
    Q_c_slider,
    f_0_slider,
    hv,
    np,
    project_circle,
):
    _f_0 = f_0_slider.value*1e6
    _Q_1, _Q_2 = Q_1_slider.value, Q_2_slider.value
    _Q_c = Q_c_slider.value

    _r_1, _r_2 = _Q_1/(2*_Q_c), _Q_2/(2*_Q_c)

    _fs = np.linspace(_f_0 - 0.25e6, _f_0 + 0.25e6, 500)

    _x = (_fs - _f_0)/_f_0

    _s21_1 = 1 - (_Q_1/_Q_c)*(1/(1 + 2j*_Q_1*_x))
    _s21_2 = 1 - (_Q_2/_Q_c)*(1/(1 + 2j*_Q_2*_x))

    _I_1, _I_2 = -1*(np.real(_s21_1) - (1 - _r_2)), -1*(np.real(_s21_2) - (1 - _r_2))
    _QQ_1, _QQ_2 = np.imag(_s21_1), np.imag(_s21_2)

    _I_star = _r_2 - (2*_r_1*_r_2)/(_r_1+_r_2)

    _I_proj, _Q_proj = project_circle(_I_1, _QQ_1, _r_1, 0.5)

    _lines = -(_I_2 - _I_1)/(_QQ_2 - _QQ_1)*(_QQ_1) + _I_1 

    print(_lines[0])
    print(1 - (2*_r_1*_r_2)/(_r_1+_r_2))
    (hv.Points((_I_1, _QQ_1)).opts(data_aspect=1)
    *hv.Points((_I_2, _QQ_2)).opts(data_aspect=1)
    *hv.Points((_I_proj, _Q_proj)).opts(data_aspect=1))
    #*hv.Points((_lines, np.zeros(len(_lines)))))

    #hv.Curve((_fs, ))
    return


@app.cell
def _(np):
    def project_circle(I, Q, r_1, r_2):
        """
        Project points on a smaller S21 circle onto the larger S21 circle.

        Draws lines from the focus/intersection point I* through each point on
        the smaller circle. Each line intersects the larger circle at the
        corresponding projected point.

        Parameters
        ----------
        I_small, Q_small : ndarray
            I and Q components of the smaller circle
        I_star : float
            I-axis coordinate of the focus/intersection point (Q = 0)
        r_large : float
            Radius of the large S21 circle; its centre lies at (1 - r_large, 0)

        Returns
        -------
        I_proj, Q_proj : ndarray
            Projected I and Q components lying on the large circle
        """

        I_star = -1*(r_2 - (2*r_1*r_2)/(r_1+r_2))
        #print(I_star)
        m = Q/(I-I_star)
        a = 1 + m**2
        b = -2*(m**2 * I_star)
        c = (m*I_star)**2 - r_2 ** 2 
        #print(b**2 - 4*a*c)

        I_proj_pos = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        I_proj_neg = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)

        mask =  (I_proj_neg <= I_star) & (I >= I_star)
        I_proj = np.where(mask, I_proj_pos, I_proj_neg)

        Q_proj = m*(I_proj-I_star)

        return I_proj, Q_proj


    return (project_circle,)


@app.function
def s21(x, Q, R):
    return 2*R*(1/(1 + 2j*Q*x)) - R


@app.cell
def _(np):
    def s21_diff(x, I_proj, Q_proj, Q, R):
        return np.abs(2*R*(1/(1 + 2j*Q*x)) - R - (I_proj + 1j*Q_proj))

    return (s21_diff,)


@app.cell
def _(np):
    def consensus_solution(x_candidates):
        """Find the two most similar solutions and average them."""
        x_candidates = np.asarray(x_candidates)
        distances = np.abs(np.subtract.outer(x_candidates, x_candidates))

        # Find minimum non-zero distance
        distances_nz = np.where(distances > 0, distances, np.inf)
        idx_i, idx_j = np.unravel_index(
            np.argmin(distances_nz), distances_nz.shape
        )

        #print(idx_i, idx_j)

        return (x_candidates[idx_i] + x_candidates[idx_j]) / 2

    return (consensus_solution,)


@app.cell
def _(np, pl):
    a, b = np.arange(0, 10, 1), np.arange(10, 20, 1)

    pl.Series(pl.DataFrame({'a': a, 'b': b}).select(pl.struct([pl.col('a'), pl.col('b')]))).struct.fields
    return


@app.cell
def _():
    return


@app.cell(column=1)
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
    routines.IQ_circle_center(
        det,
        data="targ",
        savgol_window=9,
        savgol_order=1,
        trim_window=10,
        trim_mean_points=10,
        mismatch_mean_points=10,
        savgol_workers=savgol_workers_selector.value,
        circle_fit_workers=circle_fit_workers_selector.value,
        normalize=False,
        recalc=False,
    )

    get_centered_dfs = True
    return (get_centered_dfs,)


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
def _(PREFIX, circle_prefix, det, get_centered_dfs, mo):
    mo.stop(not get_centered_dfs)

    mismatch_prefix = f"{PREFIX['remove_impedance_mismatch']}_{PREFIX['rotate']}_\
    {PREFIX['center_origin']}_{PREFIX['translate']}_\
    {PREFIX['center_origin']}_{PREFIX['rotate']}_\
    {PREFIX['remove_cable']}_{PREFIX['rotate']}"

    circle_fit_prefix = f"{PREFIX['IQ_circle_fit']}_{PREFIX['trim_tail']}_{PREFIX['trim']}_{PREFIX['remove_cable']}_{PREFIX['rotate']}"

    circle_origin_prefix = f"{PREFIX['center_origin']}_{PREFIX['translate']}_\
    {PREFIX['center_origin']}_{PREFIX['rotate']}_\
    {circle_prefix}"

    mismatch_plot_dfs = get_plot_dfs(det, mismatch_prefix)
    circle_plot_dfs = get_plot_dfs(det, circle_origin_prefix)
    return (
        circle_fit_prefix,
        circle_plot_dfs,
        mismatch_plot_dfs,
        mismatch_prefix,
    )


@app.cell
def _(PREFIX, det, get_centered_dfs, mo):
    mo.stop(not get_centered_dfs)

    _cable_prefix = f"{PREFIX['remove_cable']}_{PREFIX['rotate']}"
    _trim_prefix = f"{PREFIX['trim_tail']}_{PREFIX['trim']}_{_cable_prefix}"
    circle_prefix = f"{PREFIX['IQ_circle_fit']}_{_trim_prefix}"

    det.IQ_circle_origin(
            prefix=circle_prefix,
            circle_fit_prefix=circle_prefix,
            data='targ',
        )
    return (circle_prefix,)


@app.cell
def _(circle_fit_prefix, det, mismatch_prefix):
    det.targ.mag(prefix=mismatch_prefix, dB=False)
    proj_df = det.IQ_circle_diss_corr(prefix=mismatch_prefix,
                                                   circle_fit_prefix=circle_fit_prefix,
                                                   recalc=False,
                                                   max_workers=1)
    return


@app.cell
def _(PREFIX, det, mismatch_prefix):
    proj_prefix = f"{PREFIX['dissipation_correction']}_{mismatch_prefix}"
    proj_plot_dfs = get_plot_dfs(det, proj_prefix)
    return (proj_plot_dfs,)


@app.cell
def _(circle_plots_dashboard, create_dashboard, plot_sweep, proj_plot_dfs):
    proj_plots, _mag_phase_opts, _IQ_opts = plot_sweep(
        proj_plot_dfs, include_tones=True
    )

    proj_plots_dashboard = create_dashboard(
        proj_plots, _mag_phase_opts, _IQ_opts
    )

    circle_plots_dashboard[0]*proj_plots_dashboard[0]
    return


@app.cell
def _(circle_fit_prefix, det, mismatch_prefix):
    det.linear_frac_f(prefix=mismatch_prefix, circle_fit_prefix=circle_fit_prefix, mag_prefix='savgol0')
    det.frac_f(prefix='', data='targ', ref_f= 'mid_savgol0_FWHM_f')

    _shifts = det.get_properties(col_name=f"{circle_fit_prefix}_R").to_numpy().T[1]
    det.targ.IQ_shift(prefix=mismatch_prefix, shift_I = _shifts, name='off_res')
    det.IQ_trim(prefix=f"off_res_shift_{mismatch_prefix}", window=2, mag_prefix='savgol0')
    det.targ.mag(prefix=f"tail_trim_off_res_shift_{mismatch_prefix}", dB=False)
    return


@app.cell
def _(det, hv, mismatch_prefix, pl, tone_selector):
    _df = det.targ.get_data(['f', 'ff', f'tail_trim_off_res_shift_{mismatch_prefix}_mag'], include=tone_selector.value)

    _df = _df.rename({_col : _name for _name, _col in zip(['f', 'proj_diss_lin_ff', 'nonlin_ff', 'off_res_dist', 'lin_ff'], _df.columns)}).filter(pl.col('off_res_dist').is_not_null())

    _f, _x_lin_proj_diss, _x_nonlin, _off_res, _x_lin = _df
    _x_diff = (_x_nonlin - _x_lin_proj_diss)*1e6
    hv.Scatter((_off_res**2, _x_diff)).opts(s=10, c=_f, cmap='viridis', xlabel=r"$|z - z_{off}|^2 \propto I^2$", ylabel='$x_{measured} - x_{linear}$ [ppm]', show_grid=True)
    return


@app.cell
def _(tone_selector):
    tone_selector
    return


@app.cell
def _(circle_plot_dfs, create_dashboard, mismatch_plot_dfs, plot_sweep):
    mismatch_plots, _mag_phase_opts, _IQ_opts = plot_sweep(
        mismatch_plot_dfs, include_tones=True
    )

    circle_plots, _circle_mag_phase_opts, _circle_IQ_opts = plot_sweep(
        circle_plot_dfs, include_tones=True
    )

    mismatch_plots_dashboard = create_dashboard(
        mismatch_plots, _mag_phase_opts, _IQ_opts
    )

    circle_plots_dashboard = create_dashboard(
        circle_plots, _circle_mag_phase_opts, _circle_IQ_opts
    )

    mismatch_plots_dashboard[0]*circle_plots_dashboard[0]
    return circle_plots_dashboard, mismatch_plots, mismatch_plots_dashboard


@app.cell
def _(det, mismatch_prefix, tone_selector):
    det.targ.mag(prefix=mismatch_prefix, dB=False)
    fs, I, Q, mag = det.targ.get_data(['f', f'{mismatch_prefix}_mag', f'{mismatch_prefix}_I', f'{mismatch_prefix}_Q'], include=tone_selector.value, strict=True).to_numpy().T

    R = det.get_properties(col_name=['circle_fit_tail_trim_unwind_rotate_R'], include=tone_selector.value).item(0, 1)
    return I, Q, R, fs, mag


@app.cell
def _(det):
    det.properties
    return


@app.cell
def _():
    from scipy.optimize import root_scalar

    return (root_scalar,)


@app.cell
def _(
    I_proj,
    Q_proj,
    R,
    consensus_solution,
    det,
    fs,
    hv,
    mo,
    np,
    root_scalar,
    s21_diff,
    tone_selector,
):
    _f_low, _f_high = det.get_properties(col_name=['low_savgol0_FWHM_f', 'high_savgol0_FWHM_f'], include=tone_selector.value).to_numpy()[0][1:]
    _fwhm = _f_high - _f_low
    _f_0 = det.get_properties(col_name=['mid_savgol0_FWHM_f'], include=tone_selector.value).item(0, 1)

    _Qr = _f_0/_fwhm

    _x = (fs - _f_0)/_f_0

    _s21 = s21(_x, _Qr, R)
    _I, _Q = np.real(_s21), np.imag(_s21)
    mo.output.append(hv.Scatter((_I, _Q)).opts(s=5) * hv.Scatter((I_proj, Q_proj)).opts(s=5))

    _xx = []
    for _I_proj, _Q_proj, _x0 in zip(I_proj, Q_proj, _x):
        _xx.append(root_scalar(s21_diff, args=(_I_proj, _Q_proj, _Qr, R), x0=_x0).root)
    print(_xx)

    _x11 = 1/(2*_Qr)*np.sqrt((2*R)/(I_proj + R)-1)
    _x12 = -1/(2*_Qr)*np.sqrt((2*R)/(I_proj + R)-1)

    _a = 4*Q_proj*_Qr**2 
    _b = 4*R*_Qr
    _c = Q_proj

    _x21 = (-_b + np.sqrt(_b**2 - 4*_a*_c))/(2*_a)
    _x22 = (-_b - np.sqrt(_b**2 - 4*_a*_c))/(2*_a)

    I_x = np.array([_x11, _x11, _x12, _x12])
    Q_x = np.array([_x21, _x22, _x21, _x22])

    diff = np.abs(I_x - Q_x)
    #print(diff)
    index_array = np.argmin(diff, axis=0, keepdims=True)
    #print(index_array)
    _xxx_I = np.take_along_axis(I_x, index_array, axis=0)[0]
    _xxx_Q = np.take_along_axis(Q_x, index_array, axis=0)[0]


    print(_xxx_I - _xxx_Q)
    _xs = []
    for _xx11, _xx12, _xx21, _xx22 in zip(_x11, _x12, _x21, _x22):
        _xs.append(consensus_solution([_xx11, _xx12, _xx21, _xx22]))

    off_res_dist = (Q_proj**2 + (I_proj + R)**2)**(1/2)
    mo.output.append(hv.Scatter((off_res_dist, _x - _xx)).opts(c=fs, cmap='viridis', show_grid=True, colorbar=True, ylim=(-0.0001, 0.0001)))
    return


@app.cell
def _(I, Q, R, fs, hv, mag, np, project_circle):
    I_proj, Q_proj = [], []
    I_shift = R - mag
    I_corr = I - I_shift
    f_proj = []

    for _f, _mag, _I, _Q in zip(fs, mag, I_corr, Q):
        _I_proj, _Q_proj = project_circle(_I, _Q, _mag, R)
        I_proj.append(_I_proj), Q_proj.append(_Q_proj), f_proj.append(_f)
    I_proj, Q_proj = np.array(I_proj), np.array(Q_proj)
    #hv.Points((I_proj, Q_proj)).opts(data_aspect=1)
    #hv.Scatter((I_proj, Q_proj)).opts(data_aspect=1, c=f_proj, cmap='viridis')
    hv.Scatter((f_proj, np.arctan2(Q_proj,I_proj))).opts(s=1)*hv.Scatter((f_proj, np.arctan2(Q,I)    )).opts(s=1)#.opts(data_aspect=1)
    return I_proj, Q_proj


@app.cell(hide_code=True)
def _(mismatch_plots_dashboard, mo):
    mo.md(rf"""
    ### Centered IQ Circle (*Best Viewed as Fullscreen*)

    {mismatch_plots_dashboard[0]}
    """)
    return


@app.cell(disabled=True)
def _(
    PREFIX,
    det,
    mismatch_prefix,
    phase_fit_window,
    phase_fit_workers_selector,
):
    det.IQ_trim(
        prefix=mismatch_prefix,
        window=phase_fit_window.value,
        mean_points=15,
        use_fit=False,
        mag_prefix=f"{PREFIX['savgol_filter']}0",
        recalc=True
    )

    trim_mismatch_prefix = f"{PREFIX['trim_tail']}_{PREFIX['trim']}_{mismatch_prefix}"
    det.targ.phase(prefix=trim_mismatch_prefix, recalc=True)

    _circle_fit_prefix = f"{PREFIX['IQ_circle_fit']}_{PREFIX['trim_tail']}_{PREFIX['trim']}_{PREFIX['remove_cable']}_{PREFIX['rotate']}"
    det.phase_fit(
        prefix=trim_mismatch_prefix,
        circle_fit_prefix=_circle_fit_prefix,
        nonlinear=True,
        max_workers=phase_fit_workers_selector.value,
        recalc=True
    )
    return (trim_mismatch_prefix,)


@app.cell(disabled=True)
def _(PREFIX, det, trim_mismatch_prefix):
    phase_plot_df = det.targ.phase_plot(
        prefix=f"{PREFIX['phase_fit']}_{trim_mismatch_prefix}",
        include=det.targ.tones,
        return_df=True,
        return_fig=False,
        save_fig=False,
    )[0]
    return (phase_plot_df,)


@app.cell(disabled=True)
def _(LABELS, mismatch_plots, opts, phase_plot_df, pl, tone_selector):
    _fit_plot_opts = opts.Curve(
        xlabel=LABELS["phase"]["xlabel"], ylabel=LABELS["phase"]["ylabel"], color='orange'
    )

    _fit_plot = phase_plot_df.filter(
        pl.col("det") == tone_selector.value
    ).hvplot.line(x="f", y="phase", label='Phase Fit')

    phase_fit_plot =  mismatch_plots['phase']*_fit_plot.opts(_fit_plot_opts)
    return (phase_fit_plot,)


@app.cell(hide_code=True)
def _(mo, phase_fit_plot, phase_fit_window, tone_selector):
    mo.md(rf"""
    ### Detector Phase Fit

    {mo.lazy(mo.vstack([tone_selector, phase_fit_window, phase_fit_plot]))}
    """)
    return


@app.cell(disabled=True)
def _(
    PREFIX,
    Parameters,
    ccat_fit,
    det,
    hv,
    mo,
    tone_selector,
    trim_mismatch_prefix,
):
    _f, _I, _Q, _phase = det.targ.get_data(col_name=['f', f"{trim_mismatch_prefix}_phase", f"{trim_mismatch_prefix}_I", f"{trim_mismatch_prefix}_Q"], include=tone_selector.value, strict=True).to_numpy().T

    _circle_fit_prefix = f"{PREFIX['IQ_circle_fit']}_{PREFIX['trim_tail']}_{PREFIX['trim']}_{PREFIX['remove_cable']}_{PREFIX['rotate']}"
    _R = det.targ.get_properties(f"{_circle_fit_prefix}_R", include=tone_selector.value).to_numpy()[0][1]

    _params = Parameters()
    _params.add("gamma", -2, True, -1e4, 1e4)
    _params.add("beta", 0, True, -1e4, 1e4)


    _result = ccat_fit.phase_fit_al(_f, _phase, I=_I, Q=_Q, R=_R, params=_params, nonlinear=True)
    _params = _result.params
    _params['gamma'].value = -0.2
    mo.output.append(hv.Curve((_f, _result.model.eval(params=_params, f=_f, I=_I, Q=_Q))).opts(linewidth=1, ms=1, marker='o'))
    #mo.output.append(_result.best_values)
    #mo.output.append(hv.Curve(_result.best_fit).opts(linewidth=1, ms=1, marker='o'))
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

    return Detector, Parameters, ccat_fit, ccat_io, routines


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
        full_width=True,
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
def _(hv, mo):
    def create_dashboard(plots, mag_phase_opts, IQ_opts):
        mag_phase_layout = hv.Layout([plots["mag"], plots["phase"]]).cols(1)
        plot = mo.hstack(
            [
                plots["IQ"].opts(IQ_opts),
                mag_phase_layout.opts(*mag_phase_opts),
            ],
            widths=[1, 1],
        )
        return plots["IQ"].opts(IQ_opts),
        #return mo.vstack([tone_selector, plot])  # Best viewed in fullscreen mode
    return (create_dashboard,)


if __name__ == "__main__":
    app.run()
