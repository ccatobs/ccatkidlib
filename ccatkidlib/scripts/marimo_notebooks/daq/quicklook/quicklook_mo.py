import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(cfg_editor, init_daq_button, init_switches, mo, sys_cfg_browser):
    mo.md(rf"""
    ### Initialize R Control Object

    {
        mo.hstack(
            [
                mo.vstack(
                    [sys_cfg_browser, init_switches, init_daq_button],
                    justify="start",
                    align="center",
                ),
                cfg_editor,
            ],
            widths=[0.5, 0.5],
        )
    }
    """)
    return


@app.cell(hide_code=True)
def _(init_comb_browser, mo):
    mo.md(rf"""
    ### Set Initial Tuning

    {mo.vstack([init_comb_browser])}
    """)
    return


@app.cell(hide_code=True)
def _(
    mo,
    stream_time_select,
    sweep_steps_select,
    target_button,
    timestream_button,
):
    mo.md(rf"""
    ### Data Collection

    {mo.vstack([mo.hstack([target_button, sweep_steps_select], widths=[2, 1]), mo.hstack([timestream_button, stream_time_select], widths=[2, 1])])}
    """)
    return


@app.cell
def _(mo, targ_dashboards):
    mo.ui.tabs(targ_dashboards, lazy=True)
    return


@app.cell
def _():
    return


@app.cell(column=1)
def _(
    R,
    init_boards_switch,
    init_daq_button,
    init_drones_switch,
    mo,
    sys_cfg_path,
):
    mo.stop(not init_daq_button.value)

    RC = R(
        cfg_path=sys_cfg_path,
        init_boards=init_boards_switch.value,
        init_drones=init_drones_switch.value,
    )
    return (RC,)


@app.cell
def _(RC, init_comb_browser, mo, np):
    mo.stop(not RC or not init_comb_browser.value)

    _com_tos = RC.drone_list
    _num_drones = len(_com_tos)

    init_comb_freqs, init_comb_powers, init_comb_phis, init_comb_drives = (
        [None] * _num_drones,
        [None] * _num_drones,
        [None] * _num_drones,
        [6] * _num_drones,
    )

    _init_comb_dir = init_comb_browser.value[0].path

    for _i, _com_to in enumerate(_com_tos):
        _bid, _drid = _com_to.split(".")

        _drone_dir = _init_comb_dir / f"B{_bid}D{_drid}"

        init_comb_freqs[_i] = np.load(_drone_dir / "best_freqs.npy")
        init_comb_powers[_i] = np.load(_drone_dir / "best_powers.npy")
        init_comb_phis[_i] = np.load(_drone_dir / "best_phis.npy")
        init_comb_drives[_i] = np.load(_drone_dir / "best_drives.npy")[0]
    return init_comb_drives, init_comb_freqs, init_comb_phis, init_comb_powers


@app.cell
def _(
    RC,
    init_comb_drives,
    init_comb_freqs,
    init_comb_phis,
    init_comb_powers,
    np,
):
    RC.set_atten(drive=np.array(init_comb_drives))
    RC.take_vna_sweep()  # Take VNA Sweep to get cable delay
    RC.tune_tone_placement(
        tone_freqs=init_comb_freqs,
        tone_powers=init_comb_powers,
        tone_phis=init_comb_phis,
        method="grad",
    )
    RC.take_target_sweep()
    return


@app.cell
def _(RC, mo, sweep_steps_select, target_button):
    mo.stop(not target_button.value)

    targ_files = RC.take_target_sweep(sweep_steps=sweep_steps_select.value)
    return (targ_files,)


@app.cell
def _(Detector, ProcessPoolExecutor, RC, center_IQ, targ_files, tqdm):
    targ_dets = {}
    with ProcessPoolExecutor(max_workers=20) as ex:
        for _com_to, _file in tqdm(
            zip(RC.drone_list, targ_files), total=len(targ_files)
        ):
            _det = Detector(com_to=_com_to, targ_path=_file)
            center_IQ(det=_det, data="targ", max_workers=20, ex=ex)
            _det.IQ_circle_rotate(
                prefix="origin_shift_origin_rotate_unwind_rotate",
                data="targ",
                rotation="mismatch",
            )
            _det.targ.phase(
                prefix="mismatch_rotate_origin_shift_origin_rotate_unwind_rotate"
            )
            targ_dets[_com_to] = _det
    return (targ_dets,)


@app.cell
def _(targ_dets, tqdm):
    targ_dashboards = {}
    for _com_to, _det in tqdm(targ_dets.items()):
        _mag_plot = _det.targ.mag_plot(
            prefix="dB", grouping="groupby", include=_det.targ.tones
        )
        _IQ_plot = _det.targ.IQ_plot(
            prefix="mismatch_rotate_origin_shift_origin_rotate_unwind_rotate",
            grouping="groupby",
            include=_det.targ.tones,
        )
        _phase_plot = _det.targ.phase_plot(
            prefix="mismatch_rotate_origin_shift_origin_rotate_unwind_rotate",
            grouping="groupby",
            include=_det.targ.tones,
        )

        targ_dashboards[_com_to] = (_mag_plot + _phase_plot + _IQ_plot).cols(1).opts(sublabel_format='')
    return (targ_dashboards,)


@app.cell
def _(RC, mo, stream_time_select, timestream_button):
    mo.stop(not timestream_button.value)

    stream_files = RC.take_timestream(stream_time_select.value)
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

    return (np,)


@app.cell
def _():
    # ccatkidlib
    import ccatkidlib.io as ccat_io
    import ccatkidlib.log as ccat_log
    import ccatkidlib.analysis.utils.pair as ccat_pair
    import ccatkidlib.analysis.utils.dataframe as ccat_df

    from ccatkidlib.rfsoc.rfsoc_daq import R
    from ccatkidlib.analysis.core.network import Network
    from ccatkidlib.analysis.core.timestream import Timestream
    from ccatkidlib.analysis.core.target import Target
    from ccatkidlib.analysis.core.detector import Detector

    return Detector, R, ccat_io


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
    return


@app.cell
def _(Path, mo, os):
    HOME_DIR = Path(os.environ["HOME"])
    sys_cfg_browser = mo.ui.file_browser(
        initial_path=HOME_DIR,
        filetypes=[".yaml"],
        multiple=False,
        ignore_empty_dirs=True,
        label="Select system configuration file...",
    )
    return HOME_DIR, sys_cfg_browser


@app.cell
def _(ccat_io, json, mo, sys_cfg_browser):
    _editor_height = 750
    ext_cfg, io_cfg = {}, {}
    if _browser_value := sys_cfg_browser.value:
        sys_cfg_path = _browser_value[0].path
        ext_cfg, io_cfg = ccat_io.load_config(cfg_path=sys_cfg_path)

    cfg_editor = mo.ui.code_editor(
        value=json.dumps(io_cfg, indent=4) if io_cfg else "",
        disabled=True,
        min_height=_editor_height,
        max_height=_editor_height,
        placeholder="Configuration file contents will display here once a valid file is selected!",
    )
    return cfg_editor, io_cfg, sys_cfg_path


@app.cell
def _(mo):
    init_boards_switch = mo.ui.switch(label="Initialize Boards")
    init_drones_switch = mo.ui.switch(label="Initialize Drones")

    init_switches = mo.hstack(
        [init_boards_switch, init_drones_switch], align="start", justify="center"
    )
    return init_boards_switch, init_drones_switch, init_switches


@app.cell
def _(io_cfg, mo):
    init_daq_button = mo.ui.run_button(
        kind="success",
        disabled=io_cfg == {},
        tooltip="Click to create a new data acquisition object using the selected system configuration file.",
        label="Create DAQ Object",
        full_width=True,
    )
    return (init_daq_button,)


@app.cell
def _(HOME_DIR, mo):
    init_comb_browser = mo.ui.file_browser(
        initial_path=HOME_DIR,
        multiple=False,
        ignore_empty_dirs=True,
        label="Select directory with inital comb files...",
        selection_mode="directory",
    )
    return (init_comb_browser,)


@app.cell
def _(init_comb_browser, mo):
    sweep_steps_select = mo.ui.number(
        start=1,
        stop=10_000,
        step=1,
        value=501,
        label="Target Sweep Steps",
        full_width=True,
        debounce=True,
    )

    target_button = mo.ui.run_button(
        kind="success",
        tooltip="Click to take target sweep.",
        label="Take Target Sweep",
        full_width=True,
        disabled=not init_comb_browser.value,
    )

    stream_time_select = mo.ui.number(
        start=0,
        value=12,
        label="Timestream Length [seconds]",
        full_width=True,
        debounce=True,
    )

    timestream_button = mo.ui.run_button(
        kind="success",
        tooltip="Click to take timestream.",
        label="Take Timestream",
        full_width=True,
        disabled=not init_comb_browser.value,
    )
    return (
        stream_time_select,
        sweep_steps_select,
        target_button,
        timestream_button,
    )


@app.cell
def _():
    return


@app.cell(column=3)
def _(Detector, ProcessPoolExecutor):
    def center_IQ(
        det: Detector,
        data: str = "both",
        delay_col: str = "network_cable_delay",
        recalc: bool = False,
        max_workers=1,
        ex: ProcessPoolExecutor = None,
    ):
        """
        Center KID IQ circles at the origin
        """

        det.targ.mag(dB=True, recalc=recalc)  # Calculate magnitude
        det.targ.savgol(
            col_name="mag",
            prefix="dB",
            deriv=0,
            window=9,
            k=1,
            max_workers=min(10, max_workers),
            recalc=recalc,
            ex=ex,
        )  # Apply savgol filter to smooth out magnitude data
        det.IQ_unwind(
            data=data, delay_col=delay_col, recalc=recalc
        )  # Remove cable delay
        det.IQ_trim(
            prefix="unwind_rotate",
            window=2,
            use_fit=False,
            mag_prefix="savgol0_dB",
            recalc=recalc,
        )  # Use savgol filtered data to trim off ends of resonators (keep ``window`` linewidths of data)

        det.IQ_circle_fit(
            prefix="tail_trim_unwind_rotate",
            max_workers=min(15, max_workers),
            recalc=recalc,
            ex=ex,
        )  # Fit trimmed IQ circle data
        det.IQ_circle_real(
            prefix="unwind_rotate",
            circle_fit_col="circle_fit_tail_trim_unwind_rotate",
            loc="origin",
            data=data,
            recalc=recalc,
        )  # Rotate and translate circle to the origin
        return det

    return (center_IQ,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
