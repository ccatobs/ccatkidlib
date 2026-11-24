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
    atten_sweep_button,
    atten_sweep_range,
    atten_sweep_step,
    mo,
    stream_time_select,
    sweep_steps_select,
):
    mo.md(rf"""
    ### Attenuation Sweep

    Set attenuation sweep parameters: <br>
    {
        mo.vstack(
            [
                mo.hstack([atten_sweep_range, atten_sweep_step]),
                mo.hstack([sweep_steps_select, stream_time_select]),
            ]
        )
    }

    Start attenuation sweep data collection: <br>
    {atten_sweep_button}
    """)
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
    mo.stop(not RC)

    _com_tos = RC.drone_list
    _num_drones = len(_com_tos)

    init_comb_freqs, init_comb_powers, init_comb_phis, init_comb_drives = (
        [None] * _num_drones,
        [None] * _num_drones,
        [None] * _num_drones,
        [12] * _num_drones,
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
def _(
    RC,
    atten_sweep_button,
    atten_sweep_range,
    atten_sweep_step,
    init_comb_drives,
    mo,
    np,
    stream_time_select,
    sweep_steps_select,
):
    mo.stop(not atten_sweep_button.value)

    _sweep_start, _sweep_stop, _sweep_step = (
        atten_sweep_range.value[0],
        atten_sweep_range.value[1],
        atten_sweep_step.value,
    )

    _atten_diffs = np.arange(
        _sweep_stop, _sweep_start - _sweep_step, -1 * _sweep_step
    )
    for _diff in _atten_diffs:
        _drives = [
            max(min(_drive + _diff, 31.25), 0) for _drive in init_comb_drives
        ]
        RC.set_atten(drive=_drives)
        RC.tune_tone_placement(method="grad")
        RC.take_target_sweep(sweep_steps=sweep_steps_select.value)
        RC.take_timestream(stream_time_select.value)
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

    return R, ccat_io


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
def _(init_comb_browser):
    init_comb_browser.value
    return


@app.cell
def _(RC, mo):
    mo.stop(not RC)

    atten_sweep_step = mo.ui.number(
        start=0.25,
        stop=31,
        step=0.25,
        value=1,
        label="Step Size [dB]",
        full_width=True,
        debounce=True,
    )

    atten_sweep_range = mo.ui.range_slider(
        start=-31.25,
        stop=31.25,
        step=0.25,
        value=[-10, 10],
        label="Attenuation Sweep Range [dB]",
        show_value=True,
        full_width=True,
        debounce=True,
    )

    sweep_steps_select = mo.ui.number(
        start=1,
        stop=10_000,
        step=1,
        value=1000,
        label="Target Sweep Steps",
        full_width=True,
        debounce=True,
    )

    stream_time_select = mo.ui.number(
        start=0,
        value=12,
        label="Timestream Length [seconds]",
        full_width=True,
        debounce=True,
    )

    atten_sweep_button = mo.ui.run_button(
        kind="success",
        tooltip="Click to start attenuation sweep data collection",
        label="Start Attenuation Sweep",
        full_width=True,
    )
    return (
        atten_sweep_button,
        atten_sweep_range,
        atten_sweep_step,
        stream_time_select,
        sweep_steps_select,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
