import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(
    cfg_editor,
    fts_agent_name,
    init_daq_button,
    init_fts_button,
    init_switches,
    mo,
    sys_cfg_browser,
):
    mo.md(rf"""
    ### Initialize R Control Object

    {
        mo.vstack(
            [
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
                ),
                mo.hstack([init_fts_button, fts_agent_name], widths=[0.5, 0.5]),
            ]
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
def _(fts_pos_feed_name, hk_dir_browser, mo):
    mo.md(rf"""
    ### FTS Position Logging

    {mo.vstack([hk_dir_browser, fts_pos_feed_name])}
    """)
    return


@app.cell(hide_code=True)
def _(
    fts_range_select,
    fts_reverse,
    fts_setup_button,
    fts_slew_speed,
    fts_source_temp,
    fts_start_button,
    mo,
    stream_time_select,
):
    mo.md(rf"""
    ### FTS Data Collection

    {mo.vstack([mo.hstack([fts_slew_speed, fts_source_temp, fts_reverse, fts_range_select]), mo.hstack([fts_setup_button, fts_start_button, stream_time_select], widths=[1, 1, 0.5])])}
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
def _(OCSClient, fts_agent_name, init_fts_button, mo):
    mo.stop(not init_fts_button.value)

    FTS = OCSClient(fts_agent_name.value)
    FTS.init_stage()
    FTS.home()
    return (FTS,)


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
def _(
    FTS,
    RC,
    ccat_io,
    fts_range_select,
    fts_reverse,
    fts_setup_button,
    fts_slew_speed,
    fts_source_temp,
    mo,
):
    mo.stop(not fts_setup_button.value)

    fts_positions = fts_range_select.value
    if fts_reverse.value:
        fts_positions = fts_positions[::-1]

    ccat_io.edit_config(RC.ext_cfg, "fts_start", fts_positions[0], append=True)
    ccat_io.edit_config(RC.ext_cfg, "fts_end", fts_positions[1], append=True)
    ccat_io.edit_config(
        RC.ext_cfg, "slew_speed", fts_slew_speed.value, append=True
    )
    ccat_io.edit_config(
        RC.ext_cfg, "source_temperature", fts_source_temp.value, append=True
    )

    FTS.move_to(position=fts_positions[0])
    return (fts_positions,)


@app.cell
def _(FTS, RC, fts_positions, fts_start_button, mo, stream_time_select):
    mo.stop(not fts_start_button.value)
    FTS.move_to.start(position=fts_positions[1])
    stream_files = RC.take_timestream(stream_time_select.value)
    return


@app.cell(column=2)
def _():
    # General
    import marimo as mo

    from tqdm import tqdm
    from functools import partial
    from ocs.ocs_client import OCSClient
    from so3g.hk import load_range
    import time
    import pytz
    import datetime as dt


    # IO
    import os
    import ast
    import json
    import pickle
    from pathlib import Path

    return OCSClient, Path, dt, json, load_range, mo, os, pytz


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
def _(mo):
    fts_agent_name = mo.ui.text(
        value="det-fts",
        debounce=True,
        full_width=True,
        label="FTS Agent Instance Name",
    )
    return (fts_agent_name,)


@app.cell
def _(fts_agent_name, io_cfg, mo):
    init_daq_button = mo.ui.run_button(
        kind="success",
        disabled=io_cfg == {},
        tooltip="Click to create a new data acquisition object using the selected system configuration file.",
        label="Create DAQ Control Object",
        full_width=True,
    )

    init_fts_button = mo.ui.run_button(
        kind="success",
        disabled=not fts_agent_name.value,
        tooltip="Click to create FTS OCS Agent client.",
        label="Create FTS Control Object",
        full_width=True,
    )
    return init_daq_button, init_fts_button


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
def _(HOME_DIR, mo):
    hk_dir_browser = mo.ui.file_browser(
        initial_path=HOME_DIR,
        multiple=False,
        ignore_empty_dirs=True,
        label="Select housekeeping data directory...",
        selection_mode="directory",
    )

    fts_pos_feed_name = mo.ui.text(
        value="observatory.det-fts.feeds.position.pos",
        debounce=True,
        full_width=True,
        label="FTS Position Feed Name",
    )
    return fts_pos_feed_name, hk_dir_browser


@app.cell
def _(mo):
    fts_reverse = mo.ui.switch(value=False, label="Reverse Slew?")

    fts_range_select = mo.ui.range_slider(
        start=-70,
        stop=70,
        step=1,
        value=[-70, 70],
        label="Central Mirror Slew Positions [mm]",
        debounce=True,
        show_value=True,
        full_width=True,
    )

    fts_slew_speed = mo.ui.number(
        start=0,
        value=1,
        label="FTS Slew Speed [mm/s]",
        full_width=True,
        debounce=True,
    )

    fts_source_temp = mo.ui.number(
        start=250,
        value=500,
        label="Source Temperature [K]",
        full_width=True,
        debounce=True,
    )
    return fts_range_select, fts_reverse, fts_slew_speed, fts_source_temp


@app.cell
def _(init_comb_browser, mo):
    stream_time_select = mo.ui.number(
        start=0,
        value=150,
        label="Timestream Length [seconds]",
        full_width=True,
        debounce=True,
    )

    fts_setup_button = mo.ui.run_button(
        kind="success",
        tooltip="Click to setup FTS measurement.",
        label="Setup FTS Measurement",
        full_width=True,
        disabled=not init_comb_browser.value,
    )

    fts_start_button = mo.ui.run_button(
        kind="success",
        tooltip="Click to take FTS data.",
        label="Take FTS Data",
        full_width=True,
        disabled=not init_comb_browser.value,
    )
    return fts_setup_button, fts_start_button, stream_time_select


@app.cell
def _():
    return


@app.cell(column=3)
def _(dt, load_range, pytz):
    def load_hk(start_time, end_time, data_dir, data_name):
        eastern_tz = pytz.timezone("America/New_York")

        start_dt = dt.datetime.fromtimestamp(start_time, tz=eastern_tz)
        end_dt = dt.datetime.fromtimestamp(end_time, tz=eastern_tz)
        return load_range(start_dt, end_dt, fields=[data_name], data_dir=data_dir)[
            data_name
        ]

    return


@app.cell
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

    return


if __name__ == "__main__":
    app.run()
