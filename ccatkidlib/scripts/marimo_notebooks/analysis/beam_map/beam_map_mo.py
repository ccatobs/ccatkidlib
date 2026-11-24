import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    return


@app.cell(column=1)
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

    return


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

    return (ccat_io,)


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
    return (analysis_cfg,)


@app.cell
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
    return (data_browser,)


@app.cell
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
    return (data_dirs,)


@app.cell
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


@app.cell
def _(com_to_selector, mo, os):
    max_workers_selector = mo.ui.number(
        start=1,
        stop=os.cpu_count(),
        step=1,
        label="Max Workers",
        value=1,
        full_width=True,
    )

    transform_button = mo.ui.run_button(
        kind="success",
        label="Run Data Transformation",
        tooltip="Click to transform data for the selected drones",
        full_width=True,
        disabled=not com_to_selector.value,
    )

    pickle_name_selector = mo.ui.text(
        value="tune_full_network",
        debounce=True,
        label="Enter Custom Pickle File Name",
        full_width=True,
    )
    return


@app.cell
def _(com_to_selector, data_dirs, mo):
    _pickle_dir = data_dirs[0] / "pickle"
    _com_tos = com_to_selector.value

    pickle_dict = {}
    for _com_to in sorted(_com_tos):
        _bid, _drid = _com_to.split(".")
        if (
            _com_to_pickle := (_pickle_dir / f"B{_bid}D{_drid}" / "network")
        ).exists():
            _avail_com_to_pickle = []
            for _sess_id in _com_to_pickle.iterdir():
                for _pickle_file in _sess_id.iterdir():
                    _avail_com_to_pickle.append(_pickle_file)
            _avail_com_to_pickle = sorted(_avail_com_to_pickle)
            pickle_dict[_com_to] = mo.ui.multiselect(
                _avail_com_to_pickle,
                value=[_avail_com_to_pickle[0]],
                full_width=True,
                label="Select Pickle Files...",
                max_selections=1,
            )
        else:
            pickle_dict[_com_to] = mo.ui.text(
                "No pickle files to select", disabled=True, full_width=True
            )
    pickle_selector = mo.ui.dictionary(pickle_dict, label="Pickle File Selector")
    return (pickle_selector,)


@app.cell
def _(com_to_selector, mo, pickle_selector):
    pickle_tabs = mo.ui.tabs(
        {_drone: _selector for _drone, _selector in pickle_selector.items()}
    )

    load_pickle_button = mo.ui.run_button(
        kind="success",
        label="Load Pickle Files",
        disabled=not com_to_selector.value,
        full_width=True,
    )
    return


if __name__ == "__main__":
    app.run()
