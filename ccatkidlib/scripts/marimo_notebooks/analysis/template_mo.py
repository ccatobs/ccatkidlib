import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(mo):
    mo.md(r"""
    ### Overview

    This notebook is a template for loading RFSoC data taken using *ccatkidlib* and should be useable as the basis for analysis notebooks of any type of data set.
    """)
    return


@app.cell(hide_code=True)
def _(analysis_cfg_browser, cfg_editor, data_browser, data_desc, mo):
    mo.md(rf"""
    ### Select Data to Load

    All *ccatkidlib* data analysis objects (**Detector**, **Network**, etc.) require an `analysis_config.yaml`, which can be selected below. If you do not already have a config file setup, an example can be found at `ccatkidlib/analysis/example_analysis_config.yaml`. After a config file is chosen, the directory containing the data to load can be selected. The default behavior is to only allow loading data corresponding to a single **sess_id** (measurement) but can be modified as necessary.


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
    max_workers_selector,
    mo,
    num_pickle_files_selector,
    pickle_name_selector,
    transform_button,
):
    mo.md(rf"""
    ### Transform Data

    The typical data analysis procedure involves loading the data into **Network** objects, transforming/analyzing the data using the underlying **Detector** objects, and pickling the processed data. One often wants to only analyze the data for a subset of RFSoC drones which can be selected below. Many of the built-in data analysis methods can be multi-processed and the maximum number of CPU cores to use can be specified below. Finally, the name of the pickle file storing the processed data can be chosen below. 


    {
        mo.vstack(
            [
                mo.hstack(
                    [
                        com_to_selector,
                        max_workers_selector,
                        num_pickle_files_selector,
                        pickle_name_selector,
                    ],
                    widths=[2, 2, 2, 6],
                ),
                mo.hstack([transform_button]),
            ]
        )
    }
    """)
    return


@app.cell(hide_code=True)
def _(load_pickle_button, mo, pickle_tabs):
    mo.md(rf"""
    ### Load Pickled Data

    If the processed data was saved as pickle files, it can be loaded here without having to re-run any transformation/analysis methods. The RFSoC drones to load the data of can be selected in the **Transform Data** cell. If valid pickle files are found, they can be selected below.

    {mo.vstack([pickle_tabs, load_pickle_button])}
    """)
    return


@app.cell(column=1)
def transform_data(
    Network,
    analysis_cfg,
    ccat_pickle,
    com_to_selector,
    dump_transform,
    max_workers_selector,
    mo,
    num_pickle_files_selector,
    pickle_name_selector,
    pickle_selector,
    root_data_dir,
    transform_button,
    viz_cfg,
):
    mo.stop(not transform_button.value)

    _data_dir, _date, _sess_id = pickle_selector.value[com_to_selector.value[0]][
        0
    ].parts[-8:-5]

    for _com_to in com_to_selector.value:
        _network = Network(
            com_to=_com_to,
            sess_ids=_sess_id,
            date=_date,
            data_dir=_data_dir,
            root_data_dir=root_data_dir,
            analysis_cfg=analysis_cfg,
            viz_cfg=viz_cfg,
        )

        _network.add_columns(
            data_cols=[
                "com_to",
                "drive",
                "sense",
                "detector_type",
                "network",
            ],
            max_workers=max_workers_selector.value,
        )

        ccat_pickle.multi_dump(
            _network,
            pickle_name_selector.value,
            num_segments=num_pickle_files_selector.value,
            transform=dump_transform,
        )
    return


@app.cell
def load_pickle(
    ccat_pickle,
    com_to_selector,
    load_pickle_button,
    load_transform,
    mo,
    pickle_selector,
    root_data_dir,
):
    mo.stop(not load_pickle_button.value)

    _data_dir, _date, _sess_id = pickle_selector.value[com_to_selector.value[0]][
        0
    ].parts[-8:-5]

    networks = {
        _com_to: ccat_pickle.multi_load(
            _com_to,
            pickle_selector.value[_com_to][0].stem,
            _sess_id,
            data_dir=_data_dir,
            date=_date,
            root_data_dir=root_data_dir,
            transform=load_transform,
        )
        for _com_to in com_to_selector.value
    }
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

    return


@app.cell
def _():
    # ccatkidlib
    import ccatkidlib.io as ccat_io
    import ccatkidlib.log as ccat_log
    import ccatkidlib.analysis.utils.pickle as ccat_pickle
    import ccatkidlib.analysis.utils.pair as ccat_pair
    import ccatkidlib.analysis.utils.dataframe as ccat_df

    from ccatkidlib.rfsoc.rfsoc_daq import R
    from ccatkidlib.analysis.core.detector import Detector
    from ccatkidlib.analysis.core.network import Network

    return Network, ccat_io, ccat_pickle


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

    # Create file browser for selecting data sess_id directory
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
        label="Select Drone(s)...",
        full_width=True,
    )
    return (com_to_selector,)


@app.cell
def _(mo, os):
    # Create UI elements for transforming data
    # ----------------------------------------

    # Create selector for max number of CPU cores to use
    max_workers_selector = mo.ui.number(
        start=1,
        stop=os.cpu_count(),
        step=1,
        label="Max Workers",
        value=1,
        full_width=True,
    )

    # Create text element for choosing pickle file save name
    pickle_name_selector = mo.ui.text(
        value="network_pickle",
        debounce=True,
        label="Enter Custom Pickle File Name",
        full_width=True,
    )

    # Create selector for the number of pickle files to split Network objects into
    num_pickle_files_selector = mo.ui.number(
        start=1,
        step=1,
        label="Number of Pickle Files",
        value=1,
        full_width=True,
    )
    return (
        max_workers_selector,
        num_pickle_files_selector,
        pickle_name_selector,
    )


@app.cell
def _(com_to_selector, mo):
    # Create run button for transforming data
    transform_button = mo.ui.run_button(
        kind="success",
        label="Run Data Transformation",
        tooltip="Click to transform data for the selected drones",
        full_width=True,
        disabled=not com_to_selector.value,
    )
    return (transform_button,)


@app.cell
def _(com_to_selector, data_dirs, mo):
    # Search for pickle files
    # -----------------------

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
    # Create UI elements for selecting and loading pickle files
    # ---------------------------------------------------------
    pickle_tabs = mo.ui.tabs(
        {_drone: _selector for _drone, _selector in pickle_selector.items()}
    )

    load_pickle_button = mo.ui.run_button(
        kind="success",
        label="Load Pickle Files",
        disabled=not com_to_selector.value,
        full_width=True,
    )
    return load_pickle_button, pickle_tabs


@app.cell(column=3)
def _(Network):
    def dump_transform(network: Network) -> Network:
        """
        Transformations to run on Network objects before dumping to pickle files
        """

        return network

    return (dump_transform,)


@app.cell
def _(Network):
    def load_transform(network: Network) -> Network:
        """
        Transformations to run on Network objects loaded from pickle files
        """
        return network

    return (load_transform,)


if __name__ == "__main__":
    app.run()
