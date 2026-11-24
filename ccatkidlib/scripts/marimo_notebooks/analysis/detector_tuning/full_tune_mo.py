import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(analysis_cfg_browser, cfg_editor, data_browser, data_desc, mo):
    mo.md(rf"""
    ### Select Data to Load

    {mo.hstack([mo.vstack([analysis_cfg_browser, data_browser, data_desc]), cfg_editor])}
    """)
    return


@app.cell(hide_code=True)
def _(com_to_selector, mo):
    mo.md(rf"""
    ### Select RFSoC Drones

    {mo.hstack([com_to_selector])}
    """)
    return


@app.cell(hide_code=True)
def _(max_workers_selector, mo, pickle_name_selector, transform_button):
    mo.md(rf"""
    ### Transform Data

    {
        mo.vstack(
            [
                pickle_name_selector,
                mo.hstack(
                    [max_workers_selector, transform_button], widths=[1, 2.5]
                ),
            ]
        )
    }
    """)
    return


@app.cell(hide_code=True)
def _(load_pickle_button, mo, pickle_tabs):
    mo.md(rf"""
    ### Load Pickled Data

    {mo.vstack([pickle_tabs, load_pickle_button])}
    """)
    return


@app.cell(hide_code=True)
def _(mo, save_powers_button, tuned_comb_browser):
    mo.md(rf"""
    ### Save Tuned Comb

    {mo.vstack([tuned_comb_browser, save_powers_button])}
    """)
    return


@app.cell(column=1)
def _(
    Network,
    Path,
    ProcessPoolExecutor,
    all_transformations,
    analysis_cfg,
    ccat_log,
    com_to_selector,
    data_dirs,
    max_workers_selector,
    mo,
    pickle,
    pickle_name_selector,
    pl,
    tqdm,
    transform_button,
    viz_cfg,
):
    mo.stop(not transform_button.value)

    *_root_data_parts, _data_dir, _date, _sess_id = data_dirs[0].parts
    _root_data_dir = "/".join(_root_data_parts)[1:]

    _com_tos = com_to_selector.value
    with ProcessPoolExecutor(max_workers=max_workers_selector.value) as _ex:
        for _com_to in _com_tos:
            _network = Network(
                com_to=_com_to,
                sess_ids=_sess_id,
                date=_date,
                data_dir=_data_dir,
                root_data_dir=_root_data_dir,
                analysis_cfg=analysis_cfg,
                viz_cfg=viz_cfg,
            )

            _network.add_columns(
                data_cols=["drive", "sense", "detector_type", "network"],
                max_workers=min(15, max_workers_selector.value),
                ex=_ex,
            )

            # _network.data = _network.data.sort("drive").gather_every(2)

            _success_list = _network.data.height * [True]
            for _i, _det in tqdm(
                enumerate(_network.data["detector"]), total=_network.data.height
            ):
                try:
                    _det = _network.det_dict[_det]
                    all_transformations(
                        _det,
                        max_workers=max_workers_selector.value,
                        ex=_ex,
                        recalc=False,
                    )
                except Exception as _e:
                    ccat_log.log("ERROR", "Analysis failed with error: %s", _e)
                    _success_list[_i] = False
            _network.data = _network.data.with_columns(
                pl.Series("analysis_success", _success_list)
            )
            _network.data = _network.data.filter(pl.col("analysis_success"))
            _pickle_name = pickle_name_selector.value.split(".")[0]
            _pickle_path = Path(_network.pickle_dir) / f"{_pickle_name}.pickle"

            with open(_pickle_path, "wb") as _f:
                pickle.dump(_network, _f, pickle.HIGHEST_PROTOCOL)
            del _network
    return


@app.cell
def _(com_to_selector, load_pickle_button, mo, pickle, pickle_selector, pl):
    mo.stop(not load_pickle_button.value)

    networks = {}
    for _i, _com_to in enumerate(com_to_selector.value):
        _network = None
        for _file in pickle_selector.value[_com_to]:
            with open(_file, "rb") as _f:
                _sub_network = pickle.load(_f)
                if _network is None:
                    _network = _sub_network
                else:
                    # Need to combine DataFrames and dictionaries with mappings to Detector objects
                    _network.data = pl.concat(
                        [_network.data, _sub_network.data], how="vertical"
                    )
                    _network.det_dict = _network.det_dict | _sub_network.det_dict
        networks[_com_to] = _network
    return (networks,)


@app.cell
def _(load_pickle_button, mo, networks, pl):
    mo.stop(not load_pickle_button.value)

    all_network_props = None
    for _com_to, _network in networks.items():
        # Combine properties DataFrames
        _network_props = _network.combine_properties(
            data_cols=["drive", "sense", "detector_type", "network"]
        )

        all_network_props = (
            _network_props
            if all_network_props is None
            else pl.concat([all_network_props, _network_props], how="diagonal")
        )
    all_network_props = all_network_props.with_columns(
        pl.col("network").str.split(".").list[0].alias("array_num")
    ).with_columns(
        pl.when(pl.col("detector_type") == "Al")
        .then(f"Al " + pl.col("array_num"))
        .otherwise(pl.col("detector_type"))
        .alias("detector_array")
    )
    return (all_network_props,)


@app.cell
def _(
    ccat_io,
    drive_to_power,
    mo,
    networks,
    np,
    pl,
    save_powers_button,
    tuned_comb_browser,
):
    mo.stop(not save_powers_button.value)

    _save_drive_dir = tuned_comb_browser.value[0].path
    for _com_to, _network in networks.items():
        _bid, _drid = _com_to.split(".")
        _combined_props = None

        _med_drive = None
        for _det, _drive in _network.data.select(["detector", "drive"]).to_numpy():
            _det = _network.det_dict[_det]
            _props = _det.properties.with_columns(pl.lit(_drive).alias("drive"))
            if _drive == 5 or _drive == 5.25:
                _med_drive = _props
            _combined_props = (
                _props
                if _combined_props is None
                else pl.concat([_combined_props, _props], how="vertical")
            )

        _best_drive_df = (
            _combined_props.filter(
                ~pl.col("freq/diss").is_null(),
                pl.col("freq/diss") <= 50,
                pl.col("freq/diss") > 1,
                pl.col("max_IQ_angle_deg") <= 25,
                pl.col("max_IQ_dist_adj_f") < pl.col("min_mag_f"),
            )
            .filter(pl.col("freq/diss") == pl.col("freq/diss").max().over("det"))
            .group_by("det")
            .agg(
                pl.col("drive").first(),
                pl.col("tone_powers").first(),
                pl.col("tone_freqs").first(),
                pl.col("tone_phis").first(),
            )
            .rename({"drive": "best_drive"})
        )

        _best_powers, _drive_ref = drive_to_power(_best_drive_df)
        _found_dets = _best_drive_df["det"].to_list()
        _best_freqs = _med_drive.filter(pl.col("det").is_in(_found_dets))[
            "tone_freqs"
        ]

        _best_phis = _best_drive_df["tone_phis"].to_numpy()

        _drone_drive_dir = _save_drive_dir / f"B{_bid}D{_drid}"
        ccat_io.create_dir(_drone_drive_dir)
        np.save(_drone_drive_dir / "best_freqs.npy", _best_freqs)
        np.save(_drone_drive_dir / "best_phis.npy", _best_phis)
        np.save(_drone_drive_dir / "best_powers.npy", _best_powers)
        np.save(_drone_drive_dir / "best_drives.npy", np.array([_drive_ref]))
    return


@app.cell
def _(all_network_props, pl):
    low_freq_tuned_df = (
        all_network_props.filter(
            ~pl.col("freq/diss").is_null(),
            pl.col("freq/diss") < 50,
            pl.col("freq/diss") > 1,
            pl.col(
                "frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_white_noise"
            )
            < 5e-8,
            pl.col("max_IQ_angle_deg") < 20,
            pl.col("max_IQ_dist_adj_f") < pl.col("min_mag_f"),
        )
        .filter(
            pl.col("freq/diss")
            == pl.col("freq/diss").max().over("det", "detector_type", "network")
        )
        .group_by("detector_type", "network", "det")
        .agg(
            pl.col("drive").first(),
            pl.col("tone_powers").first(),
            pl.col("tone_freqs").first(),
            pl.col("tone_phis").first(),
            pl.col(
                "frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_white_noise"
            ).first(),
            # pl.col(
            #    "phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i"
            # ).first(),
            # pl.col(
            #    "phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a"
            # ).first(),
            pl.col("max_IQ_angle_deg").first(),
            pl.col("freq/diss").first(),
            pl.col("detector_array").first(),
        )
        .rename({"drive": "best_drive"})
    )

    high_freq_tuned_df = (
        all_network_props.filter(
            ~pl.col("freq/diss").is_null(),
            pl.col("freq/diss") < 50,
            pl.col("freq/diss") > 1,
            pl.col(
                "frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_white_noise"
            )
            < 5e-8,
            pl.col("max_IQ_angle_deg") < 20,
            pl.col("max_IQ_dist_f") > pl.col("min_mag_f"),
        )
        .filter(
            pl.col("freq/diss")
            == pl.col("freq/diss").max().over("det", "detector_type", "network")
        )
        .group_by("detector_type", "network", "det")
        .agg(
            pl.col("drive").first(),
            pl.col("tone_powers").first(),
            pl.col("tone_freqs").first(),
            pl.col("tone_phis").first(),
            pl.col(
                "frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_white_noise"
            ).first(),
            # pl.col(
            #    "phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_Q_i"
            # ).first(),
            # pl.col(
            #    "phase_fit_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_a"
            # ).first(),
            pl.col("max_IQ_angle_deg").first(),
            pl.col("freq/diss").first(),
            pl.col("detector_array").first(),
        )
        .rename({"drive": "best_drive"})
    )
    return high_freq_tuned_df, low_freq_tuned_df


@app.cell
def _(networks, pl):
    _det = networks["1.1"].data.filter(pl.col("drive") == 0)["detector"][0]
    _det = networks["1.1"].det_dict[_det]

    _det.targ.mag_plot(grouping="groupby", exclude=0)
    return


@app.cell
def _(high_freq_tuned_df, low_freq_tuned_df):
    low_hist = low_freq_tuned_df.sort("network", descending=False).hvplot.hist(
        "best_drive", groupby="network", bins=25
    )
    high_hist = high_freq_tuned_df.sort("network").hvplot.hist(
        "best_drive", bins=25, groupby="network"
    )
    low_hist * high_hist
    return


@app.cell
def _(low_freq_tuned_df, pl):
    low_freq_tuned_df.select("best_drive", "network").filter(
        pl.col("best_drive") > 4.5, pl.col("best_drive") < 6
    ).unique().sort("network", "best_drive")
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

    return Path, json, mo, os, pickle, tqdm


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

    from ccatkidlib.rfsoc.rfsoc_daq import R
    from ccatkidlib.analysis.core.detector import Detector
    from ccatkidlib.analysis.core.network import Network

    return Detector, Network, ccat_df, ccat_io, ccat_log


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
    return analysis_cfg, cfg_editor, viz_cfg


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
    return data_browser, root_data_dir


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
    return data_desc, data_dirs


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
    return max_workers_selector, pickle_name_selector, transform_button


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
    return load_pickle_button, pickle_tabs


@app.cell
def _(load_pickle_button, mo, root_data_dir):
    tuned_comb_browser = mo.ui.file_browser(
        initial_path=root_data_dir,
        selection_mode="directory",
        multiple=False,
        ignore_empty_dirs=False,
        label="Select directory in which to save tuned comb files...",
    )

    save_powers_button = mo.ui.run_button(
        kind="success",
        label="Save Tuned Comb",
        disabled=not load_pickle_button.value,
        full_width=True,
    )
    return save_powers_button, tuned_comb_browser


@app.cell(column=3)
def _(np, pl):
    def drive_to_power(drive_df, drive_ref=None, drive_step=0.25, attempts=100):
        """Convert best drive attenuations for each detector to tone powers.
        Also, choose a overall drive attenuation so that the total tone power is similar to the total tone power of the original comb.
        """
        drive_df = drive_df.with_columns(
            pl.when(pl.col("best_drive").is_null())
            .then(pl.col("best_drive").median())
            .otherwise("best_drive")
            .alias("best_drive")
        )
        curr_power, new_power = (
            drive_df.select(pl.col("tone_powers").sum()).item(),
            np.inf,
        )
        for _ in range(attempts):
            drive_df = (
                drive_df.with_columns(
                    pl.col("best_drive").median().alias("ref_drive")
                )
                if drive_ref is None
                else drive_df.with_columns(pl.lit(drive_ref).alias("ref_drive"))
            )
            drive_ref = drive_df["ref_drive"][0]
            best_powers = (
                drive_df.with_columns(
                    (pl.col("ref_drive") - pl.col("best_drive")).alias(
                        "drive_diff"
                    )
                )
                .with_columns(
                    (
                        pl.col("tone_powers") * 10 ** (pl.col("drive_diff") / 20)
                    ).alias("best_tone_powers")
                )
                .sort("det")
            )
            new_power = best_powers.select(pl.col("best_tone_powers").sum()).item()
            if new_power > curr_power:
                drive_ref -= drive_step
                if drive_ref <= 0:
                    drive_ref = 0
                    break
            else:
                break
        drive_ref = (
            round(drive_ref / drive_step) * drive_step
        )  # Round to nearest drive_step increment
        return best_powers["best_tone_powers"].to_numpy(), drive_ref

    return (drive_to_power,)


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

    return (center_IQ,)


@app.cell
def _(Detector, ProcessPoolExecutor, center_IQ):
    def fit_detectors(
        det: Detector,
        complex_fit: bool = False,
        phase_fit: bool = True,
        recalc: bool = False,
        max_workers=1,
        ex: ProcessPoolExecutor = None,
    ):
        """
        Fit KIDs
        """

        # Fit phase data of detectors using native phase fitting code (Not using kid_phase_fit. kid_phase_fit is imported above and can be used instead if preferred [but results will not be automatically added to dataframes])
        if phase_fit:
            center_IQ(
                det,
                data="targ",
                recalc=recalc,
                delay_col="network_cable_delay",
                max_workers=max_workers,
                ex=ex,
            )  # Center IQ circles
            det.IQ_circle_rotate(
                prefix="origin_shift_origin_rotate_unwind_rotate",
                data="targ",
                recalc=recalc,
                rotation="mismatch",
            )  # Rotate IQ circle to correct for impedance mismatch
            det.targ.phase(
                prefix="mismatch_rotate_origin_shift_origin_rotate_unwind_rotate",
                recalc=recalc,
            )  # Calculate target sweep phase
            det.targ.mag(
                prefix="mismatch_rotate_origin_shift_origin_rotate_unwind_rotate",
                recalc=recalc,
            )
            det.phase_fit(
                prefix="mismatch_rotate_origin_shift_origin_rotate_unwind_rotate",
                circle_fit_col="circle_fit_tail_trim_unwind_rotate",
                nonlinear=True,
                window=6,
                method="least_squares",
                max_workers=min(20, max_workers),
                ex=ex,
                recalc=recalc,
            )  # Fit phase data

        # Fit complex data using resonator_model_v3
        if complex_fit:
            det.complex_fit(
                nonlinear=True, max_workers=min(30, max_workers), recalc=recalc
            )
        return det

    return


@app.cell
def _(center_IQ, pl):
    def phase_to_f(
        detector, max_workers=1, ex=None, recalc=False, phase_bounds=0.4, k=2
    ):
        """Convert timestreams from phase to frequency"""
        # Center IQ circle and rotate by impedance mismatch angle
        center_IQ(
            detector,
            ex=ex,
            data="both",
            delay_col="network_cable_delay",
            max_workers=max_workers,
            recalc=recalc,
        )
        detector.IQ_circle_rotate(
            prefix="origin_shift_origin_rotate_unwind_rotate",
            data="both",
            rotation="mismatch",
            recalc=recalc,
        )
        detector.IQ_noise(
            prefix="mismatch_rotate_origin_shift_origin_rotate_unwind_rotate",
            use_noise_tones=False,
            recalc=recalc,
        )

        detector.targ.phase(
            prefix="mismatch_rotate_origin_shift_origin_rotate_unwind_rotate",
            recalc=recalc,
        )
        detector.stream.phase(
            prefix=[
                "mismatch_rotate_origin_shift_origin_rotate_unwind_rotate",
                "noise_rotate_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate",
            ],
            recalc=recalc,
        )

        # Convert timestream phase to fractional frequency shift in ppm
        detector.phase_to_f(
            prefix="mismatch_rotate_origin_shift_origin_rotate_unwind_rotate",
            spline_col="mismatch_rotate_origin_shift_origin_rotate_unwind_rotate",
            phase_bounds=0.4,
            k=2,
            max_workers=min(25, max_workers),
            ex=ex,
            recalc=recalc,
        )
        detector.phase_to_f(
            prefix="noise_rotate_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate",
            spline_col="mismatch_rotate_origin_shift_origin_rotate_unwind_rotate",
            phase_bounds=0.4,
            k=2,
            max_workers=min(25, max_workers),
            ex=ex,
            recalc=recalc,
        )

        detector.frac_f(
            prefix=[
                "mismatch_rotate_origin_shift_origin_rotate_unwind_rotate",
                "noise_rotate_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate",
            ],
            recalc=recalc,
        )
        frac_f_cols = [
            pl.col(f"{prefix}_{tone:0{detector.stream.padding}d}")
            for tone in detector.stream.tones
            for prefix in (
                "frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f",
                "frac_noise_rotate_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f",
            )
        ]
        detector.stream.data = detector.stream.data.with_columns(
            [(frac_f_col * 1e6).name.prefix("ppm_") for frac_f_col in frac_f_cols]
        )
        return detector

    return (phase_to_f,)


@app.cell
def _(ccat_df, pl):
    def white_noise_psd(
        detector,
        max_workers=1,
        ex=None,
        nperseg=None,
        f_threshold=180,
        recalc=False,
    ):
        def _calc_white_noise(f_col, data_col, white_noise_col):
            f_df = (
                detector.stream.get_data(col_name=f_col, strict=True)
                .unpivot(variable_name="det", value_name="psd_f")
                .with_columns(
                    pl.col("det").str.strip_prefix(f"{f_col}_").cast(pl.Int32)
                )
            )
            psd_df = (
                detector.stream.get_data(col_name=data_col)
                .unpivot(variable_name="det", value_name="psd")
                .with_columns(
                    pl.col("det").str.strip_prefix(f"{data_col}_").cast(pl.Int32)
                )
                .drop("det")
            )

            psd_f_df = pl.concat([f_df, psd_df], how="horizontal")
            psd_f_df = (
                psd_f_df.filter(
                    (~pl.col("psd_f").is_nan()) & (pl.col("psd_f") > f_threshold)
                )
                .select(
                    "det", pl.col("psd").mean().over("det").alias(white_noise_col)
                )
                .unique()
            )
            shared_cols = (
                white_noise_col
                if white_noise_col in detector._properties_df.schema
                else []
            )
            detector._properties_df = ccat_df.coalesce_join(
                detector._properties_df, psd_f_df, "det", shared_cols
            )

        # Calculate PSD
        detector.stream.psd(
            prefix="frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate",
            col_name="f",
            recalc=recalc,
            nperseg=nperseg,
            detrend="linear",
            max_workers=min(10, max_workers),
            ex=ex,
        )
        detector.stream.psd(
            prefix="frac_noise_rotate_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate",
            col_name="f",
            recalc=recalc,
            nperseg=nperseg,
            detrend="linear",
            max_workers=min(10, max_workers),
            ex=ex,
        )

        # Calculate white noise
        _calc_white_noise(
            "psd_f_frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f",
            "psd_frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f",
            "frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_white_noise",
        )

        _calc_white_noise(
            "psd_f_frac_noise_rotate_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f",
            "psd_frac_noise_rotate_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f",
            "frac_noise_rotate_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_white_noise",
        )

        detector._properties_df = detector._properties_df.with_columns(
            (
                pl.col(
                    "frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_white_noise"
                )
                / pl.col(
                    "frac_noise_rotate_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f_white_noise"
                )
            ).alias("freq/diss")
        )
        return detector

    return (white_noise_psd,)


@app.cell
def _(ccat_log, phase_to_f, white_noise_psd):
    def all_transformations(detector, max_workers=1, ex=None, recalc=False):
        ccat_log.log(
            "INFO",
            "Converting timestreams to fractional frequency...",
            name="analysis.detector",
        )
        phase_to_f(detector, max_workers=max_workers, ex=ex, recalc=recalc)

        ccat_log.log("INFO", "Calculating rt(Sxx)...", name="analysis.detector")
        white_noise_psd(detector, max_workers=max_workers, ex=ex, recalc=recalc)

        detector.is_bifurcated(max_workers=min(10, max_workers), ex=ex)

    return (all_transformations,)


if __name__ == "__main__":
    app.run()
