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
def _(
    com_to_selector,
    load_pickle_button,
    mo,
    pickle_name_selector,
    pickle_tabs,
):
    mo.md(rf"""
    ### Load Pickled Data

    {mo.vstack([mo.hstack([com_to_selector, pickle_name_selector], widths=[0.5, 1]), pickle_tabs, load_pickle_button])}
    """)
    return


@app.cell(hide_code=True)
def _(
    data_col_selector,
    fit_beams_button,
    init_fit_params,
    mo,
    offset_selector,
    power_method_selector,
    sigma_selector,
):
    mo.md(rf"""
    ### Fit Detector Beams

    {mo.vstack([mo.hstack([data_col_selector, power_method_selector], widths=[3, 1]), mo.hstack([sigma_selector, offset_selector], widths=[1, 1]), init_fit_params, fit_beams_button])}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(rf"""
    ### Plot Detector Map
    """)
    return


@app.cell(hide_code=True)
def _(mo, save_map_browser, save_map_button, save_map_name):
    mo.md(rf"""
    ### Save Detector Map

    {mo.vstack([save_map_name, save_map_browser, save_map_button])}
    """)
    return


@app.cell(column=1)
def _(
    com_to_selector,
    load_pickle_button,
    mo,
    pickle,
    pickle_selector,
    pl,
    tqdm,
):
    mo.stop(not load_pickle_button.value)

    networks = {}
    for _i, _com_to in tqdm(enumerate(com_to_selector.value)):
        _network = None
        for _file in tqdm(pickle_selector.value[_com_to]):
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
def _(
    data_col_selector,
    fit_beams_button,
    gaussian_2D_fit,
    init_fit_params,
    mo,
    networks,
    pl,
    power_method_selector,
    tqdm,
):
    mo.stop(not fit_beams_button.value)

    all_network_props = None
    for _com_to, _network in tqdm(networks.items()):
        # Combine properties DataFrames
        _network_props = _network.combine_properties(
            data_cols=["x_pos", "y_pos", "detector_type", "network"]
        )

        # Fit Gaussian to beams
        _init_params = init_fit_params.value.filter(pl.col("com_to") == _com_to)
        _network_props = gaussian_2D_fit(
            _network_props,
            amp_col=f"{data_col_selector.value}_{power_method_selector.value}_psd",
            x_guess=_init_params["X Guess"][0],
            y_guess=_init_params["Y Guess"][0],
            sigma_guess=_init_params["Std Dev Guess"][0],
            offset_guess=_init_params["Offset Guess"][0],
        )

        all_network_props = (
            _network_props
            if all_network_props is None
            else pl.concat([all_network_props, _network_props], how="diagonal")
        )

    all_network_props = (
        all_network_props.lazy()
        .rename(
            {
                _col: _col.split("gaussian_2D_fit_")[-1]
                for _col in all_network_props.select(
                    "^.*gaussian_2D_fit_.*$"
                ).columns
            }
        )
        .with_columns(pl.col("network").str.split(".").list[0].alias("array_num"))
        .with_columns(
            (pl.col("detector_type") + " " + pl.col("array_num")).alias(
                "detector_array"
            )
        )
    ).collect()
    return (all_network_props,)


@app.cell
def _(all_network_props, mo, save_map_browser, save_map_button, save_map_name):
    mo.stop(not save_map_button.value)

    _dir = save_map_browser.value[0].path
    _file_name = save_map_name.value.split(".")[0]

    all_network_props.write_parquet(_dir / f"{_file_name}.parquet")
    return


@app.cell
def _(
    all_network_props,
    data_col_selector,
    hv,
    np,
    opts,
    power_method_selector,
):
    map_center = hv.VLine(7.5).opts(
        linewidth=0.4, linestyle="-", color="purple", show_legend=False
    ) * hv.HLine(-5.5).opts(
        linewidth=0.3, linestyle="-", color="purple", show_legend=False
    )
    map_grid = hv.Overlay(  # [hv.Slope(np.tan(np.pi/3), offset) for offset in np.linspace(-96, 72, 25)] +
        # [hv.Slope(-np.tan(np.pi/3), offset) for offset in np.linspace(-82, 79, 25)] +
        [hv.VLine(offset) for offset in np.linspace(-105, 105, 25)]
        + [hv.HLine(offset) for offset in np.linspace(-105, 105, 25)]
    ).opts(
        opts.Slope(linewidth=0.25, linestyle="--", color="k", show_legend=False),
        opts.VLine(linewidth=0.25, linestyle="--", color="k", show_legend=False),
        opts.HLine(linewidth=0.25, linestyle="--", color="k", show_legend=False),
    )

    (
        map_grid
        *
        # map_center*
        all_network_props.select(
            ["^tone_.*$", "^.*gaussian_2D_fit_.*$", "network", "det"]
        )
        .rename({"network": "Network"})
        .unique()
        .sort("Network")
        # .filter(pl.col(f'{amp_col}_gaussian_2D_fit_sigma') > 5,
        #        pl.col(f'{amp_col}_gaussian_2D_fit_sigma') < 30,
        #        pl.col(f'{amp_col}_gaussian_2D_fit_amp') < 20,
        #        pl.col(f'{amp_col}_gaussian_2D_fit_amp') > 1e-7,
        #        pl.col(f'{amp_col}_gaussian_2D_fit_sigma_err') < 0.4,
        #        pl.col(f'{amp_col}_gaussian_2D_fit_x_0_err') < 0.5,
        #        pl.col(f'{amp_col}_gaussian_2D_fit_y_0_err') < 0.5)
        .hvplot.scatter(
            x=f"{data_col_selector.value}_{power_method_selector.value}_psd_gaussian_2D_fit_x_0",
            y=f"{data_col_selector.value}_{power_method_selector.value}_psd_gaussian_2D_fit_y_0",
            # c='det',
            # cmap='viridis',
            by="Network",
            data_aspect=1,
            s=150,
            xlabel="Beam Mapper X [mm]",
            ylabel="Beam Mapper Y [mm]",
            title="350 GHz Detector Position Map (Looking Into Mod-Cam)",
        )
    ).opts(
        fig_size=300,
        invert_xaxis=True,
        legend_position="top_right",
        show_grid=False,
        xlim=(-105, 105),
        ylim=(-105, 105),
    )
    return


@app.cell
def _(all_network_props):
    all_network_props
    return


@app.cell
def _(networks):
    networks['1.1'].det_dict[networks['1.1'].data['detector'][0]].stream.padding
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
    from numba import njit
    from scipy.optimize import curve_fit

    return curve_fit, njit, np, pl


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

    return ccat_io, ccat_log


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
    return analysis_cfg, cfg_editor


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
def _(mo):
    pickle_name_selector = mo.ui.text(
        value="beam_map",
        debounce=True,
        label="Pickle File Name",
        full_width=True,
    )
    return (pickle_name_selector,)


@app.cell
def _(com_to_selector, data_dirs, mo, pickle_name_selector):
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
                _avail_com_to_pickle += list(
                    _sess_id.glob(f"*{pickle_name_selector.value}*")
                )

            _avail_com_to_pickle = sorted(_avail_com_to_pickle)
            pickle_dict[_com_to] = mo.ui.multiselect(
                _avail_com_to_pickle,
                full_width=True,
                value=_avail_com_to_pickle,
                label="Select Pickle Files...",
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
def _(mo, networks):
    fit_beams_button = mo.ui.run_button(
        kind="success",
        label="Fit Beams",
        disabled=not networks,
        full_width=True,
    )

    data_col_selector = mo.ui.text(
        debounce=True,
        label="Data Column",
        full_width=True,
        value="frac_mismatch_rotate_origin_shift_origin_rotate_unwind_rotate_f",
    )

    power_method_selector = mo.ui.dropdown(
        options=["integral", "max"],
        value="integral",
        allow_select_none=False,
        label="Peak Power Extraction Method",
        full_width=True,
    )

    sigma_selector = mo.ui.number(
        start=0,
        stop=200,
        value=15,
        debounce=True,
        label="Standard Deviation Guess",
        full_width=True,
    )

    offset_selector = mo.ui.number(
        value=0,
        debounce=True,
        label="Offset Guess",
        full_width=True,
    )
    return (
        data_col_selector,
        fit_beams_button,
        offset_selector,
        power_method_selector,
        sigma_selector,
    )


@app.cell
def _(com_to_selector, mo, networks, offset_selector, pl, sigma_selector):
    _init_x, _init_y = [], []
    for _com_to, _network in networks.items():
        _med_x, _med_y = _network.data.select(
            pl.col("x_pos").median(), pl.col("y_pos").median()
        )
        _init_x += list(_med_x)
        _init_y += list(_med_y)

    _init_df = pl.DataFrame(
        {"com_to": com_to_selector.value, "X Guess": _init_x, "Y Guess": _init_y}
    ).with_columns(
        pl.lit(sigma_selector.value).alias("Std Dev Guess"),
        pl.lit(offset_selector.value).alias("Offset Guess"),
    )
    init_fit_params = mo.ui.data_editor(_init_df)
    return (init_fit_params,)


@app.cell
def _(all_network_props, mo, root_data_dir):
    save_map_browser = mo.ui.file_browser(
        initial_path=root_data_dir,
        selection_mode="directory",
        multiple=False,
        restrict_navigation=False,
        ignore_empty_dirs=False,
        label="Select directory in which to save detector map...",
    )

    save_map_name = mo.ui.text(
        debounce=True,
        label="Detector Map File Name",
        full_width=True,
        value="detector_map",
    )

    save_map_button = mo.ui.run_button(
        kind="success",
        label="Save Detector Map",
        disabled=all_network_props is None,
        full_width=True,
    )
    return save_map_browser, save_map_button, save_map_name


@app.cell
def _():
    return


@app.cell(column=3)
def _(ccat_log, curve_fit, njit, np, pl):
    @njit
    def gaussian_2D(
        coord: np.array,
        amplitude: float,
        x_0: float,
        y_0: float,
        sigma: float,
        offset: float,
    ):
        """
        Rotationally symmetric 2D Gaussian
        """
        x, y = coord

        # Calculate constants
        # -------------------
        a = 1 / (2 * sigma**2)
        c = 1 / (2 * sigma**2)

        # Calculate Gaussian amplitude for each point in grid
        # ---------------------------------------------------
        g = offset + amplitude * np.exp(
            -(a * ((x - x_0) ** 2) + c * ((y - y_0) ** 2))
        )
        return g


    def _gaussian_2D_fit(
        x, y, amplitude, x_guess, y_guess, sigma_guess, offset_guess
    ):
        coord = np.array([x.to_numpy(), y.to_numpy()])
        p0 = [
            amplitude.sort(descending=True, nulls_last=True).head(20).mean(),
            x_guess[0],
            y_guess[0],
            sigma_guess[0],
            offset_guess[0],
        ]

        bounds = (
            [0, x.min(), y.min(), np.abs(0.7 * (x[1] - x[0])), -np.inf],
            [np.inf, x.max(), y.max(), 1 * (x.max() - x.min()), np.inf],
        )

        try:
            popt, pcov = curve_fit(
                gaussian_2D,
                coord,
                amplitude,
                p0=p0,
                bounds=bounds,
                nan_policy="omit",
            )
            df = pl.DataFrame(
                {
                    "popt": [popt.tolist()],
                    "pcov": [np.sqrt(np.diag(pcov)).tolist()],
                }
            )
        except Exception as e:
            ccat_log.log("ERROR", e, name="analysis.network")
            df = pl.DataFrame({"popt": [[np.nan] * 5], "pcov": [[np.nan] * 5]})
        out_series = pl.Series(df.select(pl.struct(df.columns)))
        return out_series


    def _gaussian_2D_eval(x, y, popt):
        try:
            coord = np.array([x.to_numpy(), y.to_numpy()])
            return pl.Series(gaussian_2D(coord, *popt[0]))
        except Exception as e:
            ccat_log.log("DEBUG", e, name="analysis.network")
            return pl.Series(np.full(len(x), np.nan))


    def gaussian_2D_fit(
        df, amp_col, x_guess=0, y_guess=0, sigma_guess=15, offset_guess=0
    ):
        out_cols = [
            f"{amp_col}_gaussian_2D_fit",
            f"{amp_col}_gaussian_2D_fit_popt",
            f"{amp_col}_gaussian_2D_fit_x_0",
            f"{amp_col}_gaussian_2D_fit_y_0",
            f"{amp_col}_gaussian_2D_fit_amp",
            f"{amp_col}_gaussian_2D_fit_sigma",
            f"{amp_col}_gaussian_2D_fit_err",
            f"{amp_col}_gaussian_2D_fit_x_0_err",
            f"{amp_col}_gaussian_2D_fit_y_0_err",
            f"{amp_col}_gaussian_2D_fit_amp_err",
            f"{amp_col}_gaussian_2D_fit_sigma_err",
        ]

        fit_df = (
            df.lazy()
            .select("det", "x_pos", "y_pos", amp_col)
            .with_columns(
                pl.lit(x_guess).alias("x_guess"),
                pl.lit(y_guess).alias("y_guess"),
                pl.lit(sigma_guess).alias("sigma_guess"),
                pl.lit(offset_guess).alias("offset_guess"),
            )
            .sort("det")
            .group_by("det", maintain_order=True)
            .agg(
                pl.all(),
                pl.map_groups(
                    exprs=[
                        "x_pos",
                        "y_pos",
                        amp_col,
                        "x_guess",
                        "y_guess",
                        "sigma_guess",
                        "offset_guess",
                    ],
                    function=lambda exprs: _gaussian_2D_fit(*exprs),
                    returns_scalar=False,
                    return_dtype=pl.Struct(
                        [
                            pl.Field("popt", pl.List(pl.Float64)),
                            pl.Field("pcov", pl.List(pl.Float64)),
                        ]
                    ),
                )
                .first()
                .alias("fit_result"),
            )
            .unnest("fit_result")
            .rename({"popt": out_cols[1], "pcov": out_cols[6]})
            .explode(["x_pos", "y_pos", amp_col])
            .group_by("det", maintain_order=True)
            .agg(
                pl.all().exclude(out_cols[1], out_cols[6]),
                pl.col(out_cols[1]).first(),
                pl.col(out_cols[6]).first(),
                pl.map_groups(
                    exprs=["x_pos", "y_pos", out_cols[1]],
                    function=lambda exprs: _gaussian_2D_eval(
                        exprs[0], exprs[1], exprs[2]
                    ),
                    returns_scalar=False,
                    return_dtype=pl.Float64,
                ).alias(out_cols[0]),
            )
            .explode([out_cols[0]])
            .with_columns(
                pl.col(out_cols[1]).list.get(1).alias(out_cols[2]),
                pl.col(out_cols[1]).list.get(2).alias(out_cols[3]),
                pl.col(out_cols[1]).list.get(0).alias(out_cols[4]),
                pl.col(out_cols[1]).list.get(3).alias(out_cols[5]),
                pl.col(out_cols[6]).list.get(1).alias(out_cols[7]),
                pl.col(out_cols[6]).list.get(2).alias(out_cols[8]),
                pl.col(out_cols[6]).list.get(0).alias(out_cols[9]),
                pl.col(out_cols[6]).list.get(3).alias(out_cols[10]),
            )
            .select(out_cols)
            .drop(out_cols[1], out_cols[6])
            .collect()
        )
        if out_cols[0] in df.schema:
            df = df.drop(out_cols)
        return pl.concat([df.sort("det"), fit_df], how="horizontal")

    return (gaussian_2D_fit,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
