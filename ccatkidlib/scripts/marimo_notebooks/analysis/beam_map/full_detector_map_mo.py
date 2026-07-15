import marimo

__generated_with = "0.23.10"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(analysis_cfg_browser, cfg_editor, detector_map_browser, mo):
    mo.md(rf"""
    ### Select Data to Load

    {mo.hstack([mo.vstack([analysis_cfg_browser, detector_map_browser]), cfg_editor])}
    """)
    return


@app.cell
def _():
    return


@app.cell(column=1)
def _(detector_map_browser, mo, pl):
    mo.stop(not detector_map_browser.value)

    all_network_props = None
    for _f in detector_map_browser.value:
        _df = (pl.read_parquet(_f.path).with_columns((pl.col('detector_type') + ' ' + pl.col('network')).alias('array_network'))
                                       .with_columns(pl.when(pl.col('array_network') == 'Al 1.1')
                                                       .then(pl.lit('Al_1.2'))
                                                       .otherwise(pl.col('array_network')).alias('array_network'))
                                        .with_columns(pl.when(pl.col('array_network') == 'Al 1.2')
                                                       .then(pl.lit('Al_1.1'))
                                                       .otherwise(pl.col('array_network')).alias('array_network'))
                                        .with_columns(pl.when(pl.col('array_network') == 'Al 2.1')
                                                        .then(pl.lit('Al_2.2'))
                                                        .otherwise(pl.col('array_network')).alias('array_network'))
                                        .with_columns(pl.when(pl.col('array_network') == 'Al 2.2')
                                                       .then(pl.lit('Al_2.1'))
                                                       .otherwise(pl.col('array_network')).alias('array_network'))
                                        .with_columns(pl.when(pl.col('array_network') == 'TiN 1.1')
                                                       .then(pl.lit('TiN_1.4'))
                                                       .otherwise(pl.col('array_network')).alias('array_network'))
                                        .with_columns(pl.when(pl.col('array_network') == 'TiN 1.2')
                                                       .then(pl.lit('TiN_1.5'))
                                                       .otherwise(pl.col('array_network')).alias('array_network'))
                                        .with_columns(pl.when(pl.col('array_network') == 'TiN 1.3')
                                                                .then(pl.lit('TiN_1.6'))
                                                                .otherwise(pl.col('array_network')).alias('array_network'))
                                        .with_columns(pl.when(pl.col('array_network') == 'TiN 1.4')
                                                                .then(pl.lit('TiN_1.2'))
                                                                .otherwise(pl.col('array_network')).alias('array_network'))
                                        .with_columns(pl.when(pl.col('array_network') == 'TiN 1.5')
                                                                .then(pl.lit('TiN_1.1'))
                                                                .otherwise(pl.col('array_network')).alias('array_network'))
                                        .with_columns(pl.when(pl.col('array_network') == 'TiN 1.6')
                                                                .then(pl.lit('TiN_1.3'))
                                                                .otherwise(pl.col('array_network')).alias('array_network'))
                                        .with_columns(pl.col('array_network').str.split('_').list.join(' ').alias('array_network'))
              )




        all_network_props = _df if all_network_props is None else pl.concat([all_network_props, _df], how='vertical')
    all_network_props
    return (all_network_props,)


@app.cell
def _(glasbey, hv):
    cmap = hv.Cycle(glasbey.create_palette(palette_size=16, colorblind_safe=True, optimize_palette_search_radius=1000, cvd_severity=75, lightness_bounds=(15, 85)))
    return (cmap,)


@app.cell
def _(
    all_network_props,
    bounds_selector,
    cmap,
    grid_lines,
    hv,
    mo,
    np,
    opts,
    pl,
):
    _bounds = bounds_selector.value
    _grid_bounds = _bounds*np.tan(np.pi/6)+_bounds

    map_grid = (hv.Overlay([hv.Slope(np.tan(np.pi/6), _offset) for _offset in np.linspace(-1*_grid_bounds, _grid_bounds, grid_lines.value)] + [hv.Slope(-np.tan(np.pi/6), _offset) for _offset in np.linspace(-1*_grid_bounds, _grid_bounds, grid_lines.value)]# + 
    #[hv.VLine(_offset) for _offset in np.linspace(-1*_bounds, _bounds, grid_lines.value)] + 
    #[hv.HLine(_offset) for _offset in np.linspace(-1*_bounds, _bounds, grid_lines.value)])
    )).opts(
        opts.Slope(linewidth=0.1, linestyle="--", color="k", show_legend=False),
        opts.VLine(linewidth=0.25, linestyle="--", color="k", show_legend=False),
        opts.HLine(linewidth=0.25, linestyle="--", color="k", show_legend=False),
    )

    _filt_df = (all_network_props
        .rename({"array_network": "Network"})
        .filter(pl.col('amp') < 20,
                 pl.col('amp') > 1e-3,
                 pl.col('sigma') < 30,
                 pl.col('sigma') > 9.85,
                 pl.col('sigma_err') < 6,
                 pl.col('sigma_err') > 0,
                 pl.col('x_0_err') < 7,
                 pl.col('x_0_err') > 0,
                 pl.col('y_0_err') < 7)
        .with_columns(pl.col('centered_x') - 22,
                      pl.col('centered_y') + 28)
        .select('Network', 'centered_x', 'centered_y')
        .unique()
        .sort("Network"))

    detector_map = (
        map_grid*
        _filt_df
        .hvplot.points(
            x='centered_x',
            y='centered_y',
            by="Network",
            c=cmap,
            #cmap=cmap_selector.value,
            #cnorm='log',
            data_aspect=1,
            s=40**2,
        ).opts(show_legend=False)
    )

    _nets = all_network_props['array_network'].unique().sort().to_numpy()
    _legend_df = pl.DataFrame({'network': _nets, 'x': len(_nets)*[-1.5*_bounds], 'y': len(_nets)*[-1.5*_bounds]})
    _legend_plot = _legend_df.hvplot.points(x='x', y='y', by='network', c=cmap, label='Legend', s = 300**2).opts(show_legend=True)

    detector_map *= _legend_plot

    _map_opts = [opts.Overlay(fontscale=3.05, invert_xaxis=True, legend_position='right', show_grid=True, xlim=(-1*_bounds, _bounds), ylim=(-1*_bounds, _bounds), fig_inches=(7, 7), xlabel="Beam Mapper X Position [mm]", ylabel="Beam Mapper Y Position [mm]",) ]
    detector_map.opts(*_map_opts)
    _fig = hv.render(detector_map, backend='matplotlib')
    _ax = _fig.gca()
    _fig.suptitle(f'280 GHz Module Detector Map: {_filt_df.height} Detectors', fontweight='bold', fontsize=42, y=0.87)
    _fig.set_layout_engine('constrained')

    detector_map = mo.mpl.interactive(_ax)
    return (detector_map,)


@app.cell(hide_code=True)
def _(
    bounds_selector,
    cmap_range_selector,
    cmap_selector,
    detector_map,
    grid_lines,
    mo,
):
    mo.md(rf"""
    ### Plot Detector Map

    {mo.vstack([mo.hstack([cmap_selector, cmap_range_selector], widths=[2, 4]), mo.hstack([bounds_selector, grid_lines]), detector_map])}
    """)
    return


@app.cell
def _(hv):
    hv.output(fig='png', dpi=99)
    return


@app.cell
def _(all_network_props, pl):
    (all_network_props
        .rename({"array_network": "Network"})
        .filter(pl.col('amp') < 20,
                 pl.col('amp') > 1e-3,
                 pl.col('sigma') < 30,
                 pl.col('sigma') > 9.85,
                 pl.col('sigma_err') < 6,
                 pl.col('sigma_err') > 0,
                 pl.col('x_0_err') < 7,
                 pl.col('x_0_err') > 0,
                 pl.col('y_0_err') < 7)
        .with_columns(pl.col('centered_x') - 22,
                      pl.col('centered_y') + 28)
        .select('Network', 'centered_x', 'centered_y')
        .unique()
        .sort("Network").height)
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
    import ccatkidlib.analysis.viz.viz_utils as viz_utils
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
    import matplotlib
    import matplotlib.pyplot as plt
    import holoviews as hv
    import hvplot.polars
    import panel as pn
    import glasbey

    from holoviews import opts

    hv.extension("matplotlib")
    return glasbey, hv, matplotlib, opts


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
def _(HOME_DIR, analysis_cfg):
    root_data_dir = (
        analysis_cfg["file_paths"]["root_data_dir"] if analysis_cfg else HOME_DIR
    )
    return (root_data_dir,)


@app.cell
def _(mo, root_data_dir):
    detector_map_browser = mo.ui.file_browser(
        initial_path=root_data_dir,
        filetypes=[".parquet"],
        multiple=True,
        ignore_empty_dirs=True,
        label="Select parquet file(s) with detector map data...",
    )
    return (detector_map_browser,)


@app.cell
def _(mo):
    bounds_selector = mo.ui.slider(start=1, stop=500, step=1, value=180, label='Select Plot Bounds', debounce=True, full_width=True)
    return (bounds_selector,)


@app.cell
def _(mo):
    grid_lines = mo.ui.slider(start=1, stop=100, step=1, value=25, label='Select Number of Grid Lines', debounce=True, full_width=True)
    return (grid_lines,)


@app.cell
def _(matplotlib, mo):
    cmap_selector = mo.ui.dropdown(options=list(matplotlib.colormaps), value='Set1', allow_select_none=False, searchable=True, label='Select Detector Map Colormap', full_width=True)

    cmap_range_selector = mo.ui.range_slider(start=0, stop=1, step=0.1, label='Select Detector Map Colormap Range', full_width=True, value=[0, 1], debounce=True, show_value=True)
    return cmap_range_selector, cmap_selector


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

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
