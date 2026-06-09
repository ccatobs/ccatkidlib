import marimo

__generated_with = "0.23.2"
app = marimo.App(width="columns")


@app.cell(column=0, hide_code=True)
def _(analysis_cfg_browser, cfg_editor, data_browser, mo):
    mo.md(rf"""
    ### Select Data to Load

    {mo.hstack([mo.vstack([analysis_cfg_browser, data_browser]), cfg_editor])}
    """)
    return


@app.cell(hide_code=True)
def _(com_to_selector, mo, vna_selector):
    mo.md(rf"""
    ### Select RFSoC Drones

    {mo.hstack([com_to_selector, vna_selector])}
    """)
    return


@app.cell(hide_code=True)
def _(find_res_tabs, mo):
    mo.md(rf"""
    ### Tune Peak Finding Parameters

    {mo.vstack([find_res_tabs], align="center")}
    """)
    return


@app.cell
def _(hv, opts, vna_plots):
    hv.Layout(vna_plots).opts(
        sublabel_format="",
        shared_axes=False,
    ).opts(opts.Overlay(ylabel="$|S_{21}|$ [dB]", show_legend=False, fig_size=250)).opts(opts.Curve(aspect=3, linewidth=0.25)).cols(4)
    return


@app.cell
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
    from ccatkidlib.analysis.core.network import Network
    from ccatkidlib.analysis.core.vna import VNA

    return VNA, ccat_io


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
def _(com_to_selector, data_dirs, mo):
    _vna_dir = data_dirs[0] / "vna"
    _com_tos = com_to_selector.value

    vnas_dict = {}
    for _com_to in sorted(_com_tos):
        _bid, _drid = _com_to.split(".")
        _avail_vnas = []
        for _vna_file in (
            _com_to_vna := (_vna_dir / f"B{_bid}D{_drid}")
        ).iterdir():
            _avail_vnas.append(_vna_file)
        _avail_vnas = sorted(_avail_vnas)

        vnas_dict[_com_to] = mo.ui.multiselect(
            _avail_vnas,
            value=[_avail_vnas[-1]],
        )

    vna_selector = mo.ui.dictionary(vnas_dict, label="Select VNA File...")
    return (vna_selector,)


@app.cell
def _(VNA, analysis_cfg, hv, mo, viz_cfg, vna_selector):
    mo.stop(not vna_selector.value)
    hv.extension("matplotlib")

    vnas, vna_plots, find_res_sliders, res_vlines = {}, {}, {}, {}
    for _com_to, _vna_file in vna_selector.value.items():
        _vna = VNA(
            com_to=_com_to,
            data_path=_vna_file,
            analysis_cfg=analysis_cfg,
            viz_cfg=viz_cfg,
        )
        _vna.mag(dB=True)
        vnas[_com_to] = _vna
        vna_plots[_com_to] = _vna.mag_plot(prefix="dB", return_df=False, ms=0)

        _peak_prom_std = mo.ui.number(
            start=0,
            stop=100,
            value=3,
            debounce=True,
            full_width=True,
            label="Peak Prominence [Standard Deviations]",
        )

        _peak_dis = mo.ui.number(
            start=0,
            stop=10_000,
            value=100,
            debounce=True,
            full_width=True,
            label="Peak Distance",
        )

        _peak_width = mo.ui.range_slider(
            start=0,
            stop=1_000,
            value=[25, 200],
            show_value=True,
            debounce=True,
            full_width=True,
            label="Peak Width",
        )

        _stitch = mo.ui.switch(
            value=False,
            label="Stitch VNA Bins",
        )

        _stitch_sw = mo.ui.number(
            start=0,
            stop=250,
            value=100,
            debounce=True,
            full_width=True,
            label="Stitch Edge Size",
        )

        _remove_cont = mo.ui.switch(
            value=False,
            label="Remove Continuum",
        )

        _cont_wn = mo.ui.number(
            start=0,
            stop=10_000,
            value=300,
            debounce=True,
            full_width=True,
            label="Continuum Window",
        )

        _remove_noise = mo.ui.switch(
            value=True,
            label="Remove Noise",
        )

        _noise_wn = mo.ui.number(
            start=0,
            stop=50_000,
            value=7_500,
            debounce=True,
            full_width=True,
            label="Noise Window",
        )

        _wlen = mo.ui.number(
            start=0,
            stop=10_000,
            value=200,
            debounce=True,
            full_width=True,
            label="Window Length",
        )

        find_res_sliders[_com_to] = mo.ui.dictionary(
            {
                "peak_prom_std": _peak_prom_std,
                "peak_dis": _peak_dis,
                "peak_width": _peak_width,
                "stitch": _stitch,
                "stitch_sw": _stitch_sw,
                "remove_cont": _remove_cont,
                "cont_wn": _cont_wn,
                "remove_noise": _remove_noise,
                "noise_wn": _noise_wn,
                "wlen": _wlen,
            }
        )

    find_res_sliders = mo.ui.dictionary(
        find_res_sliders, label="Peak Finding Parameters"
    )
    return find_res_sliders, res_vlines, vna_plots, vnas


@app.cell
def _(com_to_selector, findResonators, find_res_sliders, hv, res_vlines, vnas):
    num_res = {}
    for _com_to in com_to_selector.value:
        _vna = vnas[_com_to]
        _res_find = find_res_sliders[_com_to]

        _f = _vna.data["f"].to_numpy()
        _I, _Q = _vna.get_data(["I", "Q"]).to_numpy().T
        _z = _I + 1j * _Q

        _vna_res, _vna_filt = findResonators(
            _f,
            _z,
            peak_prom_std=_res_find["peak_prom_std"].value,
            wlen=_res_find["wlen"].value,
            width_min=_res_find["peak_width"].value[0],
            width_max=_res_find["peak_width"].value[1],
            peak_dis=_res_find["peak_dis"].value,
            stitch_sw=_res_find["stitch_sw"].value,
            stitch=_res_find["stitch"].value,
            remove_noise=_res_find["remove_noise"].value,
            noise_wn=_res_find["noise_wn"].value,
            continuum_wn=_res_find["cont_wn"].value,
            remove_cont=_res_find["remove_cont"].value,
        )

        num_res[_com_to] = len(_vna_res)
        res_vlines[_com_to] = hv.VLines(
            _vna_res, label=f"{len(_vna_res)} Found Detectors"
        ).opts(color="red", linewidth=0.25)
    return (num_res,)


@app.cell
def _(find_res_sliders, hv, mo, num_res, res_vlines, vna_plots):
    hv.extension("matplotlib")
    _tabs_dict = {}
    for _com_to, _dict in find_res_sliders.items():
        _ui = mo.vstack(
            [
                mo.hstack(
                    [
                        _dict["peak_prom_std"],
                        _dict["peak_dis"],
                        _dict["wlen"],
                    ]
                ),
                _dict["peak_width"],
                mo.hstack(
                    [_dict["stitch"], _dict["remove_cont"], _dict["remove_noise"]]
                ),
                mo.hstack(
                    [_dict["stitch_sw"], _dict["cont_wn"], _dict["noise_wn"]]
                ),
                mo.mpl.interactive(
                    hv.render(
                        (vna_plots[_com_to] * res_vlines[_com_to]).opts(
                            xlabel="Frequency [Hz]",
                            ylabel="$|S_{21}|$ [dB]",
                            aspect=2,
                            fig_size=220,
                            title=f"{num_res[_com_to]} Found Detectors",
                        ),
                        backend="matplotlib",
                    ).gca()
                ),
            ]
        )

        _tabs_dict[_com_to] = _ui

    find_res_tabs = mo.ui.tabs(_tabs_dict, label="Drones", lazy=True)
    return (find_res_tabs,)


@app.cell
def _():
    return


@app.cell(column=2)
def _():
    # From primecam_readout
    def butterFilter(y, x, btype, cutoff_freqs, order=3, x_time=False):
        """Butterworth digital and analog filter.

        x, y: (1D array of floats) The data.
        btype: (str) {'lowpass', 'highpass', 'bandpass', 'bandstop'}.
        cutoff_freq: (float or 2-tuple of floats) The cutoff frequencies.
        order: (int) Filter order.
        x_time: (bool) x axis is time (default is frequency).
        """

        import numpy as np
        from scipy.signal import butter, filtfilt

        fs = np.abs(x[1] - x[0])
        nyquist = 0.5 * fs
        normal_cutoff = cutoff_freqs / nyquist
        b, a = butter(order, normal_cutoff, btype=btype, fs=fs)
        filtered_data = filtfilt(b, a, y)

        return filtered_data


    def stitchS21m(S21m, bw=500, sw=100):
        """Shift S21 mags so the sweep channel bin ends align.

        S21m: (array of floats) 1D array of S21 complex modulus.
        bw:   (int) Width of the stitch bins.
        sw:   (int) Width of slice (at ends) of each stitch bin to take median.
        """

        import numpy as np

        a = S21m.reshape(-1, bw)  # reshape into bins

        meds_i = np.median(a[:, :sw], axis=1)  # medians on left
        meds_f = np.median(a[:, -sw:], axis=-1)  # medians on right

        f = meds_i[1:] - meds_f[:-1]  # bin power misalignment
        f = np.pad(f, (1, 0), mode="constant")  # 1st bin -> 0 misalignment
        f = np.cumsum(f)  # misalignments are cumulative
        f = f.reshape((a.shape[0], 1))  # reshape for matrix addition
        a_n = a - f  # misalignment correction (stitch)

        return a_n.flatten()  # reshape to 1D and return


    def findResonators(
        f,
        Z,
        peak_prom_std=15,
        peak_prom_db=0,
        peak_dis=500,
        width_min=5,
        width_max=1000,
        stitch=True,
        stitch_sw=100,
        remove_cont=True,
        continuum_wn=300,
        remove_noise=True,
        noise_wn=30_000,
        stitch_bw=None,
        wlen=200,
    ):
        """

        f:   (1D array of floats) Frequency of S21 samples.
        Z: (1D array of complex) Forward transmission S_21 as complex.
        peak_prom_std: (float) Peak height from surroundings, in noise std multiples.
                        Uses larger of peak_prom_db or peak_prom_std.
        peak_prom_db:  (float) Peak height from surroundings, in Db.
                        Uses larger of peak_prom_db or peak_prom_std.
        peak_dis:      (int) Min distance between peaks [bins].
        width_min      (int) Peak width minimum. [bins]
        width_max      (int) Peak width maximum. [bins]
        stitch:        (bool) Whether to stitch (comb discontinuities).
        stitch_sw:     (int) Discontinuity edge size for alignment [bins].
        remove_cont:   (bool) Whether to subtract the continuum.
        continuum_wn:  (int) Continuum filter cutoff frequency [Hz].
        remove_noise:  (
        \bool) Whether to subtract noise.
        noise_wn:      (int) Noise filter cutoff frequency [Hz].
        stitch_bw:     (int) Bins width of the stitch channels.
        """

        from scipy.signal import find_peaks, medfilt
        import numpy as np

        # type enforcement
        # required since parameters can get passed as strings
        peak_prom_std = float(peak_prom_std)
        peak_prom_db = float(peak_prom_db)
        peak_dis = int(peak_dis)
        peak_width = (int(width_min), int(width_max))
        stitch_sw = int(stitch_sw)
        continuum_wn = int(continuum_wn)
        noise_wn = int(noise_wn)

        try:
            stitch_bw = int(stitch_bw)
        except:
            stitch_bw = 500  # bins bw <- steps

        x = f
        y = np.abs(Z)

        # convert Db input to linear
        peak_prom_lin = np.amax(y) * (1 - 10 ** (-peak_prom_db / 20))

        # stitch discontinuities
        if stitch:
            y = stitchS21m(y, bw=stitch_bw, sw=stitch_sw)

        if remove_cont:
            y -= butterFilter(y, x, "low", continuum_wn, order=3)

        # remove noise
        y_noise = butterFilter(y, x, "high", noise_wn, order=3)
        noise_std = np.std(y_noise)
        if remove_noise:
            y -= y_noise

        # prominence
        prom = max(peak_prom_std * noise_std, peak_prom_lin)

        # find peaks
        i_peaks, peak_properties = find_peaks(
            x=-y, prominence=prom, distance=peak_dis, width=peak_width
        )

        f_res = f[i_peaks]

        return f_res, y

    return (findResonators,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
