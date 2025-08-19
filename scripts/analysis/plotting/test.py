import pickle
import polars as pl
import numpy as np
import time
import sys

from tqdm import tqdm
from math import ceil
from pathlib import Path
from lmfit import Parameters

from ccatkidlib.analysis.core.target import Target
from ccatkidlib.analysis.core.network import Network
import ccatkidlib.analysis.viz.sweep as sweep_viz
import ccatkidlib.analysis.fit.fit as ccat_fit


import panel as pn
import datashader as ds
import holoviews as hv
import hvplot.polars

from holoviews import opts
from collections.abc import Iterable
from holoviews.operation.datashader import rasterize, datashade, dynspread, shade
from bokeh.palettes import Sunset, Plasma256, Inferno256, Cividis256, Magma256, Viridis256

pn.extension('mathjax')
hv.extension('bokeh', enable_mathjax = True)

import warnings
warnings.filterwarnings('ignore')

network_df = Network(com_to='1.1', analysis_cfg = str(Path(__file__).parent / 'analysis_config.yaml'), sess_ids = '1754105411', data_dir='coldload', date='20250802', include_targs=False)
#targ = Target(com_to='1.1', tones=-1, data_path='/ccat/data-280GHz/md0/cooldown_june/coldload/20250621/1750519793/targ/B1D2/coldload_targ_1750555057.npy')
#targ.mag()
#targ.mag(dB=True)
#targ.phase()

#sweep_viz.Mag(sweep=targ).servable()