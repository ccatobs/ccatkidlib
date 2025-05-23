from .data import Data
from pathlib import Path
import sys
import numpy as np
import gc
import pandas as pd

# Bokeh Imports
from bokeh.models import CheckboxButtonGroup, CustomJS, ColumnDataSource
from bokeh.layouts import layout, column
from bokeh.io import show
from bokeh.plotting import curdoc

# Local Imports
import ccatkidlib.rfsoc_io as rfsoc_io
import ccatkidlib.utils as utils
import ccatkidlib.analysis.pair as pair
import ccatkidlib.analysis.plot_utils as putils


class Sweep(Data):
    '''
    Class representing a sweep over a range of frequencies.
    '''

    def __init__(self, com_to, analysis_cfg = str(Path(__file__).parent / 'analysis_config.yaml'), **kwargs):
        super().__init__(com_to, analysis_cfg, **kwargs)

        self.res_freqs = None
        self.res_s21z  = None
        
        self.data_path = self.data_path[0]

    ####################
    # Plotting Methods #
    ####################

    def dashboard(self, dB = False, show_plot = True, **kwargs):
        plot_cfg = self.plot_cfg
        fig_height = 600

        kwargs['height'] = int(fig_height/2.1)
        kwargs['sizing_mode'] = 'stretch_width'
        mag_fig, mag_lyot, source, mag_glyphs = self.plot_mag(dB = dB, show_plot = False, **kwargs)
        phase_fig, phase_lyot, source, phase_glyphs = self.plot_phase(show_plot = False, source = source, **kwargs)

        kwargs['height'] = fig_height
        kwargs['width'] = fig_height
        kwargs['sizing_mode'] = 'fixed'
        IQ_fig, IQ_lyot, source, IQ_glpyhs =  self.plot_IQ(show_plot = False, source = source, **kwargs)

        sweep_col = column(mag_lyot, phase_lyot)
        lyot = layout([[sweep_col, IQ_lyot]], sizing_mode='stretch_width')
        if show_plot: 
            curdoc().add_root(lyot)
            show(lyot)
        return lyot

    def plot_mag(self, fig = None, source = None, freqs = None, s21m = None, res_freqs = None, res_s21m = None, dB = False, show_plot = True, **kwargs):
        kwargs.setdefault('y_axis_label', f"|S21| {'[dB]' if dB else ''}")

        # Get sweep data
        # ---------------
        if freqs is None: freqs = self.freqs
        if s21m is None: s21m = np.abs(self.s21z)

        s21m =  utils.convert_to_dB(s21m) if dB else s21m # Convert to dB if specified

        # Get found resonant frequencies associated with sweep
        # ----------------------------------------------------
        if res_freqs is None: res_freqs = self.res_freqs

        res_s21z = self.res_s21z
        if res_s21z is not None and (len(res_s21z) > 0):
            if res_s21m is None: res_s21m = np.abs(res_s21z)
        
        if dB and (res_s21m is not None):
            res_s21m = utils.convert_to_dB(res_s21m)

        if source is None:
            source = ColumnDataSource(data={'freqs': freqs, 's21m': s21m})
        else:
            source.data.update({'freqs':freqs})
            source.data.update({'s21m':s21m})

        # Plot Sweep
        # ----------
        fig, lyot, glyphs = self._plot_sweep('freqs', 's21m', res_freqs, res_s21m, source = source, fig = fig, show_plot = show_plot, **kwargs)
        
        return fig, lyot, source, glyphs

    def plot_phase(self, fig = None, source = None, freqs = None, phase = None, res_freqs = None, res_phase = None, show_plot = True, **kwargs):
        kwargs.setdefault('y_axis_label', f"Phase [rad]")

        # Get sweep data
        # ---------------
        if freqs is None: freqs = self.freqs
        if phase is None: phase = np.arctan2(np.imag(self.s21z), np.real(self.s21z))

        # Get found resonant frequencies associated with sweep
        # -----------------------------------------------------
        if res_freqs is None: res_freqs = self.res_freqs

        res_s21z = self.res_s21z
        if res_s21z is not None and (len(res_s21z) > 0):
            if res_phase is None: res_phase = np.arctan2(np.imag(res_s21z), np.real(res_s21z))
                
        if source is None:
            source = ColumnDataSource(data={'freqs': freqs, 'phase': phase})
        else:
            source.data.update({'freqs':freqs})
            source.data.update({'phase':phase})

        # Plot Sweep
        # ----------
        fig, lyot, glyphs = self._plot_sweep('freqs', 'phase', res_freqs, res_phase, source = source, fig = fig, show_plot = show_plot, **kwargs)
        
        return fig, lyot, source, glyphs

    def plot_IQ(self, fig = None, I = None, Q = None, res_I = None, res_Q = None, source = None, show_plot = True, **kwargs):
        # Create local copy of plot_cfg since it is used multiple times
        plot_cfg = self.plot_cfg
        sizing_mode = 'fixed'

        for key, value in kwargs.items():
            if key == 'sizing_mode':
                sizing_mode = value

        # Define default plot x label, y label, and title
        # -----------------------------------------------
        kwargs.setdefault('x_axis_label', 'I')
        kwargs.setdefault('y_axis_label', 'Q')

        kwargs.setdefault('aspect_ratio', 1)

        try:
            network = self.drone_cfg['det_config']['network']
        except:
            network = '?'
        kwargs.setdefault('title', f"Sweep of {self.drone_cfg['det_config']['detector_type']} Network {network} Taken on {utils.convert_timestamp(self.timestamp)} EST")

        if I is None: I = np.real(self.s21z)
        if Q is None: Q = np.imag(self.s21z)

        kwargs['plot_line'] = False

        if source is None:
            source = ColumnDataSource(data={'I': I, 'Q': Q})
        else:
            source.data.update({'I':I})
            source.data.update({'Q':Q})

        fig, glyphs = putils.line_scatter(fig, 'I', 'Q', source, cfg=plot_cfg, **kwargs)

        res_s21z = self.res_s21z
        if res_s21z is not None and (len(res_s21z) > 0):
            if res_I is None: res_I = np.real(res_s21z)
            if res_Q is None: res_Q = np.imag(res_s21z)

        res_glyphs = None
        if res_I is not None and res_Q is not None:
            kwargs['res_line'] = False
            fig, res_glyphs = putils.plot_res(fig, res_I, res_Q, cfg = plot_cfg, **kwargs)
        
        # Create CheckboxButton for toggling graph glyphs
        # -----------------------------------------------
        labels = ["Sweep Line", "Sweep Scatter", "Tones Scatter"]
        button_glyphs = [glyphs[0],glyphs[1]]
        if isinstance(res_glyphs, list): button_glyphs.append(res_glyphs[1])
        active = [glyph.visible for glyph in button_glyphs]
        num_buttons = len(active)

        checkbox_button = putils.create_glyph_buttons(labels, active, button_glyphs, num_buttons)

        lyot = layout(checkbox_button,fig, sizing_mode=sizing_mode)

        # Show/Save plot
        # --------------
        if show_plot:
            curdoc().add_root(lyot)
            show(lyot)
        return fig, lyot, source, [glyphs, res_glyphs]

    def _plot_sweep(self, freqs, data, res_freqs, res_data, source, fig = None, show_plot = True, **kwargs):
        # Create local copy of plot_cfg since it is used multiple times
        plot_cfg = self.plot_cfg
        sizing_mode = plot_cfg['plot_defaults']['figure']['sizing_mode']

        for key, value in kwargs.items():
            if key == 'sizing_mode':
                sizing_mode = value

        # Define default plot x label, y label, and title
        # -----------------------------------------------
        kwargs.setdefault('x_axis_label', 'Frequency [Hz]')

        try:
            network = self.drone_cfg['det_config']['network']
        except:
            network = '?'
        kwargs.setdefault('title', f"Sweep of {self.drone_cfg['det_config']['detector_type']} Network {network} Taken on {utils.convert_timestamp(self.timestamp)} EST")


        fig, glyphs = putils.line_scatter(fig, freqs, data, source, cfg = plot_cfg, **kwargs) # Plot sweep

        # Create bin BoxAnnotations if enabled
        # ------------------------------------
        boxes = []
        if plot_cfg['plot_defaults']['binboxes']['enable']:
            try:
                sweep_steps = self.drone_cfg['tones']['sweep_steps']
            except KeyError:
                sweep_steps = self.drone_cfg['tones']['N_step']
            bins = source.data['freqs'].reshape((-1, sweep_steps))
            fig, boxes = putils.plot_bin_boxes(fig, bins, **kwargs)

        res_glyphs = None
        if res_freqs is not None and res_data is not None:
            fig, res_glyphs = putils.plot_res(fig, res_freqs, res_data, cfg = plot_cfg, **kwargs)
        
        # Create CheckboxButton for toggling graph glyphs
        # -----------------------------------------------
        labels = ["Sweep Line", "Sweep Scatter"]
        button_glyphs = [glyphs[0],glyphs[1]]
        if isinstance(res_glyphs, list):
            labels += ["Tones Lines", "Tones Scatter"]
            button_glyphs += [res_glyphs[0], res_glyphs[1]]

        active = [glyph.visible for glyph in button_glyphs]
        if plot_cfg['plot_defaults']['binboxes']['enable']: 
            labels.append('Bin Boxes')
            button_glyphs += boxes            
            active.append(boxes[0].visible)
        num_buttons = len(active)
        
        checkbox_button = putils.create_glyph_buttons(labels, active, button_glyphs, num_buttons)

        lyot = layout(checkbox_button, fig, sizing_mode=sizing_mode)

        # Show/Save plot
        # --------------
        if show_plot:
            curdoc().add_root(lyot)
            show(lyot)
            #putils.save_fig(fig, self.plot_cfg, **kwargs)
        return fig, lyot, [glyphs, res_glyphs, boxes]

    #################################
    # Internal Data Loading Methods #
    #################################

    def _load_res_freqs(self):
        res_freqs = self.drone_cfg['det_config']['found_detector_freqs']
        if isinstance(res_freqs, list):
            res_freqs = np.real(res_freqs)
        else:
            try:
                res_freqs = np.real(np.load(res_freqs))
            except:
                res_freqs = None
        return res_freqs
    
    def _get_res_s21z(self):
        res_s21z = None
        res_freqs = self.res_freqs
        if res_freqs is not None and len(res_freqs) > 0:
            res_s21z = [self.s21z[np.argmin(np.abs(self.freqs - freq))] for freq in res_freqs]
        return res_s21z

    def _load_sweep(self):
        '''
        Load VNA/target sweep file
        '''
        data = {'fs': None, 'I': None, 'Q': None}
        fs, s21z = np.load(self.data_path, mmap_mode='r')
        I, Q = s21z.real, s21z.imag

        data['fs'], data['I'], data['Q'] = fs.real, I, Q

        return data
    
    #######################
    # Data Getter Methods #
    #######################

    #@method_timer
    def transform_sweep(self, name, func, res_num = None):
        '''
        Method for adding data transformed by func to self.data pandas DataFrame.
        '''
        if not name in self.data.columns:
            transformed_data = pd.DataFrame(func(None, None))
            transformed_data.columns = [name]
            self.data = pd.concat([self.data, transformed_data], axis=1).sort_index(axis=1)
        return self.data[name]

    def fs(self):
        return self.data['fs']

    #################
    # Magic Methods #
    #################

    def __getattribute__(self, name):
        if name == 'data':
            if super().__getattribute__("data") is None:
                self.data = pd.DataFrame(self._load_sweep()).sort_index(axis=1, level=0)
                num_levels = len(super().__getattribute__("data").columns.names)
                self.data.columns.names = ['Data'] if num_levels == 1 else ['Data', 'Resonators']
                self.data.index.names = ['Sample']
            gc.collect()
        elif name == 'res_freqs':
            if super().__getattribute__("res_freqs") is None: self.res_freqs = self._load_res_freqs()
        elif name == 'res_s21z':
            if super().__getattribute__("res_s21z") is None: self.res_s21z = self._get_res_s21z()

        return super().__getattribute__(name)
