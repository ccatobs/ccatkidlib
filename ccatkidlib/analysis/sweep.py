from pathlib import Path
import sys
import numpy as np



from bokeh.models import CheckboxButtonGroup, CustomJS, ColumnDataSource
from bokeh.layouts import layout
from bokeh.io import show
from bokeh.plotting import curdoc

# Local Imports
import ccatkidlib.rfsoc_io as rfsoc_io
import ccatkidlib.utils as utils
import ccatkidlib.analysis.pair as pair
import ccatkidlib.analysis.plot_utils as putils


class Sweep:
    '''
    Class representing a sweep over a range of frequencies.
    '''

    def __init__(self, com_to, analysis_cfg = str(Path(__file__).parent / 'analysis_config.yaml'), **kwargs):
        # Define sweep attributes
        # -----------------------
        self.bid, self.drid = com_to.split('.') # Baard and drone sweep was taken with

        # Sweep frequency and complex S21 data
        self.freqs = None
        self.s21z  = None

        self.res_freqs = None
        self.res_s21z  = None

        self.sweep_path = None # File path of sweep
        self.analysis_cfg, self.plot_cfg = rfsoc_io.load_config(analysis_cfg) # File path of analysis config

        # Parse key word arguments
        for key, value in kwargs.items():
            if key == 'sweep_path':
                self.sweep_path = value
            elif key == 'freqs':
                self.freqs = value
            elif key == 's21z':
                self.s21z = value

        # If full sweep path is not provided, find sweep data file based on timestamp and (optional) file path parts
        # ----------------------------------------------------------------------------------------------------------
        if self.sweep_path is None:
            # Find sweep data file using 
            data_type  = None
            timestamp  = None

            # Parse sweep data file part key word arguments
            # ---------------------------------------------
            root_data_dir = self.analysis_cfg['data_load']['root_data_dir']
            data_dir = '**'
            date = '**'
            sess_id = '**'

            for key, value in kwargs.items():
                if key == 'data_type':
                    data_type = value
                elif key == 'timestamp':
                    timestamp = value
                elif key == 'root_data_dir':
                    root_data_dir = value
                elif key == 'data_dir':
                    data_dir = value
                elif key == 'date':
                    date = value
                elif key == 'sess_id':
                    sess_id = value

            # Ensure that timestamp and type of sweep are provided to uniquely find sweep data file
            assert (timestamp is not None and data_type is not None), "Need to provide either the full path to the sweep or the sweep timestamp and type ('vna' or 'targ')!"
            
            # Try to find sweep data file using given information
            try:
                self.sweep_path = pair.get_data_file(com_to, timestamp, data_dir = data_dir, date = date, sess_id = sess_id, data_type = data_type, root_data_dir=root_data_dir)[0]
                self.timestamp = timestamp
            except:
                raise FileNotFoundError(f'Could not find {data_type} file for board {self.bid}, drone {self.drid} with timestamp {timestamp}! Check that all optional file path segments are correct!')
        else:
            self.timestamp = pair.get_timestamp(self.sweep_path)
        # Get io, ext, and drone configs associated with the sweep data file
        self.sweep_configs = pair.get_config(self.sweep_path, all_cfg=False)
        self.io_cfg = None
        self.ext_cfg = None
        self.drone_cfg = None


    ####################
    # Plotting Methods #
    ####################
    def dashboard(self, db = False, show_plot = True, **kwargs):
        pass

    def plot_mag(self, fig = None, source = None, freqs = None, s21m = None, res_freqs = None, res_s21m = None, dB = False, show_plot = True, **kwargs):
        kwargs.setdefault('y_axis_label', f"|S21| {'[dB]' if dB else ''}")

        # Get sweep data
        # ---------------
        freqs = freqs if freqs is not None else self.freqs
        s21m  = s21m if s21m is not None else np.abs(self.s21z)

        s21m =  utils.convert_to_dB(s21m) if dB else s21m # Convert to dB if specified

        # Get found resonant frequencies associated with sweep
        # -----------------------------------------------------
        res_freqs = res_freqs if res_freqs is not None else self.res_freqs
        res_s21m = res_s21m if res_s21m is not None else np.abs(self.res_s21z)
        
        res_s21m = utils.convert_to_dB(res_s21m) if dB else res_s21m

        if source is None:
            source = ColumnDataSource(data={'freqs': freqs, 's21m': s21m})
        else:
            source.data.update({'freqs':freqs})
            source.data.update({'s21m':s21m})

        # Plot Sweep
        # ----------
        fig = self._plot_sweep('freqs', 's21m', res_freqs, res_s21m, source = source, fig = fig, show_plot = show_plot, **kwargs)
        
        return fig, source

    def plot_phase(self, fig = None, source = None, freqs = None, phase = None, res_freqs = None, res_phase = None, show_plot = True, **kwargs):
        kwargs.setdefault('y_axis_label', f"Phase [rad]")

        # Get sweep data
        # ---------------
        freqs = freqs if freqs is not None else self.freqs
        phase = phase if phase is not None else np.arctan2(np.imag(self.s21z), np.real(self.s21z))

        # Get found resonant frequencies associated with sweep
        # -----------------------------------------------------
        res_freqs = res_freqs if res_freqs is not None else self.res_freqs
        res_phase = res_phase if res_phase is not None else np.arctan2(np.imag(self.res_s21z), np.real(self.res_s21z))
        
        if source is None:
            source = ColumnDataSource(data={'freqs': freqs, 'phase': phase})
        else:
            source.data.update({'freqs':freqs})
            source.data.update({'phase':phase})

        # Plot Sweep
        # ----------
        fig = self._plot_sweep('freqs', 'phase', res_freqs, res_phase, source, fig = fig, show_plot = show_plot, **kwargs)
        
        return fig, source

    def plot_IQ(self, fig = None, I = None, Q = None, res_I = None, res_Q = None, source = None, show_plot = True, **kwargs):
        # Create local copy of plot_cfg since it is used multiple times
        plot_cfg = self.plot_cfg

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

        I = I if I is not None else np.real(self.s21z)
        Q = Q if Q is not None else np.imag(self.s21z)

        kwargs['plot_line'] = False

        if source is None:
            source = ColumnDataSource(data={'I': I, 'Q': Q})
        else:
            source.data.update({'I':I})
            source.data.update({'Q':Q})

        fig, glyphs = putils.line_scatter(fig, 'I', 'Q', source, cfg=plot_cfg, **kwargs)

        res_I = res_I if res_I is not None else np.real(self.res_s21z)
        res_Q = res_Q if res_Q is not None else np.imag(self.res_s21z)
        if res_I is not None and res_Q is not None:
            kwargs['res_line'] = False
            fig, res_glyphs = putils.plot_res(fig, res_I, res_Q, cfg = plot_cfg, **kwargs)

        # Show/Save plot
        # --------------
        if show_plot:
            # Create CheckboxButton for toggling graph glyphs
            # -----------------------------------------------
            labels = ["Sweep Line", "Sweep Scatter", "Tones Scatter"]
            button_glyphs = [glyphs[0],glyphs[1], res_glyphs[1]]
            active = [glyph.visible for glyph in button_glyphs]
            checkbox_button = CheckboxButtonGroup(labels = labels, active= [i for i, active in enumerate(active) if active])
            
            update_checkbox = """
            let isVisible = btn.active.includes({ind});
            glyph.visible=isVisible;
            glyph.change.emit();
            """

            for i, glyph in enumerate(button_glyphs):
                checkbox_button.js_on_change('active', CustomJS(args=dict(btn=checkbox_button, glyph=glyph), code=update_checkbox.format(ind=i)))

            lyot = layout(checkbox_button,fig, sizing_mode='fixed')
            show(lyot)
        return fig, source

    def _plot_sweep(self, freqs, data, res_freqs, res_data, source, fig = None, show_plot = True, **kwargs):
        # Create local copy of plot_cfg since it is used multiple times
        plot_cfg = self.plot_cfg

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
            N_step = self.drone_cfg['tones']['N_step']
            bins = source.data['freqs'].reshape((-1, 500))
            fig, boxes = putils.plot_bin_boxes(fig, bins, **kwargs)

        if res_freqs is not None and res_data is not None:
            fig, res_glyphs = putils.plot_res(fig, res_freqs, res_data, cfg = plot_cfg, **kwargs)
        
        # Show/Save plot
        # --------------
        if show_plot:
            # Create CheckboxButton for toggling graph glyphs
            # -----------------------------------------------
            labels = ["Sweep Line", "Sweep Scatter", "Tones Lines", "Tones Scatter"]
            button_glyphs = [glyphs[0],glyphs[1], res_glyphs[0], res_glyphs[1]]
            active = [glyph.visible for glyph in button_glyphs]
            if plot_cfg['plot_defaults']['binboxes']['enable']: 
                labels.append('Bin Boxes')
                button_glyphs += boxes            
                active.append(boxes[0].visible)
            num_buttons = len(active)
            checkbox_button = CheckboxButtonGroup(labels = labels, active= [i for i, active in enumerate(active) if active])
            
            update_checkbox = """
            let isVisible = btn.active.includes({ind});
            glyph.visible=isVisible;
            glyph.change.emit();
            """

            for i, glyph in enumerate(button_glyphs):
                ind = i if i < num_buttons else num_buttons - 1
                checkbox_button.js_on_change('active', CustomJS(args=dict(btn=checkbox_button, glyph=glyph), code=update_checkbox.format(ind=ind)))

            lyot = layout(checkbox_button,fig, sizing_mode=plot_cfg['plot_defaults']['figure']['sizing_mode'])
            
            curdoc().add_root(lyot)
            show(lyot)
            #putils.save_fig(fig, self.plot_cfg, **kwargs)
        return fig


    #################################
    # Internal Data Loading Methods #
    #################################

    def _load_res_freqs(self):
        res_freqs =  self.drone_cfg['det_config']['found_detector_freqs']
        try:
            res_freqs = np.load(res_freqs)
        except:
            pass
        return res_freqs
    
    def _get_res_s21z(self):
        res_s21z = None
        if self.res_freqs is not None:
            res_s21z = [self.s21z[np.argmin(np.abs(self.freqs - freq))] for freq in self.res_freqs]
        return res_s21z

    def _load_sweep(self):
        freqs, s21z = np.load(self.sweep_path, mmap_mode='r')
        return freqs.real, s21z        
    
    def _load_cfg(self, id):
        cfg = None
        for i, config_path in enumerate(self.sweep_configs):
            if id in str(config_path): 
                cfg = rfsoc_io.load_config(config_path)
                self.sweep_configs.pop(i)
                break
        return cfg

    #################
    # Magic Methods #
    #################

    def __getattribute__(self, name):
        if name == 'freqs' or name == 's21z':
            if super().__getattribute__("freqs")  is None: self.freqs, self.s21z = self._load_sweep()
        elif name == 'res_freqs':
            if super().__getattribute__("res_freqs") is None: self.res_freqs = self._load_res_freqs()
        elif name == 'res_s21z':
            if super().__getattribute__("res_s21z") is None: self.res_s21z = self._get_res_s21z()
        elif name == 'io_cfg':
            if super().__getattribute__("io_cfg") is None: self.io_cfg = self._load_cfg('_io_')
        elif name == 'ext_cfg':
            if super().__getattribute__("ext_cfg") is None: self.ext_cfg = self._load_cfg('_ext_')
        elif name == 'drone_cfg':
            if super().__getattribute__("drone_cfg") is None: self.drone_cfg = self._load_cfg('_drone_')

        return super().__getattribute__(name)
