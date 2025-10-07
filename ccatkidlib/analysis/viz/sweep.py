import panel as pn
import holoviews as hv
import hvplot.polars
import param
import polars as pl
import numpy as np

from holoviews.operation.datashader import datashade, dynspread
from panel.viewable import Viewer

from ccatkidlib.analysis.core.vna import VNA
from ccatkidlib.analysis.core.target import Target


class SweepView(Viewer):
    sweep = param.ClassSelector(class_=(VNA, Target))
    view = param.String(default='IQ')

    x_col, y_col = param.String(''), param.String('')
    x_prefix, y_prefix = param.Selector(default=''), param.ListSelector(default=[''])
    tones = param.Selector(default=-1, allow_refs = True)

    widgets_visible = param.Boolean(default=True)    

    def __init__(self, **params):
        super().__init__(**params)

        x = 'I' if self.view == 'IQ' else 'f'
        y = 'Q' if x == 'I' else self.view
        self.x_col, self.y_col = x, y

        sweep = self.sweep
        x_cols, y_cols = sweep.get_data([f'.*{x}']).columns, sweep.get_data([f'.*{y}']).columns
        x_prefix_list = list(set([sweep._get_prefix(col, x) for col in x_cols]))
        y_prefix_list = list(set([sweep._get_prefix(col, y) for col in y_cols]))

        if len(x_prefix_list) == 0 or len(y_prefix_list) == 0:
            raise ValueError('Could not find data needed to create plot. Ensure that all required columns exist in the sweep.data DataFrame.')

        # Set the full lists of available prefixes
        self.param.x_prefix.objects, self.param.y_prefix.objects = x_prefix_list, y_prefix_list
        
        # Set the default prefixes to the first elements in the prefix lists
        self.x_prefix = [x_prefix_list[0]] if isinstance(self.x_prefix, param.ListSelector) else x_prefix_list[0] # Need to check if Selector or ListSelector since IQ uses ListSelector while phase/mag use Selector
        self.y_prefix = [y_prefix_list[0]] 

        tones = self.sweep.tones
        if tones is not None: self.param.tones.objects = [-1] + tones
    
    def _get_plot_data(self):
        tone = self.tones if self.tones != -1 else None

        x_prefix = self.x_prefix if isinstance(self.x_prefix, list) else [self.x_prefix]
        x_cols = [f"{x_pre}{'_' if x_pre else ''}{self.x_col}" for x_pre in x_prefix]
        y_cols = [f"{y_pre}{'_' if y_pre else ''}{self.y_col}" for y_pre in self.y_prefix]

        df = self.sweep.get_data(x_cols + y_cols, include=tone)
        if self.sweep.tones is not None:
            df_xs = [None]*len(x_cols)
            for i, x_col in enumerate(x_cols):
                df_xs[i] = df.unpivot(on=df.select(pl.col(f'^{x_col}_.*$')).columns,
                                   variable_name='temp',
                                   value_name=x_col).drop('temp')
            
            df_ys = [None]*len(y_cols)
            for i, y_col in enumerate(y_cols):
                df_ys[i] = df.unpivot(on=df.select(pl.col(f'^{y_col}_.*$')).columns,
                                      variable_name='temp',
                                      value_name=y_col).drop('temp')
            
            df = pl.concat(df_xs + df_ys, how='horizontal')
        return df

class Mag(SweepView):
    view = param.String(default='mag', readonly=True)
    rasterize = param.Boolean(default=True)

    @pn.depends('x_prefix', 'y_prefix', 'tones', 'rasterize', watch=True)
    def _plot(self):
        df = super()._get_plot_data()
        df_cols = df.columns

        plots = [None]*(len(df_cols) - 1)
        for i, df_col in enumerate(df_cols[1:]):
            fig = df.hvplot.line(x = df_cols[0],
                                 y = df_col)
            if self.rasterize: fig = datashade(fig).opts(xlabel='Frequency [Hz]',
                                                         ylabel='|S21|')
            plots[i] = fig
        return hv.Overlay(plots).opts(xlabel='Frequency [Hz]',
                                      ylabel='|S21|')

    def __panel__(self):
        tone_selector = pn.widgets.Select.from_param(self.param.tones)
        tone_selector.name = 'Tone'
        tone_selector.visible = self.widgets_visible

        x_prefix_selector = pn.widgets.Select.from_param(self.param.x_prefix)
        x_prefix_selector.name = 'Frequency Prefix'

        y_prefix_selector = pn.widgets.MultiChoice.from_param(self.param.y_prefix)
        y_prefix_selector.name = 'Magnitude Prefix'

        rasterize_switch = pn.widgets.Toggle.from_param(self.param.rasterize)
        rasterize_switch.name, rasterize_switch.button_type, rasterize_switch.button_style = 'Rasterize', 'primary', 'outline'
        
        self.widgets = [tone_selector, x_prefix_selector, y_prefix_selector, rasterize_switch]
        widgets = self.widgets if self.widgets_visible else []

        return pn.Row(pn.pane.HoloViews(self._plot, linked_axes=False),
                      pn.Column(*widgets))

class Phase(SweepView):
    view = param.String(default='phase', readonly=True)
    rasterize = param.Boolean(default=True)

    @pn.depends('x_prefix', 'y_prefix', 'tones', 'rasterize', watch=False)
    def _plot(self):
        df = super()._get_plot_data()
        df_cols = df.columns

        plots = [None]*(len(df_cols) - 1)
        for i, df_col in enumerate(df_cols[1:]):
            fig = df.hvplot.line(x = df_cols[0],
                                 y = df_col,
                                 xlabel='Frequency [Hz]',
                                 ylabel='Phase [rad]')
            if self.rasterize: fig = datashade(fig,
                                               width=800,
                                               height=400).opts(xlabel='Frequency [Hz]',
                                                                ylabel='Phase [rad]')
            plots[i] = fig
        return hv.Overlay(plots).opts(xlabel='Frequency [Hz]',
                                      ylabel='Phase [rad]')

    def __panel__(self):
        tone_selector = pn.widgets.Select.from_param(self.param.tones)
        tone_selector.name = 'Tone'
        tone_selector.visible = self.widgets_visible

        x_prefix_selector = pn.widgets.Select.from_param(self.param.x_prefix)
        x_prefix_selector.name = 'Frequency Prefix'

        y_prefix_selector = pn.widgets.MultiChoice.from_param(self.param.y_prefix)
        y_prefix_selector.name = 'Phase Prefix'

        rasterize_switch = pn.widgets.Toggle.from_param(self.param.rasterize)
        rasterize_switch.name, rasterize_switch.button_type, rasterize_switch.button_style = 'Rasterize', 'primary', 'outline'
        
        self.widgets = [tone_selector, x_prefix_selector, y_prefix_selector, rasterize_switch]
        widgets = self.widgets if self.widgets_visible else []

        return pn.Row(pn.panel(self._plot, width=800, height=400),
                      pn.Column(*widgets))
    
class IQ(SweepView):
    def __panel__(self):
        return
    
class Dashboard(Viewer):
    sweep = param.ClassSelector(class_=(VNA, Target))
    views = param.List(default=[Mag, Phase])
    tones = param.Selector(default=-1)

    def __init__(self, **params):
        super().__init__(**params)
        
        self._view_objs = [view(sweep=self.sweep, tones = self.param.tones, widgets_visible = False) for view in self.views]
        self._views = pn.Column(*self._view_objs)
    
        tones = self.sweep.tones
        if tones is not None: self.param.tones.objects = [-1] + tones
    
    def __panel__(self):
        widgets = [[self.param.tones]] + [[pn.pane.Markdown(view.view)] + view.widgets for view in self._view_objs]
        widgets = [widget for widget_list in widgets for widget in widget_list]
        return pn.Row(pn.Column(*widgets),
                      self._views)