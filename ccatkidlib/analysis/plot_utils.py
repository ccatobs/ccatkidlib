#=================================#
# rfsoc_io.py               2025 #
# Darshan Patel dp649@cornell.edu #
#=================================#

'''
Library of helper functions for plotting KID sweep and timestream data.
'''

from bokeh.plotting import figure
from bokeh.models import CheckboxButtonGroup, CustomJS, BoxAnnotation, ColumnDataSource


def plot_res(fig, res_x, res_y, cfg=None, **kwargs):
    res_line = False
    res_scatter = True

    res_line_kwargs = {}
    res_scatter_kwargs = {}
    try:
        if cfg:
            res_line = cfg['plot_defaults']['res']['plot_line']
            res_line_kwargs = cfg['plot_defaults']['res']['line']
            res_scatter = cfg['plot_defaults']['res']['plot_scatter']
            res_scatter_kwargs = cfg['plot_defaults']['res']['scatter']
    except: 
        pass
    for key, value in kwargs.items():
        if key == 'res_line':
            res_line = value
            continue
        elif key == 'res_scatter':
            res_scatter = value
            continue

        split_key = key.split('_')
        if split_key[0] == 'res':
            line_key    = '_'.join(split_key[1:])
            if line_key in res_line_kwargs:
                res_line_kwargs[line_key] = value
                continue
            
            split_key = line_key.split('_')
            if split_key[0]  == 'scatter' and '_'.join(split_key[1:]) in res_scatter_kwargs:
                res_scatter_kwargs['_'.join(split_key[1:])] = value
    line = fig.vspan(x = res_x, level = 'underlay', visible = res_line,**res_line_kwargs)
    scatter = fig.scatter(res_x, res_y, visible = res_scatter, **res_scatter_kwargs)

    return fig, [line,scatter]

def plot_bin_boxes(fig, bins, **kwargs):
    bin_colors = ['dimgray', 'lightgray']
    hatch_patterns = ['/', '\\']
    hatch_alpha = 0.01
    hatch_scale = 25
    bin_alpha = 0.15
    plot_bin_boxes = False

    for key, value in kwargs.items():
        if key == 'bin_colors':
            bin_colors = value
        elif key == 'bin_alpha':
            bin_alpha = value
        elif key == 'hatch_patterns':
            hatch_patterns = value
        elif key == 'hatch_alpha':
            hatch_alpha = value
        elif key == 'hatch_scale':
            hatch_scale = value
        elif key == 'plot_bin_boxes':
            plot_bin_boxes = value

    boxes = [BoxAnnotation(left = bin[0], right = bin[-1], visible = plot_bin_boxes,fill_color = bin_colors[i%2], fill_alpha = bin_alpha, hatch_pattern = hatch_patterns[i%2], hatch_alpha = hatch_alpha, hatch_scale = hatch_scale, level = 'underlay') for i,bin in enumerate(bins)]
    for box in boxes: fig.add_layout(box)
    return fig, boxes

def line_scatter(fig, x, y, source, cfg = None, **kwargs):
    plot_line = True
    plot_scatter = False
    line_kwargs = {}
    scatter_kwargs = {}
    try:
        if cfg: 
            plot_line = cfg['plot_defaults']['line_scatter']['plot_line']
            line_kwargs = cfg['plot_defaults']['line_scatter']['line']
            plot_scatter = cfg['plot_defaults']['line_scatter']['plot_scatter']
            scatter_kwargs = cfg['plot_defaults']['line_scatter']['scatter']
    except:
        pass

    for key, value in kwargs.items():
        if key in line_kwargs: 
            line_kwargs[key] = value
            continue
        elif key in scatter_kwargs: 
            scatter_kwargs[key] = value
            continue
        elif key == 'plot_line':
            plot_line = value
            continue
        elif key == 'plot_scatter':
            plot_scatter = value
            continue

        split_key = key.split('_')
        if split_key[0] == 'scatter' and '_'.join(split_key[1:]) in scatter_kwargs:
            scatter_kwargs['_'.join(split_key[1:])] = value  

    fig = get_fig(fig, cfg, **kwargs)
    line = fig.line(x = x, y = y, source = source, visible = plot_line, **line_kwargs)
    scatter = fig.scatter(x = x, y = y, source = source, visible = plot_scatter, **scatter_kwargs)
    return fig, [line, scatter]

def get_fig(fig, cfg, **kwargs):
    if not fig:
        fig_kwargs = {'x_axis_label':'', 'y_axis_label':'', 'title':'', 'aspect_ratio': 'auto'}
        try:
            if cfg: fig_kwargs.update(cfg['plot_defaults']['figure'])
        except:
            pass
        for key, value in kwargs.items():
            if key in fig_kwargs: fig_kwargs[key] = value
        
        fig = figure(**fig_kwargs)
    return fig

def create_glyph_buttons(labels, active, button_glyphs, num_buttons):
    checkbox_button = CheckboxButtonGroup(labels = labels, active= [i for i, active in enumerate(active) if active], sizing_mode = 'stretch_width', height = 50)
        
    update_checkbox = """
    let isVisible = btn.active.includes({ind});
    glyph.visible=isVisible;
    glyph.change.emit();
    """

    for i, glyph in enumerate(button_glyphs):
        ind = i if i < num_buttons else num_buttons - 1
        checkbox_button.js_on_change('active', CustomJS(args=dict(btn=checkbox_button, glyph=glyph), code=update_checkbox.format(ind=ind)))

    return checkbox_button