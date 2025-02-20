import matplotlib.pyplot as plt

def init_fig(figax = None, **kwargs):
    if figax:
        fig, ax = figax
    else:
        fig, ax = plt.subplots(figsize=(10, 10))
    return fig, ax