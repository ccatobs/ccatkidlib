import matplotlib.pyplot as plt
import numpy as np


def interactive_plot(xs, ys):
    """
    This function takes in two lists of arrays, xs and ys, and creates an interactive plot
    where you can switch between different plot combinations using left and right arrow keys.
    The x and y indices will always be the same.
    """

    # Global variable for tracking the current index
    current_idx = 0

    def plot_selected_elements(idx):
        """Plot the selected xs and ys elements based on the given index."""
        plt.clf()  # Clear current figure
        if len(ys.shape)>2: plt.plot(xs[idx], ys[idx], s=1, label=f"xs[{idx}] vs ys[{idx}]")
        else:
            for y in ys:
                plt.plot(xs[idx], y[idx], s=1, label=f"xs[{idx}] vs y[{idx}]")
        plt.xlabel('X values')
        plt.ylabel('Y values')
        plt.title('Selected xs vs ys plot')
        plt.legend()
        plt.draw()

    def on_key(event):
        """Handle key press events to change the plot."""
        nonlocal current_idx

        if event.key == 'right':  # Right arrow key for next plot
            current_idx = (current_idx + 1) % len(xs)  # Loop over indices
            plot_selected_elements(current_idx)

        elif event.key == 'left':  # Left arrow key for previous plot
            current_idx = (current_idx - 1) % len(xs)  # Loop over indices
            plot_selected_elements(current_idx)

    # Initial plot
    fig, ax = plt.subplots()
    plot_selected_elements(current_idx)

    # Connect the key press event
    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()


