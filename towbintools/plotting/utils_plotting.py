import os

import numpy as np
import seaborn as sns

# THIS PART IS MOSTLY ABOUT HANDLING LEGENDS, SAVING FIGURES, ETC.


def save_figure(fig, name, directory, format="svg", dpi=300, transparent=True):
    """
    Save a given matplotlib figure to the specified directory with the given name, in the chose format.

    Parameters:
        fig (matplotlib.figure.Figure) : Figure to save
        name (str) : Name of the file (without extension)
        directory (str) : Directory to save the file in
        format (str) : File format to save the figure in
        dpi (int) : Resolution of the saved figure
        transparent (bool) : Whether to save the figure with a transparent background

    Returns:
        str : Full path to the saved file
    """

    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Construct full file path
    filename = f"{name}.{format}"
    filepath = os.path.join(directory, filename)

    # Save the figure
    fig.savefig(
        filepath,
        format=format,
        dpi=dpi,
        bbox_inches="tight",
        transparent=transparent,
    )


def build_legend(single_condition_dict, legend):
    if legend is None:
        return f'Condition {int(single_condition_dict["condition_id"])}'
    else:
        legend_string = ""
        for i, (key, value) in enumerate(legend.items()):
            if value:
                legend_string += f"{single_condition_dict[key]} {value}"
            else:
                legend_string += f"{single_condition_dict[key]}"
            if i < len(legend) - 1:
                legend_string += ", "
        return legend_string


def set_scale(ax, log_scale):
    if isinstance(log_scale, bool):
        ax.set_yscale("log" if log_scale else "linear")
    elif isinstance(log_scale, tuple):
        ax.set_yscale("log" if log_scale[1] else "linear")
        ax.set_xscale("log" if log_scale[0] else "linear")
    elif isinstance(log_scale, list):
        ax.set_yscale("log" if log_scale[1] else "linear")
        ax.set_xscale("log" if log_scale[0] else "linear")


def get_colors(conditions_to_plot, colors, base_palette="colorblind"):
    if colors is None:
        colors = sns.color_palette("colorblind", len(conditions_to_plot))
    else:
        if isinstance(colors, list):
            assert len(colors) == len(
                conditions_to_plot
            ), f"Length of colors list ({len(colors)}) does not match number of conditions to plot ({len(conditions_to_plot)})"
            colors = colors
        elif isinstance(colors, dict):
            assert np.all(
                [key in colors.keys() for key in conditions_to_plot]
            ), "Some conditions to plot are not in the colors dictionary"
            colors = [colors[condition] for condition in conditions_to_plot]

    return colors
