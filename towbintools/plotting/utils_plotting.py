import os

import matplotlib.axes
import matplotlib.figure
import numpy as np
import seaborn as sns

# THIS PART IS MOSTLY ABOUT HANDLING LEGENDS, SAVING FIGURES, ETC.


def save_figure(
    fig: matplotlib.figure.Figure,
    name: str,
    directory: str,
    format: str = "svg",
    dpi: int = 300,
    transparent: bool = True,
) -> None:
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


def build_legend(single_condition_dict: dict, legend: dict | None) -> str:
    """
    Build a legend label string for a single condition.

    Parameters:
        single_condition_dict (dict) : Condition dict containing metadata fields.
        legend (dict or None) : Mapping of condition dict keys to unit/suffix strings.
            Each key whose value is truthy is appended as ``"<value> <unit>"``;
            a falsy value appends only ``"<value>"``.  If ``None``, the label
            defaults to ``"Condition <condition_id>"``.

    Returns:
        str : Legend label for the condition.
    """
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


def set_scale(ax: matplotlib.axes.Axes, log_scale: bool | tuple | list) -> None:
    """
    Set the x- and/or y-axis scale of a matplotlib Axes object.

    Parameters:
        ax (matplotlib.axes.Axes) : Axes to configure.
        log_scale (bool or tuple or list) : Scale specification.
            - ``bool``: applies to the y-axis only (``True`` → log, ``False`` → linear).
            - ``tuple`` or ``list`` of two bools ``(x_log, y_log)``: sets both axes independently.

    Returns:
        None
    """
    if isinstance(log_scale, bool):
        ax.set_yscale("log" if log_scale else "linear")
    elif isinstance(log_scale, tuple):
        ax.set_yscale("log" if log_scale[1] else "linear")
        ax.set_xscale("log" if log_scale[0] else "linear")
    elif isinstance(log_scale, list):
        ax.set_yscale("log" if log_scale[1] else "linear")
        ax.set_xscale("log" if log_scale[0] else "linear")


def get_colors(
    conditions_to_plot: list,
    colors: list | dict | None,
    base_palette: str = "colorblind",
) -> list:
    """
    Return a list of colors, one per condition, validating user-supplied values.

    Parameters:
        conditions_to_plot (list) : Ordered list of condition identifiers.
        colors (list or dict or None) : Color specification.
            - ``None``: a seaborn ``base_palette`` palette is generated automatically.
            - ``list``: must have the same length as ``conditions_to_plot``.
            - ``dict``: must contain a key for every entry in ``conditions_to_plot``;
              values are returned in the same order.
        base_palette (str) : Seaborn palette name used when ``colors`` is ``None``.
            Defaults to ``"colorblind"``.

    Returns:
        list : Colors in the same order as ``conditions_to_plot``.

    Raises:
        AssertionError : If a list ``colors`` has a different length than ``conditions_to_plot``,
            or if a dict ``colors`` is missing entries for any condition.
    """
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
