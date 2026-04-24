import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt

from .utils_data_processing import exclude_arrests_from_series_at_ecdysis
from .utils_plotting import build_legend
from .utils_plotting import get_colors


def plot_cv_at_ecdysis(
    conditions_struct: dict,
    column: str,
    conditions_to_plot: list[int],
    remove_hatch=True,
    legend=None,
    colors=None,
    x_axis_label=None,
    y_axis_label=None,
    exclude_arrests: bool = False,
):
    color_palette = get_colors(conditions_to_plot, colors)

    for i, condition in enumerate(conditions_to_plot):
        condition_dict = conditions_struct[condition]
        values = condition_dict[column]
        if remove_hatch:
            values = values[:, 1:]
        if exclude_arrests:
            values = exclude_arrests_from_series_at_ecdysis(values)
        cvs = np.nanstd(values, axis=0) / np.nanmean(values, axis=0) * 100
        label = build_legend(condition_dict, legend)
        plt.plot(cvs, label=label, marker="o", color=color_palette[i])
    # replace the ticks by [L1, L2, L3, L4]
    if remove_hatch:
        plt.xticks(range(4), ["M1", "M2", "M3", "M4"])
    else:
        plt.xticks(range(5), ["Hatch", "M1", "M2", "M3", "M4"])
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.legend()
    fig = plt.gcf()
    plt.show()
    return fig


def plot_std_at_ecdysis(
    conditions_struct: dict,
    column: str,
    conditions_to_plot: list[int],
    remove_hatch=True,
    legend=None,
    colors=None,
    x_axis_label=None,
    y_axis_label=None,
    exclude_arrests: bool = False,
):
    color_palette = get_colors(conditions_to_plot, colors)

    for i, condition in enumerate(conditions_to_plot):
        condition_dict = conditions_struct[condition]
        values = condition_dict[column]
        if remove_hatch:
            values = values[:, 1:]
        if exclude_arrests:
            values = exclude_arrests_from_series_at_ecdysis(values)
        stds = np.nanstd(values, axis=0)
        label = build_legend(condition_dict, legend)
        plt.plot(stds, label=label, marker="o", color=color_palette[i])
    if remove_hatch:
        plt.xticks(range(4), ["M1", "M2", "M3", "M4"])
    else:
        plt.xticks(range(5), ["Hatch", "M1", "M2", "M3", "M4"])
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.legend()
    fig = plt.gcf()
    plt.show()
    return fig


def plot_cv_development_percentage(
    conditions_struct: dict,
    column: str,
    conditions_to_plot: list[int],
    percentages: np.ndarray = np.linspace(0, 1, 11),
    legend=None,
    colors=None,
    x_axis_label=None,
    y_axis_label=None,
):
    color_palette = get_colors(conditions_to_plot, colors)
    for i, condition in enumerate(conditions_to_plot):
        condition_dict = conditions_struct[condition]
        values = condition_dict[column]
        percentages_index = np.clip(
            percentages * values.shape[1], 0, values.shape[1] - 1
        )
        values = values[:, percentages_index.astype(int)]
        epsilon = np.finfo(values.dtype).eps
        cvs = np.nanstd(values, axis=0) / (np.nanmean(values, axis=0) + epsilon) * 100
        label = build_legend(condition_dict, legend)
        plt.plot(
            percentages * 100, cvs, label=label, color=color_palette[i], marker="o"
        )
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.legend()
    fig = plt.gcf()
    plt.show()
    return fig


def plot_cv_rescaled_data(
    conditions_struct: dict,
    column: str,
    conditions_to_plot: list[int],
    smooth: bool = False,
    legend=None,
    colors=None,
    x_axis_label=None,
    y_axis_label=None,
):
    color_palette = get_colors(conditions_to_plot, colors)

    for i, condition in enumerate(conditions_to_plot):
        condition_dict = conditions_struct[condition]
        values = condition_dict[column]
        cvs = np.nanstd(values, axis=0) / np.nanmean(values, axis=0) * 100
        label = build_legend(condition_dict, legend)
        if smooth:
            cvs = medfilt(cvs, 7)
            # cvs = savgol_filter(cvs, 15, 3)
        plt.plot(cvs, label=label, color=color_palette[i])
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.legend()
    fig = plt.gcf()
    plt.show()
    return fig
