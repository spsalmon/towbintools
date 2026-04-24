import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.interpolate import make_interp_spline
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from .utils_plotting import build_legend
from .utils_plotting import get_colors
from .utils_plotting import set_scale
from towbintools.data_analysis import rescale_and_aggregate
from towbintools.foundation.utils import find_best_string_match

# from sklearn.linear_model import LinearRegression
# from sklearn.linear_model import RANSACRegressor

# from .utils_data_processing import exclude_arrests_from_series_at_ecdysis

# MODEL BUILDING


def _get_continuous_proportion_model(
    rescaled_series_one,
    rescaled_series_two,
    x_axis_label=None,
    y_axis_label=None,
    plot_model=True,
    remove_outliers=True,
):
    assert len(rescaled_series_one) == len(
        rescaled_series_two
    ), "The two series must have the same length."

    series_one = np.array(rescaled_series_one).flatten()
    series_two = np.array(rescaled_series_two).flatten()

    # remove elements that are nan in one of the two arrays
    correct_indices = ~np.isnan(series_one) & ~np.isnan(series_two)
    series_one = series_one[correct_indices]
    series_two = series_two[correct_indices]

    # log transform the data
    series_one = np.log(series_one)
    series_two = np.log(series_two)

    # for duplicate values, take the mean
    unique_series_one = np.unique(series_one)
    unique_series_two = np.array(
        [np.mean(series_two[series_one == value]) for value in unique_series_one]
    )

    series_one = unique_series_one
    series_two = unique_series_two

    plt.scatter(series_one, series_two)

    if x_axis_label is not None:
        plt.xlabel(x_axis_label)
    else:
        plt.xlabel("column one")

    if y_axis_label is not None:
        plt.ylabel(y_axis_label)
    else:
        plt.ylabel("column two")

    # lowess will return our "smoothed" data with a y value for at every x-value
    lowess = sm.nonparametric.lowess(series_two, series_one, frac=0.1)

    # unpack the lowess smoothed points to their values
    lowess_x = list(zip(*lowess))[0]
    lowess_y = list(zip(*lowess))[1]

    # interpolate the loess curve
    model = make_interp_spline(lowess_x, lowess_y, k=1)

    x = np.linspace(min(series_one), max(series_one), 500)
    y = model(x)

    plt.plot(x, y, color="red", linewidth=2)
    plt.show()

    return model


def _get_proportion_model(
    series_one_values,
    series_two_values,
    x_axis_label=None,
    y_axis_label=None,
    poly_degree=2,
    plot_model=True,
    remove_outliers=True,
):
    assert len(series_one_values) == len(
        series_two_values
    ), "The two series must have the same length."

    alpha = 0.05  # significance level for confidence intervals

    isolation_forest = IsolationForest()
    fitting_x = []
    fitting_y = []

    model_plot_x = []

    for i in range(series_two_values.shape[-1]):
        values_one = series_one_values[:, i].flatten()
        values_two = series_two_values[:, i].flatten()
        correct_indices = ~np.isnan(values_one) & ~np.isnan(values_two)
        values_one = values_one[correct_indices]
        values_two = values_two[correct_indices]
        values_one = np.log(values_one)
        values_two = np.log(values_two)

        model_plot_x.extend(values_one)

        if remove_outliers:
            # remove outliers using an isolation forest
            outlier_mask = (
                isolation_forest.fit_predict(np.column_stack((values_one, values_two)))
                == 1
            )
            if plot_model:
                # plot outliers as empty circles
                plt.scatter(
                    values_one[~outlier_mask],
                    values_two[~outlier_mask],
                    facecolors="none",
                    edgecolors="black",
                )

            values_one = values_one[outlier_mask]
            values_two = values_two[outlier_mask]

        fitting_x.extend(values_one)
        fitting_y.extend(values_two)

    fitting_x = np.array(fitting_x).flatten()
    fitting_y = np.array(fitting_y).flatten()

    # Fit polynomial model with OLS
    model = Pipeline(
        [
            ("poly_features", PolynomialFeatures(degree=poly_degree)),
            ("regression", LinearRegression()),
        ]
    )
    model.fit(fitting_x.reshape(-1, 1), fitting_y)

    # Calculate confidence intervals
    def get_confidence_intervals(x_pred):
        # Transform features
        poly_features = model.named_steps["poly_features"]
        X_design = poly_features.fit_transform(fitting_x.reshape(-1, 1))
        X_pred = poly_features.transform(x_pred.reshape(-1, 1))

        # Get predictions
        y_pred = model.predict(x_pred.reshape(-1, 1))

        # Calculate residuals and standard error
        y_fitted = model.predict(fitting_x.reshape(-1, 1))
        residuals = fitting_y - y_fitted
        n = len(fitting_y)
        p = X_design.shape[1]  # number of parameters

        # Mean squared error
        mse = np.sum(residuals**2) / (n - p)

        # Standard error of prediction
        # SE = sqrt(MSE * (1 + x'(X'X)^(-1)x))
        try:
            XTX_inv = np.linalg.inv(X_design.T @ X_design)
            se_pred = np.sqrt(mse * (1 + np.sum((X_pred @ XTX_inv) * X_pred, axis=1)))
        except np.linalg.LinAlgError:
            # If matrix is singular, use simpler approximation
            se_pred = np.sqrt(mse) * np.ones(len(x_pred))

        # Critical value
        t_crit = stats.t.ppf(1 - alpha / 2, n - p)

        # Confidence intervals
        ci_lower = y_pred - t_crit * se_pred
        ci_upper = y_pred + t_crit * se_pred

        return y_pred, ci_lower, ci_upper

    if plot_model:
        plt.scatter(fitting_x, fitting_y, color="black", label="Data")

        if x_axis_label is not None:
            plt.xlabel(x_axis_label)
        else:
            plt.xlabel("column one")
        if y_axis_label is not None:
            plt.ylabel(y_axis_label)
        else:
            plt.ylabel("column two")

        # Plot model with confidence intervals
        x_plot = np.linspace(np.nanmin(model_plot_x), np.nanmax(model_plot_x), 100)
        y_pred, ci_lower, ci_upper = get_confidence_intervals(x_plot)

        plt.plot(x_plot, y_pred, color="red", label="Fitted model")
        plt.fill_between(
            x_plot,
            ci_lower,
            ci_upper,
            alpha=0.3,
            color="red",
            label=f"{int((1-alpha)*100)}% CI",
        )
        plt.legend()
        plt.show()

    # Add method to model for getting confidence intervals
    model.get_confidence_intervals = get_confidence_intervals

    return model


# COMPARE MODELS FOR DIFFERENT CONDITIONS


def plot_model_comparison_at_ecdysis(
    conditions_struct,
    column_one,
    column_two,
    conditions_to_plot,
    remove_hatch=True,
    poly_degree=2,
    remove_outliers_fitting=True,
    log_scale=(True, False),
    colors=None,
    legend=None,
    x_axis_label=None,
    y_axis_label=None,
    single_plot=True,
):
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )

    r_squared_texts = []

    if single_plot:
        fig, axs = plt.subplots(1, 1)
    else:
        fig, axs = plt.subplots(1, len(conditions_to_plot), sharey=True, sharex=True)

    for i, condition_idx in enumerate(conditions_to_plot):
        if single_plot or len(conditions_to_plot) == 1:
            current_ax = axs
        else:
            current_ax = axs[i]

        condition = conditions_struct[condition_idx]

        label = build_legend(
            condition,
            legend,
        )
        column_one_values = condition[column_one]
        column_two_values = condition[column_two]

        if remove_hatch:
            column_one_values = column_one_values[:, 1:]
            column_two_values = column_two_values[:, 1:]

        model = _get_proportion_model(
            column_one_values,
            column_two_values,
            poly_degree=poly_degree,
            plot_model=False,
            remove_outliers=remove_outliers_fitting,
        )

        scatter_x = []
        scatter_y = []
        model_plot_x = []
        model_plot_y = []

        isolation_forest = IsolationForest()

        for j in range(column_two_values.shape[-1]):
            values_one = column_one_values[:, j].flatten()
            values_two = column_two_values[:, j].flatten()
            correct_indices = ~np.isnan(values_one) & ~np.isnan(values_two)
            values_one = values_one[correct_indices]
            values_two = values_two[correct_indices]
            values_one = np.log(values_one)
            values_two = np.log(values_two)

            model_plot_x.extend(values_one)
            model_plot_y.extend(values_two)

            if remove_outliers_fitting:
                # remove outliers using an isolation forest
                outlier_mask = (
                    isolation_forest.fit_predict(
                        np.column_stack((values_one, values_two))
                    )
                    == 1
                )

                # plot outliers as empty circles
                current_ax.scatter(
                    values_one[~outlier_mask],
                    values_two[~outlier_mask],
                    facecolors="none",
                    edgecolors=color_palette[i],
                    alpha=0.5,
                )

                values_one = values_one[outlier_mask]
                values_two = values_two[outlier_mask]

            scatter_x.extend(values_one)
            scatter_y.extend(values_two)

        model_plot_x = np.array(model_plot_x)
        model_plot_y = np.array(model_plot_y)

        current_ax.scatter(
            scatter_x,
            scatter_y,
            color=color_palette[i],
            alpha=0.5,
        )

        # plot the model
        x_values = np.linspace(np.nanmin(model_plot_x), np.nanmax(model_plot_x), 100)
        y_values, ci_low, ci_high = model.get_confidence_intervals(x_values)

        current_ax.plot(
            x_values,
            y_values,
            color=color_palette[i],
            linestyle="--",
            label=label,
        )

        current_ax.fill_between(
            x_values,
            ci_low,
            ci_high,
            color=color_palette[i],
            alpha=0.2,
        )

        # if not the first plot, remove the ticks on the y-axis
        if not single_plot and i > 0:
            current_ax.tick_params(axis="y", which="both", left=False, labelleft=False)

        # text box with R^2 value
        r_squared = model.score(model_plot_x.reshape(-1, 1), model_plot_y)
        textstr = f"$R^2$ = {r_squared:.2f}"

        r_squared_texts.append(textstr)

    # modify the legend to include R^2 values
    if single_plot or len(conditions_to_plot) == 1:
        handles, labels = axs.get_legend_handles_labels()
        new_labels = [
            f"{label} ({r_squared})"
            for label, r_squared in zip(labels, r_squared_texts)
        ]
        axs.legend(handles, new_labels, loc="upper left")

        plt.xlabel(x_axis_label if x_axis_label else column_one)
        plt.ylabel(y_axis_label if y_axis_label else column_two)
    else:
        # For multiple plots, set labels for each subplot and create a shared legend
        for ax in axs:
            ax.set_xlabel(x_axis_label if x_axis_label else column_one)
        axs[0].set_ylabel(y_axis_label if y_axis_label else column_two)

        # Collect legend handles and labels from all subplots
        all_handles = []
        all_labels = []
        for ax in axs:
            handles, labels = ax.get_legend_handles_labels()
            all_handles.extend(handles)
            all_labels.extend(labels)

        # Create new labels with R^2 values
        new_labels = [
            f"{label} ({r_squared})"
            for label, r_squared in zip(all_labels, r_squared_texts)
        ]

        # Create shared legend
        fig.legend(
            all_handles,
            new_labels,
            bbox_to_anchor=(1.01, 0.5),
            loc="center left",
            title=None,
            frameon=True,
        )

    fig = plt.gcf()
    plt.show()
    return fig


# COMPUTE DEVIATION FROM MODEL


def get_deviation_from_model(
    series_one_values, series_two_values, model, percentage=True
):
    deviations = []
    for i in range(series_two_values.shape[-1]):
        values_one = series_one_values[:, i].flatten()
        values_two = series_two_values[:, i].flatten()

        correct_indices = ~np.isnan(values_one) & ~np.isnan(values_two)
        values_one = values_one[correct_indices]
        values_two = values_two[correct_indices]

        if values_one.size == 0 or values_two.size == 0:
            deviations.append(np.array([]))
        else:
            try:
                log_expected_series_two = model.predict(
                    np.log(values_one).reshape(-1, 1)
                )
            except AttributeError:
                # Continuous models do not have predict method, use the model directly
                log_expected_series_two = model(np.log(values_one))
            deviation = np.exp(np.log(values_two) - log_expected_series_two) - 1

            if percentage:
                deviation = deviation * 100

            # Create full-length array with NaNs, then fill in the valid values
            full_deviation = np.full(len(correct_indices), np.nan)
            full_deviation[correct_indices] = deviation

            deviations.append(full_deviation)

    # Pad deviations to the same length with np.nan so they can be stacked into an array
    max_len = max(len(dev) for dev in deviations)
    padded_devs = [
        np.pad(dev, (0, max_len - len(dev)), constant_values=np.nan)
        for dev in deviations
    ]
    deviations = np.array(padded_devs).T
    return deviations


def plot_correlation(
    conditions_struct,
    column_one,
    column_two,
    conditions_to_plot,
    log_scale=True,
    colors=None,
    legend=None,
    x_axis_label=None,
    y_axis_label=None,
):
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )

    for i, condition_id in enumerate(conditions_to_plot):
        condition_dict = conditions_struct[condition_id]

        qc_keys = [key for key in condition_dict.keys() if "qc" in key]
        if len(qc_keys) == 1:
            column_one_qc_key = qc_keys[0]
            column_two_qc_key = qc_keys[0]
        else:
            column_one_qc_key = find_best_string_match(column_one, qc_keys)
            column_two_qc_key = find_best_string_match(column_two, qc_keys)

        _, aggregated_series_one, _, _ = rescale_and_aggregate(
            condition_dict[column_one],
            condition_dict["time"],
            condition_dict["ecdysis_index"],
            condition_dict["larval_stage_durations_time_step"],
            condition_dict[column_one_qc_key],
            aggregation="mean",
        )

        _, aggregated_series_two, _, _ = rescale_and_aggregate(
            condition_dict[column_two],
            condition_dict["time"],
            condition_dict["ecdysis_index"],
            condition_dict["larval_stage_durations_time_step"],
            condition_dict[column_two_qc_key],
            aggregation="mean",
        )

        # sort the values
        order = np.argsort(aggregated_series_one)
        aggregated_series_one = aggregated_series_one[order]
        aggregated_series_two = aggregated_series_two[order]

        label = build_legend(condition_dict, legend)

        plt.plot(
            aggregated_series_one,
            aggregated_series_two,
            color=color_palette[i],
            label=label,
        )

    if x_axis_label is not None:
        plt.xlabel(x_axis_label)
    else:
        plt.xlabel(column_one)

    if y_axis_label is not None:
        plt.ylabel(y_axis_label)
    else:
        plt.ylabel(column_two)

    set_scale(plt.gca(), log_scale)

    plt.legend()

    fig = plt.gcf()
    plt.show()

    return fig


def plot_correlation_at_ecdysis(
    conditions_struct,
    column_one,
    column_two,
    conditions_to_plot,
    remove_hatch=True,
    log_scale=True,
    colors=None,
    legend=None,
    x_axis_label=None,
    y_axis_label=None,
):
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )

    for i, condition_id in enumerate(conditions_to_plot):
        condition_dict = conditions_struct[condition_id]

        if remove_hatch:
            column_one_values = condition_dict[column_one][:, 1:]
            column_two_values = condition_dict[column_two][:, 1:]
        else:
            column_one_values = condition_dict[column_one]
            column_two_values = condition_dict[column_two]

        x = np.nanmean(column_one_values, axis=0)
        x_std = np.nanstd(column_one_values, axis=0)

        y = np.nanmean(column_two_values, axis=0)
        y_std = np.nanstd(column_two_values, axis=0)

        label = build_legend(condition_dict, legend)
        plt.errorbar(
            x,
            y,
            xerr=x_std,
            yerr=y_std,
            fmt="o",
            color=color_palette[i],
            label=label,
            capsize=3,
        )
        plt.plot(x, y, color=color_palette[i])

    if x_axis_label is not None:
        plt.xlabel(x_axis_label)
    else:
        plt.xlabel(column_one)

    if y_axis_label is not None:
        plt.ylabel(y_axis_label)
    else:
        plt.ylabel(column_two)

    set_scale(plt.gca(), log_scale)

    plt.legend()

    fig = plt.gcf()
    plt.show()

    return fig


def plot_continuous_deviation_from_model(
    conditions_struct,
    rescaled_column_one,
    rescaled_column_two,
    control_condition_id,
    conditions_to_plot,
    deviation_as_percentage=True,
    colors=None,
    log_scale=(True, False),
    legend=None,
    x_axis_label=None,
    y_axis_label=None,
    sort_values=False,
):
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )

    xlbl = rescaled_column_one
    ylbl = rescaled_column_two

    x_axis_label = x_axis_label if x_axis_label is not None else xlbl
    y_axis_label = (
        y_axis_label
        if y_axis_label is not None
        else f"deviation from modeled {rescaled_column_two}"
    )

    control_condition = conditions_struct[control_condition_id]

    control_model = _get_continuous_proportion_model(
        control_condition[rescaled_column_one],
        control_condition[rescaled_column_two],
        x_axis_label=xlbl,
        y_axis_label=ylbl,
        plot_model=True,
    )

    for i, condition_id in enumerate(conditions_to_plot):
        condition = conditions_struct[condition_id]
        rescaled_column_one_values, rescaled_column_two_values = (
            condition[rescaled_column_one],
            condition[rescaled_column_two],
        )
        residuals = get_deviation_from_model(
            rescaled_column_one_values,
            rescaled_column_two_values,
            control_model,
            percentage=deviation_as_percentage,
        )

        if sort_values:
            sorted_indices = np.argsort(rescaled_column_one_values, axis=1)
            rescaled_column_one_values = np.take_along_axis(
                rescaled_column_one_values, sorted_indices, axis=1
            )
            residuals = np.take_along_axis(residuals, sorted_indices, axis=1)

        average_column_one_values = np.nanmean(rescaled_column_one_values, axis=0)
        average_residuals = np.nanmean(residuals, axis=0)
        ste_residuals = np.nanstd(residuals, axis=0) / np.sqrt(
            np.sum(~np.isnan(residuals), axis=0)
        )

        label = build_legend(condition, legend)
        plt.plot(
            average_column_one_values,
            average_residuals,
            label=label,
            color=color_palette[i],
        )
        plt.fill_between(
            average_column_one_values,
            average_residuals - 1.96 * ste_residuals,
            average_residuals + 1.96 * ste_residuals,
            color=color_palette[i],
            alpha=0.2,
        )

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    set_scale(plt.gca(), log_scale)

    plt.legend()

    fig = plt.gcf()
    plt.show()

    return fig


def plot_deviation_from_model_at_ecdysis(
    conditions_struct,
    column_one,
    column_two,
    control_condition_id,
    conditions_to_plot,
    remove_hatch=False,
    deviation_as_percentage=True,
    log_scale=(True, False),
    colors=None,
    legend=None,
    x_axis_label=None,
    y_axis_label=None,
    poly_degree=2,
    remove_outliers_fitting=True,
):
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )

    xlbl = column_one
    ylbl = column_two

    x_axis_label = x_axis_label if x_axis_label is not None else xlbl
    y_axis_label = (
        y_axis_label
        if y_axis_label is not None
        else f"deviation from modeled {column_two}"
    )

    control_condition = conditions_struct[control_condition_id]

    column_one_values = control_condition[column_one]
    column_two_values = control_condition[column_two]

    if remove_hatch:
        column_one_values = column_one_values[:, 1:]
        column_two_values = column_two_values[:, 1:]

    control_model = _get_proportion_model(
        column_one_values,
        column_two_values,
        x_axis_label=xlbl,
        y_axis_label=ylbl,
        poly_degree=poly_degree,
        plot_model=True,
        remove_outliers=remove_outliers_fitting,
    )

    for i, condition_id in enumerate(conditions_to_plot):
        condition = conditions_struct[condition_id]
        column_one_values, column_two_values = (
            condition[column_one],
            condition[column_two],
        )
        if remove_hatch:
            column_one_values = column_one_values[:, 1:]
            column_two_values = column_two_values[:, 1:]
        deviations = get_deviation_from_model(
            column_one_values,
            column_two_values,
            control_model,
            percentage=deviation_as_percentage,
        )

        mean_column_one_values = np.nanmean(column_one_values, axis=0)
        mean_deviations = np.nanmean(deviations, axis=0)
        # std_deviations = np.nanstd(deviations, axis=0)
        ste_deviations = np.nanstd(deviations, axis=0) / np.sqrt(
            np.sum(~np.isnan(deviations), axis=0)
        )

        label = build_legend(condition, legend)
        plt.plot(
            mean_column_one_values,
            mean_deviations,
            label=label,
            color=color_palette[i],
            marker="o",
        )
        plt.errorbar(
            mean_column_one_values,
            mean_deviations,
            yerr=ste_deviations,
            color=color_palette[i],
            fmt="o",
            capsize=3,
        )

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    set_scale(plt.gca(), log_scale)

    plt.legend()

    fig = plt.gcf()
    plt.show()

    return fig


def plot_deviation_from_model_development_percentage(
    conditions_struct,
    column_one,
    column_two,
    control_condition_id,
    conditions_to_plot,
    percentages,
    deviation_as_percentage=True,
    log_scale=(True, False),
    colors=None,
    legend=None,
    x_axis_label=None,
    y_axis_label=None,
    poly_degree=2,
    remove_outliers_fitting=True,
):
    """
    Plot the percentage deviation from a model at specified development percentages for multiple conditions.

    Args:
        conditions_struct (dict): Dictionary of condition data.
        column_one (str): Key for the first variable.
        column_two (str): Key for the second variable.
        control_condition_id (str): Key for the control condition.
        conditions_to_plot (list): List of condition keys to plot.
        percentages (np.ndarray): Array of percentages (0-1) at which to sample.
        colors (list, optional): List of colors for plotting.
        legend (list, optional): Legend labels.
        x_axis_label (str, optional): Label for x-axis.
        y_axis_label (str, optional): Label for y-axis.
        poly_degree (int, optional): Degree of polynomial for model fitting.
    """
    color_palette = get_colors(conditions_to_plot, colors)

    xlbl = column_one
    ylbl = column_two

    x_axis_label = x_axis_label if x_axis_label is not None else xlbl
    y_axis_label = (
        y_axis_label
        if y_axis_label is not None
        else f"deviation from modeled {column_two}"
    )

    control_condition = conditions_struct[control_condition_id]
    column_one_values = control_condition[column_one]
    column_two_values = control_condition[column_two]

    indices = np.clip(
        (percentages * column_one_values.shape[1]).astype(int),
        0,
        column_one_values.shape[1] - 1,
    ).astype(int)

    control_one_values = column_one_values[:, indices]
    control_two_values = column_two_values[:, indices]

    # Fit the model on the control condition
    control_model = _get_proportion_model(
        control_one_values,
        control_two_values,
        x_axis_label=xlbl,
        y_axis_label=ylbl,
        poly_degree=poly_degree,
        plot_model=True,
    )

    for i, condition_id in enumerate(conditions_to_plot):
        condition = conditions_struct[condition_id]
        column_one_values, column_two_values = (
            condition[column_one],
            condition[column_two],
        )
        column_one_values = column_one_values[:, indices]
        column_two_values = column_two_values[:, indices]

        deviations = get_deviation_from_model(
            column_one_values,
            column_two_values,
            control_model,
            percentage=deviation_as_percentage,
        )

        mean_column_one_values = np.nanmean(column_one_values, axis=0)
        mean_deviations = np.nanmean(deviations, axis=0)
        # std_deviations = np.nanstd(deviations, axis=0)
        ste_deviations = np.nanstd(deviations, axis=0) / np.sqrt(
            np.sum(~np.isnan(deviations), axis=0)
        )

        label = build_legend(condition, legend)
        plt.plot(
            mean_column_one_values,
            mean_deviations,
            label=label,
            color=color_palette[i],
            marker="o",
        )
        plt.errorbar(
            mean_column_one_values,
            mean_deviations,
            yerr=ste_deviations,
            fmt="o",
            capsize=3,
            color=color_palette[i],
        )

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.legend()
    set_scale(plt.gca(), log_scale)
    fig = plt.gcf()
    plt.show()
    return fig


# def plot_model_comparison_at_ecdysis(
#     conditions_struct,
#     column_one,
#     column_two,
#     control_condition_id,
#     conditions_to_plot,
#     remove_hatch=True,
#     log_scale=(True, False),
#     colors=None,
#     legend=None,
#     x_axis_label=None,
#     y_axis_label=None,
#     percentage=True,
#     exclude_arrests=False,
#     poly_degree=3,
# ):
#     color_palette = get_colors(
#         conditions_to_plot,
#         colors,
#     )

#     xlbl = column_one

#     x_axis_label = x_axis_label if x_axis_label is not None else xlbl
#     y_axis_label = (
#         y_axis_label
#         if y_axis_label is not None
#         else f"deviation from modeled {column_two}"
#     )

#     models = {}
#     xs = {}

#     for i, condition_id in enumerate(conditions_to_plot):
#         condition = conditions_struct[condition_id]

#         model = _get_proportion_model(
#             condition[column_one],
#             condition[column_two],
#             remove_hatch,
#             exclude_arrests=exclude_arrests,
#             poly_degree=poly_degree,
#             plot_model=False,
#         )

#         column_one_values = np.log(condition[column_one])

#         x = np.linspace(np.nanmin(column_one_values), np.nanmax(column_one_values), 100)

#         models[condition_id] = model
#         xs[condition_id] = x

#     # determine the overlap of all the x values
#     x_min = np.nanmax(
#         [np.nanmin(xs[condition_id]) for condition_id in conditions_to_plot]
#     )
#     x_max = np.nanmin(
#         [np.nanmax(xs[condition_id]) for condition_id in conditions_to_plot]
#     )

#     x = np.linspace(x_min, x_max, 100)
#     control_values = np.exp(models[control_condition_id](x))

#     for i, condition_id in enumerate(conditions_to_plot):
#         plt.plot(
#             np.exp(x),
#             (np.exp(models[condition_id](x)) / control_values - 1) * 100,
#             color=color_palette[i],
#             label=build_legend(conditions_struct[condition_id], legend),
#         )

#     plt.xlabel(x_axis_label)
#     plt.ylabel(y_axis_label)

#     set_scale(plt.gca(), log_scale)

#     plt.legend()

#     fig = plt.gcf()
#     plt.show()

#     return fig


def plot_normalized_proportions_at_ecdysis(
    conditions_struct,
    column_one,
    column_two,
    control_condition_id,
    conditions_to_plot,
    colors=None,
    aggregation="mean",
    log_scale=(True, False),
    legend=None,
    x_axis_label=None,
    y_axis_label=None,
):
    color_palette = get_colors(
        conditions_to_plot,
        colors,
    )
    control_condition = conditions_struct[control_condition_id]
    control_column_one, control_column_two = (
        control_condition[column_one],
        control_condition[column_two],
    )

    aggregation_function = np.nanmean
    control_proportion = aggregation_function(
        control_column_two / control_column_one, axis=0
    )

    x_axis_label = x_axis_label if x_axis_label is not None else column_one
    y_axis_label = (
        y_axis_label
        if y_axis_label is not None
        else f"normalized {column_two} to {column_one} ratio"
    )

    for i, condition_id in enumerate(conditions_to_plot):
        condition = conditions_struct[condition_id]

        condition_column_one = condition[column_one]
        condition_column_two = condition[column_two]

        proportion = condition_column_two / condition_column_one
        normalized_proportion = proportion / control_proportion

        y = aggregation_function(normalized_proportion, axis=0)
        y_err = np.nanstd(normalized_proportion, axis=0) / np.sqrt(
            len(normalized_proportion)
        )
        x = aggregation_function(condition_column_one, axis=0)

        label = build_legend(condition, legend)

        plt.plot(x, y, label=label, color=color_palette[i])
        plt.errorbar(x, y, yerr=y_err, fmt="o", capsize=3, color=color_palette[i])

    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)

    set_scale(plt.gca(), log_scale)

    plt.legend()

    fig = plt.gcf()
    plt.show()

    return fig


def compute_deviation_from_model_at_ecdysis(
    conditions_struct,
    column_one,
    column_two,
    control_condition,
    output_column_name,
    remove_hatch=True,
    deviations_as_percentage=True,
    poly_degree=2,
    remove_outliers_fitting=True,
):
    control_condition = conditions_struct[control_condition]
    control_column_one_values = control_condition[column_one]
    control_column_two_values = control_condition[column_two]

    if remove_hatch:
        control_column_one_values = control_column_one_values[:, 1:]
        control_column_two_values = control_column_two_values[:, 1:]

    control_model = _get_proportion_model(
        control_column_one_values,
        control_column_two_values,
        poly_degree=poly_degree,
        plot_model=True,
        remove_outliers=remove_outliers_fitting,
    )

    for condition in conditions_struct:
        column_one_values, column_two_values = (
            condition[column_one],
            condition[column_two],
        )
        if remove_hatch:
            column_one_values = column_one_values[:, 1:]
            column_two_values = column_two_values[:, 1:]

        deviations = get_deviation_from_model(
            column_one_values,
            column_two_values,
            control_model,
            percentage=deviations_as_percentage,
        )
        condition[output_column_name] = deviations

    return conditions_struct


def compute_deviation_from_each_model_at_ecdysis(
    conditions_struct,
    column_one,
    column_two,
    output_column_name,
    remove_hatch=True,
    deviations_as_percentage=True,
    poly_degree=2,
    remove_outliers_fitting=True,
):
    for condition in conditions_struct:
        column_one_values, column_two_values = (
            condition[column_one],
            condition[column_two],
        )
        if remove_hatch:
            column_one_values = column_one_values[:, 1:]
            column_two_values = column_two_values[:, 1:]

        model = _get_proportion_model(
            column_one_values,
            column_two_values,
            poly_degree=poly_degree,
            plot_model=False,
            remove_outliers=remove_outliers_fitting,
        )

        deviations = get_deviation_from_model(
            column_one_values,
            column_two_values,
            model,
            percentage=deviations_as_percentage,
        )
        condition[output_column_name] = deviations

    return conditions_struct


def compute_deviation_from_model_development_percentage(
    conditions_struct,
    column_one,
    column_two,
    control_condition,
    percentages,
    output_column_name,
    deviations_as_percentage=True,
    poly_degree=2,
    remove_outliers_fitting=True,
):
    control_condition = conditions_struct[control_condition]
    control_column_one_values = control_condition[column_one]
    control_column_two_values = control_condition[column_two]

    indices = np.clip(
        (percentages * control_column_one_values.shape[1]).astype(int),
        0,
        control_column_one_values.shape[1] - 1,
    ).astype(int)

    control_column_one_values = control_column_one_values[:, indices]
    control_column_two_values = control_column_two_values[:, indices]

    control_model = _get_proportion_model(
        control_column_one_values,
        control_column_two_values,
        poly_degree=poly_degree,
        plot_model=True,
        remove_outliers=remove_outliers_fitting,
    )

    for condition in conditions_struct:
        column_one_values, column_two_values = (
            condition[column_one],
            condition[column_two],
        )
        column_one_values = column_one_values[:, indices]
        column_two_values = column_two_values[:, indices]

        deviations = get_deviation_from_model(
            column_one_values,
            column_two_values,
            control_model,
            percentage=deviations_as_percentage,
        )
        condition[output_column_name] = deviations

    return conditions_struct
