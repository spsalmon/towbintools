import bottleneck as bn
import numpy as np

from towbintools.data_analysis import rescale_series
from towbintools.data_analysis.growth_rate import (
    compute_instantaneous_growth_rate_classified,
)
from towbintools.data_analysis.time_series import (
    smooth_series_classified,
)
from towbintools.foundation.utils import find_best_string_match


def combine_series(
    conditions_struct, series_one, series_two, operation, new_series_name
):
    for condition in conditions_struct:
        series_one_values = condition[series_one]
        series_two_values = condition[series_two]

        if operation == "add":
            new_series_values = np.add(series_one_values, series_two_values)
        elif operation == "subtract":
            new_series_values = series_one_values - series_two_values
        elif operation == "multiply":
            new_series_values = series_one_values * series_two_values
        elif operation == "divide":
            new_series_values = np.divide(series_one_values, series_two_values + 1e-8)
        condition[new_series_name] = new_series_values
    return conditions_struct


def transform_series(conditions_struct, series, operation, new_series_name):
    for conditions in conditions_struct:
        series_values = conditions[series]

        if operation == "log":
            new_series_values = np.log(series_values)
        elif operation == "exp":
            new_series_values = np.exp(series_values)
        elif operation == "sqrt":
            new_series_values = np.sqrt(series_values)
        conditions[new_series_name] = new_series_values

    return conditions_struct


def compute_growth_rate(
    conditions_struct,
    series_name,
    gr_series_name,
    experiment_time=True,
    lmbda=0.0075,
    order=2,
    medfilt_window=5,
):
    for condition in conditions_struct:
        series_values = condition[series_name]
        qc_keys = [key for key in condition.keys() if "qc" in key]
        if len(qc_keys) == 1:
            qc_key = qc_keys[0]
        else:
            qc_key = find_best_string_match(series_name, qc_keys)
        qc = condition[qc_key]

        if experiment_time:
            time = condition["experiment_time_hours"]
        else:
            time = condition["time"]

        growth_rate = []
        for i in range(series_values.shape[0]):
            gr = compute_instantaneous_growth_rate_classified(
                series_values[i],
                time[i],
                qc[i],
                lmbda=lmbda,
                order=order,
                medfilt_window=medfilt_window,
            )
            growth_rate.append(gr)

        growth_rate = np.array(growth_rate)

        condition[gr_series_name] = growth_rate

    return conditions_struct


def rescale(
    conditions_struct,
    series_name,
    rescaled_series_name,
    experiment_time=True,
    n_points=100,
):
    for condition in conditions_struct:
        series_values = condition[series_name]
        qc_keys = [key for key in condition.keys() if "qc" in key]
        if len(qc_keys) == 1:
            qc_key = qc_keys[0]
        else:
            qc_key = find_best_string_match(series_name, qc_keys)
        qc = condition[qc_key]
        ecdysis = condition["ecdysis_index"]

        if experiment_time:
            time = condition["experiment_time"]
        else:
            time = condition["time"]

        _, rescaled_series = rescale_series(
            series_values, time, ecdysis, qc, n_points=n_points
        )  # shape (n_worms, 4, n_points)

        # reshape into (n_worms, 4*n_points)

        rescaled_series = rescaled_series.reshape(rescaled_series.shape[0], -1)

        condition[rescaled_series_name] = rescaled_series

    return conditions_struct


def rescale_without_flattening(
    conditions_struct,
    series_name,
    rescaled_series_name,
    experiment_time=True,
    n_points=100,
):
    for condition in conditions_struct:
        series_values = condition[series_name]
        qc_keys = [key for key in condition.keys() if "qc" in key]
        if len(qc_keys) == 1:
            qc_key = qc_keys[0]
        else:
            qc_key = find_best_string_match(series_name, qc_keys)
        qc = condition[qc_key]
        ecdysis = condition["ecdysis_index"]

        if experiment_time:
            time = condition["experiment_time"]
        else:
            time = condition["time"]

        _, rescaled_series = rescale_series(
            series_values, time, ecdysis, qc, n_points=n_points
        )  # shape (n_worms, 4, n_points)

        condition[rescaled_series_name] = rescaled_series

    return conditions_struct


def exclude_arrests_from_series_at_ecdysis(series_at_ecdysis):
    filtered_series = np.full(series_at_ecdysis.shape, np.nan)
    # keep only a value at one ecdys event if the next one is not nan
    if series_at_ecdysis.shape[0] == 1 or len(series_at_ecdysis.shape) == 1:
        for i in range(len(series_at_ecdysis)):
            if i == len(series_at_ecdysis) - 1:
                filtered_series[i] = series_at_ecdysis[i]
            elif not np.isnan(series_at_ecdysis[i + 1]):
                filtered_series[i] = series_at_ecdysis[i]
        return filtered_series
    else:
        for i in range(series_at_ecdysis.shape[0]):
            for j in range(series_at_ecdysis.shape[1]):
                if j == series_at_ecdysis.shape[1] - 1:
                    filtered_series[i, j] = series_at_ecdysis[i, j]
                elif not np.isnan(series_at_ecdysis[i, j + 1]):
                    filtered_series[i, j] = series_at_ecdysis[i, j]
        return filtered_series


def smooth_series(
    conditions_struct,
    series_name,
    smoothed_series_name,
    experiment_time=True,
    lmbda=0.0075,
    order=2,
    medfilt_window=5,
):
    for condition in conditions_struct:
        series_values = condition[series_name]
        qc_keys = [key for key in condition.keys() if "qc" in key]
        if len(qc_keys) == 1:
            qc_key = qc_keys[0]
        else:
            qc_key = find_best_string_match(series_name, qc_keys)
        qc = condition[qc_key]
        if experiment_time:
            time = condition["experiment_time_hours"]
        else:
            time = condition["time"]

        smoothed_series = []
        for i in range(len(series_values)):
            values = series_values[i]
            smoothed = smooth_series_classified(
                values,
                time[i],
                qc[i],
                lmbda=lmbda,
                order=order,
                medfilt_window=medfilt_window,
            )

            # pad with 0 until it's the same length as the original series
            if len(smoothed) < len(values):
                smoothed = np.pad(
                    smoothed,
                    (0, len(values) - len(smoothed)),
                    "constant",
                    constant_values=(np.nan, np.nan),
                )

            smoothed_series.append(smoothed)

        smoothed_series = np.array(smoothed_series)
        condition[smoothed_series_name] = smoothed_series

    return conditions_struct


def smooth_and_rescale_series(
    conditions_struct,
    series_name,
    smoothed_series_name,
    experiment_time=True,
    lmbda=0.0075,
    order=2,
    medfilt_window=5,
    n_points=100,
    flatten=True,
):
    for condition in conditions_struct:
        series_values = condition[series_name]
        qc_keys = [key for key in condition.keys() if "qc" in key]
        if len(qc_keys) == 1:
            qc_key = qc_keys[0]
        else:
            qc_key = find_best_string_match(series_name, qc_keys)
        qc = condition[qc_key]
        ecdysis = condition["ecdysis_index"]
        if experiment_time:
            time = condition["experiment_time_hours"]
        else:
            time = condition["time"]

        smoothed_series = []
        for i in range(len(series_values)):
            values = series_values[i]
            smoothed = smooth_series_classified(
                values,
                time[i],
                qc[i],
                lmbda=lmbda,
                order=order,
                medfilt_window=medfilt_window,
            )

            # pad with 0 until it's the same length as the original series
            if len(smoothed) < len(values):
                smoothed = np.pad(
                    smoothed,
                    (0, len(values) - len(smoothed)),
                    "constant",
                    constant_values=(np.nan, np.nan),
                )

            smoothed_series.append(smoothed)

        smoothed_series = np.array(smoothed_series)

        # we don't need the classification anymore
        qc = np.full_like(smoothed_series, "worm", dtype=object)

        # rescale the smoothed series
        _, rescaled_series = rescale_series(
            smoothed_series, time, ecdysis, qc, n_points=n_points
        )  # shape (n_worms, 4, n_points)

        # reshape into (n_worms, 4*n_points)
        if flatten:
            rescaled_series = rescaled_series.reshape(rescaled_series.shape[0], -1)

        condition[smoothed_series_name] = smoothed_series
        condition[f"{smoothed_series_name}_rescaled"] = rescaled_series
    return conditions_struct


def detrend_rescaled_series_population_mean(conditions_struct, rescaled_series):
    for condition in conditions_struct:
        series = condition[rescaled_series]
        detrended_series = _detrend_rescaled_series_population_mean(series)
        new_name = f"{rescaled_series}_detrended"
        condition[new_name] = detrended_series

    return conditions_struct


def _detrend_rescaled_series_population_mean(series):
    # detrend the series by subtracting the population mean
    population_mean = bn.nanmean(series, axis=0)
    detrended_series = series - population_mean

    return detrended_series
