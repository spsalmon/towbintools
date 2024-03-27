import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, medfilt
from towbintools.foundation.utils import interpolate_nans
from scipy.ndimage import uniform_filter1d

def compute_growth_rate_linear(series, time, ignore_start_fraction=0., ignore_end_fraction=0., savgol_filter_window=5, savgol_filter_order=3):
    """
    Compute the growth rate of a time series using linear regression.
    """

    # Assert that the series and time have the same length
    assert len(series) == len(time), "The series and time must have the same length."

    # Compute the number of points to ignore at the beginning and end
    num_points = len(series)
    num_ignore_start = int(ignore_start_fraction * num_points)
    num_ignore_end = int(ignore_end_fraction * num_points)

    # Assert that the fraction of points to ignore is not too large
    assert num_ignore_start + num_ignore_end < num_points, "The fraction of points to ignore is too large."

    # Correctly handle slicing when ignore fractions are 0
    if num_ignore_end == 0:
        time = time[num_ignore_start:]
        series = series[num_ignore_start:]
    else:
        time = time[num_ignore_start:-num_ignore_end]
        series = series[num_ignore_start:-num_ignore_end]
    
    # Remove extreme outliers with a small median filter
    series = medfilt(series, 3)

    # Smooth the series time series a bit more with a Savitzky-Golay filter
    series = savgol_filter(series, savgol_filter_window, savgol_filter_order)

    # Compute linear regression
    slope, intercept = np.polyfit(time, series, 1)

    return slope

def compute_growth_rate_exponential(series, time, ignore_start_fraction=0., ignore_end_fraction=0., savgol_filter_window=5, savgol_filter_order=3):
    """
    Compute the growth rate of a time series using exponential regression.
    """

    # Assert that the series and time have the same length
    assert len(series) == len(time), "The series and time must have the same length."

    # Compute the number of points to ignore at the beginning and end
    num_points = len(series)
    num_ignore_start = int(ignore_start_fraction * num_points)
    num_ignore_end = int(ignore_end_fraction * num_points)

    # Assert that the fraction of points to ignore is not too large
    assert num_ignore_start + num_ignore_end < num_points, "The fraction of points to ignore is too large."

    # Correctly handle slicing when ignore fractions are 0
    if num_ignore_end == 0:
        time = time[num_ignore_start:]
        series = series[num_ignore_start:]
    else:
        time = time[num_ignore_start:-num_ignore_end]
        series = series[num_ignore_start:-num_ignore_end]
    
    # Remove extreme outliers with a small median filter
    series = medfilt(series, 3)

    # Smooth the series time series a bit more with a Savitzky-Golay filter
    series = savgol_filter(series, savgol_filter_window, savgol_filter_order)

    # Compute exponential regression
    slope, intercept = np.polyfit(time, np.log(series), 1)

    return slope

def compute_instantaneous_growth_rate(series, time, smoothing_method = "savgol", savgol_filter_window=15, savgol_filter_order=3, moving_average_window=15):
    """
    Compute the instantaneous growth rate of a time series.
    """

    # Assert that the series and time have the same length
    assert len(series) == len(time), "The series and time must have the same length."
    
    # Remove extreme outliers with a small median filter
    series = medfilt(series, 3)

    if smoothing_method == "savgol":
        # Smooth the series time series a bit more with a Savitzky-Golay filter
        series = savgol_filter(series, savgol_filter_window, savgol_filter_order)
    elif smoothing_method == "moving_average":
        # Smooth the series time series a bit more with a moving average filter
        series = uniform_filter1d(series, size=moving_average_window)

    # Compute the instantaneous growth rate
    growth_rate = np.gradient(series, time)

    return growth_rate

def compute_growth_rate_classified(series, time, worm_type, method='exponential', ignore_start_fraction=0., ignore_end_fraction=0., savgol_filter_window=5, savgol_filter_order=3):
    """
    Compute the growth rate of a time series, using only points correctly classified as worms.
    """

    # Assert that the series, time, and worm_type have the same length
    assert len(series) == len(time) == len(worm_type), "The series, time, and worm_type must have the same length."

    # Correct the series time series
    series_worms = correct_series_time_series(series, worm_type)

    if method == 'exponential':
        growth_rate = compute_growth_rate_exponential(series_worms, time, ignore_start_fraction, ignore_end_fraction, savgol_filter_window, savgol_filter_order)
    elif method == 'linear':
        growth_rate = compute_growth_rate_linear(series_worms, time, ignore_start_fraction, ignore_end_fraction, savgol_filter_window, savgol_filter_order)
    
    return growth_rate

def compute_instantaneous_growth_rate_classified(series, time, worm_type, smoothing_method = "savgol", savgol_filter_window=15, savgol_filter_order=3, moving_average_window=15):
    """
    Compute the instantaneous growth rate of a time series, using only points correctly classified as worms.
    """

    # Assert that the series, time, and worm_type have the same length
    assert len(series) == len(time) == len(worm_type), "The series, time, and worm_type must have the same length."

    # Correct the series time series
    series_worms = correct_series_time_series(series, worm_type)
    growth_rate = compute_instantaneous_growth_rate(series_worms, time, smoothing_method, savgol_filter_window, savgol_filter_order, moving_average_window)
    
    return growth_rate

def correct_series_time_series(series, worm_type):
    """
    Remove the series of non-worms from the time series and interpolate them back.
    """

    # Set the series of non worms to NaN
    non_worms_indices = worm_type != 'worm'
    series_worms = series.copy()
    series_worms[non_worms_indices] = np.nan

    # Interpolate the NaNs
    series_worms = interpolate_nans(series_worms)
    
    return series_worms

def compute_growth_rate_per_larval_stage(series, time, worm_type, ecdysis, method = "exponential", ignore_start_fraction=0., ignore_end_fraction=0., savgol_filter_window=5, savgol_filter_order=3):
    """
    Compute the growth rate of a time series per larval stage.
    """

    # Assert that the series, time, and worm_type have the same length
    assert len(series) == len(time) == len(worm_type), "The series, time, and worm_type, must have the same length."

    # Correct the series time series
    series_worms = correct_series_time_series(series, worm_type)

    # extract ecdisis indices
    hatch_time = ecdysis['HatchTime']
    M1 = ecdysis['M1']
    M2 = ecdysis['M2']
    M3 = ecdysis['M3']
    M4 = ecdysis['M4']

    growth_rates = {}

    # Compute the growth rate per larval stage
    for i, (start, end) in enumerate(zip([hatch_time, M1, M2, M3], [M1, M2, M3, M4])):
        # check if start or end is NaN
        if np.isnan(start) or np.isnan(end):
            growth_rates[f"L{i+1}"] = np.nan
            
        else:
            series_worms_stage = series_worms[start:end]
            time_stage = time[start:end]
            worm_type_stage = worm_type[start:end]

            growth_rate_stage = compute_growth_rate_classified(series_worms_stage, time_stage, worm_type_stage, method, ignore_start_fraction, ignore_end_fraction, savgol_filter_window, savgol_filter_order)
            growth_rates[f"L{i+1}"] = growth_rate_stage
    
    return growth_rates