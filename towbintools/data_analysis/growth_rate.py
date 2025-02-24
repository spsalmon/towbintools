import numpy as np
from scipy.signal import savgol_filter, medfilt
from towbintools.data_analysis.time_series import correct_series_with_classification
from scipy.ndimage import uniform_filter1d
from towbintools.foundation.utils import interpolate_nans_infs

def compute_growth_rate_linear(series, time, ignore_start_fraction=0., ignore_end_fraction=0., savgol_filter_window=5, savgol_filter_order=3):
    """
    Compute the growth rate of a time series using linear regression.

    Parameters:
        series (np.ndarray): The series of values to compute the growth rate of.
        time (np.ndarray): The time data corresponding to the series.
        ignore_start_fraction (float): The fraction of points to ignore at the beginning of the series.
        ignore_end_fraction (float): The fraction of points to ignore at the end of the series.
        savgol_filter_window (int): The window size of the Savitzky-Golay filter.
        savgol_filter_order (int): The order of the Savitzky-Golay filter.

    Returns:
        float: The linear growth rate of the time series.
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

    # Interpolate NaN and inf values
    series = interpolate_nans_infs(series)

    # Smooth the series time series a bit more with a Savitzky-Golay filter
    series = savgol_filter(series, savgol_filter_window, savgol_filter_order)

    # Compute linear regression
    slope, intercept = np.polyfit(time, series, 1)

    return slope

def compute_growth_rate_exponential(series, time, ignore_start_fraction=0., ignore_end_fraction=0., savgol_filter_window=5, savgol_filter_order=3):
    """
    Compute the growth rate of a time series using exponential regression.

    Parameters:
        series (np.ndarray): The series of values to compute the growth rate of.
        time (np.ndarray): The time data corresponding to the series.
        ignore_start_fraction (float): The fraction of points to ignore at the beginning of the series.
        ignore_end_fraction (float): The fraction of points to ignore at the end of the series.
        savgol_filter_window (int): The window size of the Savitzky-Golay filter.
        savgol_filter_order (int): The order of the Savitzky-Golay filter.

    Returns:
        float: The exponential growth rate of the time series.
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

    # Interpolate NaN and inf values
    series = interpolate_nans_infs(series)

    # Smooth the series time series a bit more with a Savitzky-Golay filter
    series = savgol_filter(series, savgol_filter_window, savgol_filter_order)

    # Compute exponential regression
    slope, intercept = np.polyfit(time, np.log(series), 1)

    return slope

def compute_instantaneous_growth_rate(series, time, smoothing_method = "savgol", savgol_filter_window=15, savgol_filter_order=3, moving_average_window=15):
    """
    Compute the instantaneous growth rate of a time series.

    Parameters:
        series (np.ndarray): The series of values to compute the growth rate of.
        time (np.ndarray): The time data corresponding to the series.
        smoothing_method (str): The method to use for smoothing the series. Can be either 'savgol' or 'moving_average'.
        savgol_filter_window (int): The window size of the Savitzky-Golay filter.
        savgol_filter_order (int): The order of the Savitzky-Golay filter.
        moving_average_window (int): The window size of the moving average filter.

    Returns:
        np.ndarray: The instantaneous growth rate of the time series.
    """

    # Assert that the series and time have the same length
    assert len(series) == len(time), "The series and time must have the same length."
    
    # Remove extreme outliers with a small median filter
    series = medfilt(series, 3)

    # Interpolate NaN and inf values
    series = interpolate_nans_infs(series)

    if smoothing_method == "savgol":
        # Smooth the series time series a bit more with a Savitzky-Golay filter
        series = savgol_filter(series, savgol_filter_window, savgol_filter_order)
    elif smoothing_method == "moving_average":
        # Smooth the series time series a bit more with a moving average filter
        series = uniform_filter1d(series, size=moving_average_window)
    elif smoothing_method == "none":
        pass

    # Compute the instantaneous growth rate
    growth_rate = np.gradient(series, time)

    return growth_rate

def compute_growth_rate_classified(series, time, worm_type, method='exponential', ignore_start_fraction=0., ignore_end_fraction=0., savgol_filter_window=5, savgol_filter_order=3):
    """
    Compute the growth rate of a time series after correcting the non-worm points by removing them and interpolating them back.

    Parameters:
        series (np.ndarray): The time series of values.
        time (np.ndarray): The time data corresponding to the series.
        worm_type (np.ndarray): The classification of the points as either 'worm' or 'egg' or 'error'.
        method (str): The method to use for computing the growth rate. Can be either 'exponential' or 'linear'.
        ignore_start_fraction (float): The fraction of points to ignore at the beginning of the series.
        ignore_end_fraction (float): The fraction of points to ignore at the end of the series.
        savgol_filter_window (int): The window size of the Savitzky-Golay filter.
        savgol_filter_order (int): The order of the Savitzky-Golay filter.

    Returns:
        float: The growth rate of the time series.
    """

    # Assert that the series, time, and worm_type have the same length
    assert len(series) == len(time) == len(worm_type), "The series, time, and worm_type must have the same length."

    # Correct the series time series
    series_worms = correct_series_with_classification(series, worm_type)

    if method == 'exponential':
        growth_rate = compute_growth_rate_exponential(series_worms, time, ignore_start_fraction, ignore_end_fraction, savgol_filter_window, savgol_filter_order)
    elif method == 'linear':
        growth_rate = compute_growth_rate_linear(series_worms, time, ignore_start_fraction, ignore_end_fraction, savgol_filter_window, savgol_filter_order)
    
    return growth_rate

def compute_instantaneous_growth_rate_classified(series, time, worm_type, smoothing_method = "savgol", savgol_filter_window=15, savgol_filter_order=3, moving_average_window=15):
    """
    Compute the instantaneous growth rate of a time series after correcting the non-worm points by removing them and interpolating them back.

    Parameters:
        series (np.ndarray): The time series of values.
        time (np.ndarray): The time data corresponding to the series.
        worm_type (np.ndarray): The classification of the points as either 'worm' or 'egg' or 'error'.
        smoothing_method (str): The method to use for smoothing the series. Can be either 'savgol' or 'moving_average'.
        savgol_filter_window (int): The window size of the Savitzky-Golay filter.
        savgol_filter_order (int): The order of the Savitzky-Golay filter.
        moving_average_window (int): The window size of the moving average filter.

    Returns:
        np.ndarray: The instantaneous growth rate of the time series.
    """

    # Assert that the series, time, and worm_type have the same length
    assert len(series) == len(time) == len(worm_type), "The series, time, and worm_type must have the same length."

    # Correct the series time series
    series_worms = correct_series_with_classification(series, worm_type)
    growth_rate = compute_instantaneous_growth_rate(series_worms, time, smoothing_method, savgol_filter_window, savgol_filter_order, moving_average_window)
    
    return growth_rate

def compute_growth_rate_per_larval_stage(series, time, worm_type, ecdysis, method = "exponential", ignore_start_fraction=0., ignore_end_fraction=0., savgol_filter_window=5, savgol_filter_order=3):
    """
    Compute the growth rate of a time series per larval stage.

    Parameters:
        series (np.ndarray): The time series of values.
        time (np.ndarray): The time data corresponding to the series.
        worm_type (np.ndarray): The classification of the points as either 'worm' or 'egg' or 'error'.
        ecdysis (dict): The ecdysis events of the worm.
        method (str): The method to use for computing the growth rate. Can be either 'exponential' or 'linear'.
        ignore_start_fraction (float): The fraction of points to ignore at the beginning of the series.
        ignore_end_fraction (float): The fraction of points to ignore at the end of the series.
        savgol_filter_window (int): The window size of the Savitzky-Golay filter.
        savgol_filter_order (int): The order of the Savitzky-Golay filter.

    Returns:
        dict: The growth rate per larval stage.
    """

    # Assert that the series, time, and worm_type have the same length
    assert len(series) == len(time) == len(worm_type), "The series, time, and worm_type, must have the same length."

    # Correct the series time series
    series_worms = correct_series_with_classification(series, worm_type)

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

def compute_larval_stage_duration(ecdysis):
    """
    Compute the duration of each larval stage.

    Parameters:
        ecdysis (dict): The ecdysis events of the worm.

    Returns:
        dict: The duration of each larval stage.
    """

    # extract ecdisis indices
    hatch_time = ecdysis['HatchTime']
    M1 = ecdysis['M1']
    M2 = ecdysis['M2']
    M3 = ecdysis['M3']
    M4 = ecdysis['M4']

    ls_durations = {}

    for i, (start, end) in enumerate(zip([hatch_time, M1, M2, M3], [M1, M2, M3, M4])):
        # check if start or end is NaN
        if np.isnan(start) or np.isnan(end):
            ls_durations[f"L{i+1}"] = np.nan
        else:
            ls_durations[f"L{i+1}"] = end - start

    return ls_durations