import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, medfilt
from towbintools.foundation.utils import interpolate_nans
from scipy.ndimage import uniform_filter1d

def compute_growth_rate_linear(volume, time, ignore_start_fraction=0., ignore_end_fraction=0., savgol_filter_window=5, savgol_filter_order=3):
    """
    Compute the growth rate of a volume time series using linear regression.
    """

    # Assert that the volume and time have the same length
    assert len(volume) == len(time), "The volume and time must have the same length."

    # Compute the number of points to ignore at the beginning and end
    num_points = len(volume)
    num_ignore_start = int(ignore_start_fraction * num_points)
    num_ignore_end = int(ignore_end_fraction * num_points)

    # Assert that the fraction of points to ignore is not too large
    assert num_ignore_start + num_ignore_end < num_points, "The fraction of points to ignore is too large."

    # Correctly handle slicing when ignore fractions are 0
    if num_ignore_end == 0:
        time = time[num_ignore_start:]
        volume = volume[num_ignore_start:]
    else:
        time = time[num_ignore_start:-num_ignore_end]
        volume = volume[num_ignore_start:-num_ignore_end]
    
    # Remove extreme outliers with a small median filter
    volume = medfilt(volume, 3)

    # Smooth the volume time series a bit more with a Savitzky-Golay filter
    volume = savgol_filter(volume, savgol_filter_window, savgol_filter_order)

    # Compute linear regression
    slope, intercept = np.polyfit(time, volume, 1)

    return slope

def compute_growth_rate_exponential(volume, time, ignore_start_fraction=0., ignore_end_fraction=0., savgol_filter_window=5, savgol_filter_order=3):
    """
    Compute the growth rate of a volume time series using exponential regression.
    """

    # Assert that the volume and time have the same length
    assert len(volume) == len(time), "The volume and time must have the same length."

    # Compute the number of points to ignore at the beginning and end
    num_points = len(volume)
    num_ignore_start = int(ignore_start_fraction * num_points)
    num_ignore_end = int(ignore_end_fraction * num_points)

    # Assert that the fraction of points to ignore is not too large
    assert num_ignore_start + num_ignore_end < num_points, "The fraction of points to ignore is too large."

    # Correctly handle slicing when ignore fractions are 0
    if num_ignore_end == 0:
        time = time[num_ignore_start:]
        volume = volume[num_ignore_start:]
    else:
        time = time[num_ignore_start:-num_ignore_end]
        volume = volume[num_ignore_start:-num_ignore_end]
    
    # Remove extreme outliers with a small median filter
    volume = medfilt(volume, 3)

    # Smooth the volume time series a bit more with a Savitzky-Golay filter
    volume = savgol_filter(volume, savgol_filter_window, savgol_filter_order)

    # Compute exponential regression
    slope, intercept = np.polyfit(time, np.log(volume), 1)

    return slope

def compute_instantaneous_growth_rate(volume, time, smoothing_method = "savgol", savgol_filter_window=15, savgol_filter_order=3, moving_average_window=15):
    """
    Compute the instantaneous growth rate of a volume time series.
    """

    # Assert that the volume and time have the same length
    assert len(volume) == len(time), "The volume and time must have the same length."
    
    # Remove extreme outliers with a small median filter
    volume = medfilt(volume, 3)

    if smoothing_method == "savgol":
        # Smooth the volume time series a bit more with a Savitzky-Golay filter
        volume = savgol_filter(volume, savgol_filter_window, savgol_filter_order)
    elif smoothing_method == "moving_average":
        # Smooth the volume time series a bit more with a moving average filter
        volume = uniform_filter1d(volume, size=moving_average_window)

    # Compute the instantaneous growth rate
    growth_rate = np.gradient(volume, time)

    return growth_rate

def compute_growth_rate_classified(volume, time, worm_type, method='exponential', ignore_start_fraction=0., ignore_end_fraction=0., savgol_filter_window=5, savgol_filter_order=3):
    """
    Compute the growth rate of a volume time series, using only points correctly classified as worms.
    """

    # Assert that the volume, time, and worm_type have the same length
    assert len(volume) == len(time) == len(worm_type), "The volume, time, and worm_type must have the same length."

    # Correct the volume time series
    volume_worms = correct_volume_time_series(volume, worm_type)

    if method == 'exponential':
        growth_rate = compute_growth_rate_exponential(volume_worms, time, ignore_start_fraction, ignore_end_fraction, savgol_filter_window, savgol_filter_order)
    elif method == 'linear':
        growth_rate = compute_growth_rate_linear(volume_worms, time, ignore_start_fraction, ignore_end_fraction, savgol_filter_window, savgol_filter_order)
    
    return growth_rate

def compute_instantaneous_growth_rate_classified(volume, time, worm_type, smoothing_method = "savgol", savgol_filter_window=15, savgol_filter_order=3, moving_average_window=15):
    """
    Compute the instantaneous growth rate of a volume time series, using only points correctly classified as worms.
    """

    # Assert that the volume, time, and worm_type have the same length
    assert len(volume) == len(time) == len(worm_type), "The volume, time, and worm_type must have the same length."

    # Correct the volume time series
    volume_worms = correct_volume_time_series(volume, worm_type)
    growth_rate = compute_instantaneous_growth_rate(volume_worms, time, smoothing_method, savgol_filter_window, savgol_filter_order, moving_average_window)
    
    return growth_rate

def correct_volume_time_series(volume, worm_type):
    """
    Remove the volume of non-worms from the volume time series and interpolate them back.
    """

    # Set the volume of non worms to NaN
    non_worms_indices = worm_type != 'worm'
    volume_worms = volume.copy()
    volume_worms[non_worms_indices] = np.nan

    # Interpolate the NaNs
    volume_worms = interpolate_nans(volume_worms)
    
    return volume_worms

def compute_growth_rate_per_larval_stage(volume, time, worm_type, ecdysis, method = "exponential", ignore_start_fraction=0., ignore_end_fraction=0., savgol_filter_window=5, savgol_filter_order=3):
    """
    Compute the growth rate of a volume time series per larval stage.
    """

    # Assert that the volume, time, and worm_type have the same length
    assert len(volume) == len(time) == len(worm_type), "The volume, time, and worm_type, must have the same length."

    # Correct the volume time series
    volume_worms = correct_volume_time_series(volume, worm_type)

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
            volume_worms_stage = volume_worms[start:end]
            time_stage = time[start:end]
            worm_type_stage = worm_type[start:end]

            growth_rate_stage = compute_growth_rate_classified(volume_worms_stage, time_stage, worm_type_stage, method, ignore_start_fraction, ignore_end_fraction, savgol_filter_window, savgol_filter_order)
            growth_rates[f"L{i+1}"] = growth_rate_stage
    
    return growth_rates