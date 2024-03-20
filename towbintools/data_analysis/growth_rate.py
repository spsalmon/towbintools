import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

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
    
    volume = savgol_filter(volume, savgol_filter_window, savgol_filter_order)

    # Compute exponential regression
    slope, intercept = np.polyfit(time, np.log(volume), 1)

    return slope

