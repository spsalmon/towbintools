import numpy as np
from towbintools.foundation.utils import interpolate_nans

def correct_series_with_classification(series, worm_type):
    """
    Remove the points of non-worms from the time series and interpolate them back.

    Parameters:
        series (np.ndarray): The time series of values.
        worm_type (np.ndarray): The classification of the points as either 'worm' or 'egg' or 'error'.

    Returns:
        np.ndarray: The corrected time series.
    """

    # Set the series of non worms to NaN
    non_worms_indices = worm_type != 'worm'
    series_worms = series.copy()
    series_worms[non_worms_indices] = np.nan

    try:
        # Interpolate the NaNs
        series_worms = interpolate_nans(series_worms)
    except ValueError:
        print("Error in interpolation, returning original series.")
        return series
    
    return series_worms

def filter_series_with_classification(series, worm_type):
    """Remove the points of non-worms from the time series.
    
    Parameters:
        series (np.ndarray): The time series of values.
        worm_type (np.ndarray): The classification of the points as either 'worm' or 'egg' or 'error'.
        
    Returns:
        np.ndarray: The filtered time series."""
    
    # Set the series of non worms to NaN
    non_worms_indices = worm_type != 'worm'
    series_worms = series.copy()
    series_worms[non_worms_indices] = np.nan
    
    return series_worms