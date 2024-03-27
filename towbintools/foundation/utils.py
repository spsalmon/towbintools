import numpy as np


def nan_helper(
    y,
):
    """Helper to handle indices and logical indices of NaNs.

    Parameters:
        - y (np.ndarray): 1d numpy array with possible NaNs

    Returns:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices

    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def interpolate_nans(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Interpolate NaN values in a given signal.

    Uses linear interpolation to estimate and replace NaN values in the provided
    signal based on the values of non-NaN neighbors.

    Parameters:
        signal (np.ndarray): The input signal array, which might contain NaN values.

    Returns:
        np.ndarray: The signal array with NaN values interpolated.
    """

    nans, x = nan_helper(signal)
    signal[nans] = np.interp(x(nans), x(~nans), signal[~nans])
    return signal

# Exception class for the case when a method is not implemented
class NotImplementedError(Exception):
    pass