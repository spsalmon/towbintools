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

def inf_helper(
    y,
):
    """Helper to handle indices and logical indices of infinities.

    Parameters:
        - y (np.ndarray): 1d numpy array with possible infinities

    Returns:
        - infs, logical indices of infinities
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of infinities to 'equivalent' indices

    Example:
        >>> # linear interpolation of infinities
        >>> infs, x= inf_helper(y)
        >>> y[infs]= np.interp(x(infs), x(~infs), y[~infs])
    """

    return np.isinf(y), lambda z: z.nonzero()[0]

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

def interpolate_infs(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Interpolate infinity values in a given signal.

    Uses linear interpolation to estimate and replace infinity values in the provided
    signal based on the values of non-infinity neighbors.

    Parameters:
        signal (np.ndarray): The input signal array, which might contain infinity values.

    Returns:
        np.ndarray: The signal array with infinity values interpolated.
    """

    infs, x = inf_helper(signal)
    signal[infs] = np.interp(x(infs), x(~infs), signal[~infs])
    return signal

def interpolate_nans_infs(
    signal: np.ndarray,
) -> np.ndarray:
    """
    Interpolate NaN and infinity values in a given signal.

    Uses linear interpolation to estimate and replace NaN and infinity values in the provided
    signal based on the values of non-NaN and non-infinity neighbors.

    Parameters:
        signal (np.ndarray): The input signal array, which might contain NaN and infinity values.

    Returns:
        np.ndarray: The signal array with NaN and infinity values interpolated.
    """

    signal = interpolate_nans(signal)
    signal = interpolate_infs(signal)
    return signal

# Exception class for the case when a method is not implemented
class NotImplementedError(Exception):
    pass

