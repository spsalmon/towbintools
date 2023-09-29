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

# Exception class for the case when a method is not implemented
class NotImplementedError(Exception):
    pass