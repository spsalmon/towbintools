import numpy as np


def compute_fluorescence_in_mask(image, mask, aggregation="sum"):
    """Quantify fluorescence of an image in a mask.

    Parameters:
        image (np.ndarray): The image as a NumPy array.
        mask (np.ndarray): The binary mask as a NumPy array.
        aggregation (str): The aggregation method to use to quantify the fluorescence.
                             Can be one of 'sum', 'mean', 'median', 'max', 'min', or 'std'.

    Returns:
        float: Quantification of the fluorescence of the image in the mask.
    """

    if aggregation == "sum":
        return np.sum(image[mask > 0])
    elif aggregation == "mean":
        return np.mean(image[mask > 0])
    elif aggregation == "median":
        return np.median(image[mask > 0])
    elif aggregation == "max":
        return np.max(image[mask > 0])
    elif aggregation == "min":
        return np.min(image[mask > 0])
    elif aggregation == "std":
        return np.std(image[mask > 0])
    else:
        raise ValueError(
            'Aggregation must be one of "sum", "mean", "median", "max", "min", or "std".'
        )


def compute_background_fluorescence(image, foreground_mask, aggregation="mean"):
    """Estimate the background value of an image.

    Parameters:
        image (np.ndarray): The image as a NumPy array.
        foreground_mask (np.ndarray): The binary mask of the foreground as a NumPy array.
        aggregation (str): The aggregation method to use to estimate the background value.
                           Can be either 'mean', 'median', or 'min'.

    Returns:
        float: The estimated background value of the image.
    """

    background_mask = np.logical_not(foreground_mask > 0)

    if aggregation == "mean":
        return np.mean(image[background_mask])
    elif aggregation == "median":
        return np.median(image[background_mask])
    elif aggregation == "min":
        return np.min(image[background_mask])
    else:
        raise ValueError('Aggregation must be one of "mean", "median", or "min".')
