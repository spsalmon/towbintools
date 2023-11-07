import numpy as np
from skimage.measure import regionprops, shannon_entropy


def compute_worm_volume(
    straightened_worm_mask: np.ndarray,
    pixelsize: float,
) -> float:
    """
    Compute the volume of a straightened worm mask using its radius and pixel size.

    Parameters:
        straightened_worm_mask (np.ndarray): The straightened worm mask as a NumPy array.
        pixelsize (float): The size of a pixel, used for volume calculation.

    Returns:
        float: The computed volume of the worm.
    """

    worm_radius = np.sum(straightened_worm_mask, axis=0) / 2
    return np.sum(np.pi * (worm_radius**2)) * (pixelsize**3)

def compute_worm_area(
    straightened_worm_mask: np.ndarray,
    pixelsize: float,
) -> float:
    """
    Compute the area of a worm mask.

    Parameters:
        straightened_worm_mask (np.ndarray): The straightened worm mask as a NumPy array.
        pixelsize (float): The size of a pixel, used for area calculation.

    Returns:
        float: The computed are of the worm.
    """

    return np.sum(straightened_worm_mask) * (pixelsize**2)
    

def compute_worm_length(
    straightened_worm_mask: np.ndarray,
    pixelsize: float,
) -> float:
    """
    Compute the length of a straightened worm mask using the sum of its pixels along the 0-axis.

    Parameters:
        straightened_worm_mask (np.ndarray): The straightened worm mask as a NumPy array.
        pixelsize (float): The size of a pixel, used for length calculation.

    Returns:
        float: The computed length of the worm.
    """
    return np.sum(np.sum(straightened_worm_mask, axis=0) > 0) * pixelsize


def compute_worm_type_features(
    straightened_worm_mask: np.ndarray,
    pixelsize: float,
) -> list:
    """
    Compute a series of morphological features for a straightened worm mask including
    length, volume, volume per length, width measures, entropy, and region properties.

    Parameters:
        straightened_worm_mask (np.ndarray): The straightened worm mask as a NumPy array.
        pixelsize (float): The size of a pixel, used for various calculations.

    Returns:
        list: A list containing the computed features in the following order:
              [worm_length, worm_volume, volume_per_length, width_mean, width_std,
              width_cv, entropy_mask, eccentricity, solidity, permimeter]
    """
    # compute worm length
    worm_length = compute_worm_length(straightened_worm_mask, pixelsize=pixelsize)

    # compute worm volume
    worm_volume = compute_worm_volume(straightened_worm_mask, pixelsize=pixelsize)

    # compute worm width
    worm_widths = np.sum(straightened_worm_mask, axis=0) * pixelsize
    worm_widths = worm_widths[worm_widths > 0]

    width_std = np.std(worm_widths)
    width_mean = np.mean(worm_widths)
    width_cv = width_std / width_mean

    volume_per_length = worm_volume / worm_length

    # compute entropy
    entropy_mask = shannon_entropy(straightened_worm_mask)

    try:
        other_properties = regionprops(straightened_worm_mask.astype(np.uint8))[0]
        eccentricity = other_properties.eccentricity
        solidity = other_properties.solidity
        permimeter = other_properties.perimeter
    except IndexError:
        eccentricity = 0
        solidity = 0
        permimeter = 0

    return [
        worm_length,
        worm_volume,
        volume_per_length,
        width_mean,
        width_std,
        width_cv,
        entropy_mask,
        eccentricity,
        solidity,
        permimeter,
    ]
