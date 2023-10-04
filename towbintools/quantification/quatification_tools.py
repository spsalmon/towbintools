import numpy as np

def fluorescence_in_mask(image, mask, normalization = 'area'):
    """Quantify fluorescence of an image in a mask.

    Parameters:
        image (np.ndarray): The image as a NumPy array.
        mask (np.ndarray): The binary mask as a NumPy array.
        normalization (str): The normalization to apply to the fluorescence values.
                             Can be one of 'area', 'mean', 'max', or 'none'.

    Returns:
        float: Quantification of the fluorescence of the image in the mask.
    """
    if normalization == 'area':
        return np.sum(image * mask) / np.sum(mask)
    elif normalization == 'mean':
        return np.mean(image * mask)
    elif normalization == 'max':
        return np.max(image * mask)
    elif normalization == 'none':
        return np.sum(image * mask)
    else:
        raise ValueError('Normalization must be one of "area", "mean", "max", or "none".')

    