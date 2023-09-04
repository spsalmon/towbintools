import numpy as np


def normalized_variance_measure(
    image: np.ndarray,
) -> float:
    """
    Compute the normalized variance measure of an image.

    Parameters:
            image (np.ndarray): The input image as a NumPy array.

    Returns:
            float: The computed normalized variance value.
    """
    mean = np.mean(image)
    var = np.var(image)
    return np.divide(var, mean)
