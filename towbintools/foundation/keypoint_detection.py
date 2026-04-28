import numpy as np
from scipy.signal import find_peaks


def heatmap_to_keypoints_1D(
    heatmap: np.ndarray, height_threshold: float = 0.5
) -> np.ndarray:
    """
    Convert a 1D heatmap to keypoints by finding the highest peak per class.

    For each class (row of the heatmap), finds all peaks above ``height_threshold``
    and retains the single highest peak. If no peak is found for a class, NaN is
    returned for that class.

    Parameters:
        heatmap (np.ndarray): 2D array of shape ``(n_classes, length)`` where each
            row is a 1D signal for one class.
        height_threshold (float, optional): Minimum peak height to consider.
            (default: 0.5)

    Returns:
        np.ndarray: 1D array of shape ``(n_classes,)`` containing the index of the
            highest peak for each class, or NaN if no peak was found.
    """
    peaks = []
    for i in range(heatmap.shape[0]):
        try:
            peaks_i, peaks_i_dict = find_peaks(heatmap[i], height=height_threshold)
            # keep only the highest peak
            best_peak = np.argmax(peaks_i_dict["peak_heights"])
            peaks.append(peaks_i[best_peak])
        except ValueError:
            peaks.append(np.nan)
    peaks = np.array(peaks)
    return peaks
