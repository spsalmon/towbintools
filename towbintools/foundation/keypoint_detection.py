import numpy as np
from scipy.signal import find_peaks


def heatmap_to_keypoints_1D(heatmap, height_threshold=0.5):
    """
    Convert a heatmap to keypoints by finding the peaks.
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
