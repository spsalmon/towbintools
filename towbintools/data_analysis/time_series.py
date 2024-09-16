import numpy as np
from towbintools.foundation.utils import interpolate_nans
from scipy import interpolate
from scipy.signal import savgol_filter, medfilt

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

def interpolate_larval_stage(time, series, ecdysis, larval_stage, n_points = 100):
    if larval_stage < 1 or larval_stage > 4:
        raise ValueError("The larval stage must be between 1 and 4.")
    
    previous_ecdys = ecdysis[larval_stage-1]
    ecdys = ecdysis[larval_stage]

    # check that the molts are correct
    if (np.isnan(previous_ecdys) or np.isnan(ecdys)) or previous_ecdys > ecdys:
        return np.full(n_points, np.nan), np.full(n_points, np.nan)
    
    # convert ecdysis times to int to get the index
    previous_ecdys = int(previous_ecdys)
    ecdys = int(ecdys)

    interpolated_time = np.linspace(time[previous_ecdys], time[ecdys], n_points)
    interpolated_series = interpolate.interp1d(time, series, kind='linear')(interpolated_time)

    return interpolated_time, interpolated_series

def interpolate_entire_development(time, series, ecdysis, n_points = 100):
    interpolated_time = np.full((4, n_points), np.nan)
    interpolated_series = np.full((4, n_points), np.nan)
    for larval_stage in range(1, 5):
        interpolated_time_stage, interpolated_series_stage = interpolate_larval_stage(time, series, ecdysis, larval_stage, n_points)
        
        interpolated_time[larval_stage - 1, :] = interpolated_time_stage
        interpolated_series[larval_stage - 1, :] = interpolated_series_stage

    return interpolated_time, interpolated_series

def interpolate_entire_development_classified(time, series, ecdysis, worm_type, n_points = 100):
    time = filter_series_with_classification(time, worm_type)
    series = filter_series_with_classification(series, worm_type)
    
    return interpolate_entire_development(time, series, ecdysis, n_points)

def compute_exponential_series_at_time_classified(
    series: np.ndarray,
    worm_types: np.ndarray,
    time: np.ndarray,
    fit_width: int = 10,
) -> float:
    """
    Compute the value of a time series at a given time(s) using linear regression on a logarithmic transformation of the data.

    This function uses linear regression on a logarithmic transformation of the volume data to predict
    the volume at the specified hatch time and end-molts. Only data points where `worm_types` is "worm"
    are used for fitting. The function returns the volume at desired time.

    Parameters:
        volume (np.ndarray): A time series representing volume.
        worm_types (np.ndarray): An array indicating the type of each entry in the volume time series. Expected values are "worm", "egg", etc.
        time (np.ndarray): The time(s) at which the volume is to be computed.
        fit_width (int, optional): Width for the linear regression fit used in computing the volume. Default is 10.

    Returns:
        np.ndarray: Volume at desired time(s).
    """

    def compute_single_time(time: float) -> float:
        if np.isfinite(time):
            fit_x = np.arange(
                max(0, int(time - fit_width)),
                min(len(series), int(time + fit_width)),
                dtype=int,
            )

            filtered_fit_x = fit_x[np.where(worm_types[fit_x] == "worm")]
            if filtered_fit_x.size != 0:
                fit_y = np.log(series[filtered_fit_x])
                try:
                    p = np.polyfit(filtered_fit_x, fit_y, 1)
                except Exception as e:
                    print(
                        f"Caught an exception while interpolating volume at time {time}, returning nan : {e}"
                    )
                    return np.nan
                series_at_time = np.exp(np.polyval(p, time))
            else:
                series_at_time = np.nan
        else:
            series_at_time = np.nan

        return float(series_at_time)

    result = np.array([compute_single_time(t) for t in time])

    return result

def compute_series_at_time_classified(series: np.ndarray, worm_types: np.ndarray, time: np.ndarray, series_time = None, medfilt_window = 7, savgol_window = 7, savgol_order = 3, bspline_order=3) -> np.ndarray:
    """
    Compute the series at the given time points using the worm types to classify the points. The series is first corrected for incorrect segmentation, then median filtered to remove outliers, then smoothed using a Savitzky-Golay filter, and finally interpolated using b-splines.

    Parameters:
        series (np.ndarray): The time series.
        worm_types (np.ndarray): The classification of the points as either 'worm' or 'egg' or 'error'.
        time (np.ndarray): The time points at which the series is to be computed.
        series_time (np.ndarray, optional): The time points of the original series. Default is None. If None, the time points are assumed to be the indices of the series.
        medfilt_window (int, optional): The window size for the median filter. Default is 7.
        savgol_window (int, optional): The window size for the Savitzky-Golay filter. Default is 7.
        savgol_order (int, optional): The order of the Savitzky-Golay filter. Default is 3.
        bspline_order (int, optional): The order of the b-spline interpolation. Default is 3.

    Returns:
        np.ndarray: The series at the given time(s).
    """

    # Check if the series has any non nan values
    if np.all(np.isnan(series)):
        return np.full(time.shape, np.nan)

    # Interpolate the nans
    series = interpolate_nans(series)
    
    # Remove the points of non-worms from the time series and interpolate them back
    series = correct_series_with_classification(series, worm_types)

    # Median filter to remove outliers
    series = medfilt(series, medfilt_window)

    # Savitzky-Golay filter to smooth the data
    series = savgol_filter(series, savgol_window, savgol_order)

    # Interpolate the series using b-splines
    if series_time is None:
        series_time = np.arange(len(series))

    interpolated_series = interpolate.make_interp_spline(series_time, series, k=bspline_order)

    return interpolated_series(time)

def rescale_series(series, time, ecdysis, worm_type, points=None, n_points=100):
    """
    Interpolates one or multiple series and the time to have n_points points in total per larval stages.

    Parameters:
        series (np.ndarray) : Array of shape (nb_of_worms / points, length of series) containing the series to interpolate.
        time (np.ndarray) : Array of shape (nb_of_worms / points, length of series) containing the time information.
        ecdysis (np.ndarray) : Array of shape (nb_of_worms / points, 5) containing the ecdysis times.
        worm_type (np.ndarray) : Array of shape (nb_of_worms / points, length of series) containing the worm type classification
        points (list) : List of points to consider for the interpolation. If None, all points are considered.
        n_points (int) : Number of points to interpolate the series to.

    Returns:
        all_points_interpolated_time (np.ndarray) : Array of shape (nb_of_worms / points, 4, n_points) containing the interpolated time information.
        all_points_interpolated_series (np.ndarray) : Array of shape (nb_of_worms / points, 4, n_points) containing the interpolated series.
    """
    if points is not None:
        series = series[points]
        time = time[points]
        worm_type = worm_type[points]
        ecdysis = ecdysis[points]
    
    # Interpolate the time and the series

    all_points_interpolated_time = []
    all_points_interpolated_series = []

    for point in range(series.shape[0]):
        series_point, time_point, ecdysis_point, worm_type_point = series[point], time[point], ecdysis[point], worm_type[point]
        interpolated_time, interpolated_series = interpolate_entire_development_classified(time_point, series_point, ecdysis_point, worm_type_point, n_points = n_points)

        all_points_interpolated_time.append(interpolated_time)
        all_points_interpolated_series.append(interpolated_series)

    all_points_interpolated_time = np.array(all_points_interpolated_time)
    all_points_interpolated_series = np.array(all_points_interpolated_series)

    return all_points_interpolated_time, all_points_interpolated_series

def aggregate_interpolated_series(all_points_interpolated_series, all_points_interpolated_time, larval_stage_durations, aggregation='mean'):
    """
    Aggregates the interpolated series and time information.

    Parameters:
        all_points_interpolated_series (np.ndarray) : Array of shape (nb_of_worms / points, 4, n_points) containing the interpolated series.
        all_points_interpolated_time (np.ndarray) : Array of shape (nb_of_worms / points, 4, n_points) containing the interpolated time information.
        larval_stage_durations (np.ndarray) : Array of shape (nb_of_worms / points, 4) containing the larval stage durations.
        aggregation (str) : Aggregation method to use. Can be 'mean' or 'median'.

    Returns:
        rescaled_time (np.ndarray) : Array of shape (4*n_points) containing the rescaled time information.
        aggregated_series (np.ndarray) : Array of shape (4*n_points) containing the aggregated series.
        std_series (np.ndarray) : Array of shape (4*n_points) containing the standard deviation of the series.
        ste_series (np.ndarray) : Array of shape (4*n_points) containing the standard error of the series.
    """    
    if aggregation == 'mean':
        aggregation_function = np.nanmean
    elif aggregation == 'median':
        aggregation_function = np.nanmedian
    
    n_points = all_points_interpolated_time.shape[-1]

    aggregated_series = np.full((4, n_points), np.nan)
    std_series = np.full((4, n_points), np.nan)
    ste_series = np.full((4, n_points), np.nan)
    rescaled_time = np.full((4, n_points), np.nan)

    aggregated_larval_stage_durations = aggregation_function(larval_stage_durations, axis=0)

    for i in range(4):
        aggregated_series[i, :] = aggregation_function(all_points_interpolated_series[:, i, :], axis=0)
        std_series[i, :] = np.nanstd(all_points_interpolated_series[:, i, :], axis=0)
        ste_series[i, :] = np.nanstd(all_points_interpolated_series[:, i, :], axis=0) / np.sqrt(np.sum(np.isfinite(all_points_interpolated_series[:, i, :])))

        beginning = np.nansum(aggregated_larval_stage_durations[:i+1]) - aggregated_larval_stage_durations[i]
        end = np.nansum(aggregated_larval_stage_durations[:i+1])
        rescaled_time[i, :] = np.linspace(beginning, end, n_points)

    # flatten the arrays
    aggregated_series = aggregated_series.flatten()
    std_series = std_series.flatten()
    ste_series = ste_series.flatten()
    rescaled_time = rescaled_time.flatten()

    return rescaled_time, aggregated_series, std_series, ste_series



def rescale_and_aggregate(series, time, ecdysis, larval_stage_durations, worm_type, points=None, aggregation='mean', n_points=100):
    """
    Rescales and aggregates the series and time information.

    Parameters:
        series (np.ndarray) : Array of shape (nb_of_worms, length of series) containing the series to interpolate.
        time (np.ndarray) : Array of shape (nb_of_worms, length of series) containing the time information.
        ecdysis (np.ndarray) : Array of shape (nb_of_worms, 5) containing the ecdysis times.
        worm_type (np.ndarray) : Array of shape (nb_of_worms, length of series) containing the worm type classification
        points (list) : List of points to consider for the interpolation. If None, all points are considered.
        aggregation (str) : Aggregation method to use. Can be 'mean' or 'median'.
        n_points (int) : Number of points to interpolate the series to.

    Returns:
        rescaled_time (np.ndarray) : Array of shape (4*n_points) containing the rescaled time information.
        aggregated_series (np.ndarray) : Array of shape (4*n_points) containing the aggregated series.
        std_series (np.ndarray) : Array of shape (4*n_points) containing the standard deviation of the series.
        ste_series (np.ndarray) : Array of shape (4*n_points) containing the standard error of the series.
    """

    all_points_interpolated_time, all_points_interpolated_series = rescale_series(series, time, ecdysis, worm_type, points=points, n_points=n_points)
    rescaled_time, aggregated_series, std_series, ste_series = aggregate_interpolated_series(all_points_interpolated_series, all_points_interpolated_time, larval_stage_durations, aggregation=aggregation)

    return rescaled_time, aggregated_series, std_series, ste_series