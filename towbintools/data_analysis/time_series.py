import numpy as np
from scipy import interpolate
from scipy.interpolate import interp1d
from scipy.signal import medfilt
from whittaker_eilers import WhittakerSmoother

from towbintools.foundation.utils import interpolate_nans


def resize_series_to_length(series, new_length, kind="linear"):
    """
    Resize a 1D or 2D series to a new length using interpolation.

    Parameters:
        series (np.ndarray): The time series to resize.
        new_length (int): The desired length of the resized series.
        kind (str, optional): The type of interpolation to use. (default: "linear")

    Returns:
        np.ndarray: The resized series of the specified length.

    Raises:
        ValueError: If the input series is not 1D or 2D.
    """
    series = np.asarray(series)
    length = series.shape[-1]
    if length == new_length:
        return series
    x_old = np.linspace(0, 1, length)
    x_new = np.linspace(0, 1, new_length)
    if series.ndim == 1:
        f = interp1d(x_old, series, kind=kind, fill_value="extrapolate")
        return f(x_new)
    elif series.ndim == 2:
        f = interp1d(x_old, series, axis=1, kind=kind, fill_value="extrapolate")
        return f(x_new)
    else:
        raise ValueError("Input series must be 1D or 2D array.")


def pad_series_to_length(series, new_length, pad_value=0):
    """
    Pad a 1D or 2D series to a new length with a specified pad value.

    Parameters:
        series (np.ndarray): The time series to pad.
        new_length (int): The desired length of the padded series.
        pad_value (int, optional): The value to use for padding. Default is 0

    Returns:
        np.ndarray: The padded series of the specified length.

    Raises:
        ValueError: If the input series is not 1D or 2D.
    """
    series = np.asarray(series)
    length = series.shape[-1]
    if length == new_length:
        return series
    if series.ndim == 1:
        return np.pad(
            series, (0, new_length - length), mode="constant", constant_values=pad_value
        )
    elif series.ndim == 2:
        return np.pad(
            series,
            ((0, 0), (0, new_length - length)),
            mode="constant",
            constant_values=pad_value,
        )
    else:
        raise ValueError("Input series must be 1D or 2D array.")


def crop_series_to_length(series, new_length):
    """
    Crop a 1D or 2D series to a new length.

    Parameters:
        series (np.ndarray): The time series to crop.
        new_length (int): The desired length of the cropped series.

    Returns:
        np.ndarray: The cropped series of the specified length.

    Raises:
        ValueError: If the input series is not 1D or 2D.
    """
    series = np.asarray(series)
    length = series.shape[-1]
    if length == new_length:
        return series
    if series.ndim == 1:
        return series[0:new_length]
    elif series.ndim == 2:
        return series[:, 0:new_length]
    else:
        raise ValueError("Input series must be 1D or 2D array.")


def random_crop_series_to_length(series, new_length):
    """
    Randomly crop a 1D or 2D series to a new length.

    Parameters:
        series (np.ndarray): The time series to crop.
        new_length (int): The desired length of the cropped series.

    Returns:
        np.ndarray: The cropped series of the specified length.

    Raises:
        ValueError: If the input series is not 1D or 2D.
    """
    series = np.asarray(series)
    length = series.shape[-1]

    if length == new_length:
        return series
    if series.ndim == 1:
        start = np.random.randint(0, length - new_length + 1)
        return series[start : start + new_length]
    elif series.ndim == 2:
        start = np.random.randint(0, length - new_length + 1)
        return series[:, start : start + new_length]
    else:
        raise ValueError("Input series must be 1D or 2D array.")


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
    non_worms_indices = worm_type != "worm"
    series_worms = series.copy()
    series_worms[non_worms_indices] = np.nan

    try:
        # Interpolate the NaNs
        series_worms = interpolate_nans(series_worms)
    except ValueError:
        return np.full(series.shape, np.nan)

    return series_worms


def filter_series_with_classification(series, worm_type):
    """Remove the points of non-worms from the time series.

    Parameters:
        series (np.ndarray): The time series of values.
        worm_type (np.ndarray): The classification of the points as either 'worm' or 'egg' or 'error'.

    Returns:
        np.ndarray: The filtered time series."""

    # Set the series of non worms to NaN
    non_worms_indices = worm_type != "worm"
    series_worms = series.copy()
    series_worms[non_worms_indices] = np.nan

    return series_worms


def interpolate_larval_stage(series, time, ecdysis, larval_stage, n_points=100):
    if larval_stage < 1 or larval_stage > 4:
        raise ValueError("The larval stage must be between 1 and 4.")

    previous_ecdys = ecdysis[larval_stage - 1]
    ecdys = ecdysis[larval_stage]

    # check that the molts are correct
    if (np.isnan(previous_ecdys) or np.isnan(ecdys)) or previous_ecdys > ecdys:
        return np.full(n_points, np.nan), np.full(n_points, np.nan)

    # convert ecdysis times to int to get the index
    previous_ecdys = int(previous_ecdys)
    ecdys = int(ecdys)

    interpolated_time = np.linspace(time[previous_ecdys], time[ecdys], n_points)
    interpolated_series = interpolate.interp1d(time, series, kind="linear")(
        interpolated_time
    )

    return interpolated_time, interpolated_series


def interpolate_entire_development(series, time, ecdysis, n_points=100):
    interpolated_time = np.full((4, n_points), np.nan)
    interpolated_series = np.full((4, n_points), np.nan)
    for larval_stage in range(1, 5):
        (
            interpolated_time_stage,
            interpolated_series_stage,
        ) = interpolate_larval_stage(time, series, ecdysis, larval_stage, n_points)

        interpolated_time[larval_stage - 1, :] = interpolated_time_stage
        interpolated_series[larval_stage - 1, :] = interpolated_series_stage

    return interpolated_time, interpolated_series


def interpolate_entire_development_classified(
    series, time, ecdysis, worm_type, n_points=100
):
    time = filter_series_with_classification(time, worm_type)
    series = filter_series_with_classification(series, worm_type)

    return interpolate_entire_development(time, series, ecdysis, n_points)


def compute_exponential_series_at_time_classified(
    series: np.ndarray,
    time: np.ndarray,
    worm_types: np.ndarray,
    fit_width: int = 10,
) -> float:
    """
    Compute the value of a time series at a given time(s) using linear regression on a logarithmic transformation of the data.

    This function uses linear regression on a logarithmic transformation of the volume data to predict
    the volume at the specified hatch time and end-molts. Only data points where `worm_types` is "worm"
    are used for fitting. The function returns the volume at desired time.

    Parameters:
        volume (np.ndarray): A time series representing volume.
        time (np.ndarray): The time(s) at which the volume is to be computed.
        worm_types (np.ndarray): An array indicating the type of each entry in the volume time series. Expected values are "worm", "egg", etc.
        fit_width (int, optional): Width for the linear regression fit used in computing the volume. (default: 10)

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


def smooth_series_classified(
    series: np.ndarray,
    series_time,
    worm_types: np.ndarray,
    lmbda=0.0075,
    order=2,
    medfilt_window=5,
) -> np.ndarray:
    """
    Compute the series at the given time points using the worm types to classify the points. The series is first corrected for incorrect segmentation, then median filtered to remove outliers, then smoothed using Whittaker-Eilers smoothing.

    Parameters:
        series (np.ndarray): The time series.
        series_time (np.ndarray): The time points of the original series. If None, the time points are assumed to be the indices of the series.
        worm_types (np.ndarray): The classification of the points as either 'worm' or 'egg' or 'error'.
        lmbda (float, optional): The smoothing parameter for the Whittaker-Eilers smoothing. Default provides good results for our volume curves when series_time is in hours. (default: 0.0075)
        order (int, optional): The order of the penalty of the Whittaker-Eilers smoother. (default: 2)
        medfilt_window (int, optional): The window size for the median filter. (default: 5)
    Returns:
        np.ndarray: The series at the given time(s).
    """

    # Check if the series has any non nan values
    if np.all(np.isnan(series)):
        return np.full(series.shape, np.nan)

    # Interpolate the nans
    series = interpolate_nans(series)

    # Remove the points of non-worms from the time series and interpolate them back
    series = correct_series_with_classification(series, worm_types)

    smoothed_series = _smooth_series(
        series,
        series_time,
        lmbda=lmbda,
        order=order,
        medfilt_window=medfilt_window,
    )
    return smoothed_series


def smooth_series(
    series: np.ndarray,
    series_time,
    lmbda=0.0075,
    order=2,
    medfilt_window=5,
) -> np.ndarray:
    """
    Compute the series at the given time points using the worm types to classify the points. The series is first corrected for incorrect segmentation, then median filtered to remove outliers, then smoothed using Whittaker-Eilers smoothing.

    Parameters:
        series (np.ndarray): The time series.
        series_time (np.ndarray): The time points of the original series. If None, the time points are assumed to be the indices of the series.
        lmbda (float, optional): The smoothing parameter for the Whittaker-Eilers smoothing. Default provides good results for our volume curves when series_time is in hours. (default: 0.0075)
        order (int, optional): The order of the penalty of the Whittaker-Eilers smoother. (default: 2)
        medfilt_window (int, optional): The window size for the median filter. (default: 5)
    Returns:
        np.ndarray: The series at the given time(s).
    """

    # Check if the series has any non nan values
    if np.all(np.isnan(series)):
        return np.full(series.shape, np.nan)

    # Interpolate the nans
    series = interpolate_nans(series)

    smoothed_series = _smooth_series(
        series,
        series_time,
        lmbda=lmbda,
        order=order,
        medfilt_window=medfilt_window,
    )
    return smoothed_series


def _smooth_series(
    series: np.ndarray,
    series_time: np.ndarray,
    lmbda: float = 0.0075,
    order: int = 2,
    medfilt_window: int = 5,
) -> np.ndarray:
    # Median filter to remove outliers
    series = medfilt(series, medfilt_window)

    if series_time is not None:
        nan_series_time = np.isnan(series_time)
        non_nan_series = series[~nan_series_time]
        non_nan_series_time = series_time[~nan_series_time]
        whittaker_smoother = WhittakerSmoother(
            lmbda=lmbda,
            order=order,
            data_length=len(non_nan_series),
            x_input=non_nan_series_time,
        )

        smoothed_series = whittaker_smoother.smooth(non_nan_series)
    else:
        whittaker_smoother = WhittakerSmoother(
            lmbda=lmbda, order=order, data_length=len(series)
        )

        smoothed_series = whittaker_smoother.smooth(series)

    smoothed_series = np.array(smoothed_series)

    # pad the smoothed series to the original shape
    smoothed_series = np.pad(
        smoothed_series,
        (0, series.shape[0] - smoothed_series.shape[0]),
        mode="constant",
        constant_values=np.nan,
    )

    # Interpolate the nans again
    smoothed_series = interpolate_nans(smoothed_series)

    return smoothed_series


def compute_series_at_time_classified(
    series: np.ndarray,
    time: np.ndarray,
    series_time: np.ndarray,
    worm_types: np.ndarray,
    lmbda=0.0075,
    medfilt_window=5,
    bspline_order=3,
) -> np.ndarray:
    """
    Compute the series at the given time points using the worm types to classify the points. The series is first corrected for incorrect segmentation, then median filtered to remove outliers, then smoothed using a Savitzky-Golay filter, and finally interpolated using b-splines.

    Parameters:
        series (np.ndarray): The time series.
        time (np.ndarray): The time points at which the series is to be computed.
        series_time (np.ndarray): The time points of the original series. If None, the time points are assumed to be the indices of the series.
        worm_types (np.ndarray): The classification of the points as either 'worm' or 'egg' or 'error'.
        lmbda (float, optional): The smoothing parameter for the Whittaker-Eilers smoothing. Default provides good results for our volume curves when series_time is in hours. (default: 0.0075)
        medfilt_window (int, optional): The window size for the median filter. (default: 5)
        bspline_order (int, optional): The order of the b-spline interpolation. (default: 3)

    Returns:
        np.ndarray: The series at the given time(s).
    """

    if np.all(np.isnan(series)):
        # handle time being a single value
        if np.isscalar(time):
            return np.full(1, np.nan)
        else:
            return np.full(time.shape, np.nan)

    # Smooth the series
    series = smooth_series_classified(
        series,
        series_time,
        worm_types,
        medfilt_window=medfilt_window,
        lmbda=lmbda,
    )

    # Interpolate the series using b-splines
    if series_time is None:
        series_time = np.arange(len(series))

    try:
        interpolated_series = interpolate.make_interp_spline(
            series_time, series, k=bspline_order
        )
    except Exception as e:
        print(f"Caught an exception while interpolating series: {e}")
        return np.full(time.shape, np.nan)

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
        series_point, time_point, ecdysis_point, worm_type_point = (
            series[point],
            time[point],
            ecdysis[point],
            worm_type[point],
        )
        (
            interpolated_time,
            interpolated_series,
        ) = interpolate_entire_development_classified(
            series_point,
            time_point,
            ecdysis_point,
            worm_type_point,
            n_points=n_points,
        )

        all_points_interpolated_time.append(interpolated_time)
        all_points_interpolated_series.append(interpolated_series)

    all_points_interpolated_time = np.array(all_points_interpolated_time)
    all_points_interpolated_series = np.array(all_points_interpolated_series)

    return all_points_interpolated_time, all_points_interpolated_series


def aggregate_interpolated_series(
    all_points_interpolated_series,
    all_points_interpolated_time,
    larval_stage_durations,
    aggregation="mean",
):
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
    if aggregation == "mean":
        aggregation_function = np.nanmean
    elif aggregation == "median":
        aggregation_function = np.nanmedian

    n_points = all_points_interpolated_time.shape[-1]

    aggregated_series = np.full((4, n_points), np.nan)
    std_series = np.full((4, n_points), np.nan)
    ste_series = np.full((4, n_points), np.nan)
    rescaled_time = np.full((4, n_points), np.nan)

    aggregated_larval_stage_durations = aggregation_function(
        larval_stage_durations, axis=0
    )

    for i in range(4):
        aggregated_series[i, :] = aggregation_function(
            all_points_interpolated_series[:, i, :], axis=0
        )
        std_series[i, :] = np.nanstd(all_points_interpolated_series[:, i, :], axis=0)
        ste_series[i, :] = np.nanstd(
            all_points_interpolated_series[:, i, :], axis=0
        ) / np.sqrt(np.sum(np.isfinite(all_points_interpolated_series[:, i, :])))

        beginning = (
            np.nansum(aggregated_larval_stage_durations[: i + 1])
            - aggregated_larval_stage_durations[i]
        )
        end = np.nansum(aggregated_larval_stage_durations[: i + 1])
        rescaled_time[i, :] = np.linspace(beginning, end, n_points)

    # flatten the arrays
    aggregated_series = aggregated_series.flatten()
    std_series = std_series.flatten()
    ste_series = ste_series.flatten()
    rescaled_time = rescaled_time.flatten()

    return rescaled_time, aggregated_series, std_series, ste_series


def rescale_and_aggregate(
    series,
    time,
    ecdysis,
    larval_stage_durations,
    worm_type,
    points=None,
    aggregation="mean",
    n_points=100,
):
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

    (
        all_points_interpolated_time,
        all_points_interpolated_series,
    ) = rescale_series(
        series, time, ecdysis, worm_type, points=points, n_points=n_points
    )
    (
        rescaled_time,
        aggregated_series,
        std_series,
        ste_series,
    ) = aggregate_interpolated_series(
        all_points_interpolated_series,
        all_points_interpolated_time,
        larval_stage_durations,
        aggregation=aggregation,
    )

    return rescaled_time, aggregated_series, std_series, ste_series
