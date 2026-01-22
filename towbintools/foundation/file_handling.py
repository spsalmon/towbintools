import os
import re

import numpy as np
import polars as pl


def get_all_timepoints_from_dir(
    dir_path: str,
) -> list[dict]:
    """
    Retrieve all time points and corresponding image paths from a directory.

    Parameters:
        dir_path (str): The path to the directory containing the images.

    Returns:
        list: A list of dictionaries, each containing the time, point, and image path.
    """

    time_pattern = re.compile(r"Time(\d+)")
    point_pattern = re.compile(r"Point(\d+)")

    timepoint_list = []

    # Get a list of file paths in the directory (excluding subdirectories)
    image_paths = [
        os.path.join(dir_path, x)
        for x in os.listdir(dir_path)
        if not os.path.isdir(os.path.join(dir_path, x))
    ]

    for image_path in image_paths:
        time_match = time_pattern.search(image_path)
        point_match = point_pattern.search(image_path)

        # Only add to list if both time and point are found
        if time_match and point_match:
            time = int(time_match.group(1))
            point = int(point_match.group(1))

            timepoint_list.append(
                {"Time": time, "Point": point, "ImagePath": image_path}
            )

    return timepoint_list


def fill_empty_timepoints(
    filemap: pl.DataFrame,
) -> pl.DataFrame:
    """
    Fill in missing time points in a filemap dataframe with empty image paths.

    Parameters:
        filemap (pl.DataFrame): The filemap dataframe containing 'Time', 'Point', and 'ImagePath' columns.

    Returns:
        pl.DataFrame: The filled filemap dataframe with missing time points included.
    """
    all_points = (
        filemap.select(pl.col("Point")).unique(maintain_order=True).to_numpy().squeeze()
    )
    if all_points.ndim == 0:
        all_points = np.array([all_points])

    all_times = (
        filemap.select(pl.col("Time")).unique(maintain_order=True).to_numpy().squeeze()
    )
    missing_times = []

    for point in all_points:
        # Get the unique times associated with the current point.
        times_of_point = (
            filemap.filter(pl.col("Point") == point)
            .select(pl.col("Time"))
            .to_numpy()
            .squeeze()
        )

        missing = set(all_times) - set(times_of_point)
        missing_times.extend(
            [{"Time": time, "Point": point, "ImagePath": ""} for time in missing]
        )

    if missing_times:
        filemap_extended = pl.DataFrame(missing_times)
        filled_filemap = pl.concat([filemap, filemap_extended]).sort(["Point", "Time"])
    else:
        filled_filemap = filemap.sort(["Point", "Time"])

    return filled_filemap


def get_dir_filemap(
    dir_path: str,
) -> pl.DataFrame:
    """
    Get the filemap dataframe for a directory by retrieving all time points and filling in missing time points.

    Parameters:
        dir_path (str): The path to the directory containing the images.

    Returns:
        pl.DataFrame: The filemap dataframe with 'Time', 'Point', and 'ImagePath' columns.
    """
    timepoint_list = get_all_timepoints_from_dir(dir_path)
    filemap = pl.DataFrame(timepoint_list)
    filled_filemap = fill_empty_timepoints(filemap)

    return filled_filemap


def get_experiment_dir_filemap(
    dir_path: str,
    raw_dir: str = "raw",
    analysis_dir: str = "analysis",
) -> pl.DataFrame:
    """
    Get the filemap dataframe for an experiment directory.

    Retrieves time points from the 'raw' directory and fills in missing ones then adds paths
    from the 'analysis' subdirectories.

    Parameters:
        dir_path (str): Base directory path for the experiment.
        raw_dir (str): Subdirectory name for raw images. (default: "raw")
        analysis_dir (str): Subdirectory name for analysis output. (default: "analysis")

    Returns:
        pl.DataFrame: Extended filemap dataframe including both raw and analysis image paths.
    """
    raw_timepoint_list = get_all_timepoints_from_dir(os.path.join(dir_path, raw_dir))
    raw_filemap = pl.DataFrame(raw_timepoint_list)
    experiment_filemap = fill_empty_timepoints(raw_filemap)
    experiment_filemap.rename({"ImagePath": raw_dir})

    analysis_dir = os.path.join(dir_path, analysis_dir)
    if os.path.exists(analysis_dir):
        subdir_list = [x[0] for x in os.walk(analysis_dir)]
        for subdir in subdir_list:
            if subdir != analysis_dir:
                timepoint_list = get_all_timepoints_from_dir(subdir)
                filemap = pl.DataFrame(timepoint_list)
                filemap = fill_empty_timepoints(filemap)
                filemap = filemap.rename(
                    {"ImagePath": os.path.join(analysis_dir, os.path.basename(subdir))},
                )
                experiment_filemap = experiment_filemap.join(
                    filemap, on=["Time", "Point"], how="left"
                )
    experiment_filemap = experiment_filemap.fillna("")
    return experiment_filemap


def add_dir_to_experiment_filemap(
    experiment_filemap: pl.DataFrame,
    dir_path: str,
    subdir_name: str,
) -> pl.DataFrame:
    """
    Add a the images contained in a directory to an existing filemap as a new column.

    Parameters:
        experiment_filemap (pl.DataFrame): Filemap dataframe.
        dir_path (str): The path to the directory containing the images.
        subdir_name (str): The name of the new column to be added.

    Returns:
        pd.DataFrame: Updated filemap dataframe with the new column added.
    """
    subdir_filemap = get_dir_filemap(dir_path)
    subdir_filemap = subdir_filemap.rename({"ImagePath": subdir_name})
    # check if column already exists
    if subdir_name in experiment_filemap.columns:
        experiment_filemap = experiment_filemap.drop(subdir_name)
    experiment_filemap = experiment_filemap.join(
        subdir_filemap, on=["Time", "Point"], how="left"
    )
    experiment_filemap = experiment_filemap.fill_nan("").fill_null("")
    return experiment_filemap


def read_filemap(filemap_path: str, lazy_loading: bool = False) -> pl.DataFrame:
    """Read a filemap from a CSV or Parquet file using Polars."""
    if filemap_path.endswith(".parquet"):
        if lazy_loading:
            filemap = pl.scan_parquet(filemap_path)
        else:
            filemap = pl.read_parquet(filemap_path)
    else:
        if lazy_loading:
            filemap = pl.scan_csv(
                filemap_path,
                infer_schema_length=10000,
                null_values=["np.nan", "[nan]", "", "NaN", "nan", "NA", "N/A"],
            )
        else:
            filemap = pl.read_csv(
                filemap_path,
                infer_schema_length=10000,
                null_values=["np.nan", "[nan]", "", "NaN", "nan", "NA", "N/A"],
            )
    return filemap


def write_filemap(filemap: pl.DataFrame, filemap_path: str) -> None:
    """Write a filemap to a CSV or Parquet file using Polars."""
    if filemap_path.endswith(".parquet"):
        filemap.write_parquet(filemap_path)
    else:
        filemap.write_csv(filemap_path)
