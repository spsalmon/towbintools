import os
import re

import numpy as np
import pandas as pd


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

    # Iterate through each image path
    for image_path in image_paths:
        # Search for time and point independently
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
    filemap: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fill in missing time points in a filemap dataframe with empty image paths.

    Parameters:
        filemap (pd.DataFrame): The filemap dataframe containing 'Time', 'Point', and 'ImagePath' columns.

    Returns:
        pd.DataFrame: The filled filemap dataframe with missing time points included.
    """
    # Get unique points and times from the filemap dataframe.
    all_points = filemap["Point"].unique()
    all_times = filemap["Time"].unique()
    missing_times = []

    # Iterate through each point.
    for point in all_points:
        # Get the unique times associated with the current point.
        times_of_point = filemap.loc[filemap["Point"] == point, "Time"].unique()  # type: ignore
        # Find the missing times by comparing with all times.
        missing = set(all_times) - set(times_of_point)
        # Generate dictionaries with missing times and empty image paths.
        missing_times.extend(
            [{"Time": time, "Point": point, "ImagePath": ""} for time in missing]
        )

    # Create a new dataframe with the missing times and empty image paths.
    filemap_extended = pd.DataFrame(
        missing_times, columns=["Time", "Point", "ImagePath"]
    )
    # Concatenate the original filemap with the extended filemap.
    filled_filemap = pd.concat([filemap, filemap_extended]).sort_values(
        by=["Point", "Time"]
    )

    return filled_filemap


def get_dir_filemap(
    dir_path: str,
) -> pd.DataFrame:
    """
    Get the filemap dataframe for a directory by retrieving all time points and filling in missing time points.

    Parameters:
        dir_path (str): The path to the directory containing the images.

    Returns:
        pd.DataFrame: The filemap dataframe with 'Time', 'Point', and 'ImagePath' columns.
    """
    # Retrieve all time points from the directory.
    timepoint_list = get_all_timepoints_from_dir(dir_path)
    # Create a filemap dataframe from the timepoint list.
    filemap = pd.DataFrame(timepoint_list, columns=["Time", "Point", "ImagePath"])
    # Fill in missing time points in the filemap.
    filled_filemap = fill_empty_timepoints(filemap)

    return filled_filemap


def get_experiment_dir_filemap(
    dir_path: str,
    raw_dir: str = "raw",
    analysis_dir: str = "analysis",
) -> pd.DataFrame:
    """
    Get the filemap dataframe for an experiment directory.

    Retrieves time points from the 'raw' directory and fills in missing ones then adds paths
    from the 'analysis' subdirectories.

    Parameters:
        dir_path (str): Base directory path for the experiment.
        raw_dir (str): Subdirectory name for raw images. (default: "raw")
        analysis_dir (str): Subdirectory name for analysis output. (default: "analysis")

    Returns:
        pd.DataFrame: Extended filemap dataframe including both raw and analysis image paths.
    """
    raw_timepoint_list = get_all_timepoints_from_dir(os.path.join(dir_path, raw_dir))
    raw_filemap = pd.DataFrame(
        raw_timepoint_list, columns=["Time", "Point", "ImagePath"]
    )
    experiment_filemap = fill_empty_timepoints(raw_filemap)
    experiment_filemap.rename(columns={"ImagePath": "raw"}, inplace=True)

    analysis_dir = os.path.join(dir_path, analysis_dir)
    if os.path.exists(analysis_dir):
        subdir_list = [x[0] for x in os.walk(analysis_dir)]
        for subdir in subdir_list:
            if subdir != analysis_dir:
                timepoint_list = get_all_timepoints_from_dir(subdir)
                filemap = pd.DataFrame(
                    timepoint_list, columns=["Time", "Point", "ImagePath"]
                )
                filemap.rename(
                    columns={
                        "ImagePath": os.path.join(
                            analysis_dir, os.path.basename(subdir)
                        )
                    },
                    inplace=True,
                )
                experiment_filemap = experiment_filemap.merge(
                    filemap, on=["Time", "Point"], how="outer"
                )
    experiment_filemap = experiment_filemap.fillna("")
    return experiment_filemap


def add_dir_to_experiment_filemap(
    experiment_filemap: pd.DataFrame,
    dir_path: str,
    subdir_name: str,
) -> pd.DataFrame:
    """
    Add a the images contained in a directory to an existing filemap as a new column.

    Parameters:
        experiment_filemap (pd.DataFrame): Filemap dataframe.
        dir_path (str): The path to the directory containing the images.
        subdir_name (str): The name of the new column to be added.

    Returns:
        pd.DataFrame: Updated filemap dataframe with the new column added.
    """
    subdir_filemap = get_dir_filemap(dir_path)
    subdir_filemap.rename(columns={"ImagePath": subdir_name}, inplace=True)
    # check if column already exists
    if subdir_name in experiment_filemap.columns:
        experiment_filemap.drop(columns=[subdir_name], inplace=True)
    experiment_filemap = experiment_filemap.merge(
        subdir_filemap, on=["Time", "Point"], how="left"
    )
    experiment_filemap = experiment_filemap.replace(np.nan, "", regex=True)
    return experiment_filemap
