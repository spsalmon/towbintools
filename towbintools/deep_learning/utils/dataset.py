import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from towbintools.foundation import image_handling
import numpy as np
import os
import pandas as pd

from towbintools.deep_learning.utils.augmentation import (
    get_training_augmentation,
    get_prediction_augmentation,
)
from pytorch_toolbelt import inference
from torch.utils.data import DataLoader
from towbintools.deep_learning.utils.util import get_closest_lower_multiple, get_closest_upper_multiple
from joblib import Parallel, delayed
from typing import List

class TiledSegmentationDataset(Dataset):
    def __init__(
        self,
        dataset,
        image_slicers,
        channels,
        mask_column="mask",
        image_column="image",
        transform=None,
    ):
        self.images = dataset[image_column].values.tolist()
        self.ground_truth = dataset[mask_column].values.tolist()
        self.channels = channels
        self.image_slicers = image_slicers
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = image_handling.read_tiff_file(self.images[i], [self.channels])
        mask = image_handling.read_tiff_file(self.ground_truth[i])

        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        slicer = self.image_slicers[img.shape]
        tiles = slicer.split(img)

        if len(tiles[0].shape) == 2:
            tiles = [tile[np.newaxis, ...] for tile in tiles]

        tiles_ground_truth = slicer.split(mask)

        selected_tile = np.random.randint(0, len(tiles))
        img = tiles[selected_tile]
        mask = tiles_ground_truth[selected_tile]
        mask = mask[np.newaxis, ...]

        return img.astype(np.float32), mask

class SegmentationDataset(Dataset):
    def __init__(
        self,
        dataset,
        channels,
        mask_column="mask",
        image_column="image",
        transform=None,
        enforce_divisibility_by = 32,
        pad_or_crop = "pad",
        mask_pad_value = -1,
    ):
        self.images = dataset[image_column].values.tolist()
        self.ground_truth = dataset[mask_column].values.tolist()
        self.channels = channels
        self.transform = transform
        self.enforce_divisibility_by = enforce_divisibility_by
        if pad_or_crop not in ["pad", "crop"]:
            raise ValueError("pad_or_crop must be either 'pad' or 'crop'")

        if pad_or_crop == "pad":
            self.resize_function = image_handling.pad_to_dim_equally
            self.mask_resize_function = lambda dim, new_dim_x, new_dim_y: image_handling.pad_to_dim_equally(dim, new_dim_x, new_dim_y, pad_value = mask_pad_value)
            self.multiplier_function = get_closest_upper_multiple
        else:
            self.resize_function = image_handling.crop_to_dim_equally
            self.mask_resize_function = image_handling.crop_to_dim_equally
            self.multiplier_function = get_closest_lower_multiple

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = image_handling.read_tiff_file(self.images[i], [self.channels])
        mask = image_handling.read_tiff_file(self.ground_truth[i])

        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        if self.enforce_divisibility_by is not None:
            dim_x, dim_y = img.shape[-2:]

            if dim_x % self.enforce_divisibility_by != 0 or dim_y % self.enforce_divisibility_by != 0:
                new_dim_x = self.multiplier_function(dim_x, self.enforce_divisibility_by)
                new_dim_y = self.multiplier_function(dim_y, self.enforce_divisibility_by)

                img = self.resize_function(img, new_dim_x, new_dim_y)
                mask = self.mask_resize_function(mask, new_dim_x, new_dim_y)
                
        if len(img.shape) == 2:
            img = img[np.newaxis, ...]
        mask = mask[np.newaxis, ...]

        return img.astype(np.float32), mask

class SegmentationPredictionDataset(Dataset):
    def __init__(
        self,
        image_paths,
        channels,
        transform=None,
        enforce_divisibility_by = 32,
        pad_or_crop = "pad",
    ):
        self.images = image_paths
        self.channels = channels
        self.transform = transform
        self.enforce_divisibility_by = enforce_divisibility_by
        if pad_or_crop not in ["pad", "crop"]:
            raise ValueError("pad_or_crop must be either 'pad' or 'crop'")

        self.pad_or_crop = pad_or_crop

        if pad_or_crop == "pad":
            self.resize_function = image_handling.pad_to_dim_equally
            self.multiplier_function = get_closest_upper_multiple
        else:
            self.resize_function = image_handling.crop_to_dim_equally
            self.multiplier_function = get_closest_lower_multiple

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img_path = self.images[i]
        img = image_handling.read_tiff_file(img_path, [self.channels])

        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed["image"]

        if len(img.shape) == 2:
            img = img[np.newaxis, ...]

        return img_path, img.astype(np.float32), img.shape

    def collate_fn(self, batch):

        img_paths, imgs, original_shapes = zip(*batch)

        if self.enforce_divisibility_by is None:
            return img_paths, imgs, original_shapes

        # find the maximum dimensions in the batch
        if self.pad_or_crop == "pad":
            dim_x = max([shape[-2] for shape in original_shapes])
            dim_y = max([shape[-1] for shape in original_shapes])
        else:
            dim_x = min([shape[-2] for shape in original_shapes])
            dim_y = min([shape[-1] for shape in original_shapes])

        # find the new dimensions
        new_dim_x = self.multiplier_function(dim_x, self.enforce_divisibility_by)
        new_dim_y = self.multiplier_function(dim_y, self.enforce_divisibility_by)

        # resize the images
        resized_images = []
        for img in imgs:
            resized_images.append(self.resize_function(img, new_dim_x, new_dim_y))
        resized_images = np.array(resized_images, dtype=np.float32)

        return img_paths, resized_images, original_shapes

class ClassificationDataset(Dataset):
    def __init__(
        self,
        dataset,
        channels,
        n_classes,
        class_column="class",
        image_column="image",
        transform=None,
    ):
        self.images = dataset[image_column].values.tolist()
        self.ground_truth = dataset[class_column].values.astype(float).tolist()
        self.channels = channels
        self.transform = transform

        # convert ground truth to one-hot encoding
        if n_classes > 2:
            self.ground_truth = np.eye(n_classes)[self.ground_truth]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, i):
        img = image_handling.read_tiff_file(self.images[i], [self.channels])
        class_value = self.ground_truth[i]

        if self.transform is not None:
            transformed = self.transform(image=img)
            img = transformed["image"]

        if len(img.shape) == 2:
            img = img[np.newaxis, ...]

        return img.astype(np.float32), class_value

def split_dataset(dataframe, validation_size, test_size):
    # Load the dataset
    if isinstance(dataframe, str):
        dataframe = pd.read_csv(dataframe)

    # Ensure the sizes are valid
    if validation_size + test_size >= 1.0:
        raise ValueError(
            "The sum of validation_size and test_size should be less than 1."
        )

    # Calculate the size of the temporary set (validation + test)
    temp_size = validation_size + test_size

    # Split the dataframe into training and temporary set
    train_dataframe, temp_dataframe = train_test_split(
        dataframe, test_size=temp_size, random_state=42
    )

    # Calculate the proportion of validation and test set relative to the temporary set
    validation_proportion = validation_size / temp_size

    # Split the temporary set into validation and test sets
    validation_dataframe, test_dataframe = train_test_split(
        temp_dataframe, test_size=1 - validation_proportion, random_state=42
    )

    return train_dataframe, validation_dataframe, test_dataframe


def create_segmentation_training_dataframes(
    image_directories,
    mask_directories,
    save_dir,
    validation_set_ratio=0.25,
    test_set_ratio=0.1,
):
    if not isinstance(image_directories, list):
        image_directories = [image_directories]
    if not isinstance(mask_directories, list):
        mask_directories = [mask_directories]
        
    images = []
    masks = []
    for image_directory, mask_directory in zip(image_directories, mask_directories):
        images.extend(
            sorted([
                os.path.join(image_directory, file)
                for file in os.listdir(image_directory)
            ])
        )
        masks.extend(
            sorted([os.path.join(mask_directory, file) for file in os.listdir(mask_directory)])
        )

    assert len(images) == len(masks), "The number of images and masks in the directories must be equal"
    dataframe = pd.DataFrame({"image": images, "mask": masks})

    training_dataframe, validation_dataframe, test_dataframe = split_dataset(
        dataframe, validation_set_ratio, test_set_ratio
    )

    # backup the training and validation dataframes
    database_backup_dir = os.path.join(save_dir, "database_backup")
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    os.makedirs(database_backup_dir, exist_ok=True)

    training_dataframe.to_csv(
        os.path.join(database_backup_dir, f"training_dataframe_{current_date}.csv"),
        index=False,
    )
    validation_dataframe.to_csv(
        os.path.join(database_backup_dir, f"validation_dataframe_{current_date}.csv"),
        index=False,
    )
    test_dataframe.to_csv(
        os.path.join(database_backup_dir, f"test_dataframe_{current_date}.csv"),
        index=False,
    )

    return training_dataframe, validation_dataframe

def get_unique_shapes_from_tiffs(image_paths = List[str]) -> np.ndarray:
    """
    Get unique shapes from a list of TIFF images in parallel.
    
    Parameters:
        image_paths (List[str]): List of image paths to extract shapes from
        
    Returns:
        np.ndarray: Unique image shapes found in the dataframe
    """
    
    shapes = Parallel(n_jobs=-1)(
        delayed(image_handling.get_shape_from_tiff)(image_path) 
        for image_path in image_paths
    )
    
    valid_shapes = [shape for shape in shapes if shape is not None]
    
    if len(valid_shapes) == 0:
        raise ValueError("No valid shapes found in the dataframe")
    
    return np.unique(valid_shapes, axis=0)
    
def create_segmentation_dataloaders(
    training_dataframe,
    validation_dataframe,
    channels,
    batch_size=5,
    num_workers=32,
    pin_memory=True,
    train_on_tiles=True,
    tiler_params=None,
    training_transform=None,
    validation_transform=None,
):
    if not train_on_tiles:
        train_loader = DataLoader(
            SegmentationDataset(
                training_dataframe,
                channels=channels,
                mask_column="mask",
                image_column="image",
                transform=training_transform,
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            SegmentationDataset(
                validation_dataframe,
                channels=channels,
                mask_column="mask",
                image_column="image",
                transform=validation_transform,
            ),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        return train_loader, val_loader

    assert (
        tiler_params is not None
    ), "If train_on_tiles is True, tiler_params must be provided"

    image_paths = training_dataframe["image"].values.tolist()

    unique_shapes = get_unique_shapes_from_tiffs(image_paths)
    image_slicers = {tuple(shape): inference.ImageSlicer(shape, tiler_params["tile_size"], tiler_params["tile_step"]) for shape in unique_shapes}

    if training_transform is None:
        training_transform = get_training_augmentation("percentile", lo=1, hi=99)
    if validation_transform is None:
        validation_transform = get_prediction_augmentation("percentile", lo=1, hi=99)

    train_loader = DataLoader(
        TiledSegmentationDataset(
            training_dataframe,
            image_slicers,
            channels=channels,
            mask_column="mask",
            image_column="image",
            transform=training_transform,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        TiledSegmentationDataset(
            validation_dataframe,
            image_slicers,
            channels=channels,
            mask_column="mask",
            image_column="image",
            transform=validation_transform,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def create_segmentation_training_dataframes_and_dataloaders(
    image_directories,
    mask_directories,
    save_dir,
    channels,
    validation_set_ratio=0.25,
    test_set_ratio=0.1,
    batch_size=5,
    num_workers=32,
    pin_memory=True,
    train_on_tiles=True,
    tiler_params=None,
    training_transform=None,
    validation_transform=None,
):
    training_dataframe, validation_dataframe = create_segmentation_training_dataframes(
        image_directories,
        mask_directories,
        save_dir,
        validation_set_ratio=validation_set_ratio,
        test_set_ratio=test_set_ratio,
    )
    train_loader, val_loader = create_segmentation_dataloaders(
        training_dataframe,
        validation_dataframe,
        channels,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        train_on_tiles=train_on_tiles,
        tiler_params=tiler_params,
        training_transform=training_transform,
        validation_transform=validation_transform,
    )
    return training_dataframe, validation_dataframe, train_loader, val_loader


def create_segmentation_dataloaders_from_filemap(
    filemap_path,
    save_dir,
    channels,
    image_column="image",
    mask_column="mask",
    validation_set_ratio=0.25,
    test_set_ratio=0.1,
    batch_size=5,
    num_workers=32,
    pin_memory=True,
    train_on_tiles=True,
    tiler_params=None,
    training_transform=None,
    validation_transform=None,
):
    dataframe = pd.read_csv(filemap_path)
    # rename image column to "image" and mask column to "mask"
    dataframe = dataframe.rename(columns={image_column: "image", mask_column: "mask"})

    training_dataframe, validation_dataframe, test_dataframe = split_dataset(
        filemap_path, validation_set_ratio, test_set_ratio
    )

    # backup the training and validation dataframes
    database_backup_dir = os.path.join(save_dir, "database_backup")
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    os.makedirs(database_backup_dir, exist_ok=True)

    training_dataframe.to_csv(
        os.path.join(database_backup_dir, f"training_dataframe_{current_date}.csv"),
        index=False,
    )
    validation_dataframe.to_csv(
        os.path.join(database_backup_dir, f"validation_dataframe_{current_date}.csv"),
        index=False,
    )
    test_dataframe.to_csv(
        os.path.join(database_backup_dir, f"test_dataframe_{current_date}.csv"),
        index=False,
    )

    train_loader, val_loader = create_segmentation_dataloaders(
        training_dataframe,
        validation_dataframe,
        channels,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        train_on_tiles=train_on_tiles,
        tiler_params=tiler_params,
        training_transform=training_transform,
        validation_transform=validation_transform,
    )
    return training_dataframe, validation_dataframe, train_loader, val_loader
    
def create_classification_training_dataframes(
    ground_truth_csv_paths,
    image_columns,
    class_columns,
    save_dir,
    validation_set_ratio=0.25,
    test_set_ratio=0.1,
):
    if not isinstance(ground_truth_csv_paths, list):
        ground_truth_csv_paths = [ground_truth_csv_paths]
    if isinstance(image_columns, str):
        image_columns = [image_columns]
    if isinstance(class_columns, str):
        class_columns = [class_columns]

    if len(image_columns) == 1:
        image_columns = image_columns * len(ground_truth_csv_paths)
    if len(class_columns) == 1:
        image_columns = image_columns * len(class_columns)

    ground_truth_df = pd.DataFrame()
    for i, ground_truth_csv in enumerate(ground_truth_csv_paths):
        gt_df = pd.read_csv(ground_truth_csv)
        images = gt_df[image_columns[i]].values.tolist()
        classes = gt_df[class_columns[i]].values.tolist()
        new_gt_df = pd.DataFrame({"image": images, "class": classes})
        ground_truth_df = pd.concat([ground_truth_df, new_gt_df], ignore_index=True)

    training_dataframe, validation_dataframe, test_dataframe = split_dataset(
        ground_truth_df, validation_set_ratio, test_set_ratio
    )

    # backup the training and validation dataframes
    database_backup_dir = os.path.join(save_dir, "database_backup")
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    os.makedirs(database_backup_dir, exist_ok=True)

    training_dataframe.to_csv(
        os.path.join(database_backup_dir, f"training_dataframe_{current_date}.csv"),
        index=False,
    )
    validation_dataframe.to_csv(
        os.path.join(database_backup_dir, f"validation_dataframe_{current_date}.csv"),
        index=False,
    )
    test_dataframe.to_csv(
        os.path.join(database_backup_dir, f"test_dataframe_{current_date}.csv"),
        index=False,
    )

    return training_dataframe, validation_dataframe

def create_classification_dataloaders(
    training_dataframe,
    validation_dataframe,
    channels,
    n_classes,
    batch_size=64,
    num_workers=32,
    pin_memory=True,
    training_transform=None,
    validation_transform=None,
):
        
    train_loader = DataLoader(
        ClassificationDataset(
            training_dataframe,
            channels=channels,
            n_classes=n_classes,
            class_column="class",
            image_column="image",
            transform=training_transform,
            ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        ClassificationDataset(
            validation_dataframe,
            channels=channels,
            n_classes=n_classes,
            class_column="class",
            image_column="image",
            transform=validation_transform,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader