import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from towbintools.foundation import image_handling
import numpy as np
from .augmentation import grayscale_to_rgb
import os
import pandas as pd

from towbintools.deep_learning.utils.augmentation import (
    get_training_augmentation,
    get_prediction_augmentation,
)
from pytorch_toolbelt import inference
from torch.utils.data import DataLoader


# Dataset where each image is split into tiles in the first place
class OldTiledSegmentationDataloader(Dataset):
    def __init__(
        self,
        dataset,
        image_slicer,
        channels,
        mask_column,
        image_column="raw",
        transform=None,
        RGB=True,
    ):
        images = dataset[image_column].values.tolist()
        ground_truth = dataset[mask_column].values.tolist()

        self.image_tiles = []
        self.mask_tiles = []

        for image, ground_truth in zip(images, ground_truth):
            img = image_handling.read_tiff_file(image, [channels])
            mask = image_handling.read_tiff_file(ground_truth)

            if transform is not None:
                transformed = transform(image=img, mask=mask)
                img = transformed["image"]
                mask = transformed["mask"]

            tiles = image_slicer.split(img)
            if RGB:
                tiles = [grayscale_to_rgb(tile) for tile in tiles]
            tiles_mask = image_slicer.split(mask)

            self.image_tiles.extend(tiles)
            self.mask_tiles.extend(tiles_mask)

    def __len__(self):
        return len(self.image_tiles)

    def __getitem__(self, i):
        img = self.image_tiles[i]
        mask = self.mask_tiles[i]
        mask = mask[np.newaxis, ...]

        return img, mask


# Dataset where the images are split into tiles on the fly


class TiledSegmentationDataloader(Dataset):
    def __init__(
        self,
        dataset,
        image_slicer,
        channels,
        mask_column="mask",
        image_column="image",
        transform=None,
        RGB=True,
    ):
        self.images = dataset[image_column].values.tolist()
        self.ground_truth = dataset[mask_column].values.tolist()
        self.channels = channels
        self.image_slicer = image_slicer
        self.transform = transform
        self.RGB = RGB

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = image_handling.read_tiff_file(self.images[i], [self.channels])
        mask = image_handling.read_tiff_file(self.ground_truth[i])

        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        tiles = self.image_slicer.split(img)
        if self.RGB:
            tiles = [grayscale_to_rgb(tile) for tile in tiles]
        else:
            tiles = [tile[np.newaxis, ...] for tile in tiles]
        tiles_ground_truth = self.image_slicer.split(mask)

        selected_tile = np.random.randint(0, len(tiles))
        img = tiles[selected_tile]
        mask = tiles_ground_truth[selected_tile]
        mask = mask[np.newaxis, ...]

        return img, mask


class SegmentationDataloader(Dataset):
    def __init__(
        self,
        dataset,
        channels,
        mask_column="mask",
        image_column="image",
        transform=None,
        RGB=True,
    ):
        self.images = dataset[image_column].values.tolist()
        self.ground_truth = dataset[mask_column].values.tolist()
        self.channels = channels
        self.transform = transform
        self.RGB = RGB

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        img = image_handling.read_tiff_file(self.images[i], [self.channels])
        mask = image_handling.read_tiff_file(self.ground_truth[i])

        if self.transform is not None:
            transformed = self.transform(image=img, mask=mask)
            img = transformed["image"]
            mask = transformed["mask"]

        if self.RGB:
            img = grayscale_to_rgb(img)
        else:
            img = img[np.newaxis, ...]
        mask = mask[np.newaxis, ...]

        return img, mask


def create_segmentation_training_dataframes(
    image_directories, mask_directories, save_dir, train_test_split_ratio=0.25
):
    images = []
    masks = []
    for image_directory, mask_directory in zip(image_directories, mask_directories):
        images.extend(
            [
                os.path.join(image_directory, file)
                for file in os.listdir(image_directory)
            ]
        )
        masks.extend(
            [os.path.join(mask_directory, file) for file in os.listdir(mask_directory)]
        )

    dataframe = pd.DataFrame({"image": images, "mask": masks})
    training_dataframe, validation_dataframe = train_test_split(
        dataframe, test_size=train_test_split_ratio, random_state=42
    )

    # backup the training and validation dataframes
    database_backup_dir = os.path.join(save_dir, "database_backup")
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    os.makedirs(database_backup_dir, exist_ok=True)
    training_dataframe.to_csv(
        os.path.join(database_backup_dir, f"training_dataframe_{current_date}.csv")
    )
    validation_dataframe.to_csv(
        os.path.join(database_backup_dir, f"validation_dataframe_{current_date}.csv")
    )

    return training_dataframe, validation_dataframe


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
    RGB=True,
):
    if not train_on_tiles:
        train_loader = DataLoader(
            SegmentationDataloader(
                training_dataframe,
                channels=channels,
                mask_column="mask",
                image_column="image",
                transform=training_transform,
                RGB=RGB,
            ),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        val_loader = DataLoader(
            SegmentationDataloader(
                validation_dataframe,
                channels=channels,
                mask_column="mask",
                image_column="image",
                transform=validation_transform,
                RGB=RGB,
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

    first_image = image_handling.read_tiff_file(
        training_dataframe["image"].values[0], [0]
    )

    image_slicer = inference.ImageSlicer(
        first_image.shape, tiler_params["tile_size"], tiler_params["tile_step"]
    )

    if training_transform is None:
        training_transform = get_training_augmentation("percentile", lo=1, hi=99)
    if validation_transform is None:
        validation_transform = get_prediction_augmentation("percentile", lo=1, hi=99)

    train_loader = DataLoader(
        TiledSegmentationDataloader(
            training_dataframe,
            image_slicer,
            channels=channels,
            mask_column="mask",
            image_column="image",
            transform=training_transform,
            RGB=RGB,
        ),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        TiledSegmentationDataloader(
            validation_dataframe,
            image_slicer,
            channels=channels,
            mask_column="mask",
            image_column="image",
            transform=validation_transform,
            RGB=RGB,
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
    train_test_split_ratio=0.25,
    batch_size=5,
    num_workers=32,
    pin_memory=True,
    train_on_tiles=True,
    tiler_params=None,
    training_transform=None,
    validation_transform=None,
    RGB=True,
):
    training_dataframe, validation_dataframe = create_segmentation_training_dataframes(
        image_directories,
        mask_directories,
        save_dir,
        train_test_split_ratio=train_test_split_ratio,
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
        RGB=RGB,
    )
    return training_dataframe, validation_dataframe, train_loader, val_loader

def create_dataloaders_from_filemap(
        filemap_path,
        save_dir,
        channels,
        train_test_split_ratio=0.25,
        batch_size=5,
        num_workers=32,
        pin_memory=True,
        train_on_tiles=True,
        tiler_params=None,
        training_transform=None,
        validation_transform=None,
        RGB=True,
):
    dataframe = pd.read_csv(filemap_path)
    training_dataframe, validation_dataframe = train_test_split(
        dataframe, test_size=train_test_split_ratio, random_state=42
    )

    # backup the training and validation dataframes
    database_backup_dir = os.path.join(save_dir, "database_backup")
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    os.makedirs(database_backup_dir, exist_ok=True)
    training_dataframe.to_csv(
        os.path.join(database_backup_dir, f"training_dataframe_{current_date}.csv")
    )
    validation_dataframe.to_csv(
        os.path.join(database_backup_dir, f"validation_dataframe_{current_date}.csv")
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
        RGB=RGB,
    )
    return training_dataframe, validation_dataframe, train_loader, val_loader