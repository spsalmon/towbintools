from utils.dataset import TilesDatasetFly
from utils.augmentation import (
    get_mean_and_std,
    get_training_augmentation,
    get_prediction_augmentation,
)
import os

import pytorch_lightning as pl
import pandas as pd
from pytorch_toolbelt import inference

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from towbintools.foundation import image_handling
from architectures.models import LightningPretrained, LightningUnetPlusPlus

# database_csv = "/mnt/external.data/TowbinLab/plenart/20221020_Ti2_10x_green_bacteria_wbt150_small_chambers_good/analysis/report/analysis_filemap.csv"

# image_column = 'raw'
# mask_column = 'analysis/ch2_seg'

# database = pd.read_csv(database_csv).dropna(subset=[mask_column])
# database = database.dropna(subset=[image_column])

# # pick 10000 random images
# database = database.sample(n=50000, random_state=42)
# training_dataframe, validation_dataframe = train_test_split(database, test_size=0.25, random_state=42)

raw_dir = "/mnt/towbin.data/shared/btowbin/20230809_wBT23_LIPSI_for_body_mask_training/cleaned/raw/"
images = sorted([os.path.join(raw_dir, file) for file in os.listdir(raw_dir)])
mask_dir = "/mnt/towbin.data/shared/btowbin/20230809_wBT23_LIPSI_for_body_mask_training/cleaned/ch1_seg/"
masks = sorted([os.path.join(mask_dir, file) for file in os.listdir(mask_dir)])

dataframe = pd.DataFrame({"raw": images, "analysis/ch1_seg": masks})

training_dataframe, validation_dataframe = train_test_split(
    dataframe, test_size=0.25, random_state=42
)
image_column = "raw"
mask_column = "analysis/ch1_seg"

save_dir = "unet_confocal"
os.makedirs(save_dir, exist_ok=True)

# backup the training and validation dataframes
database_backup_dir = os.path.join(save_dir, "database_backup")
os.makedirs(database_backup_dir, exist_ok=True)
training_dataframe.to_csv(os.path.join(database_backup_dir, "training_dataframe.csv"))
validation_dataframe.to_csv(
    os.path.join(database_backup_dir, "validation_dataframe.csv")
)

first_image = image_handling.read_tiff_file(
    training_dataframe[image_column].values[0], [0]
)
image_slicer = inference.ImageSlicer(first_image.shape, (512, 512), (256, 256))

training_transform = get_training_augmentation("percentile", lo=1, hi=99)
validation_transform = get_prediction_augmentation("percentile", lo=1, hi=99)

checkpoint = "/home/spsalmon/towbintools/towbintools/deep_learning/unet_lightning_test/best_pretrained_low_lr.ckpt"
model = LightningPretrained(
    n_classes=1,
    architecture="UnetPlusPlus",
    encoder="efficientnet-b4",
    pretrained_weights="image-micronet",
    learning_rate=1e-5,
    normalization={"type": "percentile", "lo": 1, "hi": 99},
).load_from_checkpoint(checkpoint)

train_loader = DataLoader(
    TilesDatasetFly(
        training_dataframe,
        image_slicer,
        channel_to_segment=1,
        mask_column=mask_column,
        image_column=image_column,
        transform=training_transform,
        RGB=True,
    ),
    batch_size=5,
    shuffle=True,
    num_workers=32,
    pin_memory=True,
)
val_loader = DataLoader(
    TilesDatasetFly(
        validation_dataframe,
        image_slicer,
        channel_to_segment=1,
        mask_column=mask_column,
        image_column=image_column,
        transform=validation_transform,
        RGB=True,
    ),
    batch_size=5,
    shuffle=False,
    num_workers=32,
    pin_memory=True,
)

checkpoint_callback = pl.callbacks.ModelCheckpoint(
    dirpath="unet_confocal", save_top_k=3, monitor="val_loss"
)
swa_callback = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)


trainer = pl.Trainer(
    max_epochs=150,
    accelerator="gpu",
    strategy="ddp_find_unused_parameters_true",
    callbacks=[checkpoint_callback, swa_callback],
    accumulate_grad_batches=8,
    gradient_clip_val=0.5,
    detect_anomaly=True,
)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
print(checkpoint_callback.best_model_path)
