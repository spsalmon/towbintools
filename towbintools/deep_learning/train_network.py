from utils.dataset import TilesDatasetFly, TilesDataset
from towbintools.towbintools.deep_learning.augmentation import get_mean_and_std, get_training_augmentation, get_validation_augmentation, grayscale_to_rgb
import os
from time import perf_counter

import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from pytorch_toolbelt import inference
import pytorch_toolbelt.losses as L


from skimage.filters import threshold_otsu
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.classification import BinaryF1Score
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import albumentations as albu
from towbintools.foundation import image_handling
from utils.loss import FocalTverskyLoss

from architectures import NestedUNet
import pretrained_microscopy_models as pmm

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

dataframe = pd.DataFrame({'raw': images, 'analysis/ch1_seg': masks})

training_dataframe, validation_dataframe = train_test_split(dataframe, test_size=0.25, random_state=42)
image_column = 'raw'
mask_column = 'analysis/ch1_seg'

class LightningPretrained(pl.LightningModule):
	def __init__(self, n_classes, learning_rate, architecture, encoder, pretrained_weights, normalization):
		super().__init__()
		model = pmm.segmentation_training.create_segmentation_model(
            architecture=architecture,
            encoder = encoder,
            encoder_weights=pretrained_weights,
            classes=1,
        )
		self.model = model
		self.learning_rate = learning_rate
		self.criterion = FocalTverskyLoss()
		self.f1_score = BinaryF1Score()
		self.normalization = normalization
		self.save_hyperparameters()

	def forward(self, x):
		return self.model(x)

	def log_tb_images(self, viz_batch) -> None:

		# Get tensorboard logger
		tb_logger = None
		for logger in self.trainer.loggers:
			if isinstance(logger, pl.loggers.TensorBoardLogger):
				tb_logger = logger.experiment
				break

		if tb_logger is None:
				raise ValueError('TensorBoard Logger not found')

	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self.model(x)
		loss = self.criterion(y_hat, y)
		self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

		f1_score = self.f1_score(y_hat, y)
		self.log("train_f1_score", f1_score, on_step=False, on_epoch=True, logger=True, sync_dist=True)

		return loss

	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self.model(x)
		loss = self.criterion(y_hat, y)
		self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		
		f1_score = self.f1_score(y_hat, y)
		self.log("val_f1_score", f1_score, on_step=True, on_epoch=True, logger=True, sync_dist=True)

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
		return optimizer
	
class LightningUnetPlusPlus(pl.LightningModule):
	def __init__(self, n_classes, learning_rate, normalization):
		super().__init__()
		model = NestedUNet(num_classes=1, input_channels=1, deep_supervision=False)
		self.model = model
		self.learning_rate = learning_rate
		self.criterion = FocalTverskyLoss()
		self.f1_score = BinaryF1Score()
		self.normalization = normalization
		self.save_hyperparameters()

	def forward(self, x):
		return self.model(x)

	def log_tb_images(self, viz_batch) -> None:

		# Get tensorboard logger
		tb_logger = None
		for logger in self.trainer.loggers:
			if isinstance(logger, pl.loggers.TensorBoardLogger):
				tb_logger = logger.experiment
				break

		if tb_logger is None:
				raise ValueError('TensorBoard Logger not found')

	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self.model(x)
		loss = self.criterion(y_hat, y)
		self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

		f1_score = self.f1_score(y_hat, y)
		self.log("train_f1_score", f1_score, on_step=False, on_epoch=True, logger=True, sync_dist=True)
		return loss

	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self.model(x)
		loss = self.criterion(y_hat, y)
		self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		
		f1_score = self.f1_score(y_hat, y)
		self.log("val_f1_score", f1_score, on_step=True, on_epoch=True, logger=True, sync_dist=True)

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
		return optimizer
	
first_image = image_handling.read_tiff_file(training_dataframe[image_column].values[0], [0])
image_slicer = inference.ImageSlicer(first_image.shape, (512, 512), (256, 256))

training_transform = get_training_augmentation('percentile', lo=1, hi=99)
validation_transform = get_validation_augmentation('percentile', lo=1, hi=99)

checkpoint = "/home/spsalmon/towbintools/towbintools/deep_learning/unet_lightning_test/best_pretrained_low_lr.ckpt"
model = LightningPretrained(n_classes=1, architecture='UnetPlusPlus', encoder='efficientnet-b4', pretrained_weights="image-micronet", learning_rate=1e-5, normalization={'type': 'percentile', 'lo': 1, 'hi': 99}).load_from_checkpoint(checkpoint)

train_loader = DataLoader(TilesDatasetFly(training_dataframe, image_slicer, channel_to_segment=1, mask_column=mask_column, image_column= image_column, transform=training_transform, RGB = True), batch_size=5, shuffle=True, num_workers=32, pin_memory=True)
val_loader = DataLoader(TilesDatasetFly(validation_dataframe, image_slicer, channel_to_segment=1, mask_column=mask_column, image_column= image_column, transform=validation_transform, RGB= True), batch_size=5, shuffle=False, num_workers=32, pin_memory=True)

checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="unet_confocal", save_top_k=3, monitor="val_loss")
swa_callback = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)


trainer = pl.Trainer(max_epochs=150, accelerator="gpu", strategy='ddp_find_unused_parameters_true', callbacks=[checkpoint_callback, swa_callback], accumulate_grad_batches = 8, gradient_clip_val=0.5, detect_anomaly=True)
trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
print(checkpoint_callback.best_model_path)