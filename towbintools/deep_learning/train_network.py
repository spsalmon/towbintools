from utils.dataset import TilesDatasetFly, TilesDataset
from utils.augmentation import get_mean_and_std, get_training_augmentation, get_validation_augmentation, grayscale_to_rgb
import os
from time import perf_counter

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from pytorch_toolbelt import inference
from pytorch_toolbelt import losses as L
from utils import FocalTverskyLoss
from skimage.filters import threshold_otsu
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchmetrics.classification import BinaryF1Score, Dice
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
from unet import UNet
import albumentations as albu
from towbintools.foundation import image_handling
import pretrained_microscopy_models as pmm

database_csv = "/mnt/external.data/TowbinLab/plenart/20221020_Ti2_10x_green_bacteria_wbt150_small_chambers_good/analysis/report/analysis_filemap.csv"

image_column = 'raw'
mask_column = 'analysis/ch2_seg'

database = pd.read_csv(database_csv).dropna(subset=[mask_column])
database = database.dropna(subset=[image_column])

# pick 10000 random images
database = database.sample(n=40000, random_state=42)
training_dataframe, validation_dataframe = train_test_split(database, test_size=0.25, random_state=42)

# get mean and std of training images
training_images = training_dataframe[image_column].values
mean_and_std = Parallel(n_jobs=1, prefer='processes')(delayed(get_mean_and_std)(image_path) for image_path in tqdm(training_dataframe.sample(n=50, random_state=42)['raw'].values.tolist()))
mean_train_images = np.mean([mean for mean, std in mean_and_std])
std_train_images = np.mean([std for mean, std in mean_and_std])

class LightningPretrained(pl.LightningModule):
	def __init__(self, n_classes, learning_rate, architecture, encoder, pretrained_weights, mean, std):
		super().__init__()
		self.model = pmm.segmentation_training.create_segmentation_model(
            architecture=architecture,
            encoder = encoder,
            encoder_weights=pretrained_weights,
            classes=1,
        )
		self.learning_rate = learning_rate
		self.criterion = FocalTverskyLoss()
		self.dice = Dice()
		self.f1_score = BinaryF1Score()
		self.mean = mean
		self.std = std
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
		# Log the images (Give them different names)
		for img_idx, (image, y_true, y_pred) in enumerate(zip(*viz_batch)):
			tb_logger.add_image(f"Image/{img_idx}", image_handling.normalize_image(image.cpu().numpy(), dest_dtype=np.float32), 0)
			tb_logger.add_image(f"GroundTruth/{img_idx}", y_true*255, 0)
			tb_logger.add_image(f"Prediction/{img_idx}", y_pred, 0)

	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self.model(x)
		loss = self.criterion(y_hat, y)
		self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		dice_score = self.dice(y_hat, y)
		self.log("train_dice", dice_score, on_step=False, on_epoch=True, logger=True, sync_dist=True)
		f1_score = self.f1_score(y_hat, y)
		self.log("train_f1_score", f1_score, on_step=False, on_epoch=True, logger=True, sync_dist=True)

		return loss

	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self.model(x)
		loss = self.criterion(y_hat, y)
		self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
		
		dice_score = self.dice(y_hat, y)
		self.log("val_dice", dice_score, on_step=True, on_epoch=True, logger=True, sync_dist=True)
		f1_score = self.f1_score(y_hat, y)
		self.log("val_f1_score", f1_score, on_step=True, on_epoch=True, logger=True, sync_dist=True)

		if batch_idx == 0:
			self.log_tb_images((x, y, y_hat))

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
		return optimizer
	
first_image = image_handling.read_tiff_file(training_dataframe[image_column].values[0], [2])
image_slicer = inference.ImageSlicer(first_image.shape, (512, 512), (256, 256))

model_path = "/home/spsalmon/towbintools/towbintools/deep_learning/unet_lightning_test/epoch=22-step=11500.ckpt"
model = LightningPretrained(n_classes = 1, architecture = 'UnetPlusPlus', encoder = 'efficientnet-b4', pretrained_weights = 'image-micronet', learning_rate=0.0001, mean=mean_train_images, std=std_train_images).load_from_checkpoint(model_path)

train_loader = DataLoader(TilesDatasetFly(training_dataframe, image_slicer, channel_to_segment=2, mask_column=mask_column, image_column= image_column, transform=get_training_augmentation(model.mean, model.std)), batch_size=5, shuffle=True, num_workers=32, pin_memory=True)
val_loader = DataLoader(TilesDatasetFly(validation_dataframe, image_slicer, channel_to_segment=2, mask_column=mask_column, image_column= image_column, transform=get_validation_augmentation(model.mean, model.std)), batch_size=5, shuffle=False, num_workers=32, pin_memory=True)

checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="unet_lightning_test", save_top_k=10, monitor="val_loss")
swa_callback = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)


trainer = pl.Trainer(max_epochs=150, accelerator="gpu", strategy=pl.strategies.DDPStrategy(find_unused_parameters=True), callbacks=[checkpoint_callback, swa_callback], accumulate_grad_batches = 6, gradient_clip_val=0.5)
trainer.save_checkpoint("unet_lightning_test/test.ckpt")

# trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
# print(checkpoint_callback.best_model_path)