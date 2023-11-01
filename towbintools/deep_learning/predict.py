from utils.dataset import TilesDatasetFly, TilesDataset
from utils.augmentation import get_mean_and_std, get_training_augmentation, get_validation_augmentation, grayscale_to_rgb, get_prediction_augmentation
import os
from time import perf_counter

from pytorch_toolbelt.inference.tiles import ImageSlicer
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
from tqdm import tqdm
from tifffile import imwrite

from architectures import NestedUNet
import pretrained_microscopy_models as pmm

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

model_path = "/home/spsalmon/towbintools/towbintools/deep_learning/unet_confocal/epoch=102-step=16377.ckpt"

preprocessing_fn = get_prediction_augmentation('percentile', lo=1, hi=99, RGB=True)

images_dir = "/mnt/towbin.data/shared/ngerber/20230927_LIPSI_20x_348_297_20C_20230927_165930_303/pad2/analysis/best_planes/"
output_dir = "/mnt/towbin.data/shared/ngerber/20230927_LIPSI_20x_348_297_20C_20230927_165930_303/pad2/analysis/ch3_seg"
os.makedirs(output_dir, exist_ok=True)
images_path = [os.path.join(images_dir, image) for image in os.listdir(images_dir)]
first_image = image_handling.read_tiff_file(images_path[0], [1])

def deep_learning_segmentation(image, model, device, preprocessing_fn, tiler, RGB=True, activation=None):
	image = preprocessing_fn(image=image)['image']
	tiles = tiler.split(image)

	if RGB:
		tiles = [grayscale_to_rgb(tile) for tile in tiles]
	else:
		tiles = [torch.tensor(tile[np.newaxis, ...]).unsqueeze(0) for tile in tiles]

	# assemble tiles into a batch
	batch = torch.stack(tiles)
	batch = batch.to(device)

	# predict

	prediction_tiles = []
	with torch.no_grad():
		prediction = model(batch)
		if activation == 'sigmoid':
			prediction = torch.sigmoid(prediction)
		elif activation == 'softmax':
			prediction = torch.softmax(prediction, dim=1)
	
	# assemble tiles into an image
	for pred_tile in prediction:
		pred_tile = pred_tile.cpu().numpy()
		pred_tile = np.moveaxis(pred_tile, 0, -1)
		prediction_tiles.append(pred_tile)
	
	pred = tiler.merge(prediction_tiles)
	mask = (pred > 0.5).astype(np.uint8)

	return mask

def segment(image_path, model, device, preprocessing_fn, tiler, channels, RGB=True, activation=None, is_zstack=False):
	image = image_handling.read_tiff_file(image_path, channels_to_keep=channels)
	mask = deep_learning_segmentation(image, model, device, preprocessing_fn, tiler)
	return mask

tiler = ImageSlicer(first_image.shape, tile_size=(512, 512), tile_step=(256, 256), weight='pyramid')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = torch.load(model_path, map_location=device)
model = LightningPretrained.load_from_checkpoint(model_path, map_location=device)
model.eval()
normalization_type = model.normalization['type']
normalization_params = model.normalization

for image_path in tqdm(images_path):
	mask = segment(image_path, model, device, preprocessing_fn, tiler, [2])
	imwrite(os.path.join(output_dir, os.path.basename(image_path)), mask, dtype=np.uint8, compression='zlib')
	