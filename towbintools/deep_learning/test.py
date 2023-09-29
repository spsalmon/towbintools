import os
from time import perf_counter

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from pytorch_toolbelt import inference
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

image_csv = "/mnt/external.data/TowbinLab/plenart/20221020_Ti2_10x_green_bacteria_wbt150_small_chambers_good/analysis/report/analysis_filemap.csv"

image_dataframe = pd.read_csv(image_csv).dropna(subset=['analysis/ch2_seg'])

# pick 10000 random images

image_dataframe = image_dataframe.sample(n=1000, random_state=42)
image_dataframe = image_dataframe[['raw', 'analysis/ch2_seg']]

training_dataframe, validation_dataframe = train_test_split(image_dataframe, test_size=0.25, random_state=42)

# get mean and std of training images



def get_training_augmentation():
    train_transform = [
        albu.Flip(p=0.75),
        albu.RandomRotate90(p=1),       
        albu.GaussNoise(p=0.5),
        
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1, limit=0.25),
                albu.RandomGamma(p=1),
            ],
            p=0.50,
        ),

        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                #albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.50,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1, limit=0.3),
                albu.HueSaturationValue(p=1),
            ],
            p=0.50,
        ),
    ]
    return albu.Compose(train_transform)

def get_mean_and_std(image_path):
	image = image_handling.read_tiff_file(image_path, [2])
	return np.mean(image), np.std(image)
mean_and_std = Parallel(n_jobs=-1, prefer='processes')(delayed(get_mean_and_std)(image_path) for image_path in tqdm(training_dataframe.sample(n=10, random_state=42)['raw'].values.tolist()))

mean_train_images = [mean for mean, std in mean_and_std]
std_train_images = [std for mean, std in mean_and_std]

mean_train_images = np.mean(mean_train_images)
std_train_images = np.mean(std_train_images)


class LightningUNet(pl.LightningModule):
	def __init__(self, n_channels=1, n_classes=1, bilinear=True, learning_rate=0.001):
		super().__init__()
		self.model = UNet(n_channels=n_channels,
						  n_classes=n_classes, bilinear=bilinear)
		self.learning_rate = learning_rate
		self.criterion = FocalTverskyLoss()
		self.dice = Dice()
		self.f1_score = BinaryF1Score()

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
			tb_logger.add_image(f"GroundTruth/{img_idx}", y_true, 0)
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
		loss = FocalTverskyLoss(y_hat, y)
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

# # Dataset where each image is split into tiles in the first place
# class TilesDataset(Dataset):
# 	def __init__(self, dataset, image_slicer, transform=None):
# 		images = dataset['raw'].values.tolist()
# 		ground_truth = dataset['analysis/ch2_seg'].values.tolist()

# 		self.image_tiles = []
# 		self.ground_truth_tiles = []

# 		for image, ground_truth in zip(images, ground_truth):
# 			image = image_handling.read_tiff_file(image, [2]).astype(np.float32)

# 			if transform:
# 				image = transform(image).cpu().numpy().squeeze()

# 			ground_truth = image_handling.read_tiff_file(ground_truth)

# 			tiles = image_slicer.split(image)
# 			tiles_ground_truth = image_slicer.split(ground_truth)

# 			self.image_tiles.extend(tiles)
# 			self.ground_truth_tiles.extend(tiles_ground_truth)
		
# 	def __len__(self):
# 		return len(self.image_tiles)

# 	def __getitem__(self, i):
		
# 		img = self.image_tiles[i]
# 		img = img[np.newaxis, ...]
# 		mask = self.ground_truth_tiles[i]
# 		mask = mask[np.newaxis, ...]
		
# 		return torch.tensor(img, dtype=torch.float32), torch.tensor(mask, dtype=torch.int32)
	
# Dataset where the images are split into tiles on the fly

class TilesDataset(Dataset):
	def __init__(self, dataset, image_slicer, transform=None):
		self.images = dataset['raw'].values.tolist()
		self.ground_truth = dataset['analysis/ch2_seg'].values.tolist()
		self.image_slicer = image_slicer
		self.transform = transform

	def __len__(self):
		return len(self.images)

	def __getitem__(self, i):
		
		img = image_handling.read_tiff_file(self.images[i], [2]).astype(np.float32)
		if self.transform:
			img = self.transform(img).cpu().numpy().squeeze()

		mask = image_handling.read_tiff_file(self.ground_truth[i])

		tiles = self.image_slicer.split(img)
		tiles_ground_truth = self.image_slicer.split(mask)

		selected_tile = np.random.randint(0, len(tiles))
		img = tiles[selected_tile]
		img = img[np.newaxis, ...]
		mask = tiles_ground_truth[selected_tile]
		mask = mask[np.newaxis, ...]

		# img, mask = random_image_transformation(img, mask)

		return torch.tensor(img.copy(), dtype=torch.float32), torch.tensor(mask.copy(), dtype=torch.int32)
	

first_image = image_handling.read_tiff_file(training_dataframe['raw'].values[0], [2])
print(f'First image shape: {first_image.shape}')
transform = Compose([ToTensor(), Normalize(mean_train_images, std_train_images)])
image_slicer = inference.ImageSlicer(first_image.shape, (512, 512), (256, 256))

train_loader = DataLoader(TilesDataset(training_dataframe, image_slicer, transform=transform), batch_size=6, shuffle=True, num_workers=32, pin_memory=True)

for batch in train_loader:
	print(type(batch))
	x, y = batch
	print(type(x), type(y))
	print(x.shape, y.shape)
	break
# val_loader = DataLoader(TilesDataset(validation_dataframe, image_slicer, transform=transform), batch_size=6, shuffle=False, num_workers=32, pin_memory=True)

checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="unet_lightning_test", save_top_k=1, monitor="val_loss")
swa_callback = pl.callbacks.StochasticWeightAveraging(swa_lrs=1e-2)

model = LightningUNet(learning_rate=0.0001)
trainer = pl.Trainer(max_epochs=15, accelerator="gpu", strategy="auto", callbacks=[checkpoint_callback, swa_callback], accumulate_grad_batches = 6, gradient_clip_val=0.5)
# tuner = pl.tuner.Tuner(trainer)
# tuner.lr_find(model, train_dataloaders=train_loader, val_dataloaders=val_loader) 

trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loader)
print(checkpoint_callback.best_model_path)