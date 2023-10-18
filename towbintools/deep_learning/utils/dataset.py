from torch.utils.data import Dataset
from towbintools.foundation import image_handling
import torch
import numpy as np
from .augmentation import grayscale_to_rgb
from albumentations.pytorch import ToTensorV2

# Dataset where each image is split into tiles in the first place
class TilesDataset(Dataset):
	def __init__(self, dataset, image_slicer, channel_to_segment, mask_column, image_column = 'raw', transform=None, RGB = True):
		images = dataset[image_column].values.tolist()
		ground_truth = dataset[mask_column].values.tolist()



		self.image_tiles = []
		self.mask_tiles = []

		for image, ground_truth in zip(images, ground_truth):
			img = image_handling.read_tiff_file(image, [channel_to_segment])
			mask = image_handling.read_tiff_file(ground_truth)

			if transform is not None:
				transformed = transform(image = img, mask = mask)
				img = transformed['image']
				mask = transformed['mask']

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

class TilesDatasetFly(Dataset):
	def __init__(self, dataset, image_slicer, channel_to_segment, mask_column, image_column = 'raw', transform=None, RGB = True):
		self.images = dataset[image_column].values.tolist()
		self.ground_truth = dataset[mask_column].values.tolist()
		self.channel_to_segment = channel_to_segment
		self.image_slicer = image_slicer
		self.transform = transform
		self.RGB = RGB

	def __len__(self):
		return len(self.images)

	def __getitem__(self, i):
		
		img = image_handling.read_tiff_file(self.images[i], [self.channel_to_segment])
		mask = image_handling.read_tiff_file(self.ground_truth[i])
		
		if self.transform is not None:
			transformed = self.transform(image = img, mask = mask)
			img = transformed['image']
			mask = transformed['mask']

		# img = image_handling.normalize_image(img, dest_dtype=np.float32)

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
	
class TilesDatasetFlyScract(Dataset):
	def __init__(self, images, ground_truth, image_slicer, channel_to_segment, transform=None, RGB = True):
		self.images = images
		self.ground_truth = ground_truth
		self.channel_to_segment = channel_to_segment
		self.image_slicer = image_slicer
		self.transform = transform
		self.RGB = RGB

	def __len__(self):
		return len(self.images)

	def __getitem__(self, i):
		
		img = image_handling.read_tiff_file(self.images[i], [self.channel_to_segment])
		mask = image_handling.read_tiff_file(self.ground_truth[i])
		
		if self.transform is not None:
			transformed = self.transform(image = img, mask = mask)
			img = transformed['image']
			mask = transformed['mask']

		# img = image_handling.normalize_image(img, dest_dtype=np.float32)

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