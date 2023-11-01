import torch
import pretrained_microscopy_models as pmm
import torch.utils.model_zoo as model_zoo

import numpy as np
import os 
import random
from towbintools.foundation import image_handling
from towbintools.foundation import binary_image
import cv2
from scipy.ndimage import binary_fill_holes
import matplotlib.pyplot as plt
import numpy as np
import os
import xgboost as xgb

from csbdeep.utils import normalize
from time import perf_counter
import random
from joblib import Parallel, delayed


import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

import pickle
torch.set_num_threads(32)

def init_VGG16_micronet():

	model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16_bn', weights=None)
	url = pmm.util.get_pretrained_microscopynet_url('vgg16_bn', 'image-micronet')

	# remove classifier from model 
	model.classifier = nn.Sequential(*list(model.classifier.children())[:-8])
	# Load pretrained weights

	model.load_state_dict(model_zoo.load_url(url, map_location=torch.device('cpu')))

	model.eval()  # <- MicrosNet model for classifcation or transfer learning

	# Extract up to 'block1_conv2'
	features_seq = list(model.features)[:2]
	new_model = nn.Sequential(*features_seq)

	# Disable gradient computation (use pretrained weights)
	for param in new_model.parameters():
		param.requires_grad = False
	
	return new_model

def grayscale_to_rgb_pytorch(grayscale_img):
	img = np.expand_dims(grayscale_img, axis=0)
	stacked_img = np.stack((img,)*3, axis=0)
	stacked_img = np.squeeze(stacked_img)
	return stacked_img

def transform_img(img):
	img = normalize(img,1,99.8,axis=(0,1))
	img = image_handling.normalize_image(img, np.uint8)
	img = grayscale_to_rgb_pytorch(img)
	return img

def preprocess_ground_truth_mask(ground_truth):
	ground_truth = cv2.morphologyEx(ground_truth, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
	ground_truth = binary_fill_holes(ground_truth).astype(np.uint8)
	ground_truth = binary_image.get_biggest_object(ground_truth).astype(np.uint8)

	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
	dilated_ground_truth = (cv2.morphologyEx(ground_truth, cv2.MORPH_DILATE, kernel) > 0).astype(int)
	background = (dilated_ground_truth == 0).astype(int)
	ground_truth = (cv2.morphologyEx(ground_truth, cv2.MORPH_ERODE, kernel) > 0).astype(int)

	return ground_truth, background

def annotations_to_tensor_pytorch(feature_matrix, mask):
	'''Convert the user annotated labels from napari to tensors to train the classifier on.
	feature_matrix dim: [x, y, nb_features]
	possible mask elements: 0: not annotated, int[1,2]: class annotation
	'''
	mask = mask.squeeze()
	# Find the indices where mask is not -1
	indices = torch.nonzero(mask != -1, as_tuple=True)

	# Use those indices to extract y_labels from mask
	y_labels = mask[indices[0], indices[1]]

	# Use those indices to extract feature vectors
	X_features = feature_matrix[indices[0], indices[1]]

	X_features = X_features.cpu().numpy()
	y_labels = y_labels.cpu().numpy()

	return X_features, y_labels

def extract_features_and_ground_truth(features, ground_truth):
	X_, y_ = annotations_to_tensor_pytorch(features, ground_truth)
	return X_, y_

class IlastikLikeTrainingDataset(Dataset):
	def __init__(self, images, ground_truth, transform=None):
		self.images = images
		self.ground_truth = ground_truth
		self.transform = transform

	def __len__(self):
		return len(self.images)

	def __getitem__(self, i):
		
		img = image_handling.read_tiff_file(self.images[i], [2]).astype(np.float32)
		img = transform_img(img)
		mask = image_handling.read_tiff_file(self.ground_truth[i])
		ground_truth, background = preprocess_ground_truth_mask(mask)
		# replace 1 in ground truth with 2
		ground_truth[ground_truth == 1] = 1
		# replace everything else with -1
		ground_truth[ground_truth == 0] = -1
		# replace background in ground truth with 1
		ground_truth[background == 1] = 0    

		# Find indices where ground_truth is 0
		zero_indices = np.argwhere(ground_truth == 0)

		# Randomly select 99% of the zero indices
		num_samples = int(0.95 * len(zero_indices))
		random_indices = np.random.choice(len(zero_indices), num_samples, replace=False)
		selected_zero_indices = zero_indices[random_indices]
		# Set those pixels to -1
		ground_truth[selected_zero_indices[:, 0], selected_zero_indices[:, 1]] = -1

		return torch.tensor(img, dtype=torch.float32), ground_truth
	
def extract_vgg16_pyramid_features(model, x, ratios):
	features = []
	shapes = [(int(x.shape[2] / ratio), int(x.shape[3] / ratio)) for ratio in ratios]
	for shape in shapes:

		rescaled_batch = F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
		scaled_features = model(rescaled_batch)
		rescaled_features = F.interpolate(scaled_features, size=x.shape[2:], mode='bilinear', align_corners=False)
		rescaled_features = rescaled_features.squeeze(0)
		features.append(rescaled_features)

	global_feature = torch.concatenate(features,axis=0)
	global_feature = torch.transpose(global_feature, 0, 2)
	global_feature = torch.transpose(global_feature, 0, 1)
	return global_feature

def process_image_for_training(img, mask, model, shapes):
	features = extract_vgg16_pyramid_features(model, img, shapes)
	X, Y = extract_features_and_ground_truth(features, mask)
	print('Processing image...', X.shape)
	if X.shape[0] > 0.4*(img.shape[2]*img.shape[3]):
		print('Too many features, skipping...')
		return {'X': [], 'Y': []}
	return {'X': X, 'Y': Y}

def concatenate_predictions(predictions):
	X = []
	y = []
	for prediction in predictions:
		X.extend(prediction['X'])
		y.extend(prediction['Y'])
	return X, y

image_dir = "/mnt/external.data/TowbinLab/kstojanovski/20220401_Ti2_20x_160-182-190_pumping_25C_20220401_173300_429/analysis/ch1/"
images = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tiff')]
random.seed(42)
training_images = random.sample(images, 100)
training_ground_truth = [f.replace('ch1/', 'ch1_il/seg_') for f in training_images]

test_loader = DataLoader(IlastikLikeTrainingDataset(training_images, training_ground_truth, transform=transform_img), batch_size=1, shuffle=False, num_workers=16, pin_memory=True)
ratios = [1, 2, 4, 8]

model = init_VGG16_micronet()
model.eval()

predictions = Parallel(n_jobs=2, prefer='processes')(delayed(process_image_for_training)(img, mask, model, ratios) for img, mask in test_loader)
X, y = concatenate_predictions(predictions)

# Save X and y as pickle files
with open('./pickles/training_features', 'wb') as handle:
	pickle.dump(X, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('./pickles/training_labels', 'wb') as handle:
	pickle.dump(y, handle, protocol=pickle.HIGHEST_PROTOCOL)

params = {
    'device': 'cpu',
    'tree_method': 'hist',
    # 'subsample': 0.2,
    # 'sampling_method': 'uniform',
}

print('Training classifier...')
start = perf_counter()
Xy = xgb.QuantileDMatrix(X, y)
clf = xgb.train(params, Xy)

#save the classifier
clf.save_model("./trained_model.json")
end = perf_counter()
print(f'Training time {end - start} seconds')