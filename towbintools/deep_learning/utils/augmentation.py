import albumentations as albu
from towbintools.foundation import image_handling
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

def get_training_augmentation(mean, std):
    train_transform = [
        albu.Flip(p=0.75),
        albu.RandomRotate90(p=1),       
        albu.GaussNoise(p=0.5),
        albu.RandomGamma(p=0.5),
        
        albu.Normalize(mean=mean, std=std),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation(mean, std):
    test_transform = [
        albu.Normalize(mean=mean, std=std),
    ]
    return albu.Compose(test_transform)

def get_mean_and_std(image_path):
	image = image_handling.read_tiff_file(image_path, [2])
	return np.mean(image), np.std(image)

def grayscale_to_rgb(grayscale_img):
    # Check if the image is a pytorch tensor, if not, convert it to one
    if not isinstance(grayscale_img, torch.Tensor):
        grayscale_img = torch.tensor(grayscale_img, dtype=torch.float32)
    # Assuming grayscale_img has a shape of (H, W)
    # we will unsqueeze it to have a shape of (1, H, W)
    if len(grayscale_img.shape) == 2:
        grayscale_img = grayscale_img.unsqueeze(0)
    
    # stack the single channel image three times along the channel dimension (dimension 0)
    stacked_img = torch.cat((grayscale_img, grayscale_img, grayscale_img), 0)
    
    return stacked_img

