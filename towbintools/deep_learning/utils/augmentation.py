import albumentations as albu
from towbintools.foundation import image_handling
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
from csbdeep.utils import normalize

class NormalizeDataRange(ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        return (img - np.min(img)) / (np.max(img) - np.min(img))

    def get_transform_init_args_names(self):
        return ()
    
class NormalizeMeanStd(ImageOnlyTransform):
    def __init__(self, mean, std, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.mean = mean
        self.std = std

    def apply(self, img, **params):
        return (img - self.mean) / self.std

    def get_transform_init_args_names(self):
        return ('mean', 'std')
    
class NormalizePercentile(ImageOnlyTransform):
    def __init__(self, lo, hi, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.lo = lo
        self.hi = hi

    def apply(self, img, **params):
        return normalize(img, self.lo, self.hi)

    def get_transform_init_args_names(self):
        return ('lo', 'hi')

def get_training_augmentation():
    train_transform = [
        albu.Flip(p=0.75),
        albu.RandomRotate90(p=1),       
        albu.GaussNoise(p=0.5),
        albu.RandomGamma(p=0.5),
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

