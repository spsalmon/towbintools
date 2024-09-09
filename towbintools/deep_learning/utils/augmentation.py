import albumentations as albu
from towbintools.foundation import image_handling
import numpy as np
import torch
from albumentations.core.transforms_interface import ImageOnlyTransform, DualTransform
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
        return ("mean", "std")


class NormalizePercentile(ImageOnlyTransform):
    def __init__(self, lo, hi, axis = None, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.lo = lo
        self.hi = hi
        self.axis = axis

    def apply(self, img, **params):
        return normalize(img, self.lo, self.hi, axis=self.axis)

    def get_transform_init_args_names(self):
        return ("lo", "hi")


class GrayscaleToRGB(ImageOnlyTransform):
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply, p)

    def apply(self, img, **params):
        return grayscale_to_rgb(img)

    def get_transform_init_args_names(self):
        return ()

class CustomFlip(DualTransform):
    """Flip the input image horizontally or vertically with a given probability. Works well with images ordered in the OME-TIFF way."""
    def __init__(self, always_apply=False, p=0.5):
        super(CustomFlip, self).__init__(always_apply, p)
        self.axis = np.random.choice([-1, -2])

    def apply(self, img, **params):
        return np.flip(img, axis=self.axis)

    def apply_to_mask(self, img, **params):
        return np.flip(img, axis=self.axis)

    def get_transform_init_args_names(self):
        return ()
    
class CustomRotate90(DualTransform):
    """Rotate the input image by 90 degrees."""
    def __init__(self, always_apply=False, p=0.5):
        super(CustomRotate90, self).__init__(always_apply, p)
        self.k = np.random.choice([1, 2, 3])

    def apply(self, img, **params):
        return np.rot90(img, k=self.k, axes=(-2, -1))

    def apply_to_mask(self, img, **params):
        return np.rot90(img, k=self.k, axes=(-2, -1))

    def get_transform_init_args_names(self):
        return ()

def get_training_augmentation(normalization_type, **kwargs):
    train_transform = [
        CustomFlip(p=0.75),
        CustomRotate90(p=0.75),
        albu.GaussNoise(p=0.5),
        albu.RandomGamma(p=0.5),
    ]

    if normalization_type == "data_range":
        train_transform.append(NormalizeDataRange())
    elif normalization_type == "mean_std":
        train_transform.append(NormalizeMeanStd(kwargs["mean"], kwargs["std"]))
    elif normalization_type == "percentile":
        try:
            train_transform.append(NormalizePercentile(kwargs["lo"], kwargs["hi"], kwargs["axis"]))
        except KeyError:
            train_transform.append(NormalizePercentile(kwargs["lo"], kwargs["hi"]))
    return albu.Compose(train_transform)


def get_prediction_augmentation(normalization_type, **kwargs):
    prediction_transform = []

    if normalization_type == "data_range":
        prediction_transform.append(NormalizeDataRange())
    elif normalization_type == "mean_std":
        prediction_transform.append(NormalizeMeanStd(kwargs["mean"], kwargs["std"]))
    elif normalization_type == "percentile":
        try:
            prediction_transform.append(NormalizePercentile(kwargs["lo"], kwargs["hi"], kwargs["axis"]))
        except KeyError:
            prediction_transform.append(NormalizePercentile(kwargs["lo"], kwargs["hi"]))

    return albu.Compose(prediction_transform)


def get_mean_and_std(image_path):
    image = image_handling.read_tiff_file(image_path, [2])
    return np.mean(image), np.std(image)


def grayscale_to_rgb(grayscale_img):
    # Check if the image is a pytorch tensor, if not, convert it to one
    if not isinstance(grayscale_img, torch.Tensor):
        grayscale_img = torch.tensor(grayscale_img, dtype=torch.float32)

    assert (
        len(grayscale_img.shape) == 2 or len(grayscale_img.shape) == 3
    ), "Currently, zstacks are not supported"
    # Assuming grayscale_img has a shape of (H, W)
    # we will unsqueeze it to have a shape of (1, H, W)
    if len(grayscale_img.shape) == 2:
        grayscale_img = grayscale_img.unsqueeze(0)

    # Assuming grayscale_img has a shape of (C, H, W)
    if grayscale_img.shape[0] == 3:
        return grayscale_img

    if grayscale_img.shape[0] == 2:
        grayscale_img = torch.cat((grayscale_img, grayscale_img[0].unsqueeze(0)), 0)

    if grayscale_img.shape[0] > 3:
        raise ValueError(
            "The image has more than 3 channels, and thus cannot be converted to RGB"
        )

    # Repeat the single channel image three times along the channel dimension (dimension 0)
    return grayscale_img.repeat((3, 1, 1))
