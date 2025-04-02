import albumentations as albu
import numpy as np
from albumentations.core.transforms_interface import DualTransform
from albumentations.core.transforms_interface import ImageOnlyTransform
from csbdeep.utils import normalize

from towbintools.foundation import image_handling


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
    def __init__(self, lo, hi, axis=None, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.lo = lo
        self.hi = hi
        self.axis = axis

    def apply(self, img, **params):
        return normalize(img, self.lo, self.hi, axis=self.axis)

    def get_transform_init_args_names(self):
        return ("lo", "hi")


class EnforceNChannels(ImageOnlyTransform):
    def __init__(self, n_channels, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.n_channels = n_channels

    def apply(self, img, **params):
        return enforce_n_channels(img, self.n_channels)

    def get_transform_init_args_names(self):
        return ("n_channels",)


class CustomFlip(DualTransform):
    """Flip the input image horizontally or vertically with a given probability. Works well with images ordered in the OME-TIFF way."""

    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)
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
        super().__init__(always_apply=always_apply, p=p)
        self.axis = np.random.choice([-1, -2])
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
        albu.Defocus(p=0.5),
        albu.RandomGamma(p=0.5),
    ]

    if normalization_type == "data_range":
        train_transform.append(NormalizeDataRange())
    elif normalization_type == "mean_std":
        train_transform.append(NormalizeMeanStd(kwargs["mean"], kwargs["std"]))
    elif normalization_type == "percentile":
        try:
            train_transform.append(
                NormalizePercentile(kwargs["lo"], kwargs["hi"], kwargs["axis"])
            )
        except KeyError:
            train_transform.append(NormalizePercentile(kwargs["lo"], kwargs["hi"]))

    enforce_n_channels = kwargs.get("enforce_n_channels", None)

    if enforce_n_channels is not None:
        train_transform.append(EnforceNChannels(enforce_n_channels))

    return albu.Compose(train_transform)


def get_prediction_augmentation(normalization_type, **kwargs):
    prediction_transform = []

    if normalization_type == "data_range":
        prediction_transform.append(NormalizeDataRange())
    elif normalization_type == "mean_std":
        prediction_transform.append(NormalizeMeanStd(kwargs["mean"], kwargs["std"]))
    elif normalization_type == "percentile":
        try:
            prediction_transform.append(
                NormalizePercentile(kwargs["lo"], kwargs["hi"], kwargs["axis"])
            )
        except KeyError:
            prediction_transform.append(NormalizePercentile(kwargs["lo"], kwargs["hi"]))

    enforce_n_channels = kwargs.get("enforce_n_channels", None)

    if enforce_n_channels is not None:
        prediction_transform.append(EnforceNChannels(enforce_n_channels))

    return albu.Compose(prediction_transform)


def get_prediction_augmentation_from_model(model, enforce_n_channels=None):
    normalization_type = model.normalization["type"]
    normalization_params = model.normalization

    if normalization_type == "percentile":
        try:
            preprocessing_fn = get_prediction_augmentation(
                normalization_type=normalization_type,
                lo=normalization_params["lo"],
                hi=normalization_params["hi"],
                axis=normalization_params["axis"],
                enforce_n_channels=enforce_n_channels,
            )
        except KeyError:
            preprocessing_fn = get_prediction_augmentation(
                normalization_type=normalization_type,
                lo=normalization_params["lo"],
                hi=normalization_params["hi"],
                enforce_n_channels=enforce_n_channels,
            )
    elif normalization_type == "mean_std":
        preprocessing_fn = get_prediction_augmentation(
            normalization_type=normalization_type,
            mean=normalization_params["mean"],
            std=normalization_params["std"],
            enforce_n_channels=enforce_n_channels,
        )
    elif normalization_type == "data_range":
        preprocessing_fn = get_prediction_augmentation(
            normalization_type=normalization_type,
            enforce_n_channels=enforce_n_channels,
        )
    else:
        preprocessing_fn = get_prediction_augmentation(
            normalization_type=normalization_type,
            enforce_n_channels=enforce_n_channels,
        )

    return preprocessing_fn


def get_mean_and_std(image_path):
    image = image_handling.read_tiff_file(image_path, [2])
    return np.mean(image), np.std(image)


# def enforce_n_channels(image, n_channels):
#     if not isinstance(image, torch.Tensor):
#         image = torch.tensor(image, dtype=torch.float32)

#     assert (
#         len(image.shape) <= 3
#     ), "Currently, multichannel zstacks are not supported"

#     if len(image.shape) == 2:
#         image = image.unsqueeze(0)

#     current_channels = image.shape[0]

#     # Assuming grayscale_img has a shape of (C, H, W)
#     if current_channels == n_channels:
#         return image

#     if current_channels > n_channels:
#         raise ValueError(
#             f"The image has more channels than the specified number of channels ({n_channels})"
#         )

#     if n_channels % current_channels == 0:
#         return image.repeat((n_channels // current_channels, 1, 1))

#     else:
#         # First repeat the maximum number of times that divides evenly
#         base_repeats = n_channels // current_channels
#         remaining_channels = n_channels % current_channels

#         # Create the base repeated tensor
#         repeated = image.repeat((base_repeats, 1, 1))

#         # Add the remaining channels by selecting from the beginning
#         remaining = image[:remaining_channels]

#         # Concatenate along the channel dimension
#         return torch.cat([repeated, remaining], dim=0)


def enforce_n_channels(image, n_channels):
    if not isinstance(image, np.ndarray):
        image = np.array(image, dtype=np.float32)

    assert len(image.shape) <= 3, "Currently, multichannel zstacks are not supported"

    # Add channel dimension if necessary
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=0)

    current_channels = image.shape[0]

    # Return if already correct number of channels
    if current_channels == n_channels:
        return image

    if current_channels > n_channels:
        raise ValueError(
            f"The image has more channels than the specified number of channels ({n_channels})"
        )

    if n_channels % current_channels == 0:
        # Use np.tile instead of torch.repeat
        return np.tile(image, (n_channels // current_channels, 1, 1))
    else:
        # First repeat the maximum number of times that divides evenly
        base_repeats = n_channels // current_channels
        remaining_channels = n_channels % current_channels

        # Create the base repeated array
        repeated = np.tile(image, (base_repeats, 1, 1))

        # Add the remaining channels by selecting from the beginning
        remaining = image[:remaining_channels]

        # Concatenate along the channel dimension
        return np.concatenate([repeated, remaining], axis=0)
