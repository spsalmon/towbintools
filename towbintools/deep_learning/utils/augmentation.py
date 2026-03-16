import numpy as np
from csbdeep.utils import normalize
from monai.data import set_track_meta
from monai.transforms import Compose
from monai.transforms import MapTransform
from monai.transforms import RandGaussianSharpend
from monai.transforms import RandGaussianSmoothd
from monai.transforms import Randomizable
from monai.transforms import Transform

from towbintools.foundation import image_handling

# Avoid MetaTensor overhead when working with plain numpy arrays
set_track_meta(False)


# ---------------------------------------------------------------------------
# Intensity transforms
# ---------------------------------------------------------------------------


class NormalizeDataRange(MapTransform):
    def __init__(self, keys):
        MapTransform.__init__(self, keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if not isinstance(d[key], np.ndarray):
                d[key] = np.array(d[key])
            d[key] = (d[key] - d[key].min()) / (d[key].max() - d[key].min())
        return d


class NormalizeMeanStd(MapTransform):
    def __init__(self, keys, mean: float, std: float):
        MapTransform.__init__(self, keys)
        self.mean = mean
        self.std = std

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if not isinstance(d[key], np.ndarray):
                d[key] = np.array(d[key])
            d[key] = (d[key] - self.mean) / self.std
        return d


class NormalizePercentile(MapTransform):
    def __init__(self, keys, lo: float, hi: float, axis=None):
        MapTransform.__init__(self, keys)
        self.lo = lo
        self.hi = hi
        self.axis = axis

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if not isinstance(d[key], np.ndarray):
                d[key] = np.array(d[key])
            d[key] = normalize(d[key], self.lo, self.hi, axis=self.axis)
        return d


class EnforceNChannels(MapTransform):
    def __init__(self, keys, n_channels: int):
        MapTransform.__init__(self, keys)
        self.n_channels = n_channels

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if not isinstance(d[key], np.ndarray):
                d[key] = np.array(d[key])
            d[key] = _enforce_n_channels(d[key], self.n_channels)
        return d


# ---------------------------------------------------------------------------
# Geometric transforms (image + mask)
# ---------------------------------------------------------------------------


class CustomFlip(MapTransform, Randomizable):
    _FLIP_OPTIONS = [(-2,), (-1,), (-1, -2)]

    def __init__(self, keys, prob=0.75):
        MapTransform.__init__(self, keys)
        self.prob = prob
        self._do_transform = False
        self._axes = None

    def randomize(self, data=None):
        self._do_transform = self.R.random() < self.prob
        self._axes = self._FLIP_OPTIONS[self.R.randint(3)]

    def __call__(self, data):
        self.randomize()
        d = dict(data)
        if not self._do_transform:
            return d
        for key in self.keys:
            d[key] = np.flip(d[key], axis=self._axes)
        return d


class CustomRotate90(MapTransform, Randomizable):
    def __init__(self, keys, prob=0.75):
        MapTransform.__init__(self, keys)
        self.prob = prob
        self._do_transform = False
        self._k = None

    def randomize(self, data=None):
        self._do_transform = self.R.random() < self.prob
        self._k = self.R.choice([1, 2, 3])

    def __call__(self, data):
        self.randomize()
        d = dict(data)
        if not self._do_transform:
            return d
        for key in self.keys:
            d[key] = np.rot90(d[key], k=self._k, axes=(-2, -1))
        return d


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _build_normalization(keys, normalization_type: str, **kwargs) -> Transform:
    if normalization_type == "data_range":
        return NormalizeDataRange(keys)
    elif normalization_type == "mean_std":
        return NormalizeMeanStd(keys, kwargs["mean"], kwargs["std"])
    elif normalization_type == "percentile":
        return NormalizePercentile(keys, kwargs["lo"], kwargs["hi"], kwargs.get("axis"))
    else:
        raise ValueError(f"Unknown normalization type: {normalization_type}")


def get_training_augmentation(normalization_type: str, **kwargs) -> Compose:
    transforms = [
        CustomFlip(keys=["image", "mask"], prob=0.75),
        CustomRotate90(keys=["image", "mask"], prob=0.75),
        RandGaussianSmoothd(
            keys=["image"], prob=0.5, sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5)
        ),
        RandGaussianSharpend(keys=["image"], prob=0.5),
        _build_normalization(
            keys=["image"], normalization_type=normalization_type, **kwargs
        ),
    ]

    if (n := kwargs.get("enforce_n_channels")) is not None:
        transforms.append(EnforceNChannels(n))

    return Compose(transforms)


def get_qc_training_augmentation(normalization_type: str, **kwargs) -> Compose:
    transforms = [
        CustomFlip(keys=["image"], prob=0.75),
        _build_normalization(
            keys=["image"], normalization_type=normalization_type, **kwargs
        ),
    ]

    if (n := kwargs.get("enforce_n_channels")) is not None:
        transforms.append(EnforceNChannels(n))

    return Compose(transforms)


def get_prediction_augmentation(normalization_type: str, **kwargs) -> Compose:
    transforms = [
        _build_normalization(
            keys=["image"], normalization_type=normalization_type, **kwargs
        )
    ]

    if (n := kwargs.get("enforce_n_channels")) is not None:
        transforms.append(EnforceNChannels(n))

    return Compose(transforms)


def get_prediction_augmentation_from_model(model, enforce_n_channels=None) -> Compose:
    params = model.normalization
    return get_prediction_augmentation(
        normalization_type=params["type"],
        enforce_n_channels=enforce_n_channels,
        **{k: v for k, v in params.items() if k != "type"},
    )


def get_mean_and_std(image_path: str) -> tuple[float, float]:
    image = image_handling.read_tiff_file(image_path, [2])
    return float(np.mean(image)), float(np.std(image))


# ---------------------------------------------------------------------------
# Internal utility
# ---------------------------------------------------------------------------


def _enforce_n_channels(image: np.ndarray, n_channels: int) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        image = np.array(image, dtype=np.float32)

    assert image.ndim <= 3, "Multichannel z-stacks are not supported"

    if image.ndim == 2:
        image = np.expand_dims(image, axis=0)

    c = image.shape[0]
    if c == n_channels:
        return image
    if c > n_channels:
        raise ValueError(f"Image has {c} channels, expected at most {n_channels}")

    base = n_channels // c
    remainder = n_channels % c

    repeated = np.tile(image, (base, 1, 1))
    if remainder == 0:
        return repeated
    return np.concatenate([repeated, image[:remainder]], axis=0)
