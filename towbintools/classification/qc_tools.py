from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
from csbdeep.utils import normalize
from skimage.measure import regionprops_table
from skimage.measure._regionprops import RegionProperties

from towbintools.foundation.image_handling import pad_images_to_same_dim
from towbintools.foundation.image_handling import read_tiff_file
from towbintools.foundation.image_quality import normalized_variance_measure


def get_all_skimage_regionprops(mask_only: bool = False) -> list[str]:
    features = [
        attr
        for attr in dir(RegionProperties)
        if not attr.startswith("_")
        and isinstance(getattr(RegionProperties, attr), property)
        if "image" not in attr and "coords" not in attr
    ]
    # remove intensity based features if mask_only is True
    if mask_only:
        intensity_features = [
            "centroid_weighted",
            "centroid_weighted_local",
            "intensity_min",
            "intensity_max",
            "intensity_mean",
            "intensity_median",
            "intensity_std",
            "moments_weighted",
            "moments_weighted_central",
            "moments_weighted_hu",
            "moments_weighted_normalized",
        ]
        features = [f for f in features if f not in intensity_features]
    return features


def compute_qc_features(
    mask: Union[str, np.ndarray],
    image: Union[str, np.ndarray, None],
    features: Optional[list[str]] = None,
    channels: Optional[list[int]] = None,
):
    try:
        if isinstance(image, str):
            image = read_tiff_file(image, channels_to_keep=channels)
            # skimage expects the channel dimension last, in our case it's first if the image is not a stack
            if image.ndim == 3:
                image = np.transpose(image, (1, 2, 0))
            if image.ndim > 3:
                raise ValueError(
                    "Input image has more than 3 dimensions, which is currently not supported."
                )
        if isinstance(mask, str):
            mask = read_tiff_file(mask)

        mask = (mask > 0).astype(np.uint8)
        if np.max(mask) == 0:
            return None

        if features is None:
            features = get_all_skimage_regionprops(
                mask_only=image is None,
            )

        if image is not None:
            if image.shape != mask.shape:
                image, mask = pad_images_to_same_dim(image, mask)

            # normalize image
            image = normalize(image, 1, 99, axis=None)
            props = regionprops_table(
                mask,
                intensity_image=image,
                properties=features,
            )
            props_df = pd.DataFrame(props)
            other_features = {
                "NORMALIZED_VARIANCE_MEASURE": normalized_variance_measure(image),
            }

            other_features_df = pd.DataFrame([other_features])
            props_df = pd.concat([props_df, other_features_df], axis=1)
        else:
            props = regionprops_table(
                mask,
                properties=features,
            )
            props_df = pd.DataFrame(props)
        return props_df
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return None
