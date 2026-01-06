import numpy as np
import pandas as pd
import xgboost
from csbdeep.utils import normalize
from skimage.measure import regionprops_table
from skimage.measure._regionprops import RegionProperties

from towbintools.foundation.image_handling import pad_images_to_same_dim
from towbintools.foundation.image_quality import normalized_variance_measure


def get_all_skimage_regionprops():
    features = [
        attr
        for attr in dir(RegionProperties)
        if not attr.startswith("_")
        and isinstance(getattr(RegionProperties, attr), property)
        if "image" not in attr and "coords" not in attr
    ]
    return features


def compute_qc_features(
    image: np.ndarray,
    mask: np.ndarray,
    pixelsize: float,
    features: list[str] = get_all_skimage_regionprops(),
):
    try:
        mask = (mask > 0).astype(np.uint8)
        if np.max(mask) == 0:
            return None

        if image.shape != mask.shape:
            image, mask = pad_images_to_same_dim(image, mask)

        # normalize image
        image = normalize(image, 1, 99, axis=None)
        # ensure mask is binary
        mask = (mask > 0).astype(np.uint8)
        props = regionprops_table(
            mask, intensity_image=image, properties=features, spacing=pixelsize
        )
        props_df = pd.DataFrame(props)
        other_features = {
            "NORMALIZED_VARIANCE_MEASURE": normalized_variance_measure(image),
        }

        other_features_df = pd.DataFrame([other_features])
        props_df = pd.concat([props_df, other_features_df], axis=1)
        return props_df
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return None


def classify_worm_type(
    image: np.ndarray,
    mask: np.ndarray,
    pixelsize: float,
    classifier: xgboost.XGBClassifier,
    classes: list = ["egg", "worm", "error"],
) -> str:
    """
    Classify the type of worm based on extracted features using an XGBoost classifier.

    The function extracts features from the provided straightened mask, then utilizes
    the XGBoost classifier to predict the type of worm. The classifier is assumed
    to return class probabilities, which are then converted to a one-hot encoding
    format to derive the final prediction.

    Parameters:
        straightened_mask (np.ndarray): The straightened worm mask for feature extraction.
        pixelsize (float): The pixel size for feature computations.
        classifier (xgboost.XGBClassifier): The trained XGBoost classifier object.
        classes (list[str], optional): List of classes that can be predicted by the classifier.
                                       Default is ['worm', 'egg', 'error'].

    Returns:
        str: The predicted class of the worm ('worm', 'egg', or 'error').
    """

    features = compute_qc_features(
        image=image,
        mask=mask,
        pixelsize=pixelsize,
    )
    if features is None or features.empty:
        return "error"

    features_df = pd.DataFrame(features)
    # predict class probabilities
    class_probs = classifier.predict_proba(features_df)
    # convert to one-hot encoding
    one_hot_preds = np.zeros_like(class_probs)
    one_hot_preds[np.arange(len(class_probs)), np.argmax(class_probs, axis=1)] = 1
    # get predicted class
    pred_class = classes[np.argmax(one_hot_preds)]
    return pred_class
