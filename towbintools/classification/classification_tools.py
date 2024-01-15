import numpy as np
import xgboost

from towbintools.foundation import image_handling, worm_features
from typing import Callable


def classify_worm_type(
    straightened_mask: np.ndarray,
    pixelsize: float,
    classifier: xgboost.XGBClassifier,
    classes: list = ["worm", "egg", "error"],
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
    worm_type_features = np.array(
        [worm_features.compute_worm_type_features(straightened_mask, pixelsize)]
    )
    type_prediction = classifier.predict_proba(worm_type_features).squeeze()
    # convert proba to one hot encoding
    pred_class = np.argmax(type_prediction)
    prediction = classes[pred_class]
    return prediction


def classify_image(
    image: np.ndarray,
    features_function: Callable,
    classifier: xgboost.XGBClassifier,
    classes: list,
    **kwargs,
):
    """
    Classify images based on extracted features using a provided classifier.

    Parameters:
        images (np.ndarray): An array of images or a single image for feature extraction.
        features_function (callable): A function to extract features from the images.
        classifier (xgboost.XGBClassifier): The trained classifier object.
        classes (list[str]): List of classes that can be predicted by the classifier.
        return_proba (bool, optional): If True, return class probabilities instead of labels.
        **kwargs: Additional keyword arguments to pass to the features_function.

    Returns:
        np.ndarray or str: The predicted class of the image(s) or class probabilities.
    """

    # feature extraction
    try:
        features = features_function(image, **kwargs)
    except Exception as e:
        raise Exception(f"Error extracting features from image. {e}")
    # classification
    try:
        prediction = classifier.predict_proba(features).squeeze()
    except Exception as e:
        raise Exception(f"Error predicting class of image. {e}")

    assert len(prediction) == len(
        classes
    ), f"Number of provided classes and predicted classes do not match. len(prediction) = {len(prediction)}, len(classes) = {len(classes)}"
    # convert proba to one hot encoding
    pred_class = np.argmax(prediction)
    prediction = classes[pred_class]
    return prediction
