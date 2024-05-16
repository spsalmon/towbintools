import numpy as np
import xgboost

from towbintools.foundation import image_handling, worm_features
from typing import Callable
from joblib import Parallel, delayed
from towbintools.foundation.image_handling import check_if_zstack


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

def compute_features_of_label(current_label, mask_plane, image_plane, all_features, extra_properties, intensity_features, extra_intensity_features, num_closest=None, patches=None):
    """
    Compute a set of features for a single label, including context features and patch features.

    Parameters:
        current_label (int): The label of the current region.
        mask_plane (np.ndarray): The mask of all regions.
        image_plane (np.ndarray): The intensity image.
        all_features (list): The list of features to compute.
        extra_properties (list): The list of extra properties to compute.
        intensity_features (list): The list of intensity features to compute.
        extra_intensity_features (list): The list of extra intensity features to compute.
        num_closest (int): The number of closest regions to consider.
        patches (list): The list of patch sizes to consider.

    Returns:
        list: A list of features for the label.
    """

    mask_of_current_label = (mask_plane == current_label).astype("uint8")
    # check if image_plane has multiple channels
    if len(image_plane.shape) == 3:
        # compute all the features on the first channel and then intensity features on the other ones
        feature_vector = worm_features.compute_base_label_features(mask_of_current_label, image_plane[0], all_features, extra_properties)
        for i in range(1, image_plane.shape[0]):
            intensity_features = worm_features.compute_base_label_features(mask_of_current_label, image_plane[i], intensity_features, extra_intensity_features)
            feature_vector += intensity_features
    else:
        feature_vector = worm_features.compute_base_label_features(mask_of_current_label, image_plane, all_features, extra_properties)

    if patches is not None:
        for patch_size in patches:
            patch_features = worm_features.compute_patch_features(mask_of_current_label, image_plane, patch_size=patch_size)
            feature_vector += patch_features

    if num_closest is not None:
        context = worm_features.get_context(current_label, mask_of_current_label, mask_plane, num_closest=num_closest)
        context_features = worm_features.get_context_features(context, image_plane, all_features, extra_properties)
        feature_vector += context_features

    return feature_vector

def compute_features_of_plane(mask_plane, image_plane, all_features, extra_properties, intensity_features, extra_intensity_features, num_closest=None, patches=None, parallel=True, n_jobs=-1):
    if parallel:
        features_of_all_labels = Parallel(n_jobs=n_jobs)(delayed(compute_features_of_label)(current_label, mask_plane, image_plane, all_features, extra_properties, intensity_features, extra_intensity_features, num_closest=num_closest, patches=patches) for current_label in np.unique(mask_plane)[1:])
    else:
        features_of_all_labels = [compute_features_of_label(current_label, mask_plane, image_plane, all_features, extra_properties, intensity_features, extra_intensity_features, num_closest=num_closest, patches=patches) for current_label in np.unique(mask_plane)[1:]]
    return features_of_all_labels
    
def classify_labels(mask, image, classifier, all_features, extra_properties, intensity_features, extra_intensity_features, num_closest=None, patches=None, parallel=True, n_jobs=-1):
    if check_if_zstack(image) or len(image.shape) > 3:
        assert mask.shape[0] == image.shape[0], "The number of planes in the mask and the image should be the same."
        predictions = []
        for i in range(image.shape[0]):
            features = compute_features_of_plane(mask[i], image[i], all_features, extra_properties, intensity_features, extra_intensity_features, num_closest=num_closest, patches=patches, parallel=parallel, n_jobs=n_jobs)
            prediction = classifier.predict_proba(features)
            predictions.append(prediction)
    else:
        features = compute_features_of_plane(mask, image, all_features, extra_properties, intensity_features, extra_intensity_features, num_closest=num_closest, patches=patches, parallel=parallel, n_jobs=n_jobs)
        predictions = classifier.predict_proba(features)
    return predictions