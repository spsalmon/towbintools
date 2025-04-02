from typing import Callable

import numpy as np
import pandas as pd
import xgboost
from joblib import delayed
from joblib import Parallel

from towbintools.foundation import worm_features


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
        [worm_features.compute_mask_type_features(straightened_mask, pixelsize)]
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


def compute_features_of_label(
    current_label,
    mask_plane,
    image_plane,
    all_features,
    extra_properties,
    intensity_features,
    extra_intensity_features,
    num_closest=None,
    patches=None,
):
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
        feature_vector = worm_features.compute_base_label_features(
            mask_of_current_label,
            image_plane[0],
            all_features,
            extra_properties,
        )
        for i in range(1, image_plane.shape[0]):
            other_channel_intensity_features = (
                worm_features.compute_base_label_features(
                    mask_of_current_label,
                    image_plane[i],
                    intensity_features,
                    extra_intensity_features,
                )
            )
            feature_vector += other_channel_intensity_features
    else:
        feature_vector = worm_features.compute_base_label_features(
            mask_of_current_label, image_plane, all_features, extra_properties
        )

    if patches is not None:
        for patch_size in patches:
            if len(image_plane.shape) == 3:
                patch_features = worm_features.compute_patch_features(
                    mask_of_current_label,
                    image_plane[0],
                    patch_size=patch_size,
                )
                for i in range(1, image_plane.shape[0]):
                    patch_features += worm_features.compute_patch_features(
                        mask_of_current_label,
                        image_plane[i],
                        patch_size=patch_size,
                    )
                feature_vector += patch_features
            else:
                patch_features = worm_features.compute_patch_features(
                    mask_of_current_label, image_plane, patch_size=patch_size
                )
                feature_vector += patch_features

    if num_closest is not None:
        context = worm_features.get_context(
            current_label,
            mask_of_current_label,
            mask_plane,
            num_closest=num_closest,
        )
        if len(image_plane.shape) == 3:
            context_features = worm_features.get_context_features(
                context, image_plane[0], all_features, extra_properties
            )
            for i in range(1, image_plane.shape[0]):
                context_features += worm_features.get_context_features(
                    context,
                    image_plane[i],
                    intensity_features,
                    extra_intensity_features,
                )
            feature_vector += context_features
        else:
            context_features = worm_features.get_context_features(
                context, image_plane, all_features, extra_properties
            )
            feature_vector += context_features

    return feature_vector


def compute_features_of_plane(
    mask_plane,
    image_plane,
    all_features,
    extra_properties,
    intensity_features,
    extra_intensity_features,
    num_closest=None,
    patches=None,
    parallel=True,
    n_jobs=-1,
):
    """
    Compute a set of features for a single label, including context features and patch features for all labels in a plane.

    Parameters:
        mask_plane (np.ndarray): The mask of all regions.
        image_plane (np.ndarray): The intensity image.
        all_features (list): The list of features to compute.
        extra_properties (list): The list of extra properties to compute.
        intensity_features (list): The list of intensity features to compute.
        extra_intensity_features (list): The list of extra intensity features to compute.
        num_closest (int): The number of closest regions to consider.
        patches (list): The list of patch sizes to consider.
        parallel (bool): Whether to compute features in parallel.
        n_jobs (int): The number of jobs to run in parallel.

    Returns:
        list: A list of lists of features for all labels.
    """

    if parallel:
        features_of_all_labels = Parallel(n_jobs=n_jobs)(
            delayed(compute_features_of_label)(
                current_label,
                mask_plane,
                image_plane,
                all_features,
                extra_properties,
                intensity_features,
                extra_intensity_features,
                num_closest=num_closest,
                patches=patches,
            )
            for current_label in np.unique(mask_plane)[1:]
        )
    else:
        features_of_all_labels = [
            compute_features_of_label(
                current_label,
                mask_plane,
                image_plane,
                all_features,
                extra_properties,
                intensity_features,
                extra_intensity_features,
                num_closest=num_closest,
                patches=patches,
            )
            for current_label in np.unique(mask_plane)[1:]
        ]
    return features_of_all_labels


def classify_plane(
    mask_plane,
    image_plane,
    classifier,
    all_features,
    extra_properties,
    intensity_features,
    extra_intensity_features,
    num_closest=None,
    patches=None,
    parallel=True,
    n_jobs=-1,
    confidence_threshold=None,
):
    """
    Compute the features of all the labels in a plane and classify them using an XGBoost classifier.

    Parameters:
        mask_plane (np.ndarray): The mask of all regions.
        image_plane (np.ndarray): The intensity image.
        classifier (xgboost.XGBClassifier): The trained classifier object.
        all_features (list): The list of features to compute.
        extra_properties (list): The list of extra properties to compute.
        intensity_features (list): The list of intensity features to compute.
        extra_intensity_features (list): The list of extra intensity features to compute.
        num_closest (int): The number of closest regions to consider.
        patches (list): The list of patch sizes to consider.
        parallel (bool): Whether to compute features in parallel.
        n_jobs (int): The number of jobs to run in parallel.
        confidence_threshold (float): The confidence threshold for predictions to be considered valid.

    Returns:
        list: A list of predicted classes for all labels in the plane.
    """

    features = compute_features_of_plane(
        mask_plane,
        image_plane,
        all_features,
        extra_properties,
        intensity_features,
        extra_intensity_features,
        num_closest=num_closest,
        patches=patches,
        parallel=parallel,
        n_jobs=n_jobs,
    )
    if len(features) == 0:
        return None
    predictions = classifier.predict_proba(features)
    predicted_classes = np.argmax(predictions, axis=1)
    if confidence_threshold is not None:
        for i in range(len(predicted_classes)):
            if np.max(predictions[i]) < confidence_threshold:
                predicted_classes[i] = -1
    return predicted_classes


def classify_labels(
    mask,
    image,
    classifier,
    all_features,
    extra_properties,
    intensity_features,
    extra_intensity_features,
    num_closest=None,
    patches=None,
    parallel=True,
    n_jobs=-1,
    is_zstack=False,
    confidence_threshold=None,
):
    """
    Compute the features of all the labels in a mask and classify them using an XGBoost classifier.

    Parameters:
        mask (np.ndarray): The mask of all regions.
        image (np.ndarray): The intensity image.
        classifier (xgboost.XGBClassifier): The trained classifier object.
        all_features (list): The list of features to compute.
        extra_properties (list): The list of extra properties to compute.
        intensity_features (list): The list of intensity features to compute.
        extra_intensity_features (list): The list of extra intensity features to compute.
        num_closest (int): The number of closest regions to consider.
        patches (list): The list of patch sizes to consider.
        parallel (bool): Whether to compute features in parallel.
        n_jobs (int): The number of jobs to run in parallel.
        is_zstack (bool): Whether the image is a z-stack.
        confidence_threshold (float): The confidence threshold for predictions to be considered valid.

    Returns:
        list: A list of predicted classes for all labels in the mask.
    """

    if is_zstack or len(image.shape) > 3:
        assert (
            mask.shape[0] == image.shape[0]
        ), "The number of planes in the mask and the image should be the same."
        return [
            classify_plane(
                mask_plane,
                image_plane,
                classifier,
                all_features,
                extra_properties,
                intensity_features,
                extra_intensity_features,
                num_closest=num_closest,
                patches=patches,
                parallel=parallel,
                n_jobs=n_jobs,
                confidence_threshold=confidence_threshold,
            )
            for mask_plane, image_plane in zip(mask, image)
        ]
    else:
        return classify_plane(
            mask,
            image,
            classifier,
            all_features,
            extra_properties,
            intensity_features,
            extra_intensity_features,
            num_closest=num_closest,
            patches=patches,
            parallel=parallel,
            n_jobs=n_jobs,
            confidence_threshold=confidence_threshold,
        )


def classify_labels_features_dict(
    mask,
    image,
    clf,
    features_dict,
    parallel=True,
    n_jobs=-1,
    is_zstack=False,
    confidence_threshold=None,
):
    return classify_labels(
        mask,
        image,
        clf,
        features_dict["all_features"],
        features_dict["extra_properties"],
        features_dict["intensity_features"],
        features_dict["extra_intensity_features"],
        num_closest=features_dict["num_closest"],
        patches=features_dict["patches"],
        parallel=parallel,
        n_jobs=n_jobs,
        is_zstack=is_zstack,
        confidence_threshold=confidence_threshold,
    )


def convert_classification_to_mask(mask, classification, is_zstack=False):
    """
    Convert a classification (list of predicted classes) to a mask.

    Parameters:
        mask (np.ndarray): The mask of all regions.
        classification (list): The list of predicted classes for all labels.
        is_zstack (bool): Whether the image is a z-stack.

    Returns:
        np.ndarray: The given mask with pixel values replaced with class number + 1.
    """

    new_mask = np.zeros_like(mask)

    if is_zstack or len(mask.shape) > 2:
        for i, plane_classification in enumerate(classification):
            if plane_classification is not None:
                for j, label in enumerate(np.unique(mask[i])[1:]):
                    new_mask[i][mask[i] == label] = plane_classification[j] + 1
    else:
        if classification is not None:
            for i, label in enumerate(np.unique(mask)[1:]):
                new_mask[mask == label] = classification[i] + 1

    return new_mask


def convert_classification_to_dataframe(mask, classification, is_zstack=False):
    """
    Convert a classification (list of predicted classes) to a pandas DataFrame.

    Parameters:
        mask (np.ndarray): The mask of all regions.
        classification (list): The list of predicted classes for all labels.
        is_zstack (bool): Whether the image is a z-stack.

    Returns:
        pd.DataFrame: A DataFrame with columns "Plane", "Label", and "Class".
    """

    data = []
    if is_zstack or len(mask.shape) > 2:
        for i, plane_classification in enumerate(classification):
            if plane_classification is not None:
                for j, label in enumerate(np.unique(mask[i])[1:]):
                    data.append(
                        {
                            "Plane": i,
                            "Label": int(label),
                            "Class": plane_classification[j],
                        }
                    )
    else:
        if classification is not None:
            for i, label in enumerate(np.unique(mask)[1:]):
                data.append(
                    {
                        "Plane": 0,
                        "Label": int(label),
                        "Class": classification[i],
                    }
                )
    return pd.DataFrame(data)


def classify_labels_and_convert_to_mask(
    mask,
    image,
    classifier,
    all_features,
    extra_properties,
    intensity_features,
    extra_intensity_features,
    num_closest=None,
    patches=None,
    parallel=True,
    n_jobs=-1,
    is_zstack=False,
    confidence_threshold=None,
):
    """
    Classify all the labels in a mask using an XGBoost classifier and convert the classification to a mask.

    Parameters:
        mask (np.ndarray): The mask of all regions.
        image (np.ndarray): The intensity image.
        classifier (xgboost.XGBClassifier): The trained classifier object.
        all_features (list): The list of features to compute.
        extra_properties (list): The list of extra properties to compute.
        intensity_features (list): The list of intensity features to compute.
        extra_intensity_features (list): The list of extra intensity features to compute.
        num_closest (int): The number of closest regions to consider.
        patches (list): The list of patch sizes to consider.
        parallel (bool): Whether to compute features in parallel.
        n_jobs (int): The number of jobs to run in parallel.
        is_zstack (bool): Whether the image is a z-stack.
        confidence_threshold (float): The confidence threshold for predictions to be considered valid.

    Returns:
        np.ndarray: The given mask with pixel values replaced with class number + 1.
    """

    classification = classify_labels(
        mask,
        image,
        classifier,
        all_features,
        extra_properties,
        intensity_features,
        extra_intensity_features,
        num_closest=num_closest,
        patches=patches,
        parallel=parallel,
        n_jobs=n_jobs,
        is_zstack=is_zstack,
        confidence_threshold=confidence_threshold,
    )
    return convert_classification_to_mask(mask, classification)


def classify_labels_and_convert_to_dataframe(
    mask,
    image,
    classifier,
    all_features,
    extra_properties,
    intensity_features,
    extra_intensity_features,
    num_closest=None,
    patches=None,
    parallel=True,
    n_jobs=-1,
    is_zstack=False,
    confidence_threshold=None,
):
    """
    Classify all the labels in a mask using an XGBoost classifier and convert the classification to a pandas DataFrame.

    Parameters:
        mask (np.ndarray): The mask of all regions.
        image (np.ndarray): The intensity image.
        classifier (xgboost.XGBClassifier): The trained classifier object.
        all_features (list): The list of features to compute.
        extra_properties (list): The list of extra properties to compute.
        intensity_features (list): The list of intensity features to compute.
        extra_intensity_features (list): The list of extra intensity features to compute.
        num_closest (int): The number of closest regions to consider.
        patches (list): The list of patch sizes to consider.
        parallel (bool): Whether to compute features in parallel.
        n_jobs (int): The number of jobs to run in parallel.
        is_zstack (bool): Whether the image is a z-stack.
        confidence_threshold (float): The confidence threshold for predictions to be considered valid.

    Returns:
        pd.DataFrame: A DataFrame with columns "Plane", "Label", and "Class".
    """

    classification = classify_labels(
        mask,
        image,
        classifier,
        all_features,
        extra_properties,
        intensity_features,
        extra_intensity_features,
        num_closest=num_closest,
        patches=patches,
        parallel=parallel,
        n_jobs=n_jobs,
        is_zstack=is_zstack,
        confidence_threshold=confidence_threshold,
    )
    return convert_classification_to_dataframe(mask, classification)


def classify_labels_and_convert_to_mask_features_dict(
    mask,
    image,
    clf,
    features_dict,
    parallel=True,
    n_jobs=-1,
    is_zstack=False,
    confidence_threshold=None,
):
    classification = classify_labels_features_dict(
        mask,
        image,
        clf,
        features_dict,
        parallel=parallel,
        n_jobs=n_jobs,
        is_zstack=is_zstack,
        confidence_threshold=confidence_threshold,
    )
    return convert_classification_to_mask(mask, classification, is_zstack=is_zstack)


def classify_labels_and_convert_to_dataframe_features_dict(
    mask,
    image,
    clf,
    features_dict,
    parallel=True,
    n_jobs=-1,
    is_zstack=False,
    confidence_threshold=None,
):
    classification = classify_labels_features_dict(
        mask,
        image,
        clf,
        features_dict,
        parallel=parallel,
        n_jobs=n_jobs,
        is_zstack=is_zstack,
        confidence_threshold=confidence_threshold,
    )
    return convert_classification_to_dataframe(
        mask, classification, is_zstack=is_zstack
    )
