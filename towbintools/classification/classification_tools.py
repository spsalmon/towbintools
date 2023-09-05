import numpy as np
import xgboost

from towbintools.foundation import image_handling, worm_features


def classify_worm_type(straightened_mask: np.ndarray, pixelsize: float, 
                       classifier: xgboost.XGBClassifier, 
                       classes: list = ['worm', 'egg', 'error'],) -> str:
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
    worm_type_features = np.array([worm_features.compute_worm_type_features(straightened_mask, pixelsize)])
    type_prediction = classifier.predict_proba(worm_type_features).squeeze()
    # convert proba to one hot encoding
    pred_class = np.argmax(type_prediction)
    prediction = classes[pred_class]
    return prediction

