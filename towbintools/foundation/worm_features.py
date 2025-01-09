import numpy as np
from skimage.measure import regionprops, shannon_entropy
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
from skimage.util import img_as_ubyte
from skimage.measure import regionprops_table

AVAILABLE_WORM_FEATURES = [
    "length",
    "volume",
    "area",
    "width_mean",
    "width_median",
    "width_std",
    "width_cv",
    "width_skew",
    "width_kurtosis",
    "width_max",
    "width_middle",
]

FEATURES_TO_COMPUTE_AT_MOLT = [
    "volume",
    "length",
    "area",
    "fluo",
    "width",
]

def get_available_worm_features():
    """
    Get a list of available worm features.

    Returns:
        list: The list of available worm features.
    """
    return AVAILABLE_WORM_FEATURES

def get_features_to_compute_at_molt():
    """
    Get a list of features to compute at molt.

    Returns:
        list: The list of features to compute at molt.
    """
    return FEATURES_TO_COMPUTE_AT_MOLT

def compute_worm_volume(
    straightened_worm_mask: np.ndarray,
    pixelsize: float,
) -> float:
    """
    Compute the volume of a straightened worm mask using its radius and pixel size.

    Parameters:
        straightened_worm_mask (np.ndarray): The straightened worm mask as a NumPy array.
        pixelsize (float): The size of a pixel, used for volume calculation.

    Returns:
        float: The computed volume of the worm.
    """

    worm_radius = np.sum(straightened_worm_mask, axis=0) / 2
    return np.sum(np.pi * (worm_radius**2)) * (pixelsize**3)


def compute_worm_area(
    worm_mask: np.ndarray,
    pixelsize: float,
) -> float:
    """
    Compute the area of a worm mask.

    Parameters:
        worm_mask (np.ndarray): The worm mask as a NumPy array.
        pixelsize (float): The size of a pixel, used for area calculation.

    Returns:
        float: The computed are of the worm.
    """

    return np.sum(worm_mask) * (pixelsize**2)


def compute_worm_length(
    straightened_worm_mask: np.ndarray,
    pixelsize: float,
) -> float:
    """
    Compute the length of a straightened worm mask using the sum of its pixels along the 0-axis.

    Parameters:
        straightened_worm_mask (np.ndarray): The straightened worm mask as a NumPy array.
        pixelsize (float): The size of a pixel, used for length calculation.

    Returns:
        float: The computed length of the worm.
    """
    return np.sum(np.sum(straightened_worm_mask, axis=0) > 0) * pixelsize

def compute_worm_average_width(
    straightened_worm_mask: np.ndarray,
    pixelsize: float,
    aggregation: str = "mean",
) -> float:
    """
    Compute the average width of a straightened worm mask.

    Parameters:
        straightened_worm_mask (np.ndarray): The straightened worm mask as a NumPy array.
        pixelsize (float): The size of a pixel, used for width calculation.
        aggregation (str): The aggregation method to use to compute the width.
                           Can be either 'mean', 'median'
    
    Returns:
        float: The computed average width of the worm.
    """

    worm_widths = np.sum(straightened_worm_mask, axis=0) * pixelsize
    worm_widths = worm_widths[worm_widths > 0]

    if aggregation == "mean":
        return np.mean(worm_widths)
    elif aggregation == "median":
        return np.median(worm_widths)
    else:
        raise ValueError(
            'Aggregation must be one of "mean" or "median".'
        )

def compute_worm_width_profile(
    straightened_worm_mask: np.ndarray,
    pixelsize: float,
) -> np.ndarray:
    """
    Compute the width profile of a straightened worm mask.

    Parameters:
        straightened_worm_mask (np.ndarray): The straightened worm mask as a NumPy array.
        pixelsize (float): The size of a pixel, used for width calculation.

    Returns:
        np.ndarray: The computed width profile of the worm.
    """

    profile = np.sum(straightened_worm_mask, axis=0) * pixelsize
    return profile[profile > 0]

def compute_max_width(width_profile, window_size=10):
    """
    Compute the maximum width of a worm from its width profile by taking the maximum value and averaging the values around it.

    Parameters:
        width_profile (np.ndarray): The width profile of the worm.
        window_size (int): The size of the window to average around the maximum.

    Returns:
        float: The maximum width of the worm.
    """

    try:
        max_width_index = np.argmax(width_profile)
    except ValueError:
        return np.nan
    return np.mean(width_profile[max_width_index - window_size : max_width_index + window_size + 1])

def compute_mid_width(width_profile, window_size=10):
    """
    Compute the width of a worm at its midpoint from its width profile by averaging the values around the midpoint.

    Parameters:
        width_profile (np.ndarray): The width profile of the worm.
        window_size (int): The size of the window to average around the midpoint.

    Returns:
        float: The width of the worm at its midpoint.
    """

    mid_width_index = len(width_profile) // 2
    return np.mean(width_profile[mid_width_index - window_size : mid_width_index + window_size + 1])

def compute_worm_morphological_features(
    straightened_worm_mask: np.ndarray,
    pixelsize: float,
    features: list,
) -> dict:
    """
    Compute a set of morphological features for a straightened worm mask.

    Parameters:
        straightened_worm_mask (np.ndarray): The straightened worm mask as a NumPy array.
        pixelsize (float): The size of a pixel, used for various calculations.
        features (list): The list of features to compute.

    Returns:
        dict: A dictionary containing the computed features.
    """

    width_profile = compute_worm_width_profile(straightened_worm_mask, pixelsize=pixelsize)

    feature_dict = {}
    for feature in features:
        if feature == "length":
            feature_dict["length"] = compute_worm_length(straightened_worm_mask, pixelsize=pixelsize)
        elif feature == "volume":
            feature_dict["volume"] = compute_worm_volume(straightened_worm_mask, pixelsize=pixelsize)
        elif feature == "area":
            feature_dict["area"] = compute_worm_area(straightened_worm_mask, pixelsize=pixelsize)
        elif feature == "width_mean":
            feature_dict["width_mean"] = np.mean(width_profile)
        elif feature == "width_median":
            feature_dict["width_median"] = np.median(width_profile)
        elif feature == "width_std":
            feature_dict["width_std"] = np.std(width_profile)
        elif feature == "width_cv":
            feature_dict["width_cv"] = np.std(width_profile) / np.mean(width_profile)
        elif feature == "width_skew":
            feature_dict["width_skew"] = skew(width_profile)
        elif feature == "width_kurtosis":
            feature_dict["width_kurtosis"] = kurtosis(width_profile)
        elif feature == "width_max":
            feature_dict["width_max"] = compute_max_width(width_profile)
        elif feature == "width_middle":
            feature_dict["width_middle"] = compute_mid_width(width_profile)
        else:
            raise ValueError(f"Feature {feature} not recognized.")

    # set all features to NaN if they are 0
    for key in feature_dict:
        if feature_dict[key] == 0:
            feature_dict[key] = np.nan

    return feature_dict


def compute_worm_type_features(
    straightened_worm_mask: np.ndarray,
    pixelsize: float,
) -> list:
    """
    Compute a series of morphological features for a straightened worm mask including
    length, volume, volume per length, width measures, entropy, and region properties.

    Parameters:
        straightened_worm_mask (np.ndarray): The straightened worm mask as a NumPy array.
        pixelsize (float): The size of a pixel, used for various calculations.

    Returns:
        list: A list containing the computed features in the following order:
              [worm_length, worm_volume, volume_per_length, width_mean, width_std,
              width_cv, entropy_mask, eccentricity, solidity, permimeter]
    """
    # compute worm length
    worm_length = compute_worm_length(straightened_worm_mask, pixelsize=pixelsize)

    # compute worm volume
    worm_volume = compute_worm_volume(straightened_worm_mask, pixelsize=pixelsize)

    # compute worm width
    worm_widths = np.sum(straightened_worm_mask, axis=0) * pixelsize
    worm_widths = worm_widths[worm_widths > 0]

    width_std = np.std(worm_widths)
    width_mean = np.mean(worm_widths)
    width_cv = width_std / width_mean

    volume_per_length = worm_volume / worm_length

    # compute entropy
    entropy_mask = shannon_entropy(straightened_worm_mask)

    try:
        other_properties = regionprops(straightened_worm_mask.astype(np.uint8))[0]
        eccentricity = other_properties.eccentricity
        solidity = other_properties.solidity
        permimeter = other_properties.perimeter
    except IndexError:
        eccentricity = 0
        solidity = 0
        permimeter = 0

    return [
        worm_length,
        worm_volume,
        volume_per_length,
        width_mean,
        width_std,
        width_cv,
        entropy_mask,
        eccentricity,
        solidity,
        permimeter,
    ]

# Features for nuclei classification

def intensity_std(regionmask, intensity_image):
    """
    Compute the standard deviation of the intensity values within a region.
    
    Parameters:
        regionmask (np.ndarray): The mask of the region.
        intensity_image (np.ndarray): The intensity image.
        
    Returns:
        float: The standard deviation of the intensity values within the region."""
    return np.std(intensity_image[regionmask])

def intensity_skew(regionmask, intensity_image):
    """
    Compute the skewness of the intensity values within a region.

    Parameters:
        regionmask (np.ndarray): The mask of the region.
        intensity_image (np.ndarray): The intensity image.
    
    Returns:
        float: The skewness of the intensity values within the region."""
    
    return skew(intensity_image[regionmask])

def intensity_kurtosis(regionmask, intensity_image):
    """
    Compute the kurtosis of the intensity values within a region.

    Parameters:
        regionmask (np.ndarray): The mask of the region.
        intensity_image (np.ndarray): The intensity image.
    
    Returns:
        float: The kurtosis of the intensity values within the region."""
    
    return kurtosis(intensity_image[regionmask])

def compute_haralick_features(image):
    """
    Compute Haralick texture features for an image.

    Parameters:
        image (np.ndarray): The intensity image.

    Returns:
        list: A list of Haralick texture features.
    """

    # Calculate the Grey-Level Co-Occurrence Matrix
    glcm = graycomatrix(img_as_ubyte(image), distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    # Calculate properties
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    return [contrast, dissimilarity, homogeneity, energy, correlation]


def compute_patch_features(regionmask, intensity_image, patch_size=64):
    """
    Compute a set of features for a patch around a region.

    Parameters:
        regionmask (np.ndarray): The mask of the region.
        intensity_image (np.ndarray): The intensity image.
        patch_size (int): The size of the patch.

    Returns:
        list: A list of features for the patch.
    """

    # Get the centroid of the region
    centroid = regionprops(regionmask.astype("uint8"))[0].centroid
    # Create a patch of the defined size around the centroid
    minr = int(centroid[0] - patch_size/2)
    maxr = int(centroid[0] + patch_size/2)
    minc = int(centroid[1] - patch_size/2)
    maxc = int(centroid[1] + patch_size/2)

    if minr < 0:
        minr = 0
        maxr = patch_size
    if minc < 0:
        minc = 0
        maxc = patch_size
    if maxr >= intensity_image.shape[0]:
        maxr = intensity_image.shape[0] - 1
        minr = maxr - patch_size
    if maxc >= intensity_image.shape[1]:
        maxc = intensity_image.shape[1] - 1
        minc = maxc - patch_size

    patch = intensity_image[minr:maxr, minc:maxc]

    patch_basic_intensity_features = [np.max(patch), np.min(patch), np.mean(patch), np.std(patch), skew(patch.ravel()), kurtosis(patch.ravel())]
    patch_texture_features = compute_haralick_features(patch)
    patch_advanced_intensity_features = [shannon_entropy(patch)]

    patch_features = patch_basic_intensity_features + patch_texture_features + patch_advanced_intensity_features

    return patch_features

def compute_base_label_features(mask_of_label, intensity_image, features, extra_properties):
    """
    Compute a set of features for a single label.
    
    Parameters:
        mask_of_label (np.ndarray): The mask of the label.
        intensity_image (np.ndarray): The intensity image.
        features (list): The list of features to compute.
        extra_properties (list): The list of extra properties to compute.
        
    Returns:
        list: A list of features for the label.
    """

    properties = regionprops_table(mask_of_label, intensity_image=intensity_image, properties=features, extra_properties=extra_properties)  
    feature_vector = []
    for feature in properties:
        feature_vector.append(properties[feature][0])
    return feature_vector

def get_context(current_label, mask_of_current_label, mask_of_labels, num_closest=5):
    """
    Returns the mask of the closest labels to the current label in a label image.

    Parameters:
        current_label (int): The label of the current region.
        mask_of_current_label (np.ndarray): The mask of the current region.
        mask_of_labels (np.ndarray): The mask of all regions.
        num_closest (int): The number of closest regions to return.

    Returns:
        np.ndarray: The mask of the closest regions.
    """

    mask_of_all_other_labels = mask_of_labels.copy()
    mask_of_all_other_labels[mask_of_all_other_labels == current_label] = 0
    mask_of_all_other_labels = mask_of_all_other_labels.astype("uint8")

    if num_closest == -1:
        return mask_of_all_other_labels
    else:
        centroid_current_label = regionprops(mask_of_current_label)[0].centroid
        centroid_other_labels = regionprops(mask_of_all_other_labels)

        # find the num_closest labels
        closest_labels = sorted(centroid_other_labels, key=lambda x: np.linalg.norm(np.array(x.centroid) - np.array(centroid_current_label)))[:num_closest] # type: ignore
        closest_labels = [x.label for x in closest_labels]
        binary_mask_of_closest_labels = np.isin(mask_of_labels, closest_labels).astype("uint8")
        mask_of_closest_labels = mask_of_labels.copy()
        mask_of_closest_labels[binary_mask_of_closest_labels == 0] = 0
        return mask_of_closest_labels.astype("uint8")


def get_context_features(mask_of_labels, intensity_image, features, extra_properties):
    """
    Compute a set of features for the context of a label.

    Parameters:
        mask_of_labels (np.ndarray): The mask of all regions.
        intensity_image (np.ndarray): The intensity image.
        features (list): The list of features to compute.
        extra_properties (list): The list of extra properties to compute.

    Returns:
        list: A list of aggregated features for the context of the label (mean and std of each desired feature).
    """

    properties = regionprops_table(mask_of_labels, intensity_image=intensity_image, properties=features, extra_properties=extra_properties)
    context_feature_vector = []
    for feature in properties:
        context_feature_vector.append(np.mean(properties[feature]))
        context_feature_vector.append(np.std(properties[feature]))
    return context_feature_vector