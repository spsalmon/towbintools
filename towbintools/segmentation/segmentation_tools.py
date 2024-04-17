import sys
from typing import List, Optional, Union

import cv2
import numpy as np
import skimage
import skimage.exposure
import skimage.feature
import skimage.morphology
import torch
from skimage.filters import threshold_otsu, threshold_triangle
from skimage.util import img_as_ubyte

from towbintools.deep_learning import augmentation, util
from towbintools.foundation import binary_image, image_handling


def edge_based_segmentation(
    image: np.ndarray,
    pixelsize: float,
    sigma_canny: float = 1,
) -> np.ndarray:
    """
    Python adaptation of the OG Matlab code for Sobel-based segmentation

    Parameters:
            image (np.ndarray): The input 2D grayscale image as a NumPy array.
            pixelsize (float): Pixel size to consider when removing small objects.
            backbone (str, optional): The library to use for edge detection. Options are 'skimage' or 'opencv'. Default is 'skimage'.
            sigma_canny (float, optional): Standard deviation for the Gaussian filter used in Canny edge detection. Default is 1.

    Returns:
            np.ndarray: The segmented image as a binary mask (NumPy array).

    Raises:
            ValueError: If the input image is not 2D.
    """

    if image.ndim > 2:
        raise ValueError("Image must be 2D.")

    # image_dtype_max_value = np.iinfo(image.dtype).max
    thresh_otsu = threshold_otsu(image)
    edges = skimage.feature.canny(
        image.copy(),
        sigma=sigma_canny,
        low_threshold=thresh_otsu / 5,
        high_threshold=thresh_otsu / 2.5,
    ).astype(np.uint8)
    # edges = skimage.feature.canny(image.copy(), sigma=sigma_canny, low_threshold=image_dtype_max_value * 0.05, high_threshold=image_dtype_max_value*0.1).astype(np.uint8)

    edges = skimage.morphology.remove_small_objects(
        edges.astype(bool), 3, connectivity=2
    ).astype(np.uint8)

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 20000:
        return np.zeros_like(image).astype(np.uint8)

    edges = binary_image.connect_endpoints(edges, max_distance=200)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = (cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) > 0).astype(int)

    mask = binary_image.fill_bright_holes(image, edges, 10).astype(np.uint8)
    mask = skimage.morphology.remove_small_objects(
        mask.astype(bool), 422.5 / (pixelsize**2), connectivity=2
    ).astype(np.uint8)

    mask = binary_image.fill_bright_holes(image, mask, 5).astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        out = np.zeros_like(mask)
        for contour in contours:
            cv2.drawContours(out, [contour], -1, 1, 1)

        threshold = np.percentile(image[out > 0], 30)  # type: ignore

        final_mask = (image > threshold).astype(np.uint8)
        final_mask = skimage.morphology.remove_small_objects(
            final_mask.astype(bool), 422.5 / (pixelsize**2), connectivity=2
        ).astype(np.uint8)

    else:
        final_mask = np.zeros_like(image).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_mask = (cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel) > 0).astype(int)

    final_mask = binary_image.fill_bright_holes(image, final_mask, 10).astype(np.uint8)
    return final_mask


def deep_learning_segmentation(
    image,
    model,
    device,
    tiler,
    RGB=True,
    activation=None,
    batch_size=-1,
):
    # Split the image into tiles
    if tiler is None:
        tiles = [image]
    else:
        try:
            tiles = tiler.split(image)
        except:
            print("Error splitting image into tiles. Image shape: ", image.shape)
            sys.exit(1)

    # Prepare tiles for model input
    if RGB:
        tiles = [augmentation.grayscale_to_rgb(tile) for tile in tiles]
    else:
        tiles = [torch.tensor(tile[np.newaxis, ...]).unsqueeze(0) for tile in tiles]

    tiles_batched = (
        util.divide_batch(torch.stack(tiles), batch_size)
        if batch_size > 0
        else [torch.stack(tiles)]
    )

    prediction_tiles = []
    for batch in tiles_batched:
        batch = batch.to(device)

        # Predict
        with torch.no_grad():
            prediction = model(batch)
            if activation == "sigmoid":
                prediction = torch.sigmoid(prediction)
            elif activation == "softmax":
                prediction = torch.softmax(prediction, dim=1)

        # Post-process predictions
        for pred_tile in prediction:
            pred_tile = pred_tile.cpu().numpy()
            pred_tile = np.moveaxis(pred_tile, 0, -1)
            prediction_tiles.append(pred_tile)

    # Merge tiles to get the complete prediction
    if tiler is None:
        pred = prediction_tiles[0]
    else:
        pred = tiler.merge(prediction_tiles)
    mask = (pred > 0.5).astype(np.uint8)

    return mask


def double_threshold_segmentation(image):
    # keep bins 2**8 even though our images are 2**16 because none of the images cover the whole dynamic range of 2**16. This will bin lower abundance signal pixels into fewer histogram bins
    mask = image > _custom_threshold_otsu(image, nbins=2**8)
    mask = cv2.medianBlur(img_as_ubyte(mask), 5)
    mask = skimage.morphology.remove_small_objects(
        mask.astype(bool), 20, connectivity=2
    )
    mask = img_as_ubyte(mask)
    return mask


def segment_image(
    image: Union[str, np.ndarray],
    method: str,
    channels: Optional[List[int]] = None,
    pixelsize: float = None,
    sigma_canny: float = 1,
    preprocessing_fn=None,
    model=None,
    device=None,
    tiler=None,
    RGB=False,
    activation=None,
    batch_size=-1,
    is_zstack=True,
) -> np.ndarray:
    """
    Segment an image using the specified method.

    Parameters:
            image (Union[str, np.ndarray]): Input image. If string, it's interpreted as the path to a TIFF file. If ndarray, it's the image data directly.
            method (str): Segmentation method to use. Currently supported: "edge_based".
            channels (List[int], optional): List of channel indices to keep if reading a multi-channel TIFF file. Default is empty, meaning all channels are kept.
            pixelsize (Optional[float], optional): Pixel size to consider for edge-based segmentation. Must be specified if method is "edge_based".
            edge_based_backbone (str, optional): The library to use for edge detection in "edge_based" method. Options are 'skimage' or 'opencv'. Default is 'skimage'.
            sigma_canny (float, optional): Standard deviation for the Gaussian filter used in Canny edge detection if method is "edge_based". Default is 1.

    Returns:
            np.ndarray: The segmented image as a binary mask (NumPy array).

    Raises:
            ValueError: If method is not recognized or if pixelsize is not specified when required.
    """
    if type(image) == str:
        image = image_handling.read_tiff_file(image, channels_to_keep=channels)

    if method == "edge_based":
        if pixelsize is None:
            raise ValueError("Pixelsize must be specified for edge-based segmentation.")

        image = image_handling.normalize_image(image, dest_dtype=np.uint16)
        segment_fn = lambda x: edge_based_segmentation(
            x, pixelsize, sigma_canny=sigma_canny
        )

    elif method == "deep_learning":
        if not model:
            raise ValueError("Model must be specified for deep learning segmentation.")
        if not device:
            raise ValueError("Device must be specified for deep learning segmentation.")

        if preprocessing_fn:
            image = preprocessing_fn(image=image)["image"]
        segment_fn = lambda x: deep_learning_segmentation(
            x,
            model,
            device,
            tiler,
            RGB=RGB,
            activation=activation,
            batch_size=batch_size,
        )
    elif method == "double_threshold":
        segment_fn = double_threshold_segmentation

    else:
        raise ValueError("Invalid segmentation method.")

    if is_zstack:
        mask = np.zeros(
            (image.shape[0], image.shape[-2], image.shape[-1]), dtype=np.uint8
        )
        for i, plane in enumerate(image):
            mask[i] = segment_fn(plane).squeeze()
        return mask

    return segment_fn(image)


def _mode_limited_mean(image, nbins=2**8, histogram=None):
    """
    Brocher, Jan. (2014). Qualitative and Quantitative Evaluation of Two New Histogram Limiting Binarization Algorithms.
    International Journal of Image Processing (IJIP). 8. 30-48.
    Section 2.2 - Mode Limited Mean (MoLiM)
    """
    # basic idea of the paper and MoLiM is that most thresholding algorithms work on histograms
    # if most of the image is 'background', then the histogram will be dominated by a single background peak, which will affect how the threshold is calculated
    # MoLiM is the mean of the image with some of the background already classified as background
    # this limited mean value can either itself be a threshold, or can be used as a starting point for more sophisticated thresholding algorithms
    if histogram is None:
        hist, bin_centers = skimage.exposure.histogram(image, nbins=nbins)
    else:
        hist, bin_centers = histogram

    mode = bin_centers[hist.argmax()]
    molim = np.mean(image[image > mode])
    return molim


def _custom_threshold_otsu(image, nbins=2**8):
    # for consistent behaviour with regards to dtype and nbins, explicitly convert image to float
    # this is because skimage.filters.threshold_... and skimage.exposure.histogram ignore nbins in case of integer dtype
    # additionally, I think skimage trims histogram values below image.min() and above image.max() since they are zero
    # so all in all, when trying to troubleshoot threshold algorithm behaviour, and make it consistent across different channels, rescale the image as float in range (0, 1)

    # save the image range and dtype to later convert threshold back to what is expected
    image_range = image.min(), image.max()
    image_dtype = image.dtype
    image = skimage.exposure.rescale_intensity(image, out_range=(0, 1))
    # calculate histogram once and reuse in later thresholding steps
    histogram = skimage.exposure.histogram(image, nbins=nbins)
    hist_vals, bin_centers = histogram
    # limited_mean lies somewhere in the histogram peak for background
    limited_mean = _mode_limited_mean(image, histogram=histogram)
    # threshold_triangle by-and-large finds the threshold for the bottom of background peak
    # so it's a good starting point for otsu threshold, which can struggle in the case when background is so large as a proportion of the image that the overall histogram appears to be unimodal
    thresh_lower_bound = image.min()
    i = 0
    while thresh_lower_bound < limited_mean:
        # >= to allow possibility of thresh_lower_bound == limited_mean
        thresh_lower_bound = threshold_triangle(
            image[image >= thresh_lower_bound], nbins=nbins
        )
        if i > 64:
            break
        i += 1

    # don't start right at lower bound because the histogram information below thresh_lower_bound may be useful in determining a good threshold in the last iteration
    # for example, when thresh_lower_bound is already a decent threshold, in which case the the very information used to calculate it would have been discarded
    thresh = image.min()
    i = 0
    while thresh < thresh_lower_bound:
        # crop histogram to only include values above the current threshold
        hist_vals = hist_vals[bin_centers > thresh]
        bin_centers = bin_centers[bin_centers > thresh]
        thresh = threshold_otsu(hist=(hist_vals, bin_centers))

        if i > 64:
            break
        i += 1

    # rescale threshold back to original image range and dtype since threshold is in (0, 1)
    # rather than do the math ourselves, let skimage do it as it also handles float quantization
    thresh = skimage.exposure.rescale_intensity(
        np.array(thresh).reshape(1, 1), in_range=(0, 1), out_range=image_range
    ).astype(image_dtype)[0, 0]

    return thresh
