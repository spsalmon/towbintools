from typing import Tuple

import cv2
import numpy as np
import skimage.metrics
from tifffile import imread, imwrite
from tifffile import tiffcomment
import ome_types


def pad_to_correct_dim(
    image: np.ndarray,
    xdim: int,
    ydim: int,
) -> np.ndarray:
    """
    Pad an image to the correct dimensions by adding zeros on its right and bottom.

    Parameters:
            image (np.ndarray): The input image as a NumPy array.
            xdim (int): The desired X dimension of the padded image.
            ydim (int): The desired Y dimension of the padded image.

    Returns:
            np.ndarray: The padded image as a NumPy array.
    """
    xpad = xdim - image.shape[0]
    ypad = ydim - image.shape[1]

    # Pad the image with zeros.
    return np.pad(image, ((0, xpad), (0, ypad)), "constant", constant_values=(0, 0))


def pad_to_correct_dim_equally(
    image: np.ndarray,
    xdim: int,
    ydim: int,
) -> np.ndarray:
    """
    Pad an image equally to the correct dimensions by adding zeros on both sides.

    Parameters:
            image (np.ndarray): The input image as a NumPy array.
            xdim (int): The desired X dimension of the padded image.
            ydim (int): The desired Y dimension of the padded image.

    Returns:
            np.ndarray: The equally padded image as a NumPy array.
    """
    xpad = xdim - image.shape[0]
    ypad = ydim - image.shape[1]

    # Calculate the padding for each dimension equally.
    xpad_start = xpad // 2
    xpad_end = xpad // 2 + xpad % 2
    ypad_start = ypad // 2
    ypad_end = ypad // 2 + ypad % 2

    # Pad the image with zeros equally.
    return np.pad(
        image,
        ((xpad_start, xpad_end), (ypad_start, ypad_end)),
        "constant",
        constant_values=(0, 0),
    )


def crop_images_to_same_dim(
    image1: np.ndarray,
    image2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crop two images to the same dimensions by taking the minimum height and width.

    Parameters:
            image1 (np.ndarray): The first input image as a NumPy array.
            image2 (np.ndarray): The second input image as a NumPy array.

    Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the cropped images as NumPy arrays.
    """
    # Determine the minimum dimensions for cropping.
    min_height = min(image1.shape[0], image2.shape[0])
    min_width = min(image1.shape[1], image2.shape[1])

    # Crop the images to the same dimensions.
    image1 = image1[:min_height, :min_width, ...]
    image2 = image2[:min_height, :min_width, ...]

    return image1, image2


def pad_images_to_same_dim(
    image1: np.ndarray,
    image2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad two images to the same dimensions by taking the maximum height and width.

    Parameters:
            image1 (np.ndarray): The first input image as a NumPy array.
            image2 (np.ndarray): The second input image as a NumPy array.

    Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the padded images as NumPy arrays.
    """
    # Determine the maximum dimensions for padding.
    max_height = max(image1.shape[0], image2.shape[0])
    max_width = max(image1.shape[1], image2.shape[1])

    # Pad the images to the same dimensions.
    image1 = pad_to_correct_dim_equally(image1, max_height, max_width)
    image2 = pad_to_correct_dim_equally(image2, max_height, max_width)

    return image1, image2


def align_worm_orientation_ssim(
    image: np.ndarray,
    reference_image: np.ndarray,
) -> np.ndarray:
    """
    Align the orientation of a worm image based on structural similarity index (SSIM) comparison with a reference image.
    If the flipped image has a higher SSIM than the original image, it is returned as the aligned image; otherwise, the
    original image is returned.

    Parameters:
            image (np.ndarray): The input worm image as a NumPy array.
            reference_image (np.ndarray): The reference image for comparison as a NumPy array.

    Returns:
            np.ndarray: The aligned worm image as a NumPy array.
    """
    # Pad the input and reference images to the same dimensions.
    image_pad, reference_pad = pad_images_to_same_dim(
        image.copy(), reference_image.copy()
    )

    # Calculate the SSIM between the padded image and reference image.
    ssim = skimage.metrics.structural_similarity(
        image_pad, reference_pad, data_range=image_pad.max() - image_pad.min()
    )
    # Calculate the SSIM between the flipped image and reference image.
    ssim_flipped = skimage.metrics.structural_similarity(
        np.flip(image_pad, axis=1),
        reference_pad,
        data_range=image_pad.max() - image_pad.min(),
    )
    # Compare the SSIM values and determine the aligned image.
    if ssim_flipped > ssim:
        return np.flip(image, axis=1)
    else:
        return image


def normalize_image(
    image: np.ndarray,
    dest_dtype: type = np.uint16,
) -> np.ndarray:
    """
    Normalize an image to a specified data type's range.

    Parameters:
            image (np.ndarray): The input image as a NumPy array.
            dest_dtype (type): The desired data type for the output normalized image.
                                               Allowed values are np.uint16, np.uint8, np.float32, np.float64.

    Returns:
            np.ndarray: The normalized image with the specified data type.

    Raises:
            ValueError: If dest_dtype is not one of the allowed data types.
    """
    dtype_mapping = {
        np.uint16: cv2.CV_16U,
        np.uint8: cv2.CV_8U,
        np.float32: cv2.CV_32F,
        np.float64: cv2.CV_64F,
    }

    if dest_dtype not in dtype_mapping:
        raise ValueError(
            "dest_dtype must be one of np.uint16, np.uint8, np.float32, np.float64"
        )

    dest_dtype_cv2 = dtype_mapping[dest_dtype]
    max_value = np.iinfo(dest_dtype).max if dest_dtype in [np.uint16, np.uint8] else 1

    return cv2.normalize(image, None, 0, max_value, cv2.NORM_MINMAX, dtype=dest_dtype_cv2)  # type: ignore


def augment_contrast(
    image: np.ndarray,
    clip_limit: float = 5,
    tile_size: int = 8,
) -> np.ndarray:
    """
    Augment the contrast of an image using the CLAHE (Contrast Limited Adaptive Histogram Equalization) method.

    Parameters:
            image (np.ndarray): The input image as a NumPy array.
            clip_limit (float): The clipping limit for contrast enhancement. Values above this limit get clipped.
            tile_size (int): The size of the tile grid for the CLAHE method.

    Returns:
            np.ndarray: The contrast-enhanced image as a NumPy array of dtype np.uint16.
    """
    image = normalize_image(image, np.uint16)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    image = clahe.apply(image).astype(np.uint16)
    return image


def read_tiff_file(
    file_path: str,
    channels_to_keep: list = [],
) -> np.ndarray:
    """
    Read a TIFF file and optionally select specific channels from the image.

    Parameters:
            file_path (str): Path to the TIFF image file.
            channels_to_keep (list): List of channel indices to keep. If empty, all channels are kept.

    Returns:
            np.ndarray: The image data as a NumPy array. The number of dimensions may vary depending on the input and selected channels.
    """
    image = imread(file_path)

    if image.ndim == 2 or not channels_to_keep:
        return image

    if image.ndim == 3:
        return image[channels_to_keep, ...].squeeze()  # type: ignore
    else:
        return image[:, channels_to_keep, ...].squeeze()  # type: ignore
    
def get_image_size_metadata(file_path):
    try:
        ome_xml = tiffcomment(file_path)
        ome_metadata = ome_types.from_xml(ome_xml).images[0].pixels
        xdim = ome_metadata.size_x
        ydim = ome_metadata.size_y
        zdim = ome_metadata.size_z
        tdim = ome_metadata.size_t
        cdim = ome_metadata.size_c

        return {'x_dim': xdim, 'y_dim': ydim, 'z_dim': zdim, 't_dim': tdim, 'c_dim': cdim}
    except:
        return None
    
def check_if_zstack(file_path):
    try:
        ome_xml = tiffcomment(file_path)
        ome_metadata = ome_types.from_xml(ome_xml).images[0].pixels
        zdim = ome_metadata.size_z
        return zdim > 1
    except:
        return False
