import cv2
import numpy as np


def normalized_variance_measure(
    image: np.ndarray,
) -> float:
    """
    Compute the normalized variance measure of an image.

    Parameters:
            image (np.ndarray): The input image as a numpy array.

    Returns:
            float: The computed normalized variance value.
    """
    mean = np.mean(image)
    var = np.var(image)
    return np.divide(var, mean)


# This code is from : https://github.com/vismantic-ohtuprojekti/qualipy/blob/master/qualipy/utils/focus_measure.py

# Python implementations for focus measure operators described
# in "Analysis of focus measure operators for shape-from-focus"
# (Pattern recognition, 2012) by Pertuz et al.

# LICENSE

# The MIT License (MIT)

# Copyright (c) 2015 QualiPy developers

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.


def LAPV(img):
    """Implements the Variance of Laplacian (LAP4) focus measure
    operator. Measures the amount of edges present in the image.

    :param img: the image the measure is applied to
    :type img: np.ndarray
    :returns: np.float32 -- the degree of focus
    """
    return np.std(cv2.Laplacian(img, cv2.CV_64F)) ** 2  # type: ignore


def LAPM(img):
    """Implements the Modified Laplacian (LAP2) focus measure
    operator. Measures the amount of edges present in the image.

    :param img: the image the measure is applied to
    :type img: np.ndarray
    :returns: np.float32 -- the degree of focus
    """
    kernel = np.array([-1, 2, -1])
    laplacianX = np.abs(cv2.filter2D(img, -1, kernel))
    laplacianY = np.abs(cv2.filter2D(img, -1, kernel.T))
    return np.mean(laplacianX + laplacianY)


def TENG(img):
    """Implements the Tenengrad (TENG) focus measure operator.
    Based on the gradient of the image.

    :param img: the image the measure is applied to
    :type img: np.ndarray
    :returns: np.float32 -- the degree of focus
    """
    gaussianX = cv2.Sobel(img, cv2.CV_64F, 1, 0)  # type: ignore
    gaussianY = cv2.Sobel(img, cv2.CV_64F, 1, 0)  # type: ignore
    return np.mean(gaussianX * gaussianX + gaussianY * gaussianY)


def MLOG(img):
    """Implements the MLOG focus measure algorithm.

    :param img: the image the measure is applied to
    :type img: np.ndarray
    :returns: np.float32 -- the degree of focus
    """
    return np.max(cv2.convertScaleAbs(cv2.Laplacian(img, cv2.CV_64F)))  # type: ignore


def TENG_VARIANCE(img):
    """Implements the Tenengrad Variance focus measure operator.

    :param img: the image the measure is applied to
    :type img: np.ndarray
    :returns: np.float32 -- the degree of focus
    """

    gaussianX = cv2.Sobel(img, cv2.CV_64F, 1, 0)  # type: ignore
    gaussianY = cv2.Sobel(img, cv2.CV_64F, 1, 0)  # type: ignore

    G = np.sqrt(gaussianX**2 + gaussianY**2)
    return np.var(G)
