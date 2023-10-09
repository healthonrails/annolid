import cv2
import numpy as np
import os.path
import qimage2ndarray
from PyQt5 import QtGui
from PyQt5.QtGui import QImage


def load_image_from_path(img_path):
    """
    Load an image from the specified path.

    Args:
        img_path (str): The path to the image file.

    Returns:
        np.ndarray: The loaded image as a NumPy array.
    """
    if os.path.exists(img_path):
        cv_image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        return cv_image
    else:
        raise FileNotFoundError(f"Image not found at path: {img_path}")


def convert_qt_image_to_rgb_cv_image(qt_img):
    """
    Convert a Qt image to an 8-bit RGB OpenCV image.

    Args:
        qt_img (QImage): The Qt image to be converted.

    Returns:
        np.ndarray: The converted OpenCV image.
    """
    if (
        qt_img.format() == QImage.Format_RGB32
        or qt_img.format() == QImage.Format_ARGB32
        or qt_img.format() == QImage.Format_ARGB32_Premultiplied
    ):
        cv_image = qimage2ndarray.rgb_view(qt_img)
    else:
        cv_image = qimage2ndarray.raw_view(qt_img)

    # Normalize and convert to uint8 if needed
    if cv_image.dtype != np.uint8:
        cv2.normalize(cv_image, cv_image, 0, 255, cv2.NORM_MINMAX)
        cv_image = np.array(cv_image, dtype=np.uint8)

    # Convert grayscale image to RGB
    if len(cv_image.shape) == 2 or cv_image.shape[2] == 1:
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)

    return cv_image


def convert_qt_image_to_cv_image(in_image):
    """
    Convert a Qt image to an OpenCV image.

    Args:
        in_image (QImage): The Qt image to be converted.

    Returns:
        np.ndarray: The converted OpenCV image.
    """
    return qimage2ndarray.rgb_view(in_image)


def convert_cv_image_to_qt_image(in_mat):
    """
    Convert an OpenCV image to a Qt image.

    Args:
        in_mat (np.ndarray): The OpenCV image as a NumPy array.

    Returns:
        QImage: The converted Qt image.
    """
    return QtGui.QImage(qimage2ndarray.array2qimage(in_mat))
