import cv2
import numpy as np
import os.path
from qtpy import QtGui
from qtpy.QtGui import QImage
import qimage2ndarray


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
    Convert a Qt image to an owned, C-contiguous 8-bit RGB OpenCV image.

    Always copies pixel data before ``qt_img`` (or any temporary converted
    ``QImage``) can go out of scope, avoiding dangling views from
    ``QImage.bits()`` / ``qimage2ndarray`` across PySide6 and thread boundaries.

    Args:
        qt_img (QImage): The Qt image to be converted.

    Returns:
        np.ndarray: Owned C-contiguous ``(H, W, 3)`` ``uint8`` RGB image.
    """
    if qt_img is None or qt_img.isNull():
        return np.zeros((0, 0, 3), dtype=np.uint8)

    gray16_fmt = getattr(QImage, "Format_Grayscale16", None)
    if gray16_fmt is not None and qt_img.format() == gray16_fmt:
        width = int(qt_img.width())
        height = int(qt_img.height())
        if width <= 0 or height <= 0:
            return np.zeros((0, 0, 3), dtype=np.uint8)
        bpl = int(qt_img.bytesPerLine())
        total_bytes = (
            int(qt_img.sizeInBytes())
            if hasattr(qt_img, "sizeInBytes")
            else int(qt_img.byteCount())
        )
        ptr = qt_img.constBits() if hasattr(qt_img, "constBits") else qt_img.bits()
        if hasattr(ptr, "setsize"):
            ptr.setsize(total_bytes)
        gray16_view = np.ndarray(
            shape=(height, width),
            dtype=np.uint16,
            buffer=ptr,
            strides=(bpl, 2),
        )
        gray16_copy = np.array(gray16_view, dtype=np.uint16, copy=True, order="C")
        _ = qt_img
        norm_u8 = np.empty_like(gray16_copy, dtype=np.uint8)
        cv2.normalize(gray16_copy, norm_u8, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return np.ascontiguousarray(cv2.cvtColor(norm_u8, cv2.COLOR_GRAY2RGB))

    rgb_qimg = qt_img.convertToFormat(QImage.Format_RGB888)
    width = int(rgb_qimg.width())
    height = int(rgb_qimg.height())
    if width <= 0 or height <= 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    bpl = int(rgb_qimg.bytesPerLine())
    total_bytes = (
        int(rgb_qimg.sizeInBytes())
        if hasattr(rgb_qimg, "sizeInBytes")
        else int(rgb_qimg.byteCount())
    )
    ptr = rgb_qimg.constBits() if hasattr(rgb_qimg, "constBits") else rgb_qimg.bits()
    if hasattr(ptr, "setsize"):
        ptr.setsize(total_bytes)

    rgb_view = np.ndarray(
        shape=(height, width, 3),
        dtype=np.uint8,
        buffer=ptr,
        strides=(bpl, 3, 1),
    )
    cv_image = np.array(rgb_view, dtype=np.uint8, copy=True, order="C")
    # Keep Qt wrappers alive until the C-contiguous copy completes.
    _ = (rgb_qimg, qt_img)
    return cv_image


def convert_qt_image_to_cv_image(in_image):
    """
    Convert a Qt image to an owned, C-contiguous 8-bit RGB OpenCV image.

    Args:
        in_image (QImage): The Qt image to be converted.

    Returns:
        np.ndarray: Owned C-contiguous ``(H, W, 3)`` ``uint8`` RGB image.
    """
    return convert_qt_image_to_rgb_cv_image(in_image)


def convert_cv_image_to_qt_image(in_mat):
    """
    Convert an OpenCV image to an owned Qt image.

    Args:
        in_mat (np.ndarray): The OpenCV image as a NumPy array.

    Returns:
        QImage: Owned Qt image detached from ``in_mat``.
    """
    return QtGui.QImage(qimage2ndarray.array2qimage(in_mat)).copy()
