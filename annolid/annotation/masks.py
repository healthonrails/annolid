import cv2
import ast
import numpy as np
from scipy import ndimage
import pycocotools.mask as mask_util
from simplification.cutil import simplify_coords_vwp


def binary_mask_to_coco_rle(binary_mask):
    # Ensure the binary mask is in the correct format (numpy array with dtype=bool)
    binary_mask = np.asarray(binary_mask, dtype=np.uint8)

    # Convert the binary mask to COCO RLE format
    coco_rle = mask_util.encode(np.asfortranarray(binary_mask))
    coco_rle['counts'] = coco_rle['counts'].decode(
        'UTF-8')  # Convert bytes to string

    return coco_rle


def mask_to_polygons(mask,
                     use_convex_hull=False):
    """
    convert predicted mask to polygons
    """
    # for cv2 versions that do not support incontiguous array
    mask = np.ascontiguousarray(mask)
    mask = mask.astype(np.uint8)
    # cv2.RETER_CCOMP flag retrieves all the contours
    # the arranges them to a 2-level hierarchy.
    res = cv2.findContours(mask,
                           cv2.RETR_CCOMP,
                           cv2.CHAIN_APPROX_SIMPLE)

    hierarchy = res[-1]
    # mask is empty
    if hierarchy is None:
        return [], False

    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0

    res = res[-2]
    try:
        res_simp = simplify_coords_vwp(np.array(res[0]).reshape(-1, 2), 30.0)
        res_simp = np.array(res_simp)
        res = [np.expand_dims(res_simp, axis=1)]
    except ValueError as e:
        print('Failed to simplify the points:', str(e))

    if use_convex_hull:
        hull = []
        for i in range(len(res)):
            hull.append(cv2.convexHull(res[i], False))
            res = [x.flatten() for x in hull]
    else:
        res = [x.flatten() for x in res]

    # convert OpenCV int coordinates [0, H -1 or W-1] to
    # real value coordinate spaces
    res = [x + 0.5 for x in res if len(x) >= 6]

    return res, has_holes


def mask_perimeter(mask):
    """calculate perimeter for a given binary mask
    """
    try:
        mask = mask_util.decode(mask)
    except TypeError:
        mask = ast.literal_eval(mask)
        rle = [mask]
        mask = mask_util.decode(rle)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    perimeter = cv2.arcLength(cnt, True)
    return perimeter


def mask_area(mask):
    """Calulate the area of a RLE mask.
    """
    try:
        area = mask_util.area(mask)
    except TypeError:
        mask = ast.literal_eval(mask)
        area = mask_util.area(mask)
    return area


def mask_iou(this_mask, other_mask):
    """
    Calculate intersection over union between two RLE masks.
    """
    try:
        _iou = mask_util.iou([this_mask],
                             [other_mask],
                             [False, False])
    except Exception:
        this_mask = ast.literal_eval(this_mask)
        other_mask = ast.literal_eval(other_mask)
        _iou = mask_util.iou([this_mask],
                             [other_mask],
                             [False, False])
    return _iou.flatten()[0]


def calculate_mask_centroid(mask):
    """
    Calculate the centroid of an instance mask in COCO format.

    Arguments:
    mask -- a binary mask in COCO format (list of RLE-encoded strings)

    Returns:
    A tuple containing the x and y coordinates of the centroid in the original image.
    """
    try:
        mask_array = mask_util.decode(mask)
    except TypeError:
        mask_array = ast.literal_eval(mask)
        rle = [mask_array]
        mask_array = mask_util.decode(rle)
    # Calculate the center of mass of the binary mask
    center_of_mass = ndimage.measurements.center_of_mass(mask_array)

    # Convert the center of mass to the original image coordinates
    x = int(center_of_mass[1])
    y = int(center_of_mass[0])

    return x, y
