import cv2
import numpy as np


def mask_to_polygons(mask):
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

    res = [x.flatten() for x in res]
    # convert OpenCV int coordinates [0, H -1 or W-1] to
    # real value coordinate space.
    res = [x + 0.5 for x in res if len(x) >= 6]

    return res, has_holes
