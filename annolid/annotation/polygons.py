from shapely.geometry import Polygon
import cv2
import numpy as np
import torch


def simplify_polygons(masks_xy):
    """Simplifies polygons using OpenCV."""

    if masks_xy is None or len(masks_xy) == 0:
        return []

    simplified_polygons = []
    for mask_xy in masks_xy:
        # Check if it's a tensor and convert if necessary.
        if isinstance(mask_xy, torch.Tensor):
            mask_np = mask_xy.cpu().numpy().astype(
                np.int32)  # Convert ONLY if it's a tensor
        elif isinstance(mask_xy, np.ndarray):
            mask_np = mask_xy.astype(np.int32)  # Already a numpy array
        else:
            raise TypeError(
                f"Expected torch.Tensor or numpy.ndarray, got {type(mask_xy)}")

        #  Ensure correct data type and shape for cv2.arcLength:
        if len(mask_np) == 0:  # Handle empty masks
            continue  # skip to next mask

        mask_np_opencv = mask_np.astype(np.int32)  # Or np.float32 if needed
        mask_np_opencv = mask_np_opencv.reshape(
            (-1, 1, 2))  # Reshape to (N, 1, 2)

        # Use reshaped array here
        epsilon = 0.005 * cv2.arcLength(mask_np_opencv, True)
        approx = cv2.approxPolyDP(
            mask_np_opencv, epsilon, True)  # Use reshaped array

        # Convert back to original shape/dtype if needed (e.g., for visualization)
        approx = approx.reshape((-1, 2)).astype(mask_np.dtype)

        simplified_polygons.append(approx)

    return simplified_polygons


def polygon_iou(this_polygon_points, other_polygon_points):
    """
    Calculate iou of two polygons represented with points of [xi,yi] pairs.
    """
    polygon1 = Polygon(this_polygon_points)
    polygon2 = Polygon(other_polygon_points)
    intersect_area = polygon2.intersection(polygon1).area
    union_area = polygon1.area + polygon2.area - intersect_area
    iou = intersect_area / union_area
    return iou
