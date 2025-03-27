from shapely.geometry import Polygon
import cv2
import numpy as np
import torch
import math


def simplify_polygons(masks_xy):
    """Simplifies polygons using OpenCV."""
    if masks_xy is None or len(masks_xy) == 0:
        return []

    simplified_polygons = []
    for mask_xy in masks_xy:
        # Convert torch.Tensor to numpy.ndarray if necessary.
        if isinstance(mask_xy, torch.Tensor):
            mask_np = mask_xy.cpu().numpy().astype(np.int32)
        elif isinstance(mask_xy, np.ndarray):
            mask_np = mask_xy.astype(np.int32)
        else:
            raise TypeError(
                f"Expected torch.Tensor or numpy.ndarray, got {type(mask_xy)}")

        if len(mask_np) == 0:  # Skip empty masks
            continue

        mask_np_opencv = mask_np.astype(np.int32)
        mask_np_opencv = mask_np_opencv.reshape((-1, 1, 2))

        epsilon = 0.005 * cv2.arcLength(mask_np_opencv, True)
        approx = cv2.approxPolyDP(mask_np_opencv, epsilon, True)
        approx = approx.reshape((-1, 2)).astype(mask_np.dtype)
        simplified_polygons.append(approx)

    return simplified_polygons


def polygon_iou(this_polygon_points, other_polygon_points):
    """
    Calculate the Intersection-over-Union (IoU) of two polygons represented by lists of [x, y] points.
    """
    polygon1 = Polygon(this_polygon_points)
    polygon2 = Polygon(other_polygon_points)
    intersect_area = polygon2.intersection(polygon1).area
    union_area = polygon1.area + polygon2.area - intersect_area
    iou = intersect_area / union_area
    return iou


def polygon_features(polygon_points):
    """
    Compute geometric features of a polygon given its [x, y] points.

    Returns:
        dict: A dictionary with keys:
            - area: Polygon area.
            - perimeter: Length of the polygon boundary.
            - centroid: (x, y) coordinates of the polygon centroid.
            - bounding_box: (minx, miny, maxx, maxy) of the polygon.
            - convex_area: Area of the convex hull.
            - extent: Ratio of polygon area to bounding box area.
            - circularity: 4*pi*area/(perimeter^2), with 1 indicating a perfect circle.
            - aspect_ratio: Ratio of the bounding box's width to height.
    """
    polygon = Polygon(polygon_points)
    if not polygon.is_valid:
        # Attempt to fix the polygon if invalid (e.g., self-intersecting)
        polygon = polygon.buffer(0)

    features = {}
    features['area'] = polygon.area
    features['perimeter'] = polygon.length
    centroid = polygon.centroid
    features['centroid'] = (centroid.x, centroid.y)
    # Returns (minx, miny, maxx, maxy)
    features['bounding_box'] = polygon.bounds
    convex_hull = polygon.convex_hull
    features['convex_area'] = convex_hull.area

    # Extent: ratio of polygon area to the area of its bounding box
    minx, miny, maxx, maxy = polygon.bounds
    bbox_area = (maxx - minx) * (maxy - miny)
    features['extent'] = polygon.area / bbox_area if bbox_area > 0 else 0

    # Circularity: 4*pi*area/(perimeter^2) (1 indicates a perfect circle)
    features['circularity'] = 4 * math.pi * polygon.area / \
        (polygon.length ** 2) if polygon.length > 0 else 0

    # Aspect Ratio: width divided by height of the bounding box
    width = maxx - minx
    height = maxy - miny
    features['aspect_ratio'] = width / height if height > 0 else 0

    return features


# Example usage:
if __name__ == "__main__":
    # Define a mouse polygon
    mouse_polygon_points = np.array([[
        691.0,
        365.0
    ],
        [
        688.0,
        369.0
    ],
        [
        692.0,
        385.0
    ],
        [
        691.0,
        441.0
    ],
        [
        696.0,
        455.0
    ],
        [
        754.0,
        499.0
    ],
        [
        753.0,
        508.0
    ],
        [
        758.0,
        511.0
    ],
        [
        768.0,
        511.0
    ],
        [
        783.0,
        503.0
    ],
        [
        804.0,
        505.0
    ],
        [
        812.0,
        510.0
    ],
        [
        818.0,
        510.0
    ],
        [
        829.0,
        497.0
    ],
        [
        828.0,
        484.0
    ],
        [
        804.0,
        428.0
    ],
        [
        781.0,
        386.0
    ],
        [
        770.0,
        376.0
    ],
        [
        755.0,
        370.0
    ]])
    features = polygon_features(mouse_polygon_points)
    print("Polygon Features:")
    for key, value in features.items():
        print(f"{key}: {value}")
