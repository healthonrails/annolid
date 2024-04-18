import labelme
import numpy as np
import uuid
from shapely.geometry import Polygon, Point
from sklearn_extra.cluster import KMedoids
import json


def load_zone_json(zone_file):
    """Load zone information from the JSON file."""
    with open(zone_file, 'r') as f:
        zone_data = json.load(f)
    return zone_data


def is_point_in_polygon(point, polygon_points):
    """
    Check if a point is inside a polygon.

    Args:
        point (tuple): The coordinates of the point (x, y).
        polygon_points (list): List of tuples representing the polygon vertices.

    Returns:
        bool: True if the point is inside the polygon, False otherwise.
    """
    # Create a Shapely Point object
    point = Point(point[0], point[1])

    # Create a Shapely Polygon object
    polygon = Polygon(polygon_points)

    # Check if the point is within the polygon
    return polygon.contains(point)


def shapes_to_label(img_shape, shapes, label_name_to_value):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for shape in shapes:
        points = [point for point in shape.points]
        if len(points) < 3:
            # polygon must have more than 2 points
            continue
        label = shape.label
        group_id = shape.group_id
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.shape_type

        cls_name = label
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label_name_to_value[cls_name]

        mask = labelme.utils.shape.shape_to_mask(
            img_shape[:2], points, shape_type)
        cls[mask] = cls_id
        ins[mask] = ins_id

    return cls, ins


def masks_to_bboxes(masks):
    if masks.ndim != 3:
        raise ValueError(
            "masks.ndim must be 3, but it is {}".format(masks.ndim))
    if masks.dtype != bool:
        raise ValueError(
            "masks.dtype must be bool type, but it is {}".format(masks.dtype)
        )
    bboxes = []
    for mask in masks:
        where = np.argwhere(mask)
        if where.size > 0:  # Check if where array is not empty
            (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1
            bboxes.append((x1, y1, x2, y2))
    bboxes = np.asarray(bboxes, dtype=np.float32)
    return bboxes


def polygon_center(points):
    # Create a Shapely polygon object
    polygon = Polygon(points)

    # Calculate the centroid of the polygon
    centroid = polygon.centroid

    # Extract centroid coordinates
    centroid_x = centroid.x
    centroid_y = centroid.y

    return centroid_x, centroid_y


def extract_flow_points_in_mask(mask, flow, num_points=8):
    """
    Extracts representative points from the optical flow field using KMedoids clustering
    while considering points from a binary mask.

    'scikit-learn-extra >= 0.3.0',


    Args:
        mask (numpy.ndarray): Binary mask where non-zero values represent valid points.
        flow (numpy.ndarray): The optical flow field obtained from some method.
        num_points (int): The number of representative points to extract. Default is 8.

    Returns:
        numpy.ndarray: An array of (x, y) pairs representing 
        the representative points extracted from the flow field.
    """
    # Get valid indices from the binary mask
    valid_indices = np.argwhere(mask != 0)

    # Extract the flow vectors corresponding to valid indices
    flow_vectors = flow[valid_indices[:, 0], valid_indices[:, 1]]

    # Initialize KMedoids with num_points clusters
    kmedoids = KMedoids(n_clusters=num_points, random_state=0)

    # Fit the KMedoids model to the flow vectors
    kmedoids.fit(flow_vectors)

    # Get the cluster medoids (representative points) indices
    medoids_indices = valid_indices[kmedoids.medoid_indices_]
    # Convert indices to (x, y) pairs
    medoids_locations = np.array([(x, y) for y, x in medoids_indices])

    return medoids_locations


def sample_grid_in_polygon(polygon_points, grid_size=None):
    """
    Samples a grid with approximately the given number of points inside a polygon's bounding box.

    Args:
        polygon_points (list of tuples): List of (x, y) points defining the polygon.
        grid_size (float or None): Size of the grid cells. If None, the grid size will be adjusted
                                   to produce an 8x8 points grid. Default is None.

    Returns:
        numpy.ndarray: An array of (x, y) pairs representing 
        the grid points sampled inside the polygon's bounding box.
    """
    # Create a Shapely polygon from the given points
    polygon = Polygon(polygon_points)

    # Get the bounding box of the polygon
    min_x, min_y, max_x, max_y = polygon.bounds

    # Calculate the size of the bounding box
    bbox_width = max_x - min_x
    bbox_height = max_y - min_y

    # If grid_size is not specified, adjust it to produce an 8x8 points grid
    if grid_size is None:
        # Calculate the number of points per unit length to achieve approximately 8x8 grid
        points_per_unit_length = np.sqrt(64 / (bbox_width * bbox_height))
        # Calculate the grid size based on the number of points per unit length
        grid_size = 1 / points_per_unit_length

    # Generate grid points within the bounding box
    x_points = np.arange(np.ceil(min_x), np.floor(max_x) + 1, grid_size)
    y_points = np.arange(np.ceil(min_y), np.floor(max_y) + 1, grid_size)
    grid_points = np.array(np.meshgrid(x_points, y_points)).T.reshape(-1, 2)

    # Filter points that lie within the polygon
    valid_points = [point for point in grid_points if Point(
        point).within(polygon)]

    return np.array(valid_points)
