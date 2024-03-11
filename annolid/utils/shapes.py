import labelme
import numpy as np
import uuid
from shapely.geometry import Polygon


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
            bboxes.append((y1, x1, y2, x2))
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
    try:
        from sklearn_extra.cluster import KMedoids
    except:
        print("Please install: pip install scikit-learn-extra >= 0.3.0")
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
