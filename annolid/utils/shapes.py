import labelme
import numpy as np
import uuid
from shapely.geometry import Polygon


def shapes_to_label(img_shape, shapes, label_name_to_value):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for shape in shapes:
        points = [(point.x(), point.y()) for point in shape.points]
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
