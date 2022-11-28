from shapely.geometry import Polygon


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
