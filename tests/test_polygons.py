from annolid.annotation.polygons import polygon_iou


def test_poly_iou():
    this_polygon = [[0, 1], [1, 0], [0, 0]]
    other_polygon = [[0, 1], [1, 0], [0, 0]]
    assert polygon_iou(this_polygon, other_polygon) - 1.0 <= 0.00001
