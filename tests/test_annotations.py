import numpy as np
from annolid.annotation import masks


def test_mask_to_polygons():
    one_mask = np.ones((100, 100))
    polygons, has_holes = masks.mask_to_polygons(one_mask)
    expected_array = np.array([0.5, 0.5, 0.5, 99.5, 99.5, 99.5, 99.5, 0.5])
    assert not has_holes
    assert np.array_equal(polygons[0], expected_array)


def test_mask_to_polygons_does_not_over_simplify_large_shapes():
    import cv2

    mask = np.zeros((512, 512), dtype=np.uint8)
    cv2.circle(mask, (256, 256), 200, 1, -1)
    polygons, has_holes = masks.mask_to_polygons(mask)
    assert not has_holes
    assert polygons
    num_points = len(polygons[0]) // 2
    assert num_points >= 12
