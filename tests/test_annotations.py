import numpy as np
from annolid.annotation import masks


def test_mask_to_polygons():
    one_mask = np.ones((100, 100))
    polygons, has_holes = masks.mask_to_polygons(one_mask)
    expected_array = np.array([0.5,  0.5,  0.5,
                               99.5, 99.5, 99.5, 99.5,  0.5])
    assert not has_holes
    assert np.array_equal(polygons[0], expected_array)
