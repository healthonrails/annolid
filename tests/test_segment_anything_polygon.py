import numpy as np

from annolid.segmentation.SAM.segment_anything import (
    _compute_polygon_from_points,
    _select_best_mask_index,
)


def _square_flat(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.array([x1, y1, x2, y1, x2, y2, x1, y2], dtype=np.float32)


def test_compute_polygon_prefers_polygon_covering_positive_prompt(monkeypatch):
    def _fake_mask(*args, **kwargs):
        _ = args, kwargs
        return np.ones((32, 32), dtype=bool)

    def _fake_polys(mask):
        _ = mask
        # First polygon does not contain the positive click; second does.
        return [
            _square_flat(0, 0, 10, 10),
            _square_flat(60, 60, 95, 95),
        ], False

    monkeypatch.setattr(
        "annolid.segmentation.SAM.segment_anything._compute_mask_from_points",
        _fake_mask,
    )
    monkeypatch.setattr("annolid.annotation.masks.mask_to_polygons", _fake_polys)

    points = [[80.0, 80.0]]
    labels = [1]
    polygon = _compute_polygon_from_points(
        image_size=1024,
        decoder_session=None,
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        image_embedding=np.zeros((1,), dtype=np.float32),
        points=points,
        point_labels=labels,
    )

    assert polygon.shape[0] >= 4
    assert np.all(polygon[:, 0] >= 59.5)
    assert np.all(polygon[:, 1] >= 59.5)


def test_compute_polygon_avoids_polygon_covering_negative_prompt(monkeypatch):
    def _fake_mask(*args, **kwargs):
        _ = args, kwargs
        return np.ones((32, 32), dtype=bool)

    def _fake_polys(mask):
        _ = mask
        # Both contain the positive point, but first also contains a negative point.
        return [
            _square_flat(20, 20, 90, 90),
            _square_flat(60, 60, 95, 95),
        ], False

    monkeypatch.setattr(
        "annolid.segmentation.SAM.segment_anything._compute_mask_from_points",
        _fake_mask,
    )
    monkeypatch.setattr("annolid.annotation.masks.mask_to_polygons", _fake_polys)

    points = [[80.0, 80.0], [30.0, 30.0]]
    labels = [1, 0]
    polygon = _compute_polygon_from_points(
        image_size=1024,
        decoder_session=None,
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        image_embedding=np.zeros((1,), dtype=np.float32),
        points=points,
        point_labels=labels,
    )

    assert polygon.shape[0] >= 4
    assert np.all(polygon[:, 0] >= 59.5)
    assert np.all(polygon[:, 1] >= 59.5)


def test_select_best_mask_index_prefers_prompt_consistent_local_mask():
    masks = np.zeros((1, 2, 10, 10), dtype=np.float32)
    # Candidate 0: very large region, includes positive click.
    masks[0, 0, :, :] = 1.0
    # Candidate 1: local object region around positive click.
    masks[0, 1, 6:9, 6:9] = 1.0

    idx = _select_best_mask_index(
        masks=masks,
        scores=np.array([[0.9, 0.7]], dtype=np.float32),
        points=np.array([[7.0, 7.0]], dtype=np.float32),
        point_labels=np.array([1], dtype=np.int32),
    )
    assert idx == 1


def test_select_best_mask_index_penalizes_negative_prompt_hits():
    masks = np.zeros((1, 2, 10, 10), dtype=np.float32)
    # Candidate 0: includes both positive and negative points.
    masks[0, 0, 2:9, 2:9] = 1.0
    # Candidate 1: includes positive point, excludes negative point.
    masks[0, 1, 6:9, 6:9] = 1.0

    idx = _select_best_mask_index(
        masks=masks,
        scores=np.array([[0.95, 0.5]], dtype=np.float32),
        points=np.array([[7.0, 7.0], [3.0, 3.0]], dtype=np.float32),
        point_labels=np.array([1, 0], dtype=np.int32),
    )
    assert idx == 1
