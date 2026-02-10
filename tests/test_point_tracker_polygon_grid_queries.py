from __future__ import annotations

import torch
import pytest

pytest.importorskip("shapely")
from annolid.tracker.base import BasePointTracker


class _DummyPointTracker(BasePointTracker):
    def load_model(self):
        return None

    def process_video_frames(self, *args, **kwargs):
        return ""


def _make_tracker(*, track_polygon_grid_points: bool) -> _DummyPointTracker:
    tracker = _DummyPointTracker.__new__(_DummyPointTracker)
    tracker.video_height = 200
    tracker.video_width = 300
    tracker.device = torch.device("cpu")
    tracker.mask = None
    tracker.mask_label = None
    tracker.point_labels = []
    tracker.track_polygon_grid_points = track_polygon_grid_points
    tracker.polygon_grid_size = 10
    return tracker


def test_polygon_generates_grid_point_queries_without_description() -> None:
    tracker = _make_tracker(track_polygon_grid_points=True)
    shapes = [
        {
            "label": "teaball",
            "shape_type": "polygon",
            "points": [[20, 20], [120, 20], [120, 120], [20, 120]],
            "description": "",
        }
    ]

    queries = tracker._process_shapes(shapes, frame_number=1353)

    assert len(queries) > 0
    assert len(tracker.point_labels) == len(queries)
    assert all(label == "teaball" for label in tracker.point_labels)
    assert all(int(q[0]) == 1353 for q in queries)


def test_polygon_grid_queries_can_be_disabled() -> None:
    tracker = _make_tracker(track_polygon_grid_points=False)
    shapes = [
        {
            "label": "teaball",
            "shape_type": "polygon",
            "points": [[20, 20], [120, 20], [120, 120], [20, 120]],
            "description": "",
        }
    ]

    queries = tracker._process_shapes(shapes, frame_number=10)

    assert queries == []
    assert tracker.point_labels == []
