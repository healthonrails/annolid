from pathlib import Path
import sys
import types

import numpy as np

sys.modules.setdefault(
    "gdown", types.SimpleNamespace(cached_download=lambda *args, **kwargs: None)
)
_geometry_stub = types.SimpleNamespace(
    Polygon=lambda *args, **kwargs: None,
    Point=lambda *args, **kwargs: None,
)
sys.modules.setdefault("shapely", types.SimpleNamespace(geometry=_geometry_stub))
sys.modules.setdefault("shapely.geometry", _geometry_stub)


class _EasyDict(dict):
    __getattr__ = dict.get


sys.modules.setdefault("easydict", types.SimpleNamespace(EasyDict=_EasyDict))

from annolid.segmentation.cutie_vos.predict import (  # noqa: E402
    CutieCoreVideoProcessor,
    SeedFrame,
    SeedSegment,
)


def _seed(frame_index: int) -> SeedFrame:
    return SeedFrame(
        frame_index=frame_index,
        png_path=Path(f"seed_{frame_index:09d}.png"),
        json_path=Path(f"seed_{frame_index:09d}.json"),
    )


def test_count_tracking_instances_uses_only_active_seed_labels() -> None:
    processor = CutieCoreVideoProcessor.__new__(CutieCoreVideoProcessor)
    processor._seed_segment_lookup = {
        0: SeedSegment(
            seed=_seed(0),
            start_frame=0,
            end_frame=None,
            mask=np.array([[0, 1], [1, 0]], dtype=np.int32),
            labels_map={"_background_": 0, "mouse": 1, "teaball": 2},
            active_labels=["mouse"],
        )
    }

    assert processor._count_tracking_instances([_seed(0)]) == 1


class _DummyCache:
    def __init__(self, bbox):
        self._bbox = bbox

    def get_most_recent_bbox(self, _label):
        return self._bbox


def test_rejects_frame_sized_artifact_when_history_is_small() -> None:
    processor = CutieCoreVideoProcessor.__new__(CutieCoreVideoProcessor)
    processor.cache = _DummyCache((10.0, 10.0, 30.0, 30.0))
    processor._last_mask_area_ratio = {"mouse": 0.12}

    mask = np.ones((100, 100), dtype=bool)
    points = [[0.0, 0.0], [99.0, 0.0], [99.0, 99.0], [0.0, 99.0], [0.0, 0.0]]

    assert (
        processor._should_reject_frame_sized_prediction(
            label="mouse", mask=mask, points=points, frame_area=10000.0
        )
        is True
    )


def test_allows_normal_polygon_mask() -> None:
    processor = CutieCoreVideoProcessor.__new__(CutieCoreVideoProcessor)
    processor.cache = _DummyCache((10.0, 10.0, 60.0, 60.0))
    processor._last_mask_area_ratio = {"mouse": 0.2}

    mask = np.zeros((100, 100), dtype=bool)
    mask[20:60, 20:60] = True
    points = [[20.0, 20.0], [59.0, 20.0], [59.0, 59.0], [20.0, 59.0], [20.0, 20.0]]

    assert (
        processor._should_reject_frame_sized_prediction(
            label="mouse", mask=mask, points=points, frame_area=10000.0
        )
        is False
    )
