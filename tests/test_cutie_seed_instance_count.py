from pathlib import Path
import sys
import types
import json

import numpy as np
import torch

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

import annolid.segmentation.cutie_vos.predict as cutie_predict  # noqa: E402
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


def test_build_frame_intervals_merges_contiguous_ranges() -> None:
    intervals = CutieCoreVideoProcessor._build_frame_intervals({5, 6, 7, 10, 12, 13})
    assert intervals == [(5, 7), (10, 10), (12, 13)]


def test_segment_already_completed_uses_interval_coverage() -> None:
    segment = SeedSegment(
        seed=_seed(100),
        start_frame=100,
        end_frame=109,
        mask=np.array([[0, 1], [1, 0]], dtype=np.int32),
        labels_map={"_background_": 0, "mouse": 1},
        active_labels=["mouse"],
    )
    labeled = set(range(100, 110))
    intervals = CutieCoreVideoProcessor._build_frame_intervals(labeled)
    assert (
        CutieCoreVideoProcessor._segment_already_completed(
            segment, 109, labeled, intervals
        )
        is True
    )

    labeled.remove(105)
    intervals = CutieCoreVideoProcessor._build_frame_intervals(labeled)
    assert (
        CutieCoreVideoProcessor._segment_already_completed(
            segment, 109, labeled, intervals
        )
        is False
    )


def test_select_seed_frames_uses_nearest_prior_for_multi_seed_start() -> None:
    processor = CutieCoreVideoProcessor.__new__(CutieCoreVideoProcessor)
    processor._seed_frames = [_seed(100), _seed(200), _seed(300)]
    processor._seed_segment_lookup = {
        100: SeedSegment(
            seed=_seed(100),
            start_frame=100,
            end_frame=None,
            mask=np.array([[0, 1], [1, 0]], dtype=np.int32),
            labels_map={"_background_": 0, "mouse": 1},
            active_labels=["mouse"],
        ),
        200: SeedSegment(
            seed=_seed(200),
            start_frame=200,
            end_frame=None,
            mask=np.array([[0, 1], [1, 0]], dtype=np.int32),
            labels_map={"_background_": 0, "mouse": 1},
            active_labels=["mouse"],
        ),
        300: SeedSegment(
            seed=_seed(300),
            start_frame=300,
            end_frame=None,
            mask=np.array([[0, 1], [1, 0]], dtype=np.int32),
            labels_map={"_background_": 0, "mouse": 1},
            active_labels=["mouse"],
        ),
    }

    selected = processor._select_seed_frames_for_start(start_frame=301)
    assert [seed.frame_index for seed in selected] == [300]


def test_select_seed_frames_keeps_frame_zero_anchor_when_present() -> None:
    processor = CutieCoreVideoProcessor.__new__(CutieCoreVideoProcessor)
    processor._seed_frames = [_seed(0), _seed(100), _seed(200)]
    processor._seed_segment_lookup = {
        0: SeedSegment(
            seed=_seed(0),
            start_frame=0,
            end_frame=None,
            mask=np.array([[0, 1], [1, 0]], dtype=np.int32),
            labels_map={"_background_": 0, "mouse": 1},
            active_labels=["mouse"],
        ),
        100: SeedSegment(
            seed=_seed(100),
            start_frame=100,
            end_frame=None,
            mask=np.array([[0, 1], [1, 0]], dtype=np.int32),
            labels_map={"_background_": 0, "mouse": 1},
            active_labels=["mouse"],
        ),
        200: SeedSegment(
            seed=_seed(200),
            start_frame=200,
            end_frame=None,
            mask=np.array([[0, 1], [1, 0]], dtype=np.int32),
            labels_map={"_background_": 0, "mouse": 1},
            active_labels=["mouse"],
        ),
    }

    selected = processor._select_seed_frames_for_start(start_frame=150)
    assert [seed.frame_index for seed in selected] == [0, 100, 200]


def test_json_has_manual_seed_content_rejects_auto_motion_index(tmp_path) -> None:
    json_path = tmp_path / "seed.json"
    payload = {
        "shapes": [
            {
                "label": "mouse",
                "shape_type": "polygon",
                "points": [[1, 1], [4, 1], [4, 4], [1, 4]],
                "description": "motion_index: 0.123",
            }
        ]
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    assert CutieCoreVideoProcessor._json_has_manual_seed_content(json_path) is False


def test_json_has_manual_seed_content_accepts_user_polygon(tmp_path) -> None:
    json_path = tmp_path / "seed.json"
    payload = {
        "shapes": [
            {
                "label": "mouse",
                "shape_type": "polygon",
                "points": [[1, 1], [4, 1], [4, 4], [1, 4]],
                "description": "",
            }
        ]
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    assert CutieCoreVideoProcessor._json_has_manual_seed_content(json_path) is True


def test_discover_seed_frames_uses_cache_for_whole_video(tmp_path) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"")
    results_dir = video_path.with_suffix("")
    results_dir.mkdir(parents=True, exist_ok=True)

    def _write_seed(frame: int) -> None:
        stem = f"{results_dir.name}_{frame:09d}"
        (results_dir / f"{stem}.png").write_bytes(b"")
        (results_dir / f"{stem}.json").write_text(
            json.dumps(
                {
                    "shapes": [
                        {
                            "label": "mouse",
                            "shape_type": "polygon",
                            "points": [[1, 1], [4, 1], [4, 4], [1, 4]],
                            "description": "",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    old_cache = dict(CutieCoreVideoProcessor._DISCOVERED_SEEDS_CACHE)
    try:
        CutieCoreVideoProcessor._DISCOVERED_SEEDS_CACHE.clear()
        _write_seed(100)
        first = CutieCoreVideoProcessor.discover_seed_frames(
            str(video_path), results_dir
        )
        assert [seed.frame_index for seed in first] == [100]

        # Add another seed after first discovery; cached results should be reused.
        _write_seed(200)
        second = CutieCoreVideoProcessor.discover_seed_frames(
            str(video_path), results_dir
        )
        assert [seed.frame_index for seed in second] == [100]
    finally:
        CutieCoreVideoProcessor._DISCOVERED_SEEDS_CACHE.clear()
        CutieCoreVideoProcessor._DISCOVERED_SEEDS_CACHE.update(old_cache)


def test_process_segment_backfills_missing_seed_instance_on_start_frame(
    monkeypatch,
) -> None:
    class _DummyInferenceCore:
        def __init__(self, *_args, **_kwargs):
            pass

        def step(self, *_args, **_kwargs):
            # Model prediction misses object id 2 on the seed frame.
            return np.array([[1, 0], [0, 0]], dtype=np.int32)

    class _DummyCap:
        def __init__(self):
            self._done = False
            self._pos = 0

        def get(self, _prop):
            return self._pos

        def set(self, _prop, value):
            self._pos = int(value)

        def isOpened(self):
            return not self._done

        def read(self):
            if self._done:
                return False, None
            self._done = True
            frame = np.zeros((2, 2, 3), dtype=np.uint8)
            return True, frame

    processor = CutieCoreVideoProcessor.__new__(CutieCoreVideoProcessor)
    processor.cutie = object()
    processor.cfg = types.SimpleNamespace(amp=False)
    processor.device = "cpu"
    processor.video_folder = Path("clip")
    processor.label_registry = {"_background_": 0, "mouse": 1, "teaball": 2}
    processor._global_label_names = {}
    processor.compute_optical_flow = False
    processor.auto_missing_instance_recovery = False
    processor.continue_on_missing_instances = True
    processor.debug = False
    processor._optical_flow_kwargs = {}
    processor.optical_flow_backend = "farneback"
    processor._recent_instance_masks = {}
    processor._recent_instance_mask_frames = {}

    saved_masks = {}
    processor._should_stop = lambda _worker=None: False
    processor.commit_masks_into_permanent_memory = lambda *_args, **_kwargs: {
        "_background_": 0,
        "mouse": 1,
        "teaball": 2,
    }
    processor._build_object_mask_tensor = lambda _mask: (
        torch.zeros((2, 2, 2), dtype=torch.float32),
        [1, 2],
    )
    processor._register_active_objects = lambda _ids: None
    processor._save_annotation_with_notes = (
        lambda _filename, mask_dict, _shape, shape_notes=None: saved_masks.update(
            {
                "labels": set(mask_dict.keys()),
                "notes": dict(shape_notes or {}),
            }
        )
    )
    processor._update_recent_instance_masks = lambda *_args, **_kwargs: None

    monkeypatch.setattr(cutie_predict, "InferenceCore", _DummyInferenceCore)
    monkeypatch.setattr(
        cutie_predict, "image_to_torch", lambda frame, device=None: frame
    )
    monkeypatch.setattr(cutie_predict, "torch_prob_to_numpy_mask", lambda pred: pred)

    segment = SeedSegment(
        seed=_seed(0),
        start_frame=0,
        end_frame=0,
        mask=np.array([[1, 0], [0, 2]], dtype=np.int32),
        labels_map={"_background_": 0, "mouse": 1, "teaball": 2},
        active_labels=["mouse", "teaball"],
    )

    message, should_halt = processor._process_segment(
        cap=_DummyCap(),
        segment=segment,
        end_frame=0,
        fps=30.0,
    )

    assert should_halt is False
    assert message == "Stop at frame:\n#0"
    assert saved_masks["labels"] == {"mouse", "teaball"}
    assert saved_masks["notes"]["teaball"] == "filled_from_seed_mask(start_frame)"


def test_process_segment_suppresses_repetitive_missing_instance_logs(
    monkeypatch,
) -> None:
    class _DummyInferenceCore:
        def __init__(self, *_args, **_kwargs):
            pass

        def step(self, *_args, **_kwargs):
            # Persistently miss object id 2 after the seed frame.
            return np.array([[1, 0], [0, 0]], dtype=np.int32)

    class _DummyCap:
        def __init__(self, frames: int):
            self._frames = frames
            self._pos = 0

        def get(self, _prop):
            return self._pos

        def set(self, _prop, value):
            self._pos = int(value)

        def isOpened(self):
            return self._pos < self._frames

        def read(self):
            if self._pos >= self._frames:
                return False, None
            self._pos += 1
            frame = np.zeros((2, 2, 3), dtype=np.uint8)
            return True, frame

    processor = CutieCoreVideoProcessor.__new__(CutieCoreVideoProcessor)
    processor.cutie = object()
    processor.cfg = types.SimpleNamespace(amp=False)
    processor.device = "cpu"
    processor.video_folder = Path("clip")
    processor.label_registry = {"_background_": 0, "mouse": 1, "teaball": 2}
    processor._global_label_names = {}
    processor.compute_optical_flow = False
    processor.auto_missing_instance_recovery = False
    processor.continue_on_missing_instances = True
    processor.debug = False
    processor._optical_flow_kwargs = {}
    processor.optical_flow_backend = "farneback"
    processor._recent_instance_masks = {}
    processor._recent_instance_mask_frames = {}
    processor._should_stop = lambda _worker=None: False
    processor.commit_masks_into_permanent_memory = lambda *_args, **_kwargs: {
        "_background_": 0,
        "mouse": 1,
        "teaball": 2,
    }
    processor._build_object_mask_tensor = lambda _mask: (
        torch.zeros((2, 2, 2), dtype=torch.float32),
        [1, 2],
    )
    processor._register_active_objects = lambda _ids: None
    processor._save_annotation_with_notes = lambda *_args, **_kwargs: None
    processor._update_recent_instance_masks = lambda *_args, **_kwargs: None

    log_messages = []

    def _capture_info(msg, *args):
        if args:
            msg = msg % args
        log_messages.append(str(msg))

    monkeypatch.setattr(cutie_predict, "InferenceCore", _DummyInferenceCore)
    monkeypatch.setattr(
        cutie_predict, "image_to_torch", lambda frame, device=None: frame
    )
    monkeypatch.setattr(cutie_predict, "torch_prob_to_numpy_mask", lambda pred: pred)
    monkeypatch.setattr(cutie_predict.logger, "info", _capture_info)

    segment = SeedSegment(
        seed=_seed(0),
        start_frame=0,
        end_frame=2,
        mask=np.array([[1, 0], [0, 2]], dtype=np.int32),
        labels_map={"_background_": 0, "mouse": 1, "teaball": 2},
        active_labels=["mouse", "teaball"],
    )

    message, should_halt = processor._process_segment(
        cap=_DummyCap(frames=3),
        segment=segment,
        end_frame=2,
        fps=30.0,
    )

    assert should_halt is False
    assert message == "Stop at frame:\n#2"
    repetitive_missing_logs = [
        msg
        for msg in log_messages
        if "There is 1 missing instance in the current frame" in msg
    ]
    assert len(repetitive_missing_logs) == 1
