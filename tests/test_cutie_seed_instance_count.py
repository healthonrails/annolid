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
from annolid.utils.annotation_store import AnnotationStore  # noqa: E402


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


def test_select_seed_frames_prefers_nearest_prior_seed_without_frame_zero_anchor() -> (
    None
):
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
    assert [seed.frame_index for seed in selected] == [100, 200]


def test_resolve_start_frame_for_seed_backfill_restarts_from_zero_when_gap_exists() -> (
    None
):
    intervals = CutieCoreVideoProcessor._build_frame_intervals({0, 99})
    resolved = CutieCoreVideoProcessor._resolve_start_frame_for_seed_backfill(
        requested_start_frame=99,
        seed_frame_indices=[0, 99],
        labeled_intervals=intervals,
    )
    assert resolved == 0


def test_resolve_start_frame_for_seed_backfill_keeps_requested_when_covered() -> None:
    intervals = CutieCoreVideoProcessor._build_frame_intervals(set(range(0, 100)))
    resolved = CutieCoreVideoProcessor._resolve_start_frame_for_seed_backfill(
        requested_start_frame=99,
        seed_frame_indices=[0, 99],
        labeled_intervals=intervals,
    )
    assert resolved == 99


def test_json_has_manual_seed_content_accepts_parseable_json(tmp_path) -> None:
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

    assert CutieCoreVideoProcessor._json_has_manual_seed_content(json_path) is True


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


def test_json_has_manual_seed_content_rejects_invalid_json(tmp_path) -> None:
    json_path = tmp_path / "seed.json"
    json_path.write_text("{invalid_json", encoding="utf-8")
    assert CutieCoreVideoProcessor._json_has_manual_seed_content(json_path) is False


def test_apply_inference_brightness_contrast_is_noop_when_disabled() -> None:
    processor = CutieCoreVideoProcessor.__new__(CutieCoreVideoProcessor)
    processor._frame_preprocess_enabled = False
    processor._frame_preprocess_alpha = 1.0
    processor._frame_preprocess_beta = 0.0
    frame = np.full((3, 3, 3), 100, dtype=np.uint8)

    output = processor._apply_inference_brightness_contrast(frame)
    assert output is frame


def test_apply_inference_brightness_contrast_adjusts_frame() -> None:
    processor = CutieCoreVideoProcessor.__new__(CutieCoreVideoProcessor)
    processor._frame_preprocess_enabled = True
    processor._frame_preprocess_alpha = 1.5
    processor._frame_preprocess_beta = 8.0
    frame = np.full((4, 4, 3), 80, dtype=np.uint8)

    output = processor._apply_inference_brightness_contrast(frame)
    assert output is not None
    assert output.shape == frame.shape
    assert output.dtype == np.uint8
    assert float(output.mean()) > float(frame.mean())


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


def test_discover_seed_frames_force_refresh_bypasses_cache(tmp_path) -> None:
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

        _write_seed(200)
        refreshed = CutieCoreVideoProcessor.discover_seed_frames(
            str(video_path), results_dir, force_refresh=True
        )
        assert [seed.frame_index for seed in refreshed] == [100, 200]
    finally:
        CutieCoreVideoProcessor._DISCOVERED_SEEDS_CACHE.clear()
        CutieCoreVideoProcessor._DISCOVERED_SEEDS_CACHE.update(old_cache)


def test_discover_seed_frames_includes_motion_index_only_pair(tmp_path) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"")
    results_dir = video_path.with_suffix("")
    results_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{results_dir.name}_000000020"
    (results_dir / f"{stem}.png").write_bytes(b"")
    (results_dir / f"{stem}.json").write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": "mouse",
                        "shape_type": "polygon",
                        "points": [[1, 1], [4, 1], [4, 4], [1, 4]],
                        "description": "motion_index: 0.11",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    old_cache = dict(CutieCoreVideoProcessor._DISCOVERED_SEEDS_CACHE)
    try:
        CutieCoreVideoProcessor._DISCOVERED_SEEDS_CACHE.clear()
        seeds = CutieCoreVideoProcessor.discover_seed_frames(
            str(video_path), results_dir, force_refresh=True
        )
        assert [seed.frame_index for seed in seeds] == [20]
    finally:
        CutieCoreVideoProcessor._DISCOVERED_SEEDS_CACHE.clear()
        CutieCoreVideoProcessor._DISCOVERED_SEEDS_CACHE.update(old_cache)


def test_process_segment_does_not_backfill_missing_seed_instance_by_default(
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
    processor.auto_fill_missing_instances = False
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
    assert saved_masks["labels"] == {"mouse"}
    assert saved_masks["notes"] == {}


def test_process_segment_automatic_pause_stops_on_missing_instances(
    monkeypatch,
) -> None:
    class _DummyInferenceCore:
        def __init__(self, *_args, **_kwargs):
            pass

        def step(self, *_args, **_kwargs):
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
    processor.auto_fill_missing_instances = False
    processor.automatic_pause_enabled = True
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
        has_occlusion=True,
    )

    assert should_halt is True
    assert (
        message
        == "There is 1 missing instance in the current frame (0).\n\nMissing or occluded: teaball#0"
    )
    assert saved_masks["labels"] == {"mouse"}
    assert saved_masks["notes"] == {}


def test_process_segment_keeps_occlusion_mode_when_automatic_pause_disabled(
    monkeypatch,
) -> None:
    class _DummyInferenceCore:
        def __init__(self, *_args, **_kwargs):
            pass

        def step(self, *_args, **_kwargs):
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
    processor.auto_fill_missing_instances = False
    processor.automatic_pause_enabled = False
    processor.continue_on_missing_instances = False
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
        has_occlusion=True,
    )

    assert should_halt is False
    assert message == "Stop at frame:\n#0"
    assert saved_masks["labels"] == {"mouse"}
    assert saved_masks["notes"] == {}


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


def test_process_segment_skips_persist_for_already_labeled_frames(
    monkeypatch,
) -> None:
    class _DummyInferenceCore:
        def __init__(self, *_args, **_kwargs):
            pass

        def step(self, frame_torch, mask_tensor=None, objects=None, **_kwargs):
            _ = frame_torch, mask_tensor, objects
            return np.array([[1, 2], [0, 0]], dtype=np.int32)

    class _DummyCap:
        def __init__(self, frames=3):
            self._frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(frames)]
            self._idx = 0

        def isOpened(self):
            return self._idx < len(self._frames)

        def set(self, _prop, value):
            self._idx = int(value)
            return True

        def get(self, _prop):
            return self._idx

        def read(self):
            if self._idx >= len(self._frames):
                return False, None
            frame = self._frames[self._idx]
            self._idx += 1
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
    processor.auto_fill_missing_instances = False
    processor.continue_on_missing_instances = True
    processor.debug = False
    processor._optical_flow_kwargs = {}
    processor.optical_flow_backend = "farneback"
    processor._recent_instance_masks = {}
    processor._recent_instance_mask_frames = {}
    processor._last_saved_instance_masks = {}
    processor._should_stop = lambda _worker=None: False
    processor._flow_hsv = None
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
    processor._update_recent_instance_masks = lambda *_args, **_kwargs: None

    saved_frames = []

    def _capture_save(*_args, **_kwargs):
        saved_frames.append(int(processor._frame_number))

    processor._save_annotation_with_notes = _capture_save

    monkeypatch.setattr(cutie_predict, "InferenceCore", _DummyInferenceCore)
    monkeypatch.setattr(
        cutie_predict, "image_to_torch", lambda frame, device=None: frame
    )
    monkeypatch.setattr(cutie_predict, "torch_prob_to_numpy_mask", lambda pred: pred)

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
        existing_labeled_frames={1},
    )

    assert should_halt is False
    assert message == "Stop at frame:\n#2"
    assert saved_frames == [0, 2]


def test_process_segment_fast_skips_completed_spans_without_postprocess(
    monkeypatch,
) -> None:
    class _DummyInferenceCore:
        def __init__(self, *_args, **_kwargs):
            self.calls = 0

        def step(self, frame_torch, mask_tensor=None, objects=None, **_kwargs):
            _ = frame_torch, mask_tensor, objects
            self.calls += 1
            return np.array([[1, 2], [0, 0]], dtype=np.int32)

    class _DummyCap:
        def __init__(self, frames=4):
            self._frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(frames)]
            self._idx = 0

        def isOpened(self):
            return self._idx < len(self._frames)

        def set(self, _prop, value):
            self._idx = int(value)
            return True

        def get(self, _prop):
            return self._idx

        def read(self):
            if self._idx >= len(self._frames):
                return False, None
            frame = self._frames[self._idx]
            self._idx += 1
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
    processor.auto_fill_missing_instances = False
    processor.continue_on_missing_instances = True
    processor.debug = False
    processor._optical_flow_kwargs = {}
    processor.optical_flow_backend = "farneback"
    processor._recent_instance_masks = {}
    processor._recent_instance_mask_frames = {}
    processor._last_saved_instance_masks = {}
    processor._should_stop = lambda _worker=None: False
    processor._flow_hsv = None
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
    processor._update_recent_instance_masks = lambda *_args, **_kwargs: None

    saved_frames = []

    def _capture_save(*_args, **_kwargs):
        saved_frames.append(int(processor._frame_number))

    processor._save_annotation_with_notes = _capture_save

    convert_calls = {"count": 0}

    def _capture_convert(pred):
        convert_calls["count"] += 1
        return pred

    monkeypatch.setattr(cutie_predict, "InferenceCore", _DummyInferenceCore)
    monkeypatch.setattr(
        cutie_predict, "image_to_torch", lambda frame, device=None: frame
    )
    monkeypatch.setattr(cutie_predict, "torch_prob_to_numpy_mask", _capture_convert)

    segment = SeedSegment(
        seed=_seed(0),
        start_frame=0,
        end_frame=3,
        mask=np.array([[1, 0], [0, 2]], dtype=np.int32),
        labels_map={"_background_": 0, "mouse": 1, "teaball": 2},
        active_labels=["mouse", "teaball"],
    )

    message, should_halt = processor._process_segment(
        cap=_DummyCap(frames=4),
        segment=segment,
        end_frame=3,
        fps=30.0,
        existing_labeled_frames={1, 2},
    )

    assert should_halt is False
    assert message == "Stop at frame:\n#3"
    assert saved_frames == [0, 3]
    # Frames 1 and 2 take the fast-skip path, so conversion runs only for 0 and 3.
    assert convert_calls["count"] == 2


def test_process_segment_skips_inference_for_completed_tail_segment(
    monkeypatch,
) -> None:
    class _DummyInferenceCore:
        def __init__(self, *_args, **_kwargs):
            self.calls = 0

        def step(self, frame_torch, mask_tensor=None, objects=None, **_kwargs):
            _ = frame_torch, mask_tensor, objects
            self.calls += 1
            return np.array([[1, 2], [0, 0]], dtype=np.int32)

    class _DummyCap:
        def __init__(self, frames=4):
            self._frames = [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(frames)]
            self._idx = 0

        def isOpened(self):
            return self._idx < len(self._frames)

        def set(self, _prop, value):
            self._idx = int(value)
            return True

        def get(self, _prop):
            return self._idx

        def read(self):
            if self._idx >= len(self._frames):
                return False, None
            frame = self._frames[self._idx]
            self._idx += 1
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
    processor.auto_fill_missing_instances = False
    processor.continue_on_missing_instances = True
    processor.debug = False
    processor._optical_flow_kwargs = {}
    processor.optical_flow_backend = "farneback"
    processor._recent_instance_masks = {}
    processor._recent_instance_mask_frames = {}
    processor._last_saved_instance_masks = {}
    processor._should_stop = lambda _worker=None: False
    processor._flow_hsv = None
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

    created_cores = []

    def _core_factory(*_args, **_kwargs):
        core = _DummyInferenceCore(*_args, **_kwargs)
        created_cores.append(core)
        return core

    convert_calls = {"count": 0}

    def _capture_convert(pred):
        convert_calls["count"] += 1
        return pred

    monkeypatch.setattr(cutie_predict, "InferenceCore", _core_factory)
    monkeypatch.setattr(
        cutie_predict, "image_to_torch", lambda frame, device=None: frame
    )
    monkeypatch.setattr(cutie_predict, "torch_prob_to_numpy_mask", _capture_convert)

    segment = SeedSegment(
        seed=_seed(0),
        start_frame=0,
        end_frame=3,
        mask=np.array([[1, 0], [0, 2]], dtype=np.int32),
        labels_map={"_background_": 0, "mouse": 1, "teaball": 2},
        active_labels=["mouse", "teaball"],
    )

    message, should_halt = processor._process_segment(
        cap=_DummyCap(frames=4),
        segment=segment,
        end_frame=3,
        fps=30.0,
        existing_labeled_frames={2, 3},
    )

    assert should_halt is False
    assert message == "Stop at frame:\n#3"
    assert len(created_cores) == 1
    # Tail [2,3] is fully completed, so inference runs only for frames 0 and 1.
    assert created_cores[0].calls == 2
    assert convert_calls["count"] == 2


def test_save_annotation_falls_back_to_previous_mask_on_frame_sized_artifact(
    monkeypatch, tmp_path
) -> None:
    class _Cache:
        def __init__(self):
            self._bbox = (0.0, 0.0, 1.0, 1.0)

        def add_bbox(self, _key, bbox):
            self._bbox = tuple(float(v) for v in bbox)

        def get_most_recent_bbox(self, _key):
            return self._bbox

    class _Point:
        def __init__(self, x, y):
            self._x = float(x)
            self._y = float(y)

        def x(self):
            return self._x

        def y(self):
            return self._y

    class _PolyShape:
        def __init__(self, points):
            self.points = points

    class _FakeMaskShape:
        def __init__(self, label, flags=None, description=""):
            self.label = label
            self.flags = flags or {}
            self.description = description
            self.other_data = {}
            self.mask = None

        def toPolygons(self, epsilon=2.0):
            _ = epsilon
            mask = np.asarray(self.mask).astype(bool)
            ys, xs = np.where(mask)
            if xs.size == 0 or ys.size == 0:
                return []
            minx, maxx = int(xs.min()), int(xs.max())
            miny, maxy = int(ys.min()), int(ys.max())
            return [
                _PolyShape(
                    [
                        _Point(minx, miny),
                        _Point(maxx, miny),
                        _Point(maxx, maxy),
                        _Point(minx, maxy),
                    ]
                )
            ]

    processor = CutieCoreVideoProcessor.__new__(CutieCoreVideoProcessor)
    processor._frame_number = 1
    processor.epsilon_for_polygon = 2.0
    processor.reject_suspicious_mask_jumps = False
    processor._last_mask_area_ratio = {}
    processor._recent_instance_masks = {
        "mouse": np.array(
            [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]], dtype=bool
        )
    }
    processor._recent_instance_mask_frames = {"mouse": 0}
    processor._last_saved_instance_masks = {}
    processor.cache = _Cache()
    processor._flow_hsv = None
    processor.showing_KMedoids_in_mask = False

    processor._sanitize_full_frame_artifact = lambda label, mask, frame_area: (
        np.asarray(mask).astype(bool),
        False,
    )
    processor._is_suspicious_mask_jump = lambda label, mask, frame_area: False
    processor._save_results = lambda label, mask: (1.0, 2.0, -1.0)
    processor.save_KMedoids_in_mask = lambda label_list, mask: None

    def _reject(label, mask, points, frame_area):
        _ = label, points, frame_area
        return bool(np.count_nonzero(mask) >= (mask.size - 1))

    processor._should_reject_frame_sized_prediction = _reject

    saved = {"count": 0}

    def _capture_save_labels(**kwargs):
        saved["count"] = len(kwargs.get("label_list") or [])

    monkeypatch.setattr(cutie_predict, "MaskShape", _FakeMaskShape)
    monkeypatch.setattr(cutie_predict, "save_labels", _capture_save_labels)

    bad_mask = np.ones((4, 4), dtype=bool)
    processor._save_annotation_with_notes(
        filename=str(tmp_path / "frame.json"),
        mask_dict={"mouse": bad_mask},
        frame_shape=(4, 4, 3),
        shape_notes={},
    )

    assert saved["count"] == 1
    assert "mouse" in processor._last_saved_instance_masks
    assert np.array_equal(
        processor._last_saved_instance_masks["mouse"],
        processor._recent_instance_masks["mouse"],
    )


def test_repetitive_warning_logger_logs_first_and_periodic(monkeypatch) -> None:
    processor = CutieCoreVideoProcessor.__new__(CutieCoreVideoProcessor)
    logged = []

    def _capture_warning(msg, *args):
        if args:
            msg = msg % args
        logged.append(str(msg))

    monkeypatch.setattr(cutie_predict.logger, "warning", _capture_warning)

    for frame_idx in range(1, 5):
        processor._frame_number = frame_idx
        processor._log_repetitive_warning(
            ("frame_sized_rejected", "mouse"),
            "CUTIE frame-sized artifact rejected for 'mouse' at frame %s." % frame_idx,
            every=3,
        )

    assert any(
        "CUTIE frame-sized artifact rejected for 'mouse' at frame 1." in msg
        for msg in logged
    )
    assert any(
        "Suppressed 1 repetitive warning(s) for frame_sized_rejected/mouse." in msg
        for msg in logged
    )
    assert any(
        "CUTIE frame-sized artifact rejected for 'mouse' at frame 3." in msg
        for msg in logged
    )


def test_collect_labeled_frame_indices_persists_manual_seed_stats(tmp_path) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"")
    results_dir = video_path.with_suffix("")
    results_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 10
    stem = f"{results_dir.name}_{frame_idx:09d}"
    (results_dir / f"{stem}.png").write_bytes(b"png")
    (results_dir / f"{stem}.json").write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": "mouse",
                        "shape_type": "polygon",
                        "points": [[1, 1], [3, 1], [3, 3], [1, 3]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    processor = CutieCoreVideoProcessor.__new__(CutieCoreVideoProcessor)
    processor.video_name = str(video_path)
    processor.video_folder = results_dir
    processor._cached_labeled_frames = None
    processor._tracking_stats_cache = None
    processor._tracking_stats_dirty = False
    processor._tracking_stats_pending_updates = 0

    labeled = processor._collect_labeled_frame_indices()
    assert labeled == {frame_idx}

    stats_path = results_dir / f"{results_dir.name}_tracking_stats.json"
    assert stats_path.exists()
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    frame_stats = payload["frame_stats"][str(frame_idx)]
    assert "manual_seed" in frame_stats["sources"]
    assert frame_stats["json_exists"] is True
    assert frame_stats["png_exists"] is True
    assert payload["summary"]["manual_frames"] >= 1
    assert [frame_idx, frame_idx] in payload["summary"]["manual_segments"]


def test_save_annotation_updates_prediction_tracking_stats(
    monkeypatch, tmp_path
) -> None:
    class _Cache:
        def add_bbox(self, _key, _bbox):
            return None

        def get_most_recent_bbox(self, _key):
            return None

    class _Point:
        def __init__(self, x, y):
            self._x = float(x)
            self._y = float(y)

        def x(self):
            return self._x

        def y(self):
            return self._y

    class _PolyShape:
        def __init__(self, points):
            self.points = points

    class _FakeMaskShape:
        def __init__(self, label, flags=None, description=""):
            self.label = label
            self.flags = flags or {}
            self.description = description
            self.other_data = {}
            self.mask = None

        def toPolygons(self, epsilon=2.0):
            _ = epsilon
            mask = np.asarray(self.mask).astype(bool)
            ys, xs = np.where(mask)
            if xs.size == 0 or ys.size == 0:
                return []
            minx, maxx = int(xs.min()), int(xs.max())
            miny, maxy = int(ys.min()), int(ys.max())
            return [
                _PolyShape(
                    [
                        _Point(minx, miny),
                        _Point(maxx, miny),
                        _Point(maxx, maxy),
                        _Point(minx, maxy),
                    ]
                )
            ]

    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"")
    results_dir = video_path.with_suffix("")
    results_dir.mkdir(parents=True, exist_ok=True)

    processor = CutieCoreVideoProcessor.__new__(CutieCoreVideoProcessor)
    processor.video_name = str(video_path)
    processor.video_folder = results_dir
    processor._frame_number = 3
    processor.epsilon_for_polygon = 2.0
    processor.reject_suspicious_mask_jumps = False
    processor._last_mask_area_ratio = {}
    processor._recent_instance_masks = {}
    processor._recent_instance_mask_frames = {}
    processor._last_saved_instance_masks = {}
    processor.cache = _Cache()
    processor._flow_hsv = None
    processor.showing_KMedoids_in_mask = False
    processor._tracking_stats_cache = None
    processor._tracking_stats_dirty = False
    processor._tracking_stats_pending_updates = 0

    processor._sanitize_full_frame_artifact = lambda label, mask, frame_area: (
        np.asarray(mask).astype(bool),
        False,
    )
    processor._is_suspicious_mask_jump = lambda label, mask, frame_area: False
    processor._save_results = lambda label, mask: (1.0, 2.0, -1.0)
    processor.save_KMedoids_in_mask = lambda label_list, mask: None
    processor._should_reject_frame_sized_prediction = (
        lambda label, mask, points, frame_area: False
    )

    monkeypatch.setattr(cutie_predict, "MaskShape", _FakeMaskShape)
    monkeypatch.setattr(cutie_predict, "save_labels", lambda **_kwargs: None)

    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    frame_json = results_dir / f"{results_dir.name}_000000003.json"
    processor._save_annotation_with_notes(
        filename=str(frame_json),
        mask_dict={"mouse": mask},
        frame_shape=(4, 4, 3),
        shape_notes={},
    )
    processor._flush_tracking_stats(force=True)

    stats_path = results_dir / f"{results_dir.name}_tracking_stats.json"
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    # Normal prediction-only frames are not persisted in stats JSON.
    assert "3" not in payload["frame_stats"]
    assert payload["summary"]["manual_frames"] == 0


def test_record_prediction_segment_updates_tracking_stats(tmp_path) -> None:
    results_dir = tmp_path / "clip"
    results_dir.mkdir(parents=True, exist_ok=True)
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"")

    processor = CutieCoreVideoProcessor.__new__(CutieCoreVideoProcessor)
    processor.video_name = str(video_path)
    processor.video_folder = results_dir
    processor._tracking_stats_cache = None
    processor._tracking_stats_dirty = False
    processor._tracking_stats_pending_updates = 0

    processor._record_prediction_segment(0, 99, "halted")
    processor._flush_tracking_stats(force=True)

    stats_path = results_dir / f"{results_dir.name}_tracking_stats.json"
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    segments = payload.get("prediction_segments", [])
    assert len(segments) == 1
    assert segments[0]["start_frame"] == 0
    assert segments[0]["end_frame"] == 99
    assert segments[0]["status"] == "halted"


def test_tracking_stats_persist_missing_instance_frame_stats(tmp_path) -> None:
    processor = CutieCoreVideoProcessor.__new__(CutieCoreVideoProcessor)
    processor.video_name = str(tmp_path / "clip.mp4")
    processor.video_folder = tmp_path / "clip"
    processor._tracking_stats_cache = None
    processor._tracking_stats_dirty = False
    processor._tracking_stats_pending_updates = 0

    processor._update_tracking_frame_stat(
        12,
        source="prediction",
        missing_instance_count=2,
        missing_instance_labels=["mouse_a", "mouse_b"],
        unresolved_missing_instance_count=1,
        unresolved_missing_instance_labels=["mouse_b"],
    )
    payload = processor._build_tracking_stats_persist_payload(
        processor._load_tracking_stats()
    )

    frame_stats = payload["frame_stats"]
    assert "12" in frame_stats
    assert int(frame_stats["12"]["missing_instance_count"]) == 2
    assert int(frame_stats["12"]["unresolved_missing_instance_count"]) == 1


def test_tracking_stats_missing_instance_fields_can_be_cleared_on_rerun(
    tmp_path,
) -> None:
    processor = CutieCoreVideoProcessor.__new__(CutieCoreVideoProcessor)
    processor.video_name = str(tmp_path / "clip.mp4")
    processor.video_folder = tmp_path / "clip"
    processor._tracking_stats_cache = None
    processor._tracking_stats_dirty = False
    processor._tracking_stats_pending_updates = 0

    processor._update_tracking_frame_stat(
        12,
        source="prediction",
        missing_instance_count=1,
        missing_instance_labels=["stim_2"],
        unresolved_missing_instance_count=1,
        unresolved_missing_instance_labels=["stim_2"],
    )
    processor._update_tracking_frame_stat(
        12,
        source="prediction",
        missing_instance_count=0,
        missing_instance_labels=[],
        unresolved_missing_instance_count=0,
        unresolved_missing_instance_labels=[],
    )

    payload = processor._build_tracking_stats_persist_payload(
        processor._load_tracking_stats()
    )

    assert "12" not in payload["frame_stats"]
    assert int(payload["summary"]["missing_instance_frames"]) == 0


def test_collect_labeled_frame_indices_treats_empty_store_records_as_completed(
    tmp_path,
) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"")
    results_dir = video_path.with_suffix("")
    results_dir.mkdir(parents=True, exist_ok=True)

    store_path = results_dir / f"{results_dir.name}{AnnotationStore.STORE_SUFFIX}"
    store = AnnotationStore(store_path)
    store.append_frame({"frame": 50, "shapes": []})
    store.append_frame(
        {
            "frame": 51,
            "shapes": [{"shape_type": "polygon", "points": [[0, 0], [1, 0], [1, 1]]}],
        }
    )

    processor = CutieCoreVideoProcessor.__new__(CutieCoreVideoProcessor)
    processor.video_name = str(video_path)
    processor.video_folder = results_dir
    processor._cached_labeled_frames = None
    processor._tracking_stats_cache = None
    processor._tracking_stats_dirty = False
    processor._tracking_stats_pending_updates = 0

    labeled = processor._collect_labeled_frame_indices()
    assert 50 in labeled
    assert 51 in labeled


def test_collect_labeled_frame_indices_treats_empty_json_as_completed(tmp_path) -> None:
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"")
    results_dir = video_path.with_suffix("")
    results_dir.mkdir(parents=True, exist_ok=True)

    frame_idx = 42
    stem = f"{results_dir.name}_{frame_idx:09d}"
    (results_dir / f"{stem}.json").write_text(
        json.dumps({"shapes": []}), encoding="utf-8"
    )

    processor = CutieCoreVideoProcessor.__new__(CutieCoreVideoProcessor)
    processor.video_name = str(video_path)
    processor.video_folder = results_dir
    processor._cached_labeled_frames = None
    processor._tracking_stats_cache = None
    processor._tracking_stats_dirty = False
    processor._tracking_stats_pending_updates = 0

    labeled = processor._collect_labeled_frame_indices()
    assert frame_idx in labeled


def test_save_annotation_records_unresolved_bad_shape_stats(
    monkeypatch, tmp_path
) -> None:
    class _NoPolygonMaskShape:
        def __init__(self, label, flags=None, description=""):
            self.label = label
            self.flags = flags or {}
            self.description = description
            self.other_data = {}
            self.mask = None

        def toPolygons(self, epsilon=2.0):
            _ = epsilon
            return []

    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"")
    results_dir = video_path.with_suffix("")
    results_dir.mkdir(parents=True, exist_ok=True)

    processor = CutieCoreVideoProcessor.__new__(CutieCoreVideoProcessor)
    processor.video_name = str(video_path)
    processor.video_folder = results_dir
    processor._frame_number = 7
    processor.epsilon_for_polygon = 2.0
    processor.reject_suspicious_mask_jumps = False
    processor._last_mask_area_ratio = {}
    processor._recent_instance_masks = {}
    processor._recent_instance_mask_frames = {}
    processor._last_saved_instance_masks = {}
    processor.cache = types.SimpleNamespace(
        add_bbox=lambda *_args, **_kwargs: None,
        get_most_recent_bbox=lambda *_args, **_kwargs: None,
    )
    processor._flow_hsv = None
    processor.showing_KMedoids_in_mask = False
    processor._tracking_stats_cache = None
    processor._tracking_stats_dirty = False
    processor._tracking_stats_pending_updates = 0

    processor._sanitize_full_frame_artifact = lambda label, mask, frame_area: (
        np.asarray(mask).astype(bool),
        False,
    )
    processor._is_suspicious_mask_jump = lambda label, mask, frame_area: False
    processor._save_results = lambda label, mask: (1.0, 2.0, -1.0)
    processor.save_KMedoids_in_mask = lambda label_list, mask: None
    processor._should_reject_frame_sized_prediction = (
        lambda label, mask, points, frame_area: False
    )
    processor._repair_bad_shape_mask = lambda label, mask: (None, None)

    monkeypatch.setattr(cutie_predict, "MaskShape", _NoPolygonMaskShape)
    monkeypatch.setattr(cutie_predict, "save_labels", lambda **_kwargs: None)

    mask = np.zeros((4, 4), dtype=bool)
    mask[1:3, 1:3] = True
    frame_json = results_dir / f"{results_dir.name}_000000007.json"
    processor._save_annotation_with_notes(
        filename=str(frame_json),
        mask_dict={"mouse": mask},
        frame_shape=(4, 4, 3),
        shape_notes={},
    )
    processor._flush_tracking_stats(force=True)

    stats_path = results_dir / f"{results_dir.name}_tracking_stats.json"
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    assert payload["summary"]["bad_shape_failed_frames"] >= 1
    events = payload.get("bad_shape_events", [])
    assert any(
        event.get("frame") == 7
        and event.get("label") == "mouse"
        and event.get("resolved") is False
        for event in events
    )


def test_save_annotation_repairs_bad_shape_and_records_resolved_stats(
    monkeypatch, tmp_path
) -> None:
    class _Point:
        def __init__(self, x, y):
            self._x = float(x)
            self._y = float(y)

        def x(self):
            return self._x

        def y(self):
            return self._y

    class _PolyShape:
        def __init__(self, points):
            self.points = points

    class _AreaThresholdMaskShape:
        def __init__(self, label, flags=None, description=""):
            self.label = label
            self.flags = flags or {}
            self.description = description
            self.other_data = {}
            self.mask = None

        def toPolygons(self, epsilon=2.0):
            _ = epsilon
            mask = np.asarray(self.mask).astype(bool)
            if int(np.count_nonzero(mask)) < 4:
                return []
            ys, xs = np.where(mask)
            minx, maxx = int(xs.min()), int(xs.max())
            miny, maxy = int(ys.min()), int(ys.max())
            return [
                _PolyShape(
                    [
                        _Point(minx, miny),
                        _Point(maxx, miny),
                        _Point(maxx, maxy),
                        _Point(minx, maxy),
                    ]
                )
            ]

    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"")
    results_dir = video_path.with_suffix("")
    results_dir.mkdir(parents=True, exist_ok=True)

    processor = CutieCoreVideoProcessor.__new__(CutieCoreVideoProcessor)
    processor.video_name = str(video_path)
    processor.video_folder = results_dir
    processor._frame_number = 11
    processor.epsilon_for_polygon = 2.0
    processor.reject_suspicious_mask_jumps = False
    processor._last_mask_area_ratio = {}
    processor._recent_instance_masks = {}
    processor._recent_instance_mask_frames = {}
    processor._last_saved_instance_masks = {}
    processor.cache = types.SimpleNamespace(
        add_bbox=lambda *_args, **_kwargs: None,
        get_most_recent_bbox=lambda *_args, **_kwargs: None,
    )
    processor._flow_hsv = None
    processor.showing_KMedoids_in_mask = False
    processor._tracking_stats_cache = None
    processor._tracking_stats_dirty = False
    processor._tracking_stats_pending_updates = 0

    processor._sanitize_full_frame_artifact = lambda label, mask, frame_area: (
        np.asarray(mask).astype(bool),
        False,
    )
    processor._is_suspicious_mask_jump = lambda label, mask, frame_area: False
    processor._save_results = lambda label, mask: (1.0, 2.0, -1.0)
    processor.save_KMedoids_in_mask = lambda label_list, mask: None
    processor._should_reject_frame_sized_prediction = (
        lambda label, mask, points, frame_area: False
    )

    saved = {"count": 0}
    monkeypatch.setattr(cutie_predict, "MaskShape", _AreaThresholdMaskShape)
    monkeypatch.setattr(
        cutie_predict,
        "save_labels",
        lambda **kwargs: saved.__setitem__(
            "count", len(kwargs.get("label_list") or [])
        ),
    )

    tiny_mask = np.zeros((4, 4), dtype=bool)
    tiny_mask[2, 2] = True
    frame_json = results_dir / f"{results_dir.name}_000000011.json"
    processor._save_annotation_with_notes(
        filename=str(frame_json),
        mask_dict={"mouse": tiny_mask},
        frame_shape=(4, 4, 3),
        shape_notes={},
    )
    processor._flush_tracking_stats(force=True)

    assert saved["count"] == 1
    stats_path = results_dir / f"{results_dir.name}_tracking_stats.json"
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    events = payload.get("bad_shape_events", [])
    assert any(
        event.get("frame") == 11
        and event.get("label") == "mouse"
        and event.get("resolved") is True
        for event in events
    )
