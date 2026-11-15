"""Regression coverage for keypoint tracking, multi-animal metadata, smoothing, and QC."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest
import torch
from PIL import Image

from annolid.annotation.keypoint_visibility import (
    KeypointVisibility,
    visibility_from_labelme_shape,
)
from annolid.annotation.pose_schema import PoseSchema
from annolid.gui.mixins.prediction_execution_mixin import PredictionExecutionMixin
from annolid.postprocessing.track_qc import evaluate_track_qc
from annolid.segmentation.dino_kpseg import DinoKPSEGPredictor, TemporalFusionConfig
from annolid.segmentation.dino_kpseg.predictor import _KeypointTrackStateStore
from annolid.tracker.base import BasePointTracker
from annolid.tracker.tapnext_onnx.track import TapNextOnnxProcessor
from annolid.tracking.annotation_adapter import AnnotationAdapter
from annolid.tracking.configuration import CutieDinoTrackerConfig
from annolid.tracking.cutie_mask_manager import MaskResult
from annolid.tracking.dino_keypoint_tracker import (
    DinoKeypointTracker,
    DinoKeypointVideoProcessor,
)
from annolid.tracking.dino_kpseg_annotations import DinoKPSEGAnnotationParser
from annolid.tracking.dino_kpseg_tracker import DinoKPSEGVideoProcessor
from annolid.tracking.domain import InstanceRegistry, KeypointState
from annolid.tracking.kpseg_smoothing import KeypointSmoother
from annolid.utils.annotation_store import load_labelme_json


class _DummyExtractor:
    patch_size = 1

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._queue: List[torch.Tensor] = []

    def set_queue(self, tensors: List[torch.Tensor]) -> None:
        self._queue = list(tensors)

    def extract(self, _image, return_layer="all", normalize=True):
        if not self._queue:
            return torch.ones((2, 8, 8), dtype=torch.float32)
        return self._queue.pop(0)

    def _compute_resized_hw(self, width: int, height: int) -> Tuple[int, int]:
        return height, width


def test_annotation_adapter_disambiguates_same_class_polygons_and_preserves_group_ids(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "two_mice"
    out_dir.mkdir()
    seed_path = out_dir / "two_mice_000000000.json"
    payload = {
        "shapes": [
            {
                "label": "mouse",
                "shape_type": "polygon",
                "group_id": 0,
                "points": [[5, 5], [25, 5], [25, 25], [5, 25]],
            },
            {
                "label": "mouse",
                "shape_type": "polygon",
                "group_id": 1,
                "points": [[40, 40], [60, 40], [60, 60], [40, 60]],
            },
            {
                "label": "nose",
                "shape_type": "point",
                "group_id": 0,
                "points": [[12, 12]],
            },
            {
                "label": "nose",
                "shape_type": "point",
                "group_id": 1,
                "points": [[48, 48]],
            },
        ]
    }
    seed_path.write_text(json.dumps(payload), encoding="utf-8")

    adapter = AnnotationAdapter(image_height=80, image_width=80, persist_json=True)
    registry = adapter.read_annotation(seed_path)

    assert set(registry.instances.keys()) == {"mouse_0", "mouse_1"}
    assert registry.instances["mouse_0"].group_id == 0
    assert registry.instances["mouse_1"].group_id == 1

    # Round-trip twice to verify polygon display labels ("mouse") and group_ids
    # stay stable across repeated saves/reads.
    frame1 = adapter.write_annotation(
        frame_number=1, registry=registry, output_dir=out_dir
    )
    reloaded = adapter.read_annotation(frame1)
    frame2 = adapter.write_annotation(
        frame_number=2, registry=reloaded, output_dir=out_dir
    )
    saved = json.loads(frame2.read_text(encoding="utf-8"))

    polygons = [s for s in saved["shapes"] if s["shape_type"] == "polygon"]
    points = [s for s in saved["shapes"] if s["shape_type"] == "point"]
    assert len(polygons) == 2
    assert {s["label"] for s in polygons} == {"mouse"}
    assert {s["group_id"] for s in polygons} == {0, 1}
    assert len(points) == 2
    pts_by_gid = {s["group_id"]: s["points"][0] for s in points}
    assert pts_by_gid[0] == [12.0, 12.0]
    assert pts_by_gid[1] == [48.0, 48.0]


def test_annotation_adapter_prevents_key_collision_and_roundtrips_occluded_visibility(
    tmp_path: Path,
) -> None:
    out_dir = tmp_path / "clip"
    out_dir.mkdir()
    seed_path = out_dir / "clip_000000000.json"
    payload = {
        "shapes": [
            {
                "label": "1_nose",
                "shape_type": "point",
                "points": [[10.0, 15.0]],
                "flags": {
                    "instance_label": "mouse",
                    "display_label": "1_nose",
                    "kp_visibility": 1,
                    "kp_visible": False,
                },
                "visible": True,
            },
            {
                "label": "_nose",
                "shape_type": "point",
                "points": [[30.0, 35.0]],
                "flags": {
                    "instance_label": "mouse1",
                    "display_label": "_nose",
                },
                "visible": True,
            },
        ]
    }
    seed_path.write_text(json.dumps(payload), encoding="utf-8")

    adapter = AnnotationAdapter(image_height=64, image_width=64, persist_json=True)
    registry = adapter.read_annotation(seed_path)
    all_kps = [kp for inst in registry for kp in inst.keypoints.values()]
    assert len(all_kps) == 2
    assert len({kp.key for kp in all_kps}) == 2

    occluded_kp = next(kp for kp in all_kps if kp.instance_label == "mouse")
    visible_kp = next(kp for kp in all_kps if kp.instance_label == "mouse1")
    assert occluded_kp.visible is False
    assert visible_kp.visible is True

    written = adapter.write_annotation(
        frame_number=1, registry=registry, output_dir=out_dir
    )
    saved_shapes = json.loads(written.read_text(encoding="utf-8"))["shapes"]
    by_label = {s["label"]: s for s in saved_shapes}
    assert visibility_from_labelme_shape(by_label["1_nose"]) == int(
        KeypointVisibility.OCCLUDED
    )
    assert by_label["1_nose"]["visible"] is False


def test_inferred_left_right_symmetry_pairs_and_pose_schema_propagation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        _DummyExtractor,
    )
    cfg = CutieDinoTrackerConfig(
        symmetry_penalty=1.0,
        appearance_bundle_weight=0.0,
        baseline_similarity_weight=0.0,
        structural_consistency_weight=0.0,
    )
    tracker = DinoKeypointTracker(
        model_name="dummy", runtime_config=cfg, search_radius=2
    )
    tracker.extractor.set_queue([torch.ones((3, 4, 5), dtype=torch.float32)])  # type: ignore[attr-defined]

    registry = InstanceRegistry()
    registry.register_keypoint(
        KeypointState(
            key="mouseleft_forepaw",
            instance_label="mouse",
            label="left_forepaw",
            x=1.0,
            y=1.0,
        )
    )
    registry.register_keypoint(
        KeypointState(
            key="mouseright_forepaw",
            instance_label="mouse",
            label="right_forepaw",
            x=3.0,
            y=1.0,
        )
    )
    tracker.start(Image.fromarray(np.zeros((4, 5, 3), dtype=np.uint8)), registry, {})

    assert tracker.tracks["mouseleft_forepaw"].symmetry_partner == "mouseright_forepaw"
    assert tracker.tracks["mouseright_forepaw"].symmetry_partner == "mouseleft_forepaw"

    host = PredictionExecutionMixin.__new__(PredictionExecutionMixin)
    host._pose_schema = PoseSchema(
        keypoints=["medial_claw", "lateral_claw"],
        symmetry_pairs=[("medial_claw", "lateral_claw")],
        instances=["fly1"],
    )
    gui_cfg = CutieDinoTrackerConfig()
    host._apply_pose_schema_to_tracker_config(gui_cfg)
    assert ("medial_claw", "lateral_claw") in gui_cfg.symmetry_pairs
    assert ("fly1_medial_claw", "fly1_lateral_claw") in gui_cfg.symmetry_pairs


def test_dino_video_processor_resumes_from_persisted_tracked_frame_without_gap_jump(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _StubVideo:
        def __init__(self, _path: str) -> None:
            self.frames = [
                np.full((8, 8, 3), fill_value=idx, dtype=np.uint8) for idx in range(6)
            ]

        def get_first_frame(self):
            return self.frames[0]

        def total_frames(self) -> int:
            return len(self.frames)

        def load_frame(self, idx: int):
            return self.frames[idx] if 0 <= idx < len(self.frames) else None

        def get_fps(self) -> float:
            return 30.0

    class _RecordingMaskManager:
        instance = None

        def __init__(self, *_args, **_kwargs) -> None:
            self.enabled = True
            self.prime_frames: List[int] = []
            self.update_frames: List[int] = []
            _RecordingMaskManager.instance = self

        def reset_state(self) -> None:
            pass

        def prime(self, frame_number: int, _frame, _registry) -> None:
            self.prime_frames.append(int(frame_number))

        def update_masks(self, frame_number: int, _frame, registry):
            self.update_frames.append(int(frame_number))
            return {
                inst.label: MaskResult(
                    inst.label,
                    np.ones((8, 8), dtype=bool),
                    [(1.0, 1.0), (6.0, 1.0), (6.0, 6.0), (1.0, 6.0)],
                )
                for inst in registry
            }

    class _RecordingTracker:
        instance = None

        def __init__(self, *_args, runtime_config=None, **_kwargs) -> None:
            self.runtime_config = runtime_config or CutieDinoTrackerConfig()
            self.start_calls: List[Tuple[int, List[Tuple[float, float]]]] = []
            self.preserve_flags: List[bool] = []
            self.update_frames: List[int] = []
            _RecordingTracker.instance = self

        def reset_state(self, *, preserve_manual_anchors: bool = False) -> None:
            self.preserve_flags.append(bool(preserve_manual_anchors))

        def start(self, image, registry, _mask_lookup) -> None:
            fid = int(np.asarray(image)[0, 0, 0])
            coords = [
                (float(kp.x), float(kp.y))
                for inst in registry
                for kp in inst.keypoints.values()
            ]
            self.start_calls.append((fid, coords))

        def update(self, image, _mask_lookup):
            fid = int(np.asarray(image)[0, 0, 0])
            self.update_frames.append(fid)
            return [
                {
                    "id": "animalnose",
                    "x": float(fid),
                    "y": float(fid),
                    "visible": True,
                    "confidence": 0.9,
                    "quality": 0.9,
                }
            ]

    monkeypatch.setattr("annolid.tracking.dino_keypoint_tracker.CV2Video", _StubVideo)
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.CutieMaskManager",
        _RecordingMaskManager,
    )
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.DinoKeypointTracker",
        _RecordingTracker,
    )

    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"mp4")
    out_dir = tmp_path / "clip"
    out_dir.mkdir()

    # Frame 0 is the only manual PNG+JSON seed.
    manual_adapter = AnnotationAdapter(image_height=8, image_width=8, persist_json=True)
    seed_reg = InstanceRegistry()
    seed_reg.ensure_instance("animal").set_mask(
        bitmap=None, polygon=[(1.0, 1.0), (6.0, 1.0), (6.0, 6.0), (1.0, 6.0)]
    )
    seed_reg.register_keypoint(
        KeypointState(
            key="animalnose", instance_label="animal", label="nose", x=1.0, y=1.0
        )
    )
    manual_adapter.write_annotation(
        frame_number=0, registry=seed_reg, output_dir=out_dir
    )
    Image.new("RGB", (8, 8)).save(out_dir / "clip_000000000.png")

    # Frames 1..3 were previously tracked into AnnotationStore ONLY (persist_json=False).
    store_adapter = AnnotationAdapter(image_height=8, image_width=8, persist_json=False)
    for tracked_idx in (1, 2, 3):
        reg = InstanceRegistry()
        reg.ensure_instance("animal").set_mask(
            bitmap=None, polygon=[(1.0, 1.0), (6.0, 1.0), (6.0, 6.0), (1.0, 6.0)]
        )
        reg.register_keypoint(
            KeypointState(
                key="animalnose",
                instance_label="animal",
                label="nose",
                x=float(tracked_idx),
                y=float(tracked_idx),
            )
        )
        store_adapter.write_annotation(
            frame_number=tracked_idx, registry=reg, output_dir=out_dir
        )
        assert not (out_dir / f"clip_{tracked_idx:09d}.json").exists()

    processor = DinoKeypointVideoProcessor(
        video_path=str(video_path),
        result_folder=out_dir,
        model_name="dummy",
        runtime_config=CutieDinoTrackerConfig(
            persist_labelme_json=False,
            tracking_smoother_enable=True,
        ),
    )
    processor.process_video(start_frame=4, end_frame=5, step=1)

    mm = _RecordingMaskManager.instance
    trk = _RecordingTracker.instance
    assert mm is not None and trk is not None
    # Cutie primes at persisted frame 3 (not jumping from 0 -> 4), while DINO
    # anchors manual codebook at frame 0 then starts tracking state at frame 3!
    assert mm.prime_frames == [3]
    assert mm.update_frames == [4, 5]
    assert [f for f, _ in trk.start_calls] == [0, 3]
    assert True in trk.preserve_flags
    assert trk.update_frames == [4, 5]


def test_dino_kpseg_does_not_mutate_caller_config_and_survives_missing_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _SparseVideo:
        def __init__(self, _path: str) -> None:
            self._frame = np.zeros((24, 24, 3), dtype=np.uint8)

        def get_first_frame(self):
            return self._frame.copy()

        def total_frames(self) -> int:
            return 4

        def load_frame(self, frame_number: int):
            if frame_number == 2:
                return None
            return self._frame.copy()

        def get_fps(self) -> float:
            return 30.0

    class _DummyMM:
        def __init__(self, _video_path: Path, adapter, config) -> None:
            self.enabled = True
            self._poly = [(4.0, 4.0), (18.0, 4.0), (18.0, 18.0), (4.0, 18.0)]
            self._mask = np.ones((24, 24), dtype=bool)

        def reset_state(self) -> None:
            pass

        def prime(self, *_args, **_kwargs) -> None:
            pass

        def update_masks(self, _frame_number: int, frame: np.ndarray, registry):
            assert frame is not None
            return {
                str(inst.label): MaskResult(
                    str(inst.label), self._mask.copy(), list(self._poly)
                )
                for inst in registry
            }

    class _DummyPred:
        keypoints_xy = [(10.0, 10.0)]
        keypoint_scores = [0.9]

    class _DummyPredictor:
        def __init__(self, _weights, device=None) -> None:
            self.keypoint_names = ["nose"]
            self.meta = type("Meta", (), {"short_side": 24, "model_name": "dummy"})()
            self.device = "cpu"

        def reset_state(self) -> None:
            pass

        def seed_instance_state(self, *_args, **_kwargs) -> None:
            pass

    monkeypatch.setattr("annolid.tracking.dino_kpseg_tracker.CV2Video", _SparseVideo)
    monkeypatch.setattr(
        "annolid.tracking.dino_kpseg_tracker.CutieMaskManager", _DummyMM
    )
    monkeypatch.setattr(
        "annolid.tracking.dino_kpseg_tracker.DinoKPSEGPredictor", _DummyPredictor
    )
    monkeypatch.setattr(
        "annolid.tracking.dino_kpseg_tracker.build_instance_crops",
        lambda _f, masks, **_kw: list(masks),
    )
    monkeypatch.setattr(
        "annolid.tracking.dino_kpseg_tracker.predict_on_instance_crops",
        lambda _p, crops, **_kw: [(int(gid), _DummyPred()) for gid, _ in crops],
    )

    out_dir = tmp_path / "vid"
    out_dir.mkdir()
    (out_dir / "vid_000000000.json").write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": "mouse",
                        "shape_type": "polygon",
                        "group_id": 0,
                        "points": [[4, 4], [18, 4], [18, 18], [4, 18]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (out_dir / "vid_000000000.png").write_bytes(b"")

    caller_cfg = CutieDinoTrackerConfig(
        use_cutie_tracking=False,
        restrict_to_initial_mask=False,
        persist_labelme_json=True,
    )
    analytics_events: List[Dict[str, object]] = []
    caller_cfg.analytics_hook = analytics_events.append

    processor = DinoKPSEGVideoProcessor(
        video_path=str(tmp_path / "vid.mp4"),
        result_folder=out_dir,
        kpseg_weights="dummy.pt",
        runtime_config=caller_cfg,
    )
    # Caller's config must remain untouched while processor enforces its contract.
    assert caller_cfg.use_cutie_tracking is False
    assert caller_cfg.restrict_to_initial_mask is False
    assert processor.config.use_cutie_tracking is True
    assert processor.config.restrict_to_initial_mask is True

    processor.process_video(start_frame=0, end_frame=3, step=1)
    assert (out_dir / "vid_000000001.json").exists()
    assert not (out_dir / "vid_000000002.json").exists()
    assert (out_dir / "vid_000000003.json").exists()
    assert [e["frame"] for e in analytics_events] == [1, 3]


def test_keypoint_smoother_ignores_ungated_initial_frames_and_nan_inputs() -> None:
    smoother = KeypointSmoother(
        mode="kalman",
        fps=30.0,
        min_score=0.4,
        kalman_gate_sigma=3.0,
        max_jump_px=15.0,
    )
    # Unreliable frame 0 should NOT initialize Kalman state at (999, 999).
    p0 = smoother.smooth("m0:nose", (999.0, 999.0), score=0.05, mask_ok=False)
    assert p0 == (999.0, 999.0)

    p1 = smoother.smooth("m0:nose", (10.0, 20.0), score=0.95, mask_ok=True)
    assert p1 == pytest.approx((10.0, 20.0))

    # Non-finite input falls back cleanly to predicted state without poisoning covariance.
    p2 = smoother.smooth(
        "m0:nose", (float("nan"), float("inf")), score=0.95, mask_ok=True
    )
    assert math.isfinite(p2[0]) and math.isfinite(p2[1])
    assert p2 == pytest.approx((10.0, 20.0))

    # Extreme outlier jump is bounded by max_jump_px.
    p3 = smoother.smooth("m0:nose", (500.0, 20.0), score=0.95, mask_ok=True)
    assert p3[0] <= 25.0 + 1e-5


def test_point_tracker_preserves_multi_animal_group_ids_and_all_polygon_grid_points(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _StubVideo:
        def __init__(self, _path: str) -> None:
            pass

        def get_first_frame(self):
            return np.zeros((100, 120, 3), dtype=np.uint8)

    class _DummyTracker(BasePointTracker):
        def load_model(self):
            return None

        def process_video_frames(self, *args, **kwargs):
            return ""

    monkeypatch.setattr("annolid.tracker.base.CV2Video", _StubVideo)

    custom_out = tmp_path / "custom_results"
    custom_out.mkdir()

    # Manual seed 0 (older) and manual seed 5 (active start_frame) both label mouse 0's nose;
    # seed 5 also has mouse 1's nose, a missing COCO placeholder (kp_visibility=0), and a polygon.
    (custom_out / "custom_results_000000000.json").write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": "nose",
                        "shape_type": "point",
                        "group_id": 0,
                        "points": [[1.0, 1.0]],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (custom_out / "custom_results_000000000.png").write_bytes(b"")

    (custom_out / "custom_results_000000005.json").write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": "nose",
                        "shape_type": "point",
                        "group_id": 0,
                        "points": [[15.0, 20.0]],
                    },
                    {
                        "label": "nose",
                        "shape_type": "point",
                        "group_id": 1,
                        "points": [[75.0, 80.0]],
                    },
                    {
                        "label": "unlabeled_ear",
                        "shape_type": "point",
                        "group_id": 0,
                        "points": [[0.0, 0.0]],
                        "flags": {"kp_visibility": 0, "kp_visible": False},
                    },
                    {
                        "label": "teaball",
                        "shape_type": "polygon",
                        "group_id": 9,
                        "points": [[20, 20], [60, 20], [60, 60], [20, 60]],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (custom_out / "custom_results_000000005.png").write_bytes(b"")

    tracker = _DummyTracker(
        str(tmp_path / "raw_video.mp4"),
        results_folder=str(custom_out),
        persist_json=True,
        polygon_grid_size=15,
    )
    tracker.start_frame = 5
    tracker.end_frame = 10

    queries = tracker.load_queries()
    grid_specs = [s for s in tracker.point_specs if s.label == "teaball"]
    nose_specs = [s for s in tracker.point_specs if s.label == "nose"]

    # Mouse 0's nose deduplicated to frame 5 ([15, 20]), Mouse 1's nose kept,
    # missing placeholder skipped, and multiple polygon grid points retained.
    assert len(nose_specs) == 2
    assert {s.group_id for s in nose_specs} == {0, 1}
    assert len(grid_specs) >= 2
    assert all(s.label != "unlabeled_ear" for s in tracker.point_specs)

    pts_for_frame6 = [
        (
            np.array([float(q[1]) + 2.0, float(q[2]) + 1.0], dtype=np.float32),
            idx % 2 == 0,
        )
        for idx, q in enumerate(queries.detach().cpu().numpy())
    ]
    tracker.save_frame_json(6, pts_for_frame6, description="CoTracker")

    saved = load_labelme_json(custom_out / "custom_results_000000006.json")
    saved_shapes = saved["shapes"]
    assert len(saved_shapes) == len(pts_for_frame6)
    saved_noses = {
        s["group_id"]: s["points"][0] for s in saved_shapes if s["label"] == "nose"
    }
    assert saved_noses[0] == pytest.approx([17.0, 21.0])
    assert saved_noses[1] == pytest.approx([77.0, 81.0])
    saved_grid = [s for s in saved_shapes if s["label"] == "teaball"]
    assert len(saved_grid) == len(grid_specs)


def test_tapnext_online_preserves_future_frame_seed_until_chunk_reaches_it() -> None:
    class _FakeVideoLoader:
        def total_frames(self) -> int:
            return 5

        def load_frame(self, frame_number: int) -> np.ndarray:
            return np.full((8, 8, 3), frame_number, dtype=np.uint8)

    processor = TapNextOnnxProcessor.__new__(TapNextOnnxProcessor)
    processor.video_width = 8
    processor.video_height = 8
    processor._input_width = 8
    processor._input_height = 8
    processor.visibility_threshold = 0.0
    processor.video_loader = _FakeVideoLoader()
    processor.start_frame = 0
    processor.end_frame = 4
    processor.device = torch.device("cpu")
    # Query 0 seeded at t=0, Query 1 seeded at future frame t=3 (x=42, y=24).
    processor.queries = torch.tensor(
        [[0.0, 5.0, 5.0], [3.0, 42.0, 24.0]], dtype=torch.float32
    )
    processor._temporal_window = 2
    processor.pred_worker = None
    processor._stop_triggered = False
    processor._should_stop_callback = lambda: False
    processor._ensure_model = lambda: object()

    seen_second_query: List[List[float]] = []

    def _run_query_batches(_session, frames, *, queries, chunk_start_frame):
        seen_second_query.append(queries[1].detach().cpu().tolist())
        tracks = np.zeros((len(frames), 2, 2), dtype=np.float32)
        tracks[:, 0, :] = [6.0, 6.0]
        # Before reaching t=3, model output for query 1 would be bogus (0, 0).
        tracks[:, 1, :] = (
            [0.0, 0.0] if chunk_start_frame + len(frames) - 1 < 3 else [43.0, 25.0]
        )
        return tracks, np.ones((len(frames), 2), dtype=bool)

    processor._run_query_batches = _run_query_batches
    processor.save_frame_json = lambda *_args, **_kwargs: None

    processor._process_video_online(
        grid_size=10, grid_query_frame=0, need_visualize=False
    )
    # Chunk [2, 3] (index 2) MUST still see the true future seed [3.0, 42.0, 24.0]!
    assert seen_second_query[0] == pytest.approx([3.0, 42.0, 24.0])
    assert seen_second_query[1] == pytest.approx([3.0, 42.0, 24.0])
    assert seen_second_query[2] == pytest.approx([3.0, 42.0, 24.0])
    # Once chunk [2, 3] completes (saved_until=3), chunk [3, 4] propagates [3.0, 43.0, 25.0].
    assert seen_second_query[3] == pytest.approx([3.0, 43.0, 25.0])


def test_track_qc_ignores_occluded_keypoints_and_separates_multi_animal_group_ids(
    tmp_path: Path,
) -> None:
    ann_dir = tmp_path / "pose_qc"
    ann_dir.mkdir()
    pose_schema = PoseSchema(
        keypoints=["nose", "tail_base"],
        edges=[("nose", "tail_base")],
    )

    for frame_idx in range(6):
        # Mouse 0 always has bone length 10px; on frame 4 its tail_base is occluded
        # and held at a far-away coordinate (200, 200) — must NOT trigger an outlier!
        # Mouse 1 normally has bone length 10px, and on frame 5 has a visible spike to 80px!
        m0_tail_occluded = frame_idx == 4
        m1_tail_x = 130.0 if frame_idx == 5 else 60.0
        shapes = [
            {
                "label": "mouse",
                "shape_type": "polygon",
                "group_id": 0,
                "points": [[8, 8], [24, 8], [24, 24], [8, 24]],
            },
            {
                "label": "mouse",
                "shape_type": "polygon",
                "group_id": 1,
                "points": [[48, 8], [64, 8], [64, 24], [48, 24]],
            },
            {
                "label": "nose",
                "shape_type": "point",
                "group_id": 0,
                "points": [[10.0, 10.0]],
            },
            {
                "label": "tail_base",
                "shape_type": "point",
                "group_id": 0,
                "points": [[200.0, 200.0] if m0_tail_occluded else [20.0, 10.0]],
                "flags": (
                    {"kp_visibility": 1, "kp_visible": False}
                    if m0_tail_occluded
                    else {}
                ),
            },
            {
                "label": "nose",
                "shape_type": "point",
                "group_id": 1,
                "points": [[50.0, 10.0]],
            },
            {
                "label": "tail_base",
                "shape_type": "point",
                "group_id": 1,
                "points": [[m1_tail_x, 10.0]],
            },
        ]
        (ann_dir / f"pose_qc_{frame_idx:09d}.json").write_text(
            json.dumps({"shapes": shapes}), encoding="utf-8"
        )

    report = evaluate_track_qc(ann_dir, pose_schema=pose_schema)
    edge_issues = [i for i in report.issues if i.code == "anatomical_edge_outlier"]
    assert len(edge_issues) == 1
    assert edge_issues[0].anchor_frame == 5
    assert edge_issues[0].instances == ("mouse_1",)


def test_dino_kpseg_temporal_fusion_gates_low_confidence_and_respects_mask_and_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _IdentityHead(torch.nn.Module):
        in_dim = 1

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class _StubExtractor:
        patch_size = 1

        def quantize_mask(self, mask_u8: np.ndarray) -> torch.Tensor:
            return torch.from_numpy((mask_u8 > 0).astype(np.float32))

    predictor = DinoKPSEGPredictor.__new__(DinoKPSEGPredictor)
    predictor.meta = type("Meta", (), {"threshold": 0.5})()
    predictor.keypoint_names = ["nose"]
    predictor.flip_idx = None
    predictor._symmetric_pairs = []
    predictor._track_state = _KeypointTrackStateStore()
    predictor._temporal_fusion = TemporalFusionConfig()
    predictor._max_track_instances = 256
    predictor.device = torch.device("cpu")
    predictor.extractor = _StubExtractor()
    predictor.head = _IdentityHead()

    predictor.set_temporal_fusion_config(
        {
            "enable": True,
            "alpha": 0.6,
            "low_conf_threshold": 0.5,
            "max_instances": 2,
        }
    )
    assert predictor._max_track_instances == 2

    # Frame 0: high-confidence peak at (2.5, 2.5) (logit +6 => sigmoid ~0.9975).
    f0 = torch.full((1, 10, 10), -10.0, dtype=torch.float32)
    f0[0, 2, 2] = 6.0
    pred0 = predictor.predict_from_features(f0, frame_shape=(10, 10), instance_id=1)
    assert pred0.keypoints_xy[0] == pytest.approx((2.5, 2.5))

    # Frame 1: low-confidence peak at (8.5, 2.5) (logit -1.1 => sigmoid ~0.2497 < 0.5).
    # Temporal fusion pulls it toward (2.5, 2.5) while keeping raw score un-inflated.
    f1 = torch.full((1, 10, 10), -10.0, dtype=torch.float32)
    f1[0, 2, 8] = -1.1
    pred1 = predictor.predict_from_features(f1, frame_shape=(10, 10), instance_id=1)
    assert 2.5 < pred1.keypoints_xy[0][0] < 8.5
    assert pred1.keypoint_scores[0] < 0.3

    # Frame 2: high-confidence jump to (7.5, 2.5) (logit +6 >= 0.5) has zero lag!
    f2 = torch.full((1, 10, 10), -10.0, dtype=torch.float32)
    f2[0, 2, 7] = 6.0
    pred2 = predictor.predict_from_features(f2, frame_shape=(10, 10), instance_id=1)
    assert pred2.keypoints_xy[0] == pytest.approx((7.5, 2.5))

    # Mask safety: if a disconnected mask has foreground at x=0..1 and x=8..9 with a
    # background gap at x=2..7, a low-conf peak at x=1.5 must NOT fuse into the gap!
    split_mask = np.zeros((10, 10), dtype=np.uint8)
    split_mask[:, :2] = 255
    split_mask[:, 8:] = 255
    f3 = torch.full((1, 10, 10), -10.0, dtype=torch.float32)
    f3[0, 2, 1] = -2.2  # sigmoid ~0.099 -> would blend near x ~ 5 (outside mask)
    pred3 = predictor.predict_from_features(
        f3, frame_shape=(10, 10), mask=split_mask, instance_id=1
    )
    assert pred3.keypoints_xy[0] == pytest.approx((1.5, 2.5))

    # Instance cap pruning (max_instances=2).
    for iid in (10, 11, 12):
        predictor.seed_instance_state(iid, keypoints_xy=[(1.0, 1.0)])
    predictor._track_state.prune_to_max_instances(
        max_instances=predictor._max_track_instances
    )
    assert len(predictor._track_state._by_instance) == 2

    # Verify DinoKPSEGVideoProcessor forwards CutieDinoTrackerConfig.kpseg_temporal_fusion_*.
    captured_fusion: List[object] = []

    class _CapturingPredictor:
        def __init__(self, _weights, device=None, temporal_fusion=None) -> None:
            captured_fusion.append(temporal_fusion)
            self.keypoint_names = ["nose"]
            self.meta = type("Meta", (), {"short_side": 16, "model_name": "dummy"})()
            self.device = "cpu"

    class _MiniVideo:
        def __init__(self, _path: str) -> None:
            pass

        def get_first_frame(self):
            return np.zeros((16, 16, 3), dtype=np.uint8)

        def total_frames(self) -> int:
            return 2

        def get_fps(self) -> float:
            return 30.0

    monkeypatch.setattr("annolid.tracking.dino_kpseg_tracker.CV2Video", _MiniVideo)
    monkeypatch.setattr(
        "annolid.tracking.dino_kpseg_tracker.CutieMaskManager",
        lambda *_a, **_kw: object(),
    )
    monkeypatch.setattr(
        "annolid.tracking.dino_kpseg_tracker.DinoKPSEGPredictor", _CapturingPredictor
    )

    DinoKPSEGVideoProcessor(
        video_path=str(tmp_path / "mini.mp4"),
        result_folder=tmp_path / "mini",
        kpseg_weights="dummy.pt",
        runtime_config=CutieDinoTrackerConfig(
            kpseg_temporal_fusion_enable=True,
            kpseg_temporal_fusion_alpha=0.4,
            kpseg_temporal_fusion_low_conf_threshold=0.45,
            kpseg_temporal_fusion_max_instances=12,
        ),
    )
    assert len(captured_fusion) == 1
    forwarded = TemporalFusionConfig.coerce(captured_fusion[0])
    assert forwarded == TemporalFusionConfig(
        enable=True,
        alpha=0.4,
        low_conf_threshold=0.45,
        max_instances=12,
    )


def test_overlapping_polygon_point_disambiguation_and_ambiguous_guard(
    tmp_path: Path,
) -> None:
    adapter = AnnotationAdapter(image_height=80, image_width=100, persist_json=False)
    parser = DinoKPSEGAnnotationParser(
        image_height=80, image_width=100, adapter=adapter
    )

    # 1. Landmark uniqueness elimination + nose-to-nose 1-to-1 overlap resolution:
    #    Mouse 0 covers x=[10..50], Mouse 1 covers x=[35..75] (overlap x=[35..50]).
    #    Both plain "nose" points lie inside the overlap at (37, 25) and (48, 25),
    #    while Mouse 0 has "tail_base" outside the overlap at (14, 25) and Mouse 1's
    #    "ear" at (46, 25) is resolved after Mouse 0's "ear" is at (18, 25).
    social_json = tmp_path / "social_000000000.json"
    social_json.write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": "mouse",
                        "shape_type": "polygon",
                        "group_id": 0,
                        "points": [[10, 10], [50, 10], [50, 40], [10, 40]],
                    },
                    {
                        "label": "mouse",
                        "shape_type": "polygon",
                        "group_id": 1,
                        "points": [[35, 10], [75, 10], [75, 40], [35, 40]],
                    },
                    {"label": "ear", "shape_type": "point", "points": [[18, 25]]},
                    {"label": "ear", "shape_type": "point", "points": [[46, 25]]},
                    {"label": "nose", "shape_type": "point", "points": [[37, 25]]},
                    {"label": "nose", "shape_type": "point", "points": [[48, 25]]},
                ]
            }
        ),
        encoding="utf-8",
    )

    reg = adapter.read_annotation(social_json)
    m0_kps = {
        kp.label: (kp.x, kp.y) for kp in reg.instances["mouse_0"].keypoints.values()
    }
    m1_kps = {
        kp.label: (kp.x, kp.y) for kp in reg.instances["mouse_1"].keypoints.values()
    }
    assert m0_kps["ear"] == (18.0, 25.0)
    assert m1_kps["ear"] == (46.0, 25.0)
    assert m0_kps["nose"] == (37.0, 25.0)
    assert m1_kps["nose"] == (48.0, 25.0)

    manual = parser.read_manual_annotation(0, social_json)
    assert manual.keypoints_by_instance["0"]["nose"] == (37.0, 25.0)
    assert manual.keypoints_by_instance["1"]["nose"] == (48.0, 25.0)
    assert manual.keypoints_by_instance["1"]["ear"] == (46.0, 25.0)

    # 2. Instance-prefixed plain point labels inside 100% concentric polygons:
    prefixed_json = tmp_path / "prefixed_000000000.json"
    prefixed_json.write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": "intruder",
                        "shape_type": "polygon",
                        "points": [[10, 10], [50, 10], [50, 50], [10, 50]],
                    },
                    {
                        "label": "resident",
                        "shape_type": "polygon",
                        "points": [[10, 10], [50, 10], [50, 50], [10, 50]],
                    },
                    {
                        "label": "intruder_nose",
                        "shape_type": "point",
                        "points": [[25, 25]],
                    },
                    {
                        "label": "resident_nose",
                        "shape_type": "point",
                        "points": [[30, 30]],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    pref_reg = adapter.read_annotation(prefixed_json)
    intruder_kp = next(iter(pref_reg.instances["intruder"].keypoints.values()))
    resident_kp = next(iter(pref_reg.instances["resident"].keypoints.values()))
    assert intruder_kp.label == "nose"
    assert intruder_kp.storage_label == "intruder_nose"
    assert resident_kp.label == "nose"
    assert resident_kp.storage_label == "resident_nose"

    # 3. Genuinely ambiguous concentric unanchored polygons refuse to guess:
    ambig_json = tmp_path / "ambig_000000000.json"
    ambig_json.write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": "mouse",
                        "shape_type": "polygon",
                        "group_id": 0,
                        "points": [[10, 10], [50, 10], [50, 50], [10, 50]],
                    },
                    {
                        "label": "mouse",
                        "shape_type": "polygon",
                        "group_id": 1,
                        "points": [[10, 10], [50, 10], [50, 50], [10, 50]],
                    },
                    {"label": "nose", "shape_type": "point", "points": [[30, 30]]},
                ]
            }
        ),
        encoding="utf-8",
    )
    ambig_reg = adapter.read_annotation(ambig_json)
    assert not ambig_reg.instances["mouse_0"].keypoints
    assert not ambig_reg.instances["mouse_1"].keypoints
    assert "nose" in ambig_reg.instances
    assert len(adapter.last_ambiguous_points) == 1
    assert adapter.last_ambiguous_points[0]["containing_instances"] == [
        "mouse_0",
        "mouse_1",
    ]

    ambig_manual = parser.read_manual_annotation(0, ambig_json)
    assert ambig_manual.keypoints_by_instance == {}
    assert len(ambig_manual.ambiguous_keypoints) == 1


def test_concentric_overlapping_polygons_resolved_by_temporal_prior_and_gui_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clip_dir = tmp_path / "concentric_clip"
    clip_dir.mkdir()

    # Frame 0: separated animals establish unambiguous pose coordinates.
    f0_json = clip_dir / "concentric_clip_000000000.json"
    f0_json.write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": "mouse",
                        "shape_type": "polygon",
                        "group_id": 0,
                        "points": [[5, 10], [25, 10], [25, 35], [5, 35]],
                    },
                    {
                        "label": "mouse",
                        "shape_type": "polygon",
                        "group_id": 1,
                        "points": [[45, 10], [65, 10], [65, 35], [45, 35]],
                    },
                    {
                        "label": "nose",
                        "shape_type": "point",
                        "group_id": 0,
                        "points": [[15, 22]],
                    },
                    {
                        "label": "nose",
                        "shape_type": "point",
                        "group_id": 1,
                        "points": [[55, 22]],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (clip_dir / "concentric_clip_000000000.png").write_bytes(b"")

    # Frame 1: 100% concentric polygons [10..60] x [10..35] with plain unassociated
    # "nose" and "tail_base" points (no group_id, no instance_label).
    f1_json = clip_dir / "concentric_clip_000000001.json"
    f1_json.write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": "mouse",
                        "shape_type": "polygon",
                        "group_id": 0,
                        "points": [[10, 10], [60, 10], [60, 35], [10, 35]],
                    },
                    {
                        "label": "mouse",
                        "shape_type": "polygon",
                        "group_id": 1,
                        "points": [[10, 10], [60, 10], [60, 35], [10, 35]],
                    },
                    {"label": "nose", "shape_type": "point", "points": [[18, 22]]},
                    {"label": "nose", "shape_type": "point", "points": [[52, 22]]},
                    {"label": "tail_base", "shape_type": "point", "points": [[12, 22]]},
                    {"label": "tail_base", "shape_type": "point", "points": [[58, 22]]},
                ]
            }
        ),
        encoding="utf-8",
    )
    (clip_dir / "concentric_clip_000000001.png").write_bytes(b"")

    adapter = AnnotationAdapter(image_height=60, image_width=80, persist_json=False)
    # Standalone read resolves Frame 1 via same-stem neighbor lookup (Frame 0),
    # and cascades resolved "nose" anchors into Pass 2 for "tail_base"!
    reg1 = adapter.read_annotation(f1_json)
    assert adapter.last_ambiguous_points == []
    m0 = {kp.label: (kp.x, kp.y) for kp in reg1.instances["mouse_0"].keypoints.values()}
    m1 = {kp.label: (kp.x, kp.y) for kp in reg1.instances["mouse_1"].keypoints.values()}
    assert m0 == {"nose": (18.0, 22.0), "tail_base": (12.0, 22.0)}
    assert m1 == {"nose": (52.0, 22.0), "tail_base": (58.0, 22.0)}

    parser = DinoKPSEGAnnotationParser(image_height=60, image_width=80, adapter=adapter)
    manual1 = parser.read_manual_annotation(1, f1_json)
    assert manual1.ambiguous_keypoints == ()
    assert manual1.keypoints_by_instance["0"] == {
        "nose": (18.0, 22.0),
        "tail_base": (12.0, 22.0),
    }
    assert manual1.keypoints_by_instance["1"] == {
        "nose": (52.0, 22.0),
        "tail_base": (58.0, 22.0),
    }

    # GUI pre-flight check: an isolated unanchored concentric seed frame prompts
    # and jumps to that frame when the user clicks Yes.
    solo_dir = tmp_path / "solo_ambig"
    solo_dir.mkdir()
    (solo_dir / "solo_ambig_000000004.json").write_text(
        json.dumps(
            {
                "imageHeight": 60,
                "imageWidth": 80,
                "shapes": [
                    {
                        "label": "mouse",
                        "shape_type": "polygon",
                        "group_id": 0,
                        "points": [[10, 10], [50, 10], [50, 50], [10, 50]],
                    },
                    {
                        "label": "mouse",
                        "shape_type": "polygon",
                        "group_id": 1,
                        "points": [[10, 10], [50, 10], [50, 50], [10, 50]],
                    },
                    {"label": "nose", "shape_type": "point", "points": [[30, 30]]},
                ],
            }
        ),
        encoding="utf-8",
    )
    (solo_dir / "solo_ambig_000000004.png").write_bytes(b"")

    jumped_frames: List[int] = []
    host = PredictionExecutionMixin.__new__(PredictionExecutionMixin)
    host.video_results_folder = str(solo_dir)
    host.video_processor = None
    host.tr = lambda s: s  # type: ignore[attr-defined]
    host._is_dino_kpseg_tracker_model = lambda *_a: False  # type: ignore[attr-defined]
    host._is_dino_keypoint_model = lambda *_a: True  # type: ignore[attr-defined]
    host._show_frame_on_canvas = lambda f: jumped_frames.append(int(f))  # type: ignore[attr-defined]

    monkeypatch.setattr(
        "annolid.gui.mixins.prediction_execution_mixin.QtWidgets.QMessageBox.question",
        lambda *_a, **_kw: 16384,  # QtWidgets.QMessageBox.Yes
    )
    monkeypatch.setattr(
        "annolid.gui.mixins.prediction_execution_mixin.QtWidgets.QMessageBox.Yes",
        16384,
    )
    assert host._confirm_or_jump_on_ambiguous_keypoint_seeds("DINOv3", "") is False
    assert jumped_frames == [4]


def test_gui_shape_editing_binds_points_on_overlapping_polygons() -> None:
    from types import SimpleNamespace
    from annolid.gui.mixins.shape_editing_mixin import ShapeEditingMixin
    from annolid.gui.shape import Shape

    poly0 = Shape(label="mouse", shape_type="polygon", group_id=0)
    poly0.points = [[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]]
    poly1 = Shape(label="mouse", shape_type="polygon", group_id=1)
    poly1.points = [[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]]

    class _Canvas:
        def __init__(self) -> None:
            self.createMode = "point"
            self.shapes: List[Shape] = [poly0, poly1]
            self.selectedShapes: List[Shape] = []
            self.pending_xy = (25.0, 25.0)

        def setLastLabel(self, text: str, flags: dict):  # noqa: N802
            pt = Shape(label=text, shape_type="point", flags=dict(flags))
            pt.points = [[self.pending_xy[0], self.pending_xy[1]]]
            self.shapes.append(pt)
            return [pt]

    class _Host(ShapeEditingMixin):
        def __init__(self) -> None:
            self.canvas = _Canvas()
            self.labelList = SimpleNamespace(
                clearSelection=lambda: None,
                selectedItems=lambda: [],
            )
            self.actions = SimpleNamespace(
                editMode=SimpleNamespace(setEnabled=lambda _v: None),
                undoLastPoint=SimpleNamespace(setEnabled=lambda _v: None),
                undo=SimpleNamespace(setEnabled=lambda _v: None),
            )
            self.labels_next = ["nose", "nose", "tail_base"]
            self.keypoint_sequence_widget = SimpleNamespace(
                is_sequence_enabled=lambda: True,
                consume_next_label=lambda: self.labels_next.pop(0),
                auto_save_on_click=lambda: False,
            )
            self.added: List[Shape] = []
            self.prompt_calls = 0

        def addLabel(self, shape, rebuild_unique=True):  # noqa: N802
            _ = rebuild_unique
            self.added.append(shape)

        def setDirty(self) -> None:  # noqa: N802
            pass

        def _prompt_overlapping_polygon_owner(self, _shape, containing_entries):
            self.prompt_calls += 1
            return containing_entries[1]

    host = _Host()
    # 1. Preselected poly0 binds first "nose" to mouse_0 without prompting.
    host.canvas.selectedShapes = [poly0]
    assert host._try_apply_keypoint_sequence_labeling() is True
    assert host.added[0].group_id == 0
    assert host.added[0].flags["instance_label"] == "mouse_0"
    assert host.prompt_calls == 0

    # 2. Second "nose" inside concentric overlap auto-binds to poly1 (the only
    #    containing polygon still missing "nose") without prompting!
    host.canvas.selectedShapes = []
    host.canvas.pending_xy = (35.0, 25.0)
    assert host._try_apply_keypoint_sequence_labeling() is True
    assert host.added[1].group_id == 1
    assert host.added[1].flags["instance_label"] == "mouse_1"
    assert host.prompt_calls == 0

    # 3. Brand-new landmark ("tail_base") with neither polygon preselected invokes
    #    _prompt_overlapping_polygon_owner and binds to the user's chosen polygon.
    host.canvas.pending_xy = (30.0, 30.0)
    assert host._try_apply_keypoint_sequence_labeling() is True
    assert host.prompt_calls == 1
    assert host.added[2].group_id == 1
    assert host.added[2].flags["instance_label"] == "mouse_1"


def test_dino_tracker_multi_animal_hungarian_constraints_and_smoother_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from annolid.tracking.dino_keypoint_types import Candidate

    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        _DummyExtractor,
    )

    # 1. Hungarian global assignment beats Greedy when top-choice patch collides:
    greedy_tracker = DinoKeypointTracker(
        model_name="dummy",
        runtime_config=CutieDinoTrackerConfig(
            tracking_enable=True,
            tracking_matcher_algorithm="greedy",
        ),
    )
    hungarian_tracker = DinoKeypointTracker(
        model_name="dummy",
        runtime_config=CutieDinoTrackerConfig(
            tracking_enable=True,
            tracking_matcher_algorithm="hungarian",
        ),
    )
    cand_pool = {
        "m0_nose": [
            Candidate(
                rc=(1, 1),
                xy=(1.5, 1.5),
                score=0.90,
                similarity=0.90,
                descriptor=torch.ones(2),
            ),
            Candidate(
                rc=(1, 2),
                xy=(2.5, 1.5),
                score=0.88,
                similarity=0.88,
                descriptor=torch.ones(2),
            ),
        ],
        "m1_nose": [
            Candidate(
                rc=(1, 1),
                xy=(1.5, 1.5),
                score=0.89,
                similarity=0.89,
                descriptor=torch.ones(2),
            ),
            Candidate(
                rc=(5, 5),
                xy=(5.5, 5.5),
                score=0.20,
                similarity=0.20,
                descriptor=torch.ones(2),
            ),
        ],
    }
    g_res = greedy_tracker._resolve_assignments(cand_pool)
    h_res = hungarian_tracker._resolve_assignments(cand_pool)
    assert g_res["m0_nose"].rc == (1, 1) and g_res["m1_nose"].rc == (5, 5)  # type: ignore[union-attr]
    assert h_res["m0_nose"].rc == (1, 2) and h_res["m1_nose"].rc == (1, 1)  # type: ignore[union-attr]

    # 2. Velocity & dropout constraints under tracking_enable=True:
    constrained_cfg = CutieDinoTrackerConfig(
        tracking_enable=True,
        tracking_matcher_algorithm="hungarian",
        tracking_constraint_max_velocity_px=2.5,
        tracking_enforce_skeletal_constraints=True,
        tracking_constraint_limb_std_multiplier=1.5,
        tracking_instance_dropout_timeout_frames=2,
        tracking_instance_match_min_similarity=0.35,
        pixel_refine_enabled=False,
        keypoint_cluster_refine=False,
        dinov3_positional_debias=False,
        dinov3_backward_consistency=False,
        appearance_bundle_weight=0.0,
        support_probe_weight=0.0,
        context_weight=0.0,
        part_shared_weight=0.0,
        structural_consistency_weight=0.0,
    )
    tracker = DinoKeypointTracker(
        model_name="dummy", runtime_config=constrained_cfg, search_radius=8
    )

    f0 = torch.zeros((2, 10, 10), dtype=torch.float32)
    f0[:, :, :] = torch.tensor([[[0.0]], [[1.0]]])
    f0[:, 1, 1] = torch.tensor([1.0, 0.0])  # seed at (1.5, 1.5)

    # Frame 1: tracker _is_fresh_start clears on first update; keep peak at (1, 1).
    f1 = f0.clone()
    # Frame 2: deceptive identical peak teleports far away to (8, 8) (step ~9.9px > 2.5px)!
    f2 = torch.zeros((2, 10, 10), dtype=torch.float32)
    f2[:, :, :] = torch.tensor([[[0.0]], [[1.0]]])
    f2[:, 8, 8] = torch.tensor([1.0, 0.0])
    # Frame 3: second consecutive miss triggers dropout_timeout=2 template reset!
    f3 = f2.clone()

    tracker.extractor.set_queue([f0, f1, f2, f3])  # type: ignore[attr-defined]
    reg = InstanceRegistry()
    reg.register_keypoint(
        KeypointState(key="m0nose", instance_label="m0", label="nose", x=1.5, y=1.5)
    )
    blank = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
    tracker.start(blank, reg, {})
    res1 = tracker.update(blank, {})
    assert res1[0]["visible"] is True

    # Mutate live descriptor slightly so we can verify dropout reset restores reference_descriptor.
    tracker.tracks["m0nose"].descriptor = torch.tensor([0.6, 0.8])
    res2 = tracker.update(blank, {})
    assert res2[0]["visible"] is False
    assert res2[0]["misses"] == 1

    res3 = tracker.update(blank, {})
    assert res3[0]["visible"] is False
    assert res3[0]["misses"] == 2
    assert torch.allclose(
        tracker.tracks["m0nose"].descriptor,
        tracker.tracks["m0nose"].reference_descriptor,
    )

    # 3. DinoKeypointVideoProcessor honors tracking_smoother_mode and parameters:
    class _MiniVideo:
        def __init__(self, _p: str) -> None:
            pass

        def get_first_frame(self):
            return np.zeros((16, 16, 3), dtype=np.uint8)

        def total_frames(self) -> int:
            return 2

        def get_fps(self) -> float:
            return 25.0

    monkeypatch.setattr("annolid.tracking.dino_keypoint_tracker.CV2Video", _MiniVideo)
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.CutieMaskManager",
        lambda *_a, **_kw: object(),
    )
    proc = DinoKeypointVideoProcessor(
        video_path=str(tmp_path / "m.mp4"),
        result_folder=tmp_path / "m_out",
        model_name="dummy",
        runtime_config=CutieDinoTrackerConfig(
            tracking_smoother_enable=True,
            tracking_smoother_mode="one_euro",
            tracking_smoother_fps=60.0,
            tracking_smoother_one_euro_min_cutoff=1.7,
            tracking_smoother_one_euro_beta=0.15,
        ),
    )
    assert proc._smoother is not None
    assert proc._smoother.mode == "one_euro"
    assert proc._smoother.fps == pytest.approx(60.0)
    assert proc._smoother.one_euro_min_cutoff == pytest.approx(1.7)
    assert proc._smoother.one_euro_beta == pytest.approx(0.15)


def test_nested_concentric_single_frame_and_ambiguous_point_policies(
    tmp_path: Path,
) -> None:
    from annolid.tracking.overlap_assignment import AmbiguousKeypointError

    # 1. Single isolated frame with genuinely nested concentric polygons (outer 100x100, inner 40x40):
    #    - 1 plain point at [50, 50] resolves to nested inner polygon (mouse_1).
    #    - 2 plain points at [50, 50] (deep inside inner) and [32, 50] (rim of inner, deep inside outer)
    #      resolve to mouse_1 and mouse_0 respectively with zero ambiguity.
    nested_dir = tmp_path / "nested_solo"
    nested_dir.mkdir()
    nested_one_path = nested_dir / "nested_one_000000000.json"
    nested_one_path.write_text(
        json.dumps(
            {
                "version": "5.3.1",
                "flags": {},
                "imagePath": "nested_one_000000000.png",
                "imageHeight": 120,
                "imageWidth": 120,
                "shapes": [
                    {
                        "label": "mouse",
                        "points": [
                            [10.0, 10.0],
                            [110.0, 10.0],
                            [110.0, 110.0],
                            [10.0, 110.0],
                        ],
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {},
                    },
                    {
                        "label": "mouse",
                        "points": [
                            [40.0, 40.0],
                            [80.0, 40.0],
                            [80.0, 80.0],
                            [40.0, 80.0],
                        ],
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {},
                    },
                    {
                        "label": "nose",
                        "points": [[60.0, 60.0]],
                        "group_id": None,
                        "shape_type": "point",
                        "flags": {},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    adapter = AnnotationAdapter(image_height=120, image_width=120)
    reg_one = adapter.read_annotation(nested_one_path)
    assert adapter.last_ambiguous_points == []
    kps_one = {
        kp.instance_label: kp for inst in reg_one for kp in inst.keypoints.values()
    }
    assert set(kps_one.keys()) == {"mouse_1"}

    nested_two_path = nested_dir / "nested_two_000000000.json"
    nested_two_path.write_text(
        json.dumps(
            {
                "version": "5.3.1",
                "flags": {},
                "imagePath": "nested_two_000000000.png",
                "imageHeight": 120,
                "imageWidth": 120,
                "shapes": [
                    {
                        "label": "mouse",
                        "points": [
                            [10.0, 10.0],
                            [110.0, 10.0],
                            [110.0, 110.0],
                            [10.0, 110.0],
                        ],
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {},
                    },
                    {
                        "label": "mouse",
                        "points": [
                            [40.0, 40.0],
                            [80.0, 40.0],
                            [80.0, 80.0],
                            [40.0, 80.0],
                        ],
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {},
                    },
                    {
                        "label": "nose",
                        "points": [[60.0, 60.0]],
                        "group_id": None,
                        "shape_type": "point",
                        "flags": {},
                    },
                    {
                        "label": "nose",
                        "points": [[42.0, 60.0]],
                        "group_id": None,
                        "shape_type": "point",
                        "flags": {},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    reg_two = adapter.read_annotation(nested_two_path)
    assert adapter.last_ambiguous_points == []
    kps_two = {
        kp.instance_label: kp for inst in reg_two for kp in inst.keypoints.values()
    }
    assert set(kps_two.keys()) == {"mouse_0", "mouse_1"}
    assert kps_two["mouse_1"].x == pytest.approx(60.0)
    assert kps_two["mouse_0"].x == pytest.approx(42.0)

    # 2. Identical duplicate polygons on a single isolated frame:
    #    - "withhold" (default): withholds and records last_ambiguous_points
    #    - "error": raises AmbiguousKeypointError on both AnnotationAdapter and DinoKPSEGAnnotationParser
    #    - "polygon_order": assigns deterministically in polygon declaration order without collision
    dup_dir = tmp_path / "dup_solo"
    dup_dir.mkdir()
    dup_path = dup_dir / "dup_000000000.json"
    dup_path.write_text(
        json.dumps(
            {
                "version": "5.3.1",
                "flags": {},
                "imagePath": "dup_000000000.png",
                "imageHeight": 64,
                "imageWidth": 64,
                "shapes": [
                    {
                        "label": "mouse",
                        "points": [
                            [10.0, 10.0],
                            [50.0, 10.0],
                            [50.0, 50.0],
                            [10.0, 50.0],
                        ],
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {},
                    },
                    {
                        "label": "mouse",
                        "points": [
                            [10.0, 10.0],
                            [50.0, 10.0],
                            [50.0, 50.0],
                            [10.0, 50.0],
                        ],
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {},
                    },
                    {
                        "label": "nose",
                        "points": [[25.0, 30.0]],
                        "group_id": None,
                        "shape_type": "point",
                        "flags": {},
                    },
                    {
                        "label": "nose",
                        "points": [[35.0, 30.0]],
                        "group_id": None,
                        "shape_type": "point",
                        "flags": {},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    assert (
        CutieDinoTrackerConfig(
            ambiguous_point_policy="invalid_mode"
        ).ambiguous_point_policy
        == "withhold"
    )
    with pytest.raises(ValueError, match="ambiguous_point_policy"):
        AnnotationAdapter(
            image_height=64,
            image_width=64,
            ambiguous_point_policy="invalid_mode",
        )

    err_cfg = CutieDinoTrackerConfig(ambiguous_point_policy="error")
    err_adapter = AnnotationAdapter(
        image_height=64,
        image_width=64,
        ambiguous_point_policy=err_cfg.ambiguous_point_policy,
    )
    with pytest.raises(AmbiguousKeypointError) as exc_info:
        err_adapter.read_annotation(dup_path)
    assert len(exc_info.value.ambiguous_points) == 2
    assert len(err_adapter.last_ambiguous_points) == 2

    err_parser = DinoKPSEGAnnotationParser(
        image_height=64,
        image_width=64,
        adapter=err_adapter,
    )
    with pytest.raises(AmbiguousKeypointError):
        err_parser.read_manual_annotation(0, dup_path)

    order_adapter = AnnotationAdapter(
        image_height=64,
        image_width=64,
        ambiguous_point_policy="polygon_order",
    )
    reg_ord = order_adapter.read_annotation(dup_path)
    assert order_adapter.last_ambiguous_points == []
    kps_ord = {
        kp.instance_label: kp for inst in reg_ord for kp in inst.keypoints.values()
    }
    assert set(kps_ord.keys()) == {"mouse_0", "mouse_1"}
    assert kps_ord["mouse_0"].x == pytest.approx(25.0)
    assert kps_ord["mouse_1"].x == pytest.approx(35.0)


def test_track_qc_resolves_plain_multi_animal_points_and_flags_ambiguous_ownership(
    tmp_path: Path,
) -> None:
    from annolid.engine import cli as engine_cli

    # 1. Clean 2-animal directory where polygons carry group_ids (0, 1) while points are
    #    plain and ungrouped (group_id=None):
    #    Previously all plain points collapsed onto "__single__", overwriting across animals.
    clean_dir = tmp_path / "qc_clean"
    clean_dir.mkdir()
    for frame_idx in range(6):
        dx = float(frame_idx * 2)
        (clean_dir / f"vid_{frame_idx:09d}.json").write_text(
            json.dumps(
                {
                    "version": "5.3.1",
                    "imagePath": f"vid_{frame_idx:09d}.png",
                    "imageHeight": 100,
                    "imageWidth": 200,
                    "shapes": [
                        {
                            "label": "mouse",
                            "points": [
                                [10.0 + dx, 20.0],
                                [50.0 + dx, 20.0],
                                [50.0 + dx, 60.0],
                                [10.0 + dx, 60.0],
                            ],
                            "group_id": 0,
                            "shape_type": "polygon",
                            "flags": {},
                        },
                        {
                            "label": "mouse",
                            "points": [
                                [120.0 + dx, 20.0],
                                [160.0 + dx, 20.0],
                                [160.0 + dx, 60.0],
                                [120.0 + dx, 60.0],
                            ],
                            "group_id": 1,
                            "shape_type": "polygon",
                            "flags": {},
                        },
                        {
                            "label": "nose",
                            "points": [[20.0 + dx, 40.0]],
                            "group_id": None,
                            "shape_type": "point",
                            "flags": {},
                        },
                        {
                            "label": "tail",
                            "points": [[40.0 + dx, 40.0]],
                            "group_id": None,
                            "shape_type": "point",
                            "flags": {},
                        },
                        {
                            "label": "nose",
                            "points": [[130.0 + dx, 40.0]],
                            "group_id": None,
                            "shape_type": "point",
                            "flags": {},
                        },
                        {
                            "label": "tail",
                            "points": [[150.0 + dx, 40.0]],
                            "group_id": None,
                            "shape_type": "point",
                            "flags": {},
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
    pose_schema = PoseSchema(
        keypoints=["nose", "tail"],
        edges=[("nose", "tail")],
    )
    clean_report = evaluate_track_qc(clean_dir, pose_schema=pose_schema)
    assert clean_report.warning_count == 0
    assert clean_report.error_count == 0
    assert engine_cli.main(["track-qc", str(clean_dir), "--strict"]) == 0

    # 2. Isolated identical-polygon frame with ungrouped plain point triggers
    #    ambiguous_keypoint_ownership warning and exits 2 under --strict
    #    (both when polygons have distinct group_ids and when polygons are ungrouped).
    ambig_dir = tmp_path / "qc_ambig"
    ambig_dir.mkdir()
    (ambig_dir / "ambig_000000000.json").write_text(
        json.dumps(
            {
                "version": "5.3.1",
                "imagePath": "ambig_000000000.png",
                "imageHeight": 64,
                "imageWidth": 64,
                "shapes": [
                    {
                        "label": "mouse",
                        "points": [
                            [10.0, 10.0],
                            [50.0, 10.0],
                            [50.0, 50.0],
                            [10.0, 50.0],
                        ],
                        "group_id": 0,
                        "shape_type": "polygon",
                        "flags": {},
                    },
                    {
                        "label": "mouse",
                        "points": [
                            [10.0, 10.0],
                            [50.0, 10.0],
                            [50.0, 50.0],
                            [10.0, 50.0],
                        ],
                        "group_id": 1,
                        "shape_type": "polygon",
                        "flags": {},
                    },
                    {
                        "label": "nose",
                        "points": [[30.0, 30.0]],
                        "group_id": None,
                        "shape_type": "point",
                        "flags": {},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    ambig_report = evaluate_track_qc(ambig_dir)
    assert ambig_report.error_count == 0
    ambig_issues = [
        issue
        for issue in ambig_report.issues
        if issue.code == "ambiguous_keypoint_ownership"
    ]
    assert len(ambig_issues) == 1
    assert ambig_issues[0].severity == "warning"
    assert any(
        "ambiguous_keypoint_ownership" in bout.issue_codes
        for bout in ambig_report.review_queue
    )
    assert engine_cli.main(["track-qc", str(ambig_dir), "--strict"]) == 2

    ungrouped_ambig_dir = tmp_path / "qc_ungrouped_ambig"
    ungrouped_ambig_dir.mkdir()
    (ungrouped_ambig_dir / "ambig_000000000.json").write_text(
        json.dumps(
            {
                "version": "5.3.1",
                "imagePath": "ambig_000000000.png",
                "imageHeight": 64,
                "imageWidth": 64,
                "shapes": [
                    {
                        "label": "mouse",
                        "points": [
                            [10.0, 10.0],
                            [50.0, 10.0],
                            [50.0, 50.0],
                            [10.0, 50.0],
                        ],
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {},
                    },
                    {
                        "label": "mouse",
                        "points": [
                            [10.0, 10.0],
                            [50.0, 10.0],
                            [50.0, 50.0],
                            [10.0, 50.0],
                        ],
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {},
                    },
                    {
                        "label": "nose",
                        "points": [[30.0, 30.0]],
                        "group_id": None,
                        "shape_type": "point",
                        "flags": {},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    ungrouped_report = evaluate_track_qc(ungrouped_ambig_dir)
    assert any(
        issue.code == "ambiguous_keypoint_ownership"
        for issue in ungrouped_report.issues
    )


def test_pose_dataset_and_yolo_conversion_disambiguate_overlapping_plain_points(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace

    from annolid.annotation.labelme2yolo import Labelme2YOLO
    from annolid.segmentation.dino_kpseg.data import (
        DinoKPSEGPoseDataset,
        summarize_labelme_pose_labels,
    )

    # Two overlapping ungrouped "mouse" polygons of equal area (60x60):
    # - Polygon 0: [10..70, 10..70] (centroid 40, 40) with anchor "tail" at (15, 40)
    # - Polygon 1: [30..90, 10..70] (centroid 60, 40)
    # - Two plain "nose" points in the overlap: (35, 40) and (45, 40).
    #   Note: BOTH (35, 40) and (45, 40) are closer to Polygon 0's centroid (40, 40)
    #   than Polygon 1's centroid (60, 40), which previously caused greedy centroid
    #   assignment to overwrite Polygon 0's nose and leave Polygon 1 empty.
    ds_dir = tmp_path / "overlap_pose"
    ds_dir.mkdir()
    img_path = ds_dir / "frame_000000000.png"
    Image.new("RGB", (100, 80), color=(0, 0, 0)).save(img_path)
    json_path = ds_dir / "frame_000000000.json"
    json_path.write_text(
        json.dumps(
            {
                "version": "5.3.1",
                "imagePath": img_path.name,
                "imageHeight": 80,
                "imageWidth": 100,
                "shapes": [
                    {
                        "label": "mouse",
                        "points": [
                            [10.0, 10.0],
                            [70.0, 10.0],
                            [70.0, 70.0],
                            [10.0, 70.0],
                        ],
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {},
                    },
                    {
                        "label": "mouse",
                        "points": [
                            [30.0, 10.0],
                            [90.0, 10.0],
                            [90.0, 70.0],
                            [30.0, 70.0],
                        ],
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {},
                    },
                    {
                        "label": "tail",
                        "points": [[15.0, 40.0]],
                        "group_id": None,
                        "shape_type": "point",
                        "flags": {},
                    },
                    {
                        "label": "nose",
                        "points": [[35.0, 40.0]],
                        "group_id": None,
                        "shape_type": "point",
                        "flags": {},
                    },
                    {
                        "label": "nose",
                        "points": [[45.0, 40.0]],
                        "group_id": None,
                        "shape_type": "point",
                        "flags": {},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    # 1. Labelme2YOLO preserves both animals' noses without overwriting onto Polygon 0.
    converter = Labelme2YOLO(str(ds_dir))
    converter.create_yolo_dataset_dirs()
    converter.json_to_text("train/", json_path.name)
    yolo_txt = ds_dir / "YOLO_dataset" / "labels" / "train" / "frame_000000000.txt"
    lines = [
        ln.strip()
        for ln in yolo_txt.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(lines) == 2
    nose_idx = converter.keypoint_labels_order.index("nose")
    dims = int(converter.kpt_shape[1])
    parsed_nose_xs = []
    for ln in lines:
        vals = [float(t) for t in ln.split()[1:]]
        kpts_flat = vals[4:]
        parsed_nose_xs.append(kpts_flat[nose_idx * dims] * 100.0)
    assert sorted(parsed_nose_xs) == pytest.approx([35.0, 45.0])

    # 2. DinoKPSEGPoseDataset infers per_instance (2 instances, not collapsed into gid=0)
    #    and preserves distinct bboxes & nose coordinates.
    class _MiniPoseExtractor:
        patch_size = 16
        model_id = "stub"
        cfg = SimpleNamespace(short_side=64, layers=(-1,))
        model = SimpleNamespace(config=SimpleNamespace(hidden_size=8))

        def extract(
            self, pil: Image.Image, *, return_type: str = "torch"
        ) -> torch.Tensor:
            w, h = pil.size
            return torch.zeros(
                (8, max(1, h // 16), max(1, w // 16)), dtype=torch.float32
            )

    ds = DinoKPSEGPoseDataset(
        [img_path],
        kpt_count=2,
        kpt_dims=3,
        radius_px=4.0,
        extractor=_MiniPoseExtractor(),  # type: ignore[arg-type]
        label_format="labelme",
        label_paths=[json_path],
        keypoint_names=["nose", "tail"],
        instance_mode="auto",
        cache_dir=None,
    )
    assert ds.instance_mode == "per_instance"
    assert len(ds) == 2
    inst_records = ds._load_pose_instances_with_bbox(json_path)
    assert len(inst_records) == 2
    assert inst_records[0][0][0] == pytest.approx(0.40)  # polygon 0 cx = 40 / 100
    assert inst_records[1][0][0] == pytest.approx(0.60)  # polygon 1 cx = 60 / 100
    assert inst_records[0][1][0, 0] == pytest.approx(0.35)  # animal 0 nose x
    assert inst_records[1][1][0, 0] == pytest.approx(0.45)  # animal 1 nose x

    # 3. summarize_labelme_pose_labels surfaces unresolved identical-polygon ambiguity in issues.
    ambig_dir = tmp_path / "ambig_pose_summary"
    ambig_dir.mkdir()
    ambig_img = ambig_dir / "ambig.png"
    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(ambig_img)
    ambig_json = ambig_dir / "ambig.json"
    ambig_json.write_text(
        json.dumps(
            {
                "version": "5.3.1",
                "imagePath": ambig_img.name,
                "imageHeight": 64,
                "imageWidth": 64,
                "shapes": [
                    {
                        "label": "mouse",
                        "points": [
                            [10.0, 10.0],
                            [50.0, 10.0],
                            [50.0, 50.0],
                            [10.0, 50.0],
                        ],
                        "group_id": None,
                        "shape_type": "polygon",
                    },
                    {
                        "label": "mouse",
                        "points": [
                            [10.0, 10.0],
                            [50.0, 10.0],
                            [50.0, 50.0],
                            [10.0, 50.0],
                        ],
                        "group_id": None,
                        "shape_type": "polygon",
                    },
                    {
                        "label": "nose",
                        "points": [[30.0, 30.0]],
                        "group_id": None,
                        "shape_type": "point",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    summary = summarize_labelme_pose_labels(
        [ambig_img],
        label_paths=[ambig_json],
        keypoint_names=["nose"],
        kpt_dims=3,
    )
    assert any(
        "Ambiguous overlapping keypoint" in item for item in summary.example_issues
    )
