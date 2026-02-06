from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from annolid.tracking.configuration import CutieDinoTrackerConfig
from annolid.tracking.cutie_mask_manager import MaskResult
from annolid.tracking.dino_kpseg_tracker import DinoKPSEGVideoProcessor


@dataclass
class _DummyPred:
    keypoints_xy: List[Tuple[float, float]]
    keypoint_scores: List[float]


class _DummyVideo:
    def __init__(self, _path: str) -> None:
        self._frame = np.zeros((32, 48, 3), dtype=np.uint8)

    def get_first_frame(self):
        return self._frame.copy()

    def total_frames(self) -> int:
        return 3

    def load_frame(self, _frame_number: int):
        return self._frame.copy()

    def get_fps(self) -> float:
        return 30.0


class _DummyCutieMaskManager:
    def __init__(self, _video_path: Path, adapter, config) -> None:
        self.adapter = adapter
        self.config = config
        self.enabled = True
        self._last_polygon: Optional[List[Tuple[float, float]]] = None
        self._last_mask: Optional[np.ndarray] = None

    def reset_state(self) -> None:
        self._last_polygon = None
        self._last_mask = None

    def prime(self, _frame_number: int, _frame: np.ndarray, registry) -> None:
        for instance in registry:
            if instance.mask_bitmap is not None:
                self._last_mask = instance.mask_bitmap.astype(bool)
            if instance.polygon is not None:
                self._last_polygon = [tuple(pt) for pt in instance.polygon]

    def update_masks(
        self, _frame_number: int, _frame: np.ndarray, registry
    ) -> Dict[str, MaskResult]:
        results: Dict[str, MaskResult] = {}
        for instance in registry:
            if self._last_mask is None or self._last_polygon is None:
                continue
            results[str(instance.label)] = MaskResult(
                instance_label=str(instance.label),
                mask_bitmap=np.array(self._last_mask, copy=True),
                polygon=list(self._last_polygon),
            )
        return results


class _DummyKpsegPredictor:
    def __init__(self, _weights, device=None) -> None:
        self.keypoint_names = ["k0", "k1"]
        self.meta = type("Meta", (), {"short_side": 32, "model_name": "dummy"})()
        self.device = "cpu"

    def reset_state(self) -> None:
        return None

    def seed_instance_state(self, *_args, **_kwargs) -> None:
        return None

    def predict(self, _frame_bgr, stabilize_lr=False):
        return _DummyPred(
            keypoints_xy=[(1.0, 1.0), (2.0, 2.0)], keypoint_scores=[1.0, 1.0]
        )


def test_polygon_only_seed_tracks_masks_then_predicts_keypoints(tmp_path, monkeypatch):
    # Patch heavy components used by DinoKPSEGVideoProcessor.
    monkeypatch.setattr("annolid.tracking.dino_kpseg_tracker.CV2Video", _DummyVideo)
    monkeypatch.setattr(
        "annolid.tracking.dino_kpseg_tracker.CutieMaskManager", _DummyCutieMaskManager
    )
    monkeypatch.setattr(
        "annolid.tracking.dino_kpseg_tracker.DinoKPSEGPredictor", _DummyKpsegPredictor
    )

    def _build_instance_crops(_frame_bgr, instance_masks, pad_px, use_mask_gate):
        return list(instance_masks)

    def _predict_on_instance_crops(_predictor, crops, stabilize_lr):
        for gid, _mask in crops:
            yield (
                int(gid),
                _DummyPred(
                    keypoints_xy=[(10.0, 10.0), (20.0, 20.0)],
                    keypoint_scores=[0.9, 0.9],
                ),
            )

    monkeypatch.setattr(
        "annolid.tracking.dino_kpseg_tracker.build_instance_crops",
        _build_instance_crops,
    )
    monkeypatch.setattr(
        "annolid.tracking.dino_kpseg_tracker.predict_on_instance_crops",
        _predict_on_instance_crops,
    )

    out_dir = tmp_path / "video"
    out_dir.mkdir()

    # Create a minimal LabelMe JSON with a polygon only (no point keypoints).
    polygon = [[5.0, 5.0], [30.0, 5.0], [30.0, 25.0], [5.0, 25.0], [5.0, 5.0]]
    seed_json = {
        "shapes": [
            {
                "label": "0",
                "shape_type": "polygon",
                "group_id": 0,
                "points": polygon,
            }
        ]
    }
    (out_dir / "video_000000000.json").write_text(
        json.dumps(seed_json), encoding="utf-8"
    )
    # `find_manual_labeled_json_files` considers a frame "manual" when the PNG exists
    # alongside its JSON (or via AnnotationStore). A zero-byte file is sufficient here.
    (out_dir / "video_000000000.png").write_bytes(b"")

    cfg = CutieDinoTrackerConfig(
        kpseg_apply_mode="always", use_cutie_tracking=True, persist_labelme_json=True
    )
    processor = DinoKPSEGVideoProcessor(
        video_path=str(tmp_path / "dummy.mp4"),
        result_folder=out_dir,
        kpseg_weights="dummy.pt",
        runtime_config=cfg,
    )

    message = processor.process_video(start_frame=0, end_frame=2, step=1)
    assert "completed" in message.lower()

    produced = out_dir / "video_000000001.json"
    assert produced.exists()
    payload = json.loads(produced.read_text(encoding="utf-8"))
    shapes = payload.get("shapes", [])
    assert any(shape.get("shape_type") == "polygon" for shape in shapes)
    assert any(shape.get("shape_type") == "point" for shape in shapes)


def test_kpseg_projects_points_back_into_instance_polygon(tmp_path, monkeypatch):
    monkeypatch.setattr("annolid.tracking.dino_kpseg_tracker.CV2Video", _DummyVideo)
    monkeypatch.setattr(
        "annolid.tracking.dino_kpseg_tracker.CutieMaskManager", _DummyCutieMaskManager
    )
    monkeypatch.setattr(
        "annolid.tracking.dino_kpseg_tracker.DinoKPSEGPredictor", _DummyKpsegPredictor
    )

    def _build_instance_crops(_frame_bgr, instance_masks, pad_px, use_mask_gate):
        return list(instance_masks)

    def _predict_on_instance_crops(_predictor, crops, stabilize_lr):
        for gid, _mask in crops:
            yield (
                int(gid),
                _DummyPred(
                    keypoints_xy=[(100.0, 100.0), (120.0, 140.0)],
                    keypoint_scores=[0.95, 0.95],
                ),
            )

    monkeypatch.setattr(
        "annolid.tracking.dino_kpseg_tracker.build_instance_crops",
        _build_instance_crops,
    )
    monkeypatch.setattr(
        "annolid.tracking.dino_kpseg_tracker.predict_on_instance_crops",
        _predict_on_instance_crops,
    )

    out_dir = tmp_path / "video"
    out_dir.mkdir()

    polygon = [[5.0, 5.0], [30.0, 5.0], [30.0, 25.0], [5.0, 25.0], [5.0, 5.0]]
    seed_json = {
        "shapes": [
            {
                "label": "0",
                "shape_type": "polygon",
                "group_id": 0,
                "points": polygon,
            }
        ]
    }
    (out_dir / "video_000000000.json").write_text(
        json.dumps(seed_json), encoding="utf-8"
    )
    (out_dir / "video_000000000.png").write_bytes(b"")

    cfg = CutieDinoTrackerConfig(
        kpseg_apply_mode="always",
        use_cutie_tracking=True,
        persist_labelme_json=True,
        kpseg_use_mask_gate=False,
        restrict_to_initial_mask=False,
        mask_enforce_position=False,
        mask_enforce_reject_outside=False,
    )
    processor = DinoKPSEGVideoProcessor(
        video_path=str(tmp_path / "dummy.mp4"),
        result_folder=out_dir,
        kpseg_weights="dummy.pt",
        runtime_config=cfg,
    )
    # The Cutie+DinoKPSEG processor enforces inside-mask guarantees.
    assert processor.config.kpseg_use_mask_gate is True
    assert processor.config.restrict_to_initial_mask is True
    assert processor.config.mask_enforce_position is True
    assert processor.config.mask_enforce_reject_outside is True
    processor.process_video(start_frame=0, end_frame=2, step=1)

    payload = json.loads((out_dir / "video_000000001.json").read_text(encoding="utf-8"))
    point_shapes = [
        shape
        for shape in payload.get("shapes", [])
        if shape.get("shape_type") == "point"
    ]
    assert point_shapes
    for point_shape in point_shapes:
        x, y = point_shape["points"][0]
        assert 5.0 <= float(x) <= 30.0
        assert 5.0 <= float(y) <= 25.0


def test_kpseg_multi_instance_points_remain_in_their_own_polygons(
    tmp_path, monkeypatch
):
    monkeypatch.setattr("annolid.tracking.dino_kpseg_tracker.CV2Video", _DummyVideo)

    class _DummyCutieMaskManagerMulti:
        def __init__(self, _video_path: Path, adapter, config) -> None:
            self.adapter = adapter
            self.config = config
            self.enabled = True
            self._last_by_label: Dict[
                str, Tuple[np.ndarray, List[Tuple[float, float]]]
            ] = {}

        def reset_state(self) -> None:
            self._last_by_label = {}

        def prime(self, _frame_number: int, _frame: np.ndarray, registry) -> None:
            for instance in registry:
                if instance.mask_bitmap is None or instance.polygon is None:
                    continue
                self._last_by_label[str(instance.label)] = (
                    instance.mask_bitmap.astype(bool),
                    [tuple(pt) for pt in instance.polygon],
                )

        def update_masks(
            self, _frame_number: int, _frame: np.ndarray, registry
        ) -> Dict[str, MaskResult]:
            results: Dict[str, MaskResult] = {}
            for instance in registry:
                stored = self._last_by_label.get(str(instance.label))
                if stored is None:
                    continue
                mask, polygon = stored
                results[str(instance.label)] = MaskResult(
                    instance_label=str(instance.label),
                    mask_bitmap=np.array(mask, copy=True),
                    polygon=list(polygon),
                )
            return results

    monkeypatch.setattr(
        "annolid.tracking.dino_kpseg_tracker.CutieMaskManager",
        _DummyCutieMaskManagerMulti,
    )
    monkeypatch.setattr(
        "annolid.tracking.dino_kpseg_tracker.DinoKPSEGPredictor", _DummyKpsegPredictor
    )

    def _build_instance_crops(_frame_bgr, instance_masks, pad_px, use_mask_gate):
        return list(instance_masks)

    def _predict_on_instance_crops(_predictor, crops, stabilize_lr):
        for gid, _mask in crops:
            # Deliberately place peaks near the other instance; projection should
            # pull each prediction back into its own polygon.
            if int(gid) == 0:
                xy = [(36.0, 20.0), (37.0, 21.0)]
            else:
                xy = [(8.0, 8.0), (9.0, 9.0)]
            yield (
                int(gid),
                _DummyPred(
                    keypoints_xy=xy,
                    keypoint_scores=[0.9, 0.9],
                ),
            )

    monkeypatch.setattr(
        "annolid.tracking.dino_kpseg_tracker.build_instance_crops",
        _build_instance_crops,
    )
    monkeypatch.setattr(
        "annolid.tracking.dino_kpseg_tracker.predict_on_instance_crops",
        _predict_on_instance_crops,
    )

    out_dir = tmp_path / "video"
    out_dir.mkdir()
    seed_json = {
        "shapes": [
            {
                "label": "mouse0",
                "shape_type": "polygon",
                "group_id": 0,
                "points": [[5, 5], [15, 5], [15, 15], [5, 15]],
            },
            {
                "label": "mouse1",
                "shape_type": "polygon",
                "group_id": 1,
                "points": [[25, 15], [38, 15], [38, 28], [25, 28]],
            },
        ]
    }
    (out_dir / "video_000000000.json").write_text(
        json.dumps(seed_json), encoding="utf-8"
    )
    (out_dir / "video_000000000.png").write_bytes(b"")

    cfg = CutieDinoTrackerConfig(
        kpseg_apply_mode="always", use_cutie_tracking=True, persist_labelme_json=True
    )
    processor = DinoKPSEGVideoProcessor(
        video_path=str(tmp_path / "dummy.mp4"),
        result_folder=out_dir,
        kpseg_weights="dummy.pt",
        runtime_config=cfg,
    )
    processor.process_video(start_frame=0, end_frame=2, step=1)

    payload = json.loads((out_dir / "video_000000001.json").read_text(encoding="utf-8"))
    points_by_gid = {}
    for shape in payload.get("shapes", []):
        if shape.get("shape_type") != "point":
            continue
        gid = int(shape.get("group_id"))
        x, y = shape["points"][0]
        points_by_gid.setdefault(gid, []).append((float(x), float(y)))

    assert points_by_gid.get(0)
    assert points_by_gid.get(1)
    for x, y in points_by_gid[0]:
        assert 5.0 <= x <= 15.0
        assert 5.0 <= y <= 15.0
    for x, y in points_by_gid[1]:
        assert 25.0 <= x <= 38.0
        assert 15.0 <= y <= 28.0
