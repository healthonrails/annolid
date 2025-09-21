import json
import math
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pytest
import torch
from PIL import Image

from annolid.tracking.annotation_adapter import AnnotationAdapter
from annolid.tracking.configuration import CutieDinoTrackerConfig
from annolid.tracking.cutie_mask_manager import CutieMaskManager, MaskResult
from annolid.tracking.domain import (
    InstanceRegistry,
    KeypointState,
    combine_labels,
)
from annolid.tracking.dino_keypoint_tracker import DinoKeypointTracker, SupportProbe


class DummyExtractor:
    patch_size = 1

    def __init__(self, _cfg):
        self._queue: List[torch.Tensor] = []

    def set_queue(self, tensors: List[torch.Tensor]) -> None:
        self._queue = list(tensors)

    def queue(self, tensor: torch.Tensor) -> None:
        self._queue.append(tensor)

    def extract(self, _image, return_layer="all", normalize=True):
        if not self._queue:
            raise AssertionError("No features queued for DummyExtractor")
        tensor = self._queue.pop(0)
        return tensor

    def _compute_resized_hw(self, width: int, height: int) -> Tuple[int, int]:
        return height, width


def test_annotation_adapter_roundtrip_preserves_keypoints_and_masks(tmp_path):
    source_dir = tmp_path / "testvideo"
    source_dir.mkdir()
    input_fixture = Path("tests/golden/input_frame.json")
    expected_fixture = Path("tests/golden/expected_shapes.json")

    json_path = source_dir / "testvideo_000000000.json"
    shutil.copy(input_fixture, json_path)
    image_path = source_dir / "testvideo_000000000.png"
    Image.new("RGB", (120, 100)).save(image_path)

    adapter = AnnotationAdapter(image_height=100, image_width=120)
    frame_number, registry = adapter.load_initial_state(source_dir)
    assert frame_number == 0

    instance = registry.ensure_instance("animal")
    mask_bitmap = np.zeros((100, 120), dtype=bool)
    mask_bitmap[6:61, 6:51] = True
    instance.set_mask(
        bitmap=mask_bitmap,
        polygon=[
            (6.0, 6.0),
            (50.0, 6.0),
            (50.0, 60.0),
            (6.0, 60.0),
            (6.0, 6.0),
        ],
    )

    registry.apply_tracker_results(
        [
            {
                "id": "animalnose",
                "x": 15.5,
                "y": 22.3,
                "visible": True,
            }
        ]
    )

    output_path = adapter.write_annotation(
        frame_number=frame_number,
        registry=registry,
        output_dir=source_dir,
    )
    produced = json.loads(Path(output_path).read_text())
    expected = json.loads(expected_fixture.read_text())

    produced_shapes = sorted(produced["shapes"], key=lambda shape: shape["label"])
    expected_shapes = sorted(expected["shapes"], key=lambda shape: shape["label"])
    assert produced_shapes == expected_shapes

    keypoint_flags = produced_shapes[0]["flags"]
    mask_flags = produced_shapes[1]["flags"]
    assert keypoint_flags["instance_label"] == mask_flags["instance_label"]


def test_combine_labels_avoids_duplicate_names():
    assert combine_labels("animal", "nose") == "animalnose"
    assert combine_labels("nose", "nose") == "nose"
    assert combine_labels("", "nose") == "nose"


def test_mask_manager_applies_fallback(tmp_path):
    adapter = AnnotationAdapter(image_height=10, image_width=10)
    config = CutieDinoTrackerConfig(
        use_cutie_tracking=False,
        mask_dilation_iterations=1,
        mask_dilation_kernel=3,
        max_mask_fallback_frames=2,
    )
    manager = CutieMaskManager(tmp_path / "video.mp4", adapter, config)
    base_mask = np.zeros((10, 10), dtype=bool)
    base_mask[2:5, 2:5] = True
    polygon = manager._mask_to_polygon(base_mask)
    manager._last_results = {
        "animal": MaskResult("animal", base_mask, polygon)
    }
    manager._mask_miss_counts = {"animal": 0}

    updated = manager._apply_fallbacks({}, ["animal"])
    assert "animal" in updated
    assert updated["animal"].mask_bitmap.any()


def test_mask_fallback_bonus_guides_assignment(monkeypatch):
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        DummyExtractor,
    )
    config = CutieDinoTrackerConfig(
        mask_similarity_bonus=0.5,
        appearance_bundle_weight=0.0,
        baseline_similarity_weight=0.0,
        structural_consistency_weight=0.0,
        symmetry_penalty=0.0,
    )
    tracker = DinoKeypointTracker(
        model_name="dummy",
        runtime_config=config,
        search_radius=1,
    )
    extractor = tracker.extractor

    start_features = torch.zeros((2, 4, 4), dtype=torch.float32)
    start_features[:, 1, 1] = torch.tensor([1.0, 0.0], dtype=torch.float32)

    update_features = torch.zeros((2, 4, 4), dtype=torch.float32)
    update_features[:, 1, 1] = torch.tensor(
        [0.6, math.sqrt(1.0 - 0.6 ** 2)], dtype=torch.float32
    )
    update_features[:, 1, 2] = torch.tensor(
        [0.95, math.sqrt(1.0 - 0.95 ** 2)], dtype=torch.float32
    )

    extractor.set_queue([start_features, update_features])

    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    image = Image.fromarray(frame)

    registry = InstanceRegistry()
    registry.register_keypoint(
        KeypointState(
            key="animalnose",
            instance_label="animal",
            label="nose",
            x=1.0,
            y=1.0,
        )
    )

    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True

    tracker.start(image, registry, {"animal": mask})
    tracker.prev_gray = None

    results = tracker.update(image, {})
    assert len(results) == 1
    result = results[0]
    assert result["label"] == "animalnose"
    assert result["visible"] is True
    assert result["x"] == pytest.approx(1.5, abs=1e-6)
    assert tracker._mask_miss_counts.get("animal") == 1


def test_symmetry_pairs_prevent_swaps(monkeypatch):
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        DummyExtractor,
    )
    config = CutieDinoTrackerConfig(
        symmetry_pairs=(("left", "right"),),
        symmetry_penalty=0.6,
        appearance_bundle_weight=0.0,
        baseline_similarity_weight=0.0,
        structural_consistency_weight=0.0,
        mask_similarity_bonus=0.0,
        velocity_smoothing=0.0,
    )
    tracker = DinoKeypointTracker(
        model_name="dummy",
        runtime_config=config,
        search_radius=2,
    )
    extractor = tracker.extractor

    start_features = torch.zeros((3, 4, 5), dtype=torch.float32)
    start_features[:, 1, 1] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    start_features[:, 1, 3] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

    update_features = torch.zeros((3, 4, 5), dtype=torch.float32)
    update_features[:, 1, 1] = torch.tensor([0.1, 0.995, 0.0])
    update_features[:, 1, 3] = torch.tensor([0.995, 0.1, 0.0])

    extractor.set_queue([start_features, update_features])

    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    image = Image.fromarray(frame)

    registry = InstanceRegistry()
    registry.register_keypoint(
        KeypointState(
            key="animalleft",
            instance_label="animal",
            label="left",
            x=1.0,
            y=1.0,
        )
    )
    registry.register_keypoint(
        KeypointState(
            key="animalright",
            instance_label="animal",
            label="right",
            x=3.0,
            y=1.0,
        )
    )

    tracker.start(image, registry, {})
    tracker.prev_gray = None

    results = tracker.update(image, {})
    assert len(results) == 2
    result_map = {item["label"]: item for item in results}
    left_result = result_map["animalleft"]
    right_result = result_map["animalright"]

    assert left_result["x"] == pytest.approx(1.5, abs=1e-6)
    assert right_result["x"] == pytest.approx(3.5, abs=1e-6)
    assert left_result["symmetry_partner"] == "animalright"
    assert right_result["symmetry_partner"] == "animalleft"
    assert left_result["symmetry_sign"] == 1.0
    assert right_result["symmetry_sign"] == -1.0


def test_support_probes_prefer_contextual_candidate(monkeypatch):
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        DummyExtractor,
    )

    def fake_support(self, track_key, feats, patch_rc, grid_hw, patch_mask):
        descriptor = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        descriptor = descriptor / (descriptor.norm() + 1e-12)
        return [SupportProbe(offset_rc=(0, 2), descriptor=descriptor, weight=1.0)]

    monkeypatch.setattr(
        DinoKeypointTracker,
        "_sample_support_probes",
        fake_support,
    )

    config = CutieDinoTrackerConfig(
        support_probe_count=1,
        support_probe_sigma=0.1,
        support_probe_radius=2,
        support_probe_weight=0.8,
        support_probe_mask_only=False,
        support_probe_mask_bonus=0.0,
        appearance_bundle_weight=0.0,
        baseline_similarity_weight=0.0,
        structural_consistency_weight=0.0,
        symmetry_penalty=0.0,
        mask_similarity_bonus=0.0,
    )

    tracker = DinoKeypointTracker(
        model_name="dummy",
        runtime_config=config,
        search_radius=1,
    )
    extractor = tracker.extractor

    start_features = torch.zeros((3, 4, 5), dtype=torch.float32)
    start_features[:, 1, 1] = torch.tensor([1.0, 0.0, 0.0])
    start_features[:, 1, 3] = torch.tensor([0.0, 1.0, 0.0])

    update_features = torch.zeros((3, 4, 5), dtype=torch.float32)
    update_features[:, 1, 1] = torch.tensor([1.0, 0.0, 0.0])
    update_features[:, 1, 2] = torch.tensor([1.0, 0.0, 0.0])
    update_features[:, 1, 3] = torch.tensor([0.0, 0.0, 1.0])
    update_features[:, 1, 4] = torch.tensor([0.0, 1.0, 0.0])

    extractor.set_queue([start_features, update_features])

    frame = np.zeros((4, 5, 3), dtype=np.uint8)
    image = Image.fromarray(frame)

    registry = InstanceRegistry()
    registry.register_keypoint(
        KeypointState(
            key="animalnose",
            instance_label="animal",
            label="nose",
            x=1.0,
            y=1.0,
        )
    )

    tracker.start(image, registry, {})
    tracker.prev_gray = None

    tracker.update(image, {})
    track_state = tracker.tracks["animalnose"]
    assert track_state.patch_rc == (1, 2)
