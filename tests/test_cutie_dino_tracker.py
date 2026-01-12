import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
from annolid.tracking.dino_keypoint_tracker import (
    DinoKeypointTracker,
    DinoKeypointVideoProcessor,
    SupportProbe,
)


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
    json_path.unlink()

    output_path = adapter.write_annotation(
        frame_number=frame_number,
        registry=registry,
        output_dir=source_dir,
    )
    produced = json.loads(Path(output_path).read_text())
    expected = json.loads(expected_fixture.read_text())

    produced_shapes = sorted(
        produced["shapes"], key=lambda shape: shape["label"])
    expected_shapes = sorted(
        expected["shapes"], key=lambda shape: shape["label"])
    assert produced_shapes == expected_shapes

    assert produced_shapes[0]["instance_label"] == produced_shapes[1]["instance_label"]


def test_annotation_adapter_uses_latest_manual_frame(tmp_path):
    source_dir = tmp_path / "clip"
    source_dir.mkdir()
    input_fixture = Path("tests/golden/input_frame.json")

    for frame in (0, 42, 101):
        json_path = source_dir / f"clip_{frame:09d}.json"
        shutil.copy(input_fixture, json_path)
        image_path = source_dir / f"clip_{frame:09d}.png"
        Image.new("RGB", (120, 100)).save(image_path)

    base_time = time.time()
    # Mark frame 101 as the newest manual edit and keep others older.
    os.utime(source_dir / "clip_000000101.json", (base_time, base_time))
    os.utime(source_dir / "clip_000000042.json",
             (base_time - 60, base_time - 60))
    os.utime(source_dir / "clip_000000000.json",
             (base_time - 120, base_time - 120))

    adapter = AnnotationAdapter(image_height=100, image_width=120)
    frame_number, _ = adapter.load_initial_state(source_dir)
    assert frame_number == 101


def test_tracker_enforces_mask_position(monkeypatch):
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        DummyExtractor,
    )
    config = CutieDinoTrackerConfig(
        mask_enforce_position=True,
        mask_enforce_search_radius=3,
        mask_descriptor_weight=0.0,
        appearance_bundle_weight=0.0,
        baseline_similarity_weight=0.0,
        structural_consistency_weight=0.0,
        symmetry_penalty=0.0,
        support_probe_weight=0.0,
        motion_prior_penalty_weight=0.0,
        motion_search_gain=0.0,
        motion_search_miss_boost=0.0,
    )

    tracker = DinoKeypointTracker(
        model_name="dummy",
        runtime_config=config,
        search_radius=1,
    )
    extractor = tracker.extractor

    # start with a simple 3x3 grid where the centre is distinctive
    start_features = torch.zeros((2, 3, 3), dtype=torch.float32)
    start_features[:, 1, 1] = torch.tensor([1.0, 0.0], dtype=torch.float32)

    # make the future frame strongly favour a corner outside the mask
    update_features = torch.zeros((2, 3, 3), dtype=torch.float32)
    update_features[:, 2, 2] = torch.tensor([1.0, 0.0], dtype=torch.float32)
    update_features[:, 1, 1] = torch.tensor([0.0, 1.0], dtype=torch.float32)

    extractor.set_queue([start_features, update_features])

    frame = np.zeros((3, 3, 3), dtype=np.uint8)
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

    # mask only retains the middle pixel (1,1)
    mask = np.zeros((3, 3), dtype=bool)
    mask[1, 1] = True

    tracker.start(image, registry, {"animal": mask})
    results = tracker.update(image, {"animal": mask})

    assert len(results) == 1
    result = results[0]
    assert int(math.floor(result["x"])) == 1
    assert int(math.floor(result["y"])) == 1
    assert result["visible"] is True


def test_tracker_rejects_mask_violation_when_radius_too_small(monkeypatch):
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        DummyExtractor,
    )
    config = CutieDinoTrackerConfig(
        mask_enforce_position=True,
        mask_enforce_search_radius=1,
        mask_descriptor_weight=0.0,
        appearance_bundle_weight=0.0,
        baseline_similarity_weight=0.0,
        structural_consistency_weight=0.0,
        symmetry_penalty=0.0,
        support_probe_weight=0.0,
        motion_prior_penalty_weight=0.0,
        motion_search_gain=0.0,
        motion_search_miss_boost=0.0,
    )

    tracker = DinoKeypointTracker(
        model_name="dummy",
        runtime_config=config,
        search_radius=2,
    )
    extractor = tracker.extractor

    start_features = torch.zeros((2, 3, 3), dtype=torch.float32)
    start_features[:, 0, 0] = torch.tensor([1.0, 0.0], dtype=torch.float32)

    update_features = torch.zeros((2, 3, 3), dtype=torch.float32)
    update_features[:, 2, 2] = torch.tensor([1.0, 0.0], dtype=torch.float32)
    update_features[:, 0, 0] = torch.tensor([0.0, 1.0], dtype=torch.float32)

    extractor.set_queue([start_features, update_features])

    frame = np.zeros((3, 3, 3), dtype=np.uint8)
    image = Image.fromarray(frame)

    registry = InstanceRegistry()
    registry.register_keypoint(
        KeypointState(
            key="animalnose",
            instance_label="animal",
            label="nose",
            x=0.0,
            y=0.0,
        )
    )

    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True

    tracker.start(image, registry, {"animal": mask})
    results = tracker.update(image, {"animal": mask})

    assert len(results) == 1
    result = results[0]
    assert result["visible"] is False
    assert result["misses"] == 1
    assert tracker.tracks["animalnose"].misses == 1


def test_tracker_roi_uses_polygon_bbox(monkeypatch):
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        DummyExtractor,
    )
    tracker = DinoKeypointTracker(model_name="dummy", search_radius=1)
    extractor: DummyExtractor = tracker.extractor  # type: ignore[assignment]

    image = Image.new("RGB", (64, 64))
    registry = InstanceRegistry()
    instance = registry.ensure_instance("animal")
    polygon = [
        (30.0, 20.0),
        (40.0, 20.0),
        (40.0, 30.0),
        (30.0, 30.0),
        (30.0, 20.0),
    ]
    instance.set_mask(bitmap=None, polygon=polygon)
    registry.register_keypoint(
        KeypointState(
            key="animalnose",
            instance_label="animal",
            label="nose",
            x=35.0,
            y=25.0,
        )
    )

    expected_roi = (14, 4, 57, 47)
    roi_width = expected_roi[2] - expected_roi[0]
    roi_height = expected_roi[3] - expected_roi[1]
    extractor.set_queue([torch.zeros((2, roi_height, roi_width))])

    tracker.start(image, registry, mask_lookup=None)
    assert tracker._roi_box == expected_roi


def test_tracker_roi_defaults_to_full_frame_without_polygons(monkeypatch):
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        DummyExtractor,
    )
    tracker = DinoKeypointTracker(model_name="dummy", search_radius=1)
    extractor: DummyExtractor = tracker.extractor  # type: ignore[assignment]

    image = Image.new("RGB", (64, 64))
    registry = InstanceRegistry()
    registry.register_keypoint(
        KeypointState(
            key="animalnose",
            instance_label="animal",
            label="nose",
            x=50.0,
            y=10.0,
        )
    )

    expected_roi = (0, 0, 64, 64)
    roi_width = expected_roi[2] - expected_roi[0]
    roi_height = expected_roi[3] - expected_roi[1]
    extractor.set_queue([torch.zeros((2, roi_height, roi_width))])

    tracker.start(image, registry, mask_lookup=None)
    assert tracker._roi_box == expected_roi


def test_tracker_reset_state_clears_history(monkeypatch):
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        DummyExtractor,
    )
    tracker = DinoKeypointTracker(
        model_name="dummy",
        runtime_config=CutieDinoTrackerConfig(
            appearance_bundle_weight=0.0,
            baseline_similarity_weight=0.0,
            structural_consistency_weight=0.0,
            symmetry_penalty=0.0,
            support_probe_weight=0.0,
            motion_prior_penalty_weight=0.0,
            motion_search_gain=0.0,
            motion_search_miss_boost=0.0,
        ),
        search_radius=1,
    )
    extractor = tracker.extractor
    start_features = torch.zeros((2, 2, 2), dtype=torch.float32)
    start_features[:, 0, 0] = torch.tensor([1.0, 0.0], dtype=torch.float32)
    extractor.set_queue([start_features])

    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    image = Image.fromarray(frame)

    registry = InstanceRegistry()
    registry.register_keypoint(
        KeypointState(
            key="animalnose",
            instance_label="animal",
            label="nose",
            x=0.0,
            y=0.0,
        )
    )

    tracker.start(image, registry, {})
    assert tracker.prev_gray is not None
    assert tracker.tracks

    tracker.reset_state()
    assert tracker.prev_gray is None
    assert tracker.tracks == {}


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


def test_cutie_mask_manager_reset_state(tmp_path):
    adapter = AnnotationAdapter(image_height=10, image_width=10)
    config = CutieDinoTrackerConfig(use_cutie_tracking=True)
    manager = CutieMaskManager(tmp_path / "video.mp4", adapter, config)

    manager._processor = object()
    manager._core = object()
    manager._device = "cuda"
    manager._label_to_value = {"animal": 1}
    manager._value_to_label = {1: "animal"}
    manager._initialized = True
    manager._last_results = {"animal": MaskResult(
        "animal", np.ones((2, 2), dtype=bool), [])}
    manager._mask_miss_counts = {"animal": 2}

    manager.reset_state()

    assert manager._processor is None
    assert manager._core is None
    assert manager._device is None
    assert manager._label_to_value == {}
    assert manager._value_to_label == {}
    assert manager._initialized is False
    assert manager._last_results == {}
    assert manager._mask_miss_counts == {}


def test_video_processor_resets_on_manual_resume(tmp_path, monkeypatch):
    class StubVideo:
        def __init__(self, _path):
            self.frames = [
                np.full((4, 4, 3), fill_value=i, dtype=np.uint8)
                for i in range(7)
            ]

        def get_first_frame(self):
            return self.frames[0]

        def total_frames(self):
            return len(self.frames)

        def load_frame(self, index):
            if 0 <= index < len(self.frames):
                return self.frames[index]
            return None

    class StubMaskManager:
        instance = None

        def __init__(self, *args, **kwargs):
            self.enabled = True
            self.reset_calls: List[bool] = []
            self.prime_calls: List[int] = []
            self.update_calls: List[int] = []
            self._initialized = False
            self._last_results: Dict[str, MaskResult] = {}
            StubMaskManager.instance = self

        def ready(self) -> bool:
            return self.enabled and self._initialized

        def reset_state(self) -> None:
            self.reset_calls.append(True)
            self._initialized = False
            self._last_results = {}

        def prime(self, frame_number: int, _frame, _registry) -> None:
            self.prime_calls.append(frame_number)
            self._initialized = True

        def update_masks(self, frame_number: int, _frame, _registry):
            if self.ready():
                self.update_calls.append(frame_number)
            results: Dict[str, MaskResult] = {}
            polygon = [
                (0.0, 0.0),
                (0.0, 1.0),
                (1.0, 1.0),
                (1.0, 0.0),
            ]
            for instance in _registry:
                mask_bitmap = np.ones((4, 4), dtype=bool)
                result = MaskResult(instance.label, mask_bitmap, polygon)
                self._last_results[instance.label] = result
                results[instance.label] = result
            return results

    class StubTracker:
        instance = None

        def __init__(self, *args, runtime_config=None, **_kwargs):
            self.runtime_config = runtime_config or CutieDinoTrackerConfig()
            self.reset_calls: List[bool] = []
            self.start_frames: List[int] = []
            self.update_frames: List[int] = []
            self._active = False
            self.start_masks: List[Dict[str, np.ndarray]] = []
            StubTracker.instance = self

        def reset_state(self) -> None:
            self.reset_calls.append(True)
            self._active = False

        def start(self, image, _registry, mask_lookup) -> None:
            frame_array = np.array(image)
            frame_id = int(frame_array[0, 0, 0])
            self.start_frames.append(frame_id)
            captured_masks: Dict[str, np.ndarray] = {}
            if mask_lookup:
                for label, mask in mask_lookup.items():
                    captured_masks[label] = np.array(mask, copy=True)
            self.start_masks.append(captured_masks)
            self._active = True

        def update(self, image, _mask_lookup):
            if not self._active:
                return []
            frame_array = np.array(image)
            frame_id = int(frame_array[0, 0, 0])
            self.update_frames.append(frame_id)
            return []

    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.CV2Video",
        StubVideo,
    )
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.CutieMaskManager",
        StubMaskManager,
    )
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.DinoKeypointTracker",
        StubTracker,
    )

    video_path = tmp_path / "clip.mp4"
    video_path.write_text("video")
    result_dir = tmp_path / "clip"
    result_dir.mkdir()

    adapter = AnnotationAdapter(image_height=4, image_width=4)

    def write_manual(frame_number: int, x: float, y: float) -> Path:
        registry = InstanceRegistry()
        registry.register_keypoint(
            KeypointState(
                key="animalnose",
                instance_label="animal",
                label="nose",
                x=x,
                y=y,
            )
        )
        json_path = adapter.write_annotation(
            frame_number=frame_number,
            registry=registry,
            output_dir=result_dir,
        )
        image_path = result_dir / f"clip_{frame_number:09d}.png"
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(image_path)
        return Path(json_path)

    json_frame_three = write_manual(3, 1.0, 1.0)
    json_frame_five = write_manual(5, 2.0, 2.0)

    now = time.time()
    os.utime(json_frame_three, (now, now))
    os.utime(json_frame_five, (now - 60, now - 60))

    processor = DinoKeypointVideoProcessor(
        video_path=str(video_path),
        result_folder=result_dir,
        model_name="dummy",
        runtime_config=CutieDinoTrackerConfig(),
    )

    processor.process_video()

    mask_manager = StubMaskManager.instance
    tracker = StubTracker.instance

    assert mask_manager is not None
    assert tracker is not None

    assert len(mask_manager.reset_calls) == 2
    assert mask_manager.prime_calls == [3, 5]
    assert mask_manager.update_calls == [4, 6]

    assert len(tracker.reset_calls) == 2
    assert tracker.start_frames == [3, 5]
    assert tracker.update_frames == [4, 6]
    assert len(tracker.start_masks) == 2
    assert tracker.start_masks[0] == {}
    assert tracker.start_masks[1]["animal"].any()


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
    assert result["x"] == pytest.approx(2.5, abs=1e-6)
    assert tracker._mask_miss_counts.get("animal") == 1


def test_symmetry_pairs_prevent_swaps(monkeypatch):
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        DummyExtractor,
    )
    config = CutieDinoTrackerConfig(
        symmetry_pairs=(("left", "right"),),
        symmetry_penalty=1.0,
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
    start_features[:, 1, 1] = torch.tensor(
        [1.0, 0.0, 0.0], dtype=torch.float32)
    start_features[:, 1, 3] = torch.tensor(
        [0.0, 1.0, 0.0], dtype=torch.float32)

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

    assert left_result["x"] == pytest.approx(3.5, abs=1e-6)
    assert right_result["x"] == pytest.approx(1.5, abs=1e-6)
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


def test_gaussian_refine_shifts_keypoint_towards_secondary_peak(monkeypatch):
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        DummyExtractor,
    )
    config = CutieDinoTrackerConfig(
        keypoint_refine_radius=1,
        keypoint_refine_sigma=1.0,
        keypoint_refine_temperature=0.2,
        mask_descriptor_weight=0.0,
        appearance_bundle_weight=0.0,
        baseline_similarity_weight=0.0,
        structural_consistency_weight=0.0,
        symmetry_penalty=0.0,
        support_probe_weight=0.0,
        motion_prior_penalty_weight=0.0,
        motion_search_gain=0.0,
        motion_search_miss_boost=0.0,
        mask_similarity_bonus=0.0,
        velocity_smoothing=0.0,
    )
    tracker = DinoKeypointTracker(
        model_name="dummy",
        runtime_config=config,
        search_radius=1,
        min_similarity=0.0,
        momentum=0.0,
        reference_weight=0.0,
    )
    extractor = tracker.extractor

    start_features = torch.zeros((2, 3, 4), dtype=torch.float32)
    start_features[:, 1, 1] = torch.tensor([1.0, 0.0], dtype=torch.float32)

    update_features = torch.zeros((2, 3, 4), dtype=torch.float32)
    update_features[:, 1, 2] = torch.tensor([1.0, 0.0], dtype=torch.float32)
    update_features[:, 1, 1] = torch.tensor(
        [0.9, math.sqrt(1.0 - 0.9 ** 2)], dtype=torch.float32
    )

    extractor.set_queue([start_features, update_features])

    frame = np.zeros((3, 4, 3), dtype=np.uint8)
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

    results = tracker.update(image, {})
    assert len(results) == 1
    assert results[0]["visible"] is True
    assert results[0]["x"] == pytest.approx(2.231, abs=0.05)


def test_motion_penalty_prefers_nearby_candidate_over_far_peak(monkeypatch):
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        DummyExtractor,
    )
    config = CutieDinoTrackerConfig(
        mask_descriptor_weight=0.0,
        appearance_bundle_weight=0.0,
        baseline_similarity_weight=0.0,
        structural_consistency_weight=0.0,
        symmetry_penalty=0.0,
        support_probe_weight=0.0,
        mask_similarity_bonus=0.0,
        velocity_smoothing=0.0,
        keypoint_refine_radius=0,
        motion_search_tighten=1.0,
        motion_search_gain=0.0,
        motion_search_flow_gain=0.0,
        motion_search_min_radius=1.0,
        motion_search_max_radius=16.0,
        motion_search_miss_boost=0.0,
        motion_prior_penalty_weight=1.0,
        motion_prior_soft_radius_px=1.0,
        motion_prior_radius_factor=1.0,
        motion_prior_miss_relief=0.0,
        motion_prior_flow_relief=0.0,
    )

    tracker = DinoKeypointTracker(
        model_name="dummy",
        runtime_config=config,
        search_radius=8,
        min_similarity=0.0,
        momentum=0.0,
        reference_weight=0.0,
    )
    extractor = tracker.extractor

    start_features = torch.zeros((2, 1, 9), dtype=torch.float32)
    start_features[:, 0, 1] = torch.tensor([1.0, 0.0], dtype=torch.float32)

    update_features = torch.zeros((2, 1, 9), dtype=torch.float32)
    update_features[:, 0, 2] = torch.tensor(
        [0.9, math.sqrt(1.0 - 0.9 ** 2)], dtype=torch.float32
    )
    update_features[:, 0, 7] = torch.tensor(
        [0.95, math.sqrt(1.0 - 0.95 ** 2)], dtype=torch.float32
    )

    extractor.set_queue([start_features, update_features])

    frame = np.zeros((1, 9, 3), dtype=np.uint8)
    image = Image.fromarray(frame)

    registry = InstanceRegistry()
    registry.register_keypoint(
        KeypointState(
            key="animalnose",
            instance_label="animal",
            label="nose",
            x=1.0,
            y=0.0,
        )
    )

    tracker.start(image, registry, {})
    tracker.prev_gray = None

    tracker.update(image, {})
    assert tracker.tracks["animalnose"].patch_rc == (0, 2)


def test_body_prior_rejects_ear_to_tail_swap_on_mask(monkeypatch):
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        DummyExtractor,
    )
    config = CutieDinoTrackerConfig(
        mask_descriptor_weight=0.0,
        appearance_bundle_weight=0.0,
        baseline_similarity_weight=0.0,
        structural_consistency_weight=1.0,
        symmetry_penalty=0.0,
        support_probe_weight=0.0,
        motion_prior_penalty_weight=0.0,
        motion_prior_soft_radius_px=100.0,
        motion_prior_radius_factor=1.0,
        motion_search_gain=0.0,
        motion_search_flow_gain=0.0,
        motion_search_miss_boost=0.0,
        keypoint_refine_radius=0,
        mask_enforce_position=False,
    )

    tracker = DinoKeypointTracker(
        model_name="dummy",
        runtime_config=config,
        search_radius=8,
        min_similarity=0.0,
        momentum=0.0,
        reference_weight=0.0,
    )
    extractor = tracker.extractor

    start_features = torch.zeros((2, 2, 9), dtype=torch.float32)
    start_features[:, 0, 1] = torch.tensor([1.0, 0.0], dtype=torch.float32)

    update_features = torch.zeros((2, 2, 9), dtype=torch.float32)
    update_features[:, 0, 7] = torch.tensor([1.0, 0.0], dtype=torch.float32)

    extractor.set_queue([start_features, update_features])

    frame = np.zeros((2, 9, 3), dtype=np.uint8)
    image = Image.fromarray(frame)

    registry = InstanceRegistry()
    registry.register_keypoint(
        KeypointState(
            key="animalnose",
            instance_label="animal",
            label="nose",
            x=1.0,
            y=0.0,
        )
    )

    mask = np.ones((2, 9), dtype=bool)
    tracker.start(image, registry, {"animal": mask})
    tracker.prev_gray = None

    results = tracker.update(image, {"animal": mask})
    assert results[0]["visible"] is True
    assert results[0]["misses"] == 0
    assert tracker.tracks["animalnose"].patch_rc == (0, 1)
