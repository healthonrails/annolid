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
    KeypointTrack,
    PixelFlowEstimate,
    PixelRefineOptions,
    RefineRegion,
    SupportProbe,
)


class DummyExtractor:
    patch_size = 1

    def __init__(self, _cfg):
        self.cfg = _cfg
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


class CoarsePatchDummyExtractor(DummyExtractor):
    patch_size = 16


def test_tracker_uses_runtime_dino_model_when_constructor_is_default(monkeypatch):
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        DummyExtractor,
    )
    selected_model = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    config = CutieDinoTrackerConfig(dinov3_model_name=selected_model)

    tracker = DinoKeypointTracker(
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        runtime_config=config,
    )

    assert tracker.extractor.cfg.model_name == selected_model
    assert config.patch_model_name == selected_model
    assert config.dinov3_model_name == selected_model


def test_tracker_explicit_constructor_dino_model_wins(monkeypatch):
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        DummyExtractor,
    )
    selected_model = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    explicit_model = "facebook/dinov3-vitb16-pretrain-lvd1426"
    config = CutieDinoTrackerConfig(dinov3_model_name=selected_model)

    tracker = DinoKeypointTracker(
        model_name=explicit_model,
        runtime_config=config,
    )

    assert tracker.extractor.cfg.model_name == explicit_model
    assert config.patch_model_name == explicit_model
    assert config.dinov3_model_name == explicit_model


def test_pixel_refine_options_normalize_runtime_values() -> None:
    config = CutieDinoTrackerConfig(
        pixel_refine_enabled=True,
        pixel_refine_weight=2.0,
        pixel_refine_window=10,
        pixel_refine_max_error=-1.0,
        pixel_refine_max_jump_px=-3.0,
    )

    options = PixelRefineOptions.from_runtime(config)

    assert options.enabled is True
    assert options.weight == 1.0
    assert options.window == 11
    assert options.max_error == 0.0
    assert options.max_jump_px == 0.0
    assert options.error_confidence(1000.0) == 1.0


def test_frame_motion_uses_lk_position_without_adding_velocity(monkeypatch) -> None:
    tracker = object.__new__(DinoKeypointTracker)
    track = KeypointTrack(
        key="animalnose",
        storage_label="animalnose",
        instance_label="animal",
        display_label="nose",
        patch_rc=(1, 1),
        descriptor=torch.ones(1),
        reference_descriptor=torch.ones(1),
        velocity=(4.0, -2.0),
        last_position=(10.0, 10.0),
    )
    lk_estimate = PixelFlowEstimate(xy=(13.0, 9.0), error=1.0)
    monkeypatch.setattr(
        tracker,
        "_track_pixel_flow",
        lambda *_args, **_kwargs: lk_estimate,
    )

    motion = tracker._estimate_frame_motion(
        track,
        prev_gray=np.zeros((24, 24), dtype=np.uint8),
        frame_gray=np.zeros((24, 24), dtype=np.uint8),
        dense_flow=None,
    )

    assert motion.predicted_xy == pytest.approx((13.0, 9.0))
    assert motion.flow_vec == pytest.approx((3.0, -1.0))
    assert motion.pixel_flow is lk_estimate


def test_frame_motion_samples_dense_flow_at_last_emitted_point(monkeypatch) -> None:
    tracker = object.__new__(DinoKeypointTracker)
    track = KeypointTrack(
        key="animalnose",
        storage_label="animalnose",
        instance_label="animal",
        display_label="nose",
        patch_rc=(2, 2),
        descriptor=torch.ones(1),
        reference_descriptor=torch.ones(1),
        velocity=(8.0, 8.0),
        last_position=(12.0, 10.0),
    )
    monkeypatch.setattr(
        tracker,
        "_track_pixel_flow",
        lambda *_args, **_kwargs: None,
    )
    dense_flow = np.zeros((24, 24, 2), dtype=np.float32)
    dense_flow[10, 12] = (2.5, -1.5)

    motion = tracker._estimate_frame_motion(
        track,
        prev_gray=np.zeros((24, 24), dtype=np.uint8),
        frame_gray=np.zeros((24, 24), dtype=np.uint8),
        dense_flow=dense_flow,
    )

    assert motion.predicted_xy == pytest.approx((14.5, 8.5))
    assert motion.flow_vec == pytest.approx((2.5, -1.5))
    assert motion.pixel_flow is None


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

    produced_shapes = sorted(produced["shapes"], key=lambda shape: shape["label"])
    expected_shapes = sorted(expected["shapes"], key=lambda shape: shape["label"])
    assert produced_shapes == expected_shapes

    assert produced_shapes[0]["instance_label"] == produced_shapes[1]["instance_label"]


def test_annotation_adapter_associates_plain_points_with_containing_polygon(
    tmp_path,
):
    source_dir = tmp_path / "mouse"
    source_dir.mkdir()
    json_path = source_dir / "mouse_000000000.json"
    json_path.write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": "mouse",
                        "shape_type": "polygon",
                        "group_id": 0,
                        "points": [[20, 20], [60, 20], [60, 60], [20, 60]],
                    },
                    {
                        "label": "teaball",
                        "shape_type": "polygon",
                        "group_id": 1,
                        "points": [[80, 70], [100, 70], [100, 90], [80, 90]],
                    },
                    {
                        "label": "ear",
                        "shape_type": "point",
                        "points": [[50, 50]],
                    },
                    {
                        "label": "tailbase",
                        "shape_type": "point",
                        "points": [[25, 25]],
                    },
                    {
                        "label": "unassigned",
                        "shape_type": "point",
                        "points": [[5, 5]],
                    },
                    {
                        "label": "grouped_marker",
                        "shape_type": "point",
                        "group_id": 1,
                        "points": [[5, 5]],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    adapter = AnnotationAdapter(image_height=100, image_width=120)
    registry = adapter.read_annotation(json_path)

    assert set(registry.instances) == {"mouse", "teaball", "unassigned"}
    mouse_keypoints = registry.instances["mouse"].keypoints
    assert set(mouse_keypoints) == {"mouseear", "mousetailbase"}
    assert mouse_keypoints["mouseear"].storage_label == "ear"
    assert mouse_keypoints["mousetailbase"].storage_label == "tailbase"
    assert "teaballgrouped_marker" in registry.instances["teaball"].keypoints

    output_path = adapter.write_annotation(
        frame_number=1,
        registry=registry,
        output_dir=source_dir,
    )
    output = json.loads(output_path.read_text(encoding="utf-8"))
    point_shapes = {
        shape["label"]: shape
        for shape in output["shapes"]
        if shape["shape_type"] == "point"
    }
    assert point_shapes["ear"]["instance_label"] == "mouse"
    assert point_shapes["ear"]["display_label"] == "ear"
    assert point_shapes["tailbase"]["instance_label"] == "mouse"

    roundtrip = adapter.read_annotation(output_path)
    assert roundtrip.instances["mouse"].keypoints["mouseear"].storage_label == "ear"


def test_annotation_adapter_explicit_instance_wins_over_polygon_inference(tmp_path):
    json_path = tmp_path / "seed.json"
    json_path.write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": "mouse",
                        "shape_type": "polygon",
                        "points": [[10, 10], [40, 10], [40, 40], [10, 40]],
                    },
                    {
                        "label": "ear",
                        "shape_type": "point",
                        "points": [[20, 20]],
                        "flags": {
                            "instance_label": "explicit_animal",
                            "display_label": "ear",
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    registry = AnnotationAdapter(image_height=50, image_width=50).read_annotation(
        json_path
    )

    assert "explicit_animal" in registry.instances
    assert "explicit_animalear" in registry.instances["explicit_animal"].keypoints
    assert not registry.instances["mouse"].keypoints


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
    os.utime(source_dir / "clip_000000042.json", (base_time - 60, base_time - 60))
    os.utime(source_dir / "clip_000000000.json", (base_time - 120, base_time - 120))

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
    manager._last_results = {"animal": MaskResult("animal", base_mask, polygon)}
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
    manager._last_results = {
        "animal": MaskResult("animal", np.ones((2, 2), dtype=bool), [])
    }
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
                np.full((4, 4, 3), fill_value=i, dtype=np.uint8) for i in range(7)
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


def test_video_processor_skips_finished_frames_between_seeded_frames(
    tmp_path, monkeypatch
):
    class StubVideo:
        def __init__(self, _path):
            self.frames = [
                np.full((4, 4, 3), fill_value=i, dtype=np.uint8) for i in range(7)
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
            self.update_calls: List[int] = []
            self._initialized = False
            self._last_results: Dict[str, MaskResult] = {}
            StubMaskManager.instance = self

        def ready(self) -> bool:
            return self.enabled and self._initialized

        def reset_state(self) -> None:
            self._initialized = False
            self._last_results = {}

        def prime(self, frame_number: int, _frame, _registry) -> None:
            self._initialized = True

        def update_masks(self, frame_number: int, _frame, _registry):
            if self.ready():
                self.update_calls.append(frame_number)
            return {}

    class StubTracker:
        instance = None

        def __init__(self, *args, runtime_config=None, **_kwargs):
            self.runtime_config = runtime_config or CutieDinoTrackerConfig()
            self.update_frames: List[int] = []
            self._active = False
            StubTracker.instance = self

        def reset_state(self) -> None:
            self._active = False

        def start(self, _image, _registry, _mask_lookup) -> None:
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

    def write_seed(frame_number: int, x: float, y: float) -> Path:
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
        (result_dir / f"clip_{frame_number:09d}.png").write_bytes(b"")
        return Path(json_path)

    frame_three = write_seed(3, 1.0, 1.0)
    write_seed(5, 2.0, 2.0)

    # Existing completed output between adjacent seeds (3, 5) should be skipped.
    existing_registry = InstanceRegistry()
    existing_registry.register_keypoint(
        KeypointState(
            key="animalnose",
            instance_label="animal",
            label="nose",
            x=1.5,
            y=1.5,
        )
    )
    adapter.write_annotation(
        frame_number=4,
        registry=existing_registry,
        output_dir=result_dir,
    )

    now = time.time()
    os.utime(frame_three, (now, now))

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
    assert 4 not in mask_manager.update_calls
    assert 4 not in tracker.update_frames
    assert 6 in tracker.update_frames


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
        [0.6, math.sqrt(1.0 - 0.6**2)], dtype=torch.float32
    )
    update_features[:, 1, 2] = torch.tensor(
        [0.95, math.sqrt(1.0 - 0.95**2)], dtype=torch.float32
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


def test_mask_fallback_keeps_previous_roi_when_mask_drops(monkeypatch):
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        DummyExtractor,
    )
    config = CutieDinoTrackerConfig(
        mask_similarity_bonus=1.0,
        max_mask_fallback_frames=2,
        mask_dilation_iterations=0,
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
        min_similarity=0.0,
        momentum=0.0,
        reference_weight=0.0,
    )
    extractor = tracker.extractor

    image = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    registry = InstanceRegistry()
    registry.register_keypoint(
        KeypointState(
            key="animalnose",
            instance_label="animal",
            label="nose",
            x=30.0,
            y=30.0,
        )
    )

    mask = np.zeros((64, 64), dtype=bool)
    mask[30, 30] = True
    expected_roi = (14, 14, 47, 47)
    roi_height = expected_roi[3] - expected_roi[1]
    roi_width = expected_roi[2] - expected_roi[0]
    start_features = torch.zeros((2, roi_height, roi_width), dtype=torch.float32)
    start_features[:, 16, 16] = torch.tensor([1.0, 0.0], dtype=torch.float32)

    update_features = torch.zeros((2, roi_height, roi_width), dtype=torch.float32)
    update_features[:, 16, 16] = torch.tensor(
        [0.6, math.sqrt(1.0 - 0.6**2)], dtype=torch.float32
    )
    update_features[:, 16, 17] = torch.tensor(
        [0.95, math.sqrt(1.0 - 0.95**2)], dtype=torch.float32
    )

    extractor.set_queue([start_features, update_features])

    tracker.start(image, registry, {"animal": mask})
    tracker.prev_gray = None
    assert tracker._roi_box == expected_roi

    results = tracker.update(image, {})

    assert tracker._roi_box == expected_roi
    assert tracker._mask_miss_counts.get("animal") == 1
    assert results[0]["visible"] is True
    assert results[0]["x"] == pytest.approx(30.5, abs=1e-6)


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
        dinov3_positional_debias=False,
        dinov3_backward_consistency=False,
        keypoint_cluster_refine=False,
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
        dinov3_positional_debias=False,
        dinov3_backward_consistency=False,
        keypoint_cluster_refine=False,
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
        [0.9, math.sqrt(1.0 - 0.9**2)], dtype=torch.float32
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


def test_pixel_flow_refine_moves_keypoint_with_subpatch_joint_motion(monkeypatch):
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        CoarsePatchDummyExtractor,
    )
    config = CutieDinoTrackerConfig(
        pixel_refine_enabled=True,
        pixel_refine_weight=1.0,
        pixel_refine_window=15,
        pixel_refine_max_error=100.0,
        pixel_refine_max_jump_px=8.0,
        mask_descriptor_weight=0.0,
        appearance_bundle_weight=0.0,
        baseline_similarity_weight=0.0,
        structural_consistency_weight=0.0,
        symmetry_penalty=0.0,
        support_probe_weight=0.0,
        motion_prior_penalty_weight=0.0,
        motion_search_gain=0.0,
        motion_search_flow_gain=0.0,
        motion_search_miss_boost=0.0,
        keypoint_refine_radius=0,
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

    start_features = torch.zeros((2, 2, 2), dtype=torch.float32)
    start_features[:, 0, 0] = torch.tensor([1.0, 0.0], dtype=torch.float32)
    update_features = torch.zeros((2, 2, 2), dtype=torch.float32)
    update_features[:, 0, 0] = torch.tensor([1.0, 0.0], dtype=torch.float32)
    extractor.set_queue([start_features, update_features])

    start_frame = np.zeros((32, 32, 3), dtype=np.uint8)
    start_frame[8:13, 8:13] = 255
    update_frame = np.zeros((32, 32, 3), dtype=np.uint8)
    update_frame[8:13, 10:15] = 255
    start_image = Image.fromarray(start_frame)
    update_image = Image.fromarray(update_frame)

    registry = InstanceRegistry()
    registry.register_keypoint(
        KeypointState(
            key="flyjoint",
            instance_label="fly",
            label="joint",
            x=10.0,
            y=10.0,
        )
    )

    tracker.start(start_image, registry, {})
    results = tracker.update(update_image, {})

    assert len(results) == 1
    assert results[0]["visible"] is True
    assert results[0]["x"] == pytest.approx(12.0, abs=0.35)
    assert results[0]["y"] == pytest.approx(10.0, abs=0.35)


def test_positional_debias_prefers_semantic_match_over_coordinate_bias(monkeypatch):
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        DummyExtractor,
    )
    config = CutieDinoTrackerConfig(
        dinov3_positional_debias=True,
        dinov3_positional_debias_components=2,
        dinov3_positional_debias_strength=1.0,
        pixel_refine_enabled=False,
        mask_descriptor_weight=0.0,
        appearance_bundle_weight=0.0,
        baseline_similarity_weight=0.0,
        structural_consistency_weight=0.0,
        symmetry_penalty=0.0,
        support_probe_weight=0.0,
        motion_prior_penalty_weight=0.0,
        motion_search_gain=0.0,
        motion_search_flow_gain=0.0,
        motion_search_miss_boost=0.0,
        keypoint_refine_radius=0,
        context_weight=0.0,
        part_shared_weight=0.0,
    )
    tracker = DinoKeypointTracker(
        model_name="dummy",
        runtime_config=config,
        search_radius=4,
        min_similarity=0.0,
        momentum=0.0,
        reference_weight=0.0,
    )
    extractor = tracker.extractor

    def coordinate_biased_features(semantic_col: int) -> torch.Tensor:
        features = torch.zeros((2, 1, 5), dtype=torch.float32)
        features[1, 0, :] = torch.linspace(-2.0, 2.0, 5)
        features[0, 0, semantic_col] = 1.0
        return features

    start_features = coordinate_biased_features(1)
    update_features = coordinate_biased_features(3)
    extractor.set_queue([start_features, update_features])

    frame = np.zeros((1, 5, 3), dtype=np.uint8)
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
    results = tracker.update(image, {})

    assert len(results) == 1
    assert results[0]["visible"] is True
    assert tracker.tracks["animalnose"].patch_rc == (0, 3)


def test_positional_debias_uses_svd_channel_basis_when_available():
    tracker = object.__new__(DinoKeypointTracker)

    def svd_basis(_feats, *, components):
        assert components == 1
        return torch.tensor([[0.0], [1.0]], dtype=torch.float32)

    tracker._svd_positional_debias_basis = svd_basis

    features = torch.tensor(
        [
            [[1.0, 0.0, 1.0]],
            [[0.5, 2.0, -0.5]],
        ],
        dtype=torch.float32,
    )

    debiased = tracker._positionally_debias_feature_grid(
        DinoKeypointTracker._normalize_feature_grid(features),
        components=1,
        strength=1.0,
    )

    assert torch.max(torch.abs(debiased[1])).item() == pytest.approx(0.0, abs=1e-6)
    assert debiased[0, 0, 0].item() == pytest.approx(1.0, abs=1e-6)
    assert debiased[0, 0, 2].item() == pytest.approx(1.0, abs=1e-6)


def test_backward_consistency_prefers_candidate_mapping_to_previous_keypoint(
    monkeypatch,
):
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        DummyExtractor,
    )
    config = CutieDinoTrackerConfig(
        dinov3_backward_consistency=True,
        dinov3_backward_consistency_weight=0.3,
        dinov3_backward_consistency_tolerance=0,
        pixel_refine_enabled=False,
        mask_descriptor_weight=0.0,
        appearance_bundle_weight=0.0,
        baseline_similarity_weight=0.0,
        structural_consistency_weight=0.0,
        symmetry_penalty=0.0,
        support_probe_weight=0.0,
        motion_prior_penalty_weight=0.0,
        motion_search_gain=0.0,
        motion_search_flow_gain=0.0,
        motion_search_miss_boost=0.0,
        keypoint_refine_radius=0,
        context_weight=0.0,
        part_shared_weight=0.0,
    )
    tracker = DinoKeypointTracker(
        model_name="dummy",
        runtime_config=config,
        search_radius=4,
        min_similarity=0.0,
        momentum=0.0,
        reference_weight=0.0,
    )
    extractor = tracker.extractor

    start_features = torch.zeros((3, 1, 5), dtype=torch.float32)
    start_features[:, 0, 1] = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    start_features[:, 0, 3] = torch.tensor(
        [0.95, math.sqrt(1.0 - 0.95**2), 0.0], dtype=torch.float32
    )

    update_features = torch.zeros((3, 1, 5), dtype=torch.float32)
    update_features[:, 0, 2] = torch.tensor(
        [0.93, 0.0, math.sqrt(1.0 - 0.93**2)], dtype=torch.float32
    )
    update_features[:, 0, 4] = torch.tensor(
        [0.95, math.sqrt(1.0 - 0.95**2), 0.0], dtype=torch.float32
    )

    extractor.set_queue([start_features, update_features])

    frame = np.zeros((1, 5, 3), dtype=np.uint8)
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
    results = tracker.update(image, {})

    assert len(results) == 1
    assert results[0]["visible"] is True
    assert tracker.tracks["animalnose"].patch_rc == (0, 2)


def test_connected_peak_refine_aggregates_only_seed_component():
    config = CutieDinoTrackerConfig(
        keypoint_cluster_refine=True,
        keypoint_cluster_refine_radius=2,
        keypoint_cluster_refine_drop=0.12,
        keypoint_refine_temperature=0.2,
    )
    tracker = object.__new__(DinoKeypointTracker)
    tracker.runtime_config = config
    tracker.keypoint_refine_temperature = config.keypoint_refine_temperature

    refine_region = RefineRegion(
        logits=np.array([[0.0, 1.0, 0.94, 0.2, 0.98]], dtype=np.float32),
        r_min=0,
        c_min=0,
        valid_mask=None,
    )
    refined_x, refined_y, confidence = tracker._refine_keypoint_xy_from_connected_peak(
        center_rc=(0, 1),
        fallback_xy=(1.0, 0.0),
        refine_region=refine_region,
        patch_centers_x=np.arange(5, dtype=np.float32),
        patch_centers_y=np.array([0.0], dtype=np.float32),
    )

    assert 1.0 < refined_x < 2.0
    assert refined_y == pytest.approx(0.0)
    assert confidence > 0.0


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
        dinov3_positional_debias=False,
        dinov3_backward_consistency=False,
        keypoint_cluster_refine=False,
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
        [0.9, math.sqrt(1.0 - 0.9**2)], dtype=torch.float32
    )
    update_features[:, 0, 7] = torch.tensor(
        [0.95, math.sqrt(1.0 - 0.95**2)], dtype=torch.float32
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
        dinov3_positional_debias=False,
        dinov3_backward_consistency=False,
        keypoint_cluster_refine=False,
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
