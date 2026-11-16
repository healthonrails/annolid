"""Regression and integration tests for two-mice simulation keypoint tracking improvements."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from annolid.features.dinov3_extractor import Dinov3Config, Dinov3FeatureExtractor
from annolid.features.google_vision_extractor import GoogleVisionFeatureExtractor
from annolid.tracking.annotation_adapter import AnnotationAdapter
from annolid.tracking.configuration import CutieDinoTrackerConfig
from annolid.tracking.cutie_mask_manager import CutieMaskManager
from annolid.tracking.dino_keypoint_tracker import DinoKeypointTracker, RefineRegion
from annolid.tracking.domain import InstanceRegistry, KeypointState


BODY_PART_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "two_mice_body_part_tiny"


class _QueueExtractor:
    """Deterministic stub extractor returning queued feature grids."""

    patch_size = 8

    def __init__(self, cfg: Dinov3Config) -> None:
        self.cfg = cfg
        self._queue: List[torch.Tensor] = []

    def set_queue(self, tensors: List[torch.Tensor]) -> None:
        self._queue = list(tensors)

    def extract(
        self, image: Image.Image, return_layer: str = "last", normalize: bool = True
    ) -> torch.Tensor:
        _ = (return_layer, normalize)
        if not self._queue:
            w, h = image.size
            gh = max(1, h // self.patch_size)
            gw = max(1, w // self.patch_size)
            return F.normalize(torch.ones((4, gh, gw), dtype=torch.float32), dim=0)
        return self._queue.pop(0)

    def _compute_resized_hw(self, width: int, height: int) -> Tuple[int, int]:
        gh = max(1, math.ceil(height / self.patch_size)) * self.patch_size
        gw = max(1, math.ceil(width / self.patch_size)) * self.patch_size
        return gh, gw


class _LocalImageDescriptorExtractor:
    """Fast offline image descriptor extractor encoding multi-scale RGB + Sobel structure."""

    patch_size = 8

    def __init__(self, cfg: Dinov3Config) -> None:
        self.cfg = cfg

    def _compute_resized_hw(self, width: int, height: int) -> Tuple[int, int]:
        new_h = max(
            self.patch_size, math.ceil(height / self.patch_size) * self.patch_size
        )
        new_w = max(
            self.patch_size, math.ceil(width / self.patch_size) * self.patch_size
        )
        return new_h, new_w

    def extract(
        self, image: Image.Image, return_layer: str = "last", normalize: bool = True
    ) -> torch.Tensor:
        _ = return_layer
        new_h, new_w = self._compute_resized_hw(*image.size)
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        gmag = np.hypot(gx, gy)
        blur = cv2.GaussianBlur(resized, (9, 9), 2.0)
        diff = resized - blur

        channels = np.concatenate(
            [
                resized,
                diff * 2.5,
                gx[..., None],
                gy[..., None],
                gmag[..., None],
            ],
            axis=-1,
        )  # [H, W, 9]
        tensor = torch.from_numpy(channels).permute(2, 0, 1).unsqueeze(0)
        pooled = F.avg_pool2d(
            tensor, kernel_size=self.patch_size, stride=self.patch_size
        )
        wide = F.avg_pool2d(pooled, kernel_size=3, stride=1, padding=1)
        feat = torch.cat([pooled, wide], dim=1).squeeze(0)
        if normalize:
            feat = feat / (feat.norm(dim=0, keepdim=True) + 1e-8)
        return feat


def test_compute_resized_hw_caps_elongated_roi_while_preserving_standard_frames() -> (
    None
):
    ext = object.__new__(Dinov3FeatureExtractor)
    ext.cfg = Dinov3Config(short_side=768, max_long_side=1408, patch_size=16)
    ext.patch_size = 16

    # Standard 4:3 (640x480) and 16:9 (1920x1080) preserve full short_side=768.
    assert ext._compute_resized_hw(640, 480) == (768, 1024)
    assert ext._compute_resized_hw(1920, 1080) == (768, 1376)

    # Elongated wide two-mouse band ROI (640x180) is bounded to max_long_side=1408
    # instead of exploding to 2736x768 (>8,200 tokens).
    new_h, new_w = ext._compute_resized_hw(640, 180)
    assert new_w <= 1408
    assert new_h % 16 == 0 and new_w % 16 == 0
    assert (new_h // 16) * (new_w // 16) < 2500

    g_ext = object.__new__(GoogleVisionFeatureExtractor)
    g_ext.cfg = Dinov3Config(short_side=768, max_long_side=1408, patch_size=16)
    g_ext.patch_size = 16
    g_ext._snap_multiple = 48
    gh, gw = g_ext._compute_resized_hw(640, 160)
    assert gw <= 1440
    assert gh % 48 == 0 and gw % 48 == 0


def test_dinov3_extract_skips_all_hidden_states_on_default_last_and_recovers_from_oom() -> (
    None
):
    ext = object.__new__(Dinov3FeatureExtractor)
    ext.cfg = Dinov3Config(
        short_side=32, max_long_side=64, patch_size=16, use_amp=False
    )
    ext.patch_size = 16
    ext._num_special_tokens = 1
    ext.num_hidden_layers = 12
    ext._layers = tuple(range(12))
    ext._mean = (0.5, 0.5, 0.5)
    ext._std = (0.5, 0.5, 0.5)
    ext.device = torch.device("mps")

    calls: List[Tuple[str, bool]] = []

    class _FakeOutput:
        def __init__(self, tokens: torch.Tensor) -> None:
            self.last_hidden_state = tokens

    class _FakeModel(torch.nn.Module):
        def to(self, device, *args, **kwargs):  # type: ignore[override]
            ext._moved_to = str(device)
            return self

        def forward(
            self, pixel_values: torch.Tensor, output_hidden_states: bool = False
        ):
            calls.append((str(ext.device), bool(output_hidden_states)))
            if str(ext.device) == "mps":
                raise RuntimeError("MPS backend out of memory on private pool")
            h_g = pixel_values.shape[-2] // 16
            w_g = pixel_values.shape[-1] // 16
            tokens = torch.ones((1, 1 + h_g * w_g, 8), dtype=torch.float32)
            return _FakeOutput(tokens)

    ext.model = _FakeModel()
    ext._preprocess = lambda pil: torch.zeros((1, 3, 32, 32), dtype=torch.float32)

    out = ext.extract(Image.new("RGB", (32, 32)), return_layer="last")
    assert out.shape == (8, 2, 2)
    assert ext.device == torch.device("cpu")
    assert calls == [("mps", False), ("cpu", False)]


def test_cutie_mask_manager_migrates_to_cpu_on_accelerator_oom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = AnnotationAdapter(image_height=6, image_width=8)
    config = CutieDinoTrackerConfig(use_cutie_tracking=True)
    manager = CutieMaskManager(tmp_path / "clip.mp4", adapter, config)

    class _FakeProc:
        def __init__(
            self, _video_path: str, *, use_cpu_only: bool = False, **_kwargs
        ) -> None:
            self.device = "cpu" if use_cpu_only else "mps"
            self.cutie = object()
            self.cfg = object()

        def _prepare_frame_tensor(self, _frame: np.ndarray) -> torch.Tensor:
            return torch.zeros((3, 6, 8), dtype=torch.float32)

    class _FakeCore:
        def __init__(self, _cutie: object, cfg: object) -> None:
            self.cfg = cfg

        def step(
            self, _frame: torch.Tensor, mask: torch.Tensor | None = None, **_kwargs
        ) -> torch.Tensor:
            if manager._device == "mps" and mask is None:
                raise RuntimeError("MPS backend out of memory")
            out = torch.zeros((6, 8), dtype=torch.int64)
            out[1:5, 2:6] = 1
            return out

    monkeypatch.setattr(
        "annolid.tracking.cutie_mask_manager.CutieCoreVideoProcessor", _FakeProc
    )
    monkeypatch.setattr("annolid.tracking.cutie_mask_manager.InferenceCore", _FakeCore)

    registry = InstanceRegistry()
    seed_mask = np.zeros((6, 8), dtype=bool)
    seed_mask[1:5, 1:5] = True
    registry.ensure_instance("mouse_1").set_mask(bitmap=seed_mask, polygon=None)

    frame0 = np.zeros((6, 8, 3), dtype=np.uint8)
    frame1 = np.ones((6, 8, 3), dtype=np.uint8)
    manager.prime(0, frame0, registry)
    assert manager._device == "mps"

    results = manager.update_masks(1, frame1, registry)
    assert manager._device == "cpu"
    assert manager.enabled is True
    assert "mouse_1" in results
    assert np.any(results["mouse_1"].mask_bitmap)


def test_mask_to_patch_preserves_thin_tail_filament() -> None:
    tracker = object.__new__(DinoKeypointTracker)
    mask = np.zeros((120, 160), dtype=bool)
    # 1-pixel-wide horizontal tail filament at y=43 across x=10..90
    mask[43, 10:90] = True

    patch_mask = tracker._mask_to_patch(mask, (15, 20))
    # Every patch column touched by x=10..89 (cols 1..11) must remain active.
    assert int(np.count_nonzero(patch_mask)) >= 10


def test_occluded_seed_keypoint_does_not_steal_visible_sibling_or_pollute_codebook(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        _QueueExtractor,
    )
    cfg = CutieDinoTrackerConfig.from_preset(
        "rodent_30fps_occlusions",
        use_cutie_tracking=False,
        tracking_enable=True,
        tracking_matcher_algorithm="hungarian",
        dinov3_positional_debias=False,
        pixel_refine_enabled=False,
        pixel_region_refine_enabled=False,
        keypoint_refine_radius=0,
        keypoint_cluster_refine=False,
        region_descriptor_radius=0,
        region_subpatch_sampling=False,
        context_weight=0.0,
        support_probe_weight=0.0,
        appearance_bundle_weight=0.0,
    )
    tracker = DinoKeypointTracker(
        model_name="dummy",
        runtime_config=cfg,
        search_radius=4,
        reference_support_radius=0,
    )

    # 8x8 patch grid (64x64 image):
    # (2, 2) = ear feature [1, 0, 0, 0]
    # (2, 5) = neck feature [0, 1, 0, 0]
    # (5, 2) = visible contralateral right_forepaw feature [0, 0, 1, 0]
    feat0 = torch.zeros((4, 8, 8), dtype=torch.float32)
    feat0[0, 2, 2] = 1.0  # left_ear at (20, 20)
    feat0[1, 2, 5] = 1.0  # neck at (44, 20)
    feat0[2, 5, 2] = 1.0  # right_forepaw (visible) at (20, 44)
    feat0[3, :, :] = 0.05
    feat0 = F.normalize(feat0, dim=0)

    # Frame 1: left_forepaw remains tucked under left_ear (no distinct paw peak near (2, 2))
    feat1 = feat0.clone()

    # Frame 2: left_forepaw emerges clearly at patch (1, 1) -> pixel (12, 12) with paw feature!
    feat2 = feat0.clone()
    feat2[:, 1, 1] = F.normalize(torch.tensor([0.0, 0.0, 1.0, 0.05]), dim=0)

    tracker.extractor.set_queue([feat0, feat1, feat2])  # type: ignore[attr-defined]

    registry = InstanceRegistry()
    registry.register_keypoint(
        KeypointState(
            key="mouse_1::left_ear",
            instance_label="mouse_1",
            label="left_ear",
            x=20.0,
            y=20.0,
            visible=True,
        )
    )
    registry.register_keypoint(
        KeypointState(
            key="mouse_1::neck",
            instance_label="mouse_1",
            label="neck",
            x=44.0,
            y=20.0,
            visible=True,
        )
    )
    registry.register_keypoint(
        KeypointState(
            key="mouse_1::right_forepaw",
            instance_label="mouse_1",
            label="right_forepaw",
            x=20.0,
            y=44.0,
            visible=True,
        )
    )
    # Seeded occluded directly underneath left_ear!
    registry.register_keypoint(
        KeypointState(
            key="mouse_1::left_forepaw",
            instance_label="mouse_1",
            label="left_forepaw",
            x=20.0,
            y=20.0,
            visible=False,
        )
    )

    mask = np.ones((64, 64), dtype=bool)
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    tracker.start(img, registry, {"mouse_1": mask})

    # Occluded seed must NOT pollute manual codebook and must inherit homologous paw descriptor.
    assert "mouse_1::left_forepaw" not in tracker._manual_anchor_codebooks
    assert tracker.tracks["mouse_1::left_forepaw"].seeded_occluded is True
    assert tracker.tracks["mouse_1::left_forepaw"].has_clean_seed_descriptor is True

    # Frame 1: left_ear stays visible on its true patch (2, 2); occluded left_forepaw stays visible=False.
    res1 = {r["id"]: r for r in tracker.update(img, {"mouse_1": mask})}
    assert res1["mouse_1::left_ear"]["visible"] is True
    assert (
        math.hypot(
            res1["mouse_1::left_ear"]["x"] - 20.0, res1["mouse_1::left_ear"]["y"] - 20.0
        )
        < 4.0
    )
    assert res1["mouse_1::left_forepaw"]["visible"] is False

    # Frame 2: left_forepaw emerges at (12, 12) away from left_ear and re-acquires visible=True!
    res2 = {r["id"]: r for r in tracker.update(img, {"mouse_1": mask})}
    assert res2["mouse_1::left_forepaw"]["visible"] is True
    assert (
        math.hypot(
            res2["mouse_1::left_forepaw"]["x"] - 12.0,
            res2["mouse_1::left_forepaw"]["y"] - 12.0,
        )
        < 4.0
    )
    assert tracker.tracks["mouse_1::left_forepaw"].seeded_occluded is False


def test_cross_instance_mask_exclusion_and_partial_mask_collapse_resilience(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        _QueueExtractor,
    )
    cfg = CutieDinoTrackerConfig(
        mask_enforce_position=True,
        mask_enforce_snap_radius=6,
        mask_enforce_reject_outside=True,
        dinov3_positional_debias=False,
        pixel_refine_enabled=False,
        keypoint_refine_radius=0,
        structural_consistency_weight=0.0,
        appearance_bundle_weight=0.0,
        support_probe_weight=0.0,
        motion_prior_penalty_weight=0.0,
    )
    tracker = DinoKeypointTracker(
        model_name="dummy", runtime_config=cfg, search_radius=6
    )

    feat0 = torch.zeros((4, 8, 8), dtype=torch.float32)
    feat0[0, 3, 2] = 1.0  # mouse_1 nose at (20, 28)
    feat0[1, 3, 5] = 1.0  # mouse_2 spine at (44, 28)
    feat0[3, :, :] = 0.05
    feat0 = F.normalize(feat0, dim=0)

    # Frame 1: a distractor peak for mouse_1 appears inside mouse_2's mask at (3, 5),
    # while mouse_1's valid peak is at (3, 2); meanwhile mouse_2's mask collapses to a tiny
    # 10x10 head patch away from (3, 5) while its strong DINO peak stays at (3, 5).
    feat1 = torch.zeros((4, 8, 8), dtype=torch.float32)
    feat1[0, 3, 5] = 1.0  # stronger distractor inside mouse_2!
    feat1[0, 3, 2] = 0.85  # valid peak inside mouse_1
    feat1[1, 3, 5] = 0.98  # mouse_2 spine peak outside truncated mask
    feat1[3, :, :] = 0.05
    feat1 = F.normalize(feat1, dim=0)

    tracker.extractor.set_queue([feat0, feat1])  # type: ignore[attr-defined]

    reg = InstanceRegistry()
    reg.register_keypoint(
        KeypointState(
            key="mouse_1::nose", instance_label="mouse_1", label="nose", x=20.0, y=28.0
        )
    )
    reg.register_keypoint(
        KeypointState(
            key="mouse_2::spine_1",
            instance_label="mouse_2",
            label="spine_1",
            x=44.0,
            y=28.0,
        )
    )
    m1_seed = np.zeros((64, 64), dtype=bool)
    m1_seed[16:48, 8:30] = True
    m2_seed = np.zeros((64, 64), dtype=bool)
    m2_seed[16:48, 32:60] = True

    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    tracker.start(img, reg, {"mouse_1": m1_seed, "mouse_2": m2_seed})

    # On frame 1, mouse_2's mask shrinks below 52% of seed area (partial collapse),
    # while still covering (44, 28) for cross-instance test via a second region:
    m1_f1 = m1_seed.copy()
    m2_f1 = np.zeros((64, 64), dtype=bool)
    m2_f1[24:34, 40:50] = True  # area=100 vs seed area=896 (cur_span=10 < 0.72*29.9)
    # Add a second mouse_2 point at (52, 28) outside the collapsed 40..50 box!
    tracker.tracks["mouse_2::spine_1"].last_position = (52.0, 28.0)
    feat1[1, 3, 5] = 0.0
    feat1[1, 3, 6] = 0.98  # patch (3, 6) -> x=52, y=28 (outside collapsed m2_f1!)
    feat1 = F.normalize(feat1, dim=0)
    tracker.extractor.set_queue([feat1])  # type: ignore[attr-defined]

    res = {
        r["id"]: r for r in tracker.update(img, {"mouse_1": m1_f1, "mouse_2": m2_f1})
    }
    # mouse_1::nose did NOT bleed into mouse_2's exclusive mask at (3, 5):
    assert (
        math.hypot(res["mouse_1::nose"]["x"] - 20.0, res["mouse_1::nose"]["y"] - 28.0)
        < 4.0
    )
    # mouse_2::spine_1 survived partial mask collapse at (52, 28):
    assert res["mouse_2::spine_1"]["visible"] is True
    assert (
        math.hypot(
            res["mouse_2::spine_1"]["x"] - 52.0, res["mouse_2::spine_1"]["y"] - 28.0
        )
        < 4.0
    )


def test_connected_peak_refine_confidence_prefers_sharp_peak_over_flat_plateau() -> (
    None
):
    cfg = CutieDinoTrackerConfig(
        keypoint_cluster_refine=True,
        keypoint_cluster_refine_radius=2,
        keypoint_cluster_refine_drop=0.15,
        keypoint_refine_sigma=1.0,
        keypoint_refine_temperature=0.25,
    )
    tracker = object.__new__(DinoKeypointTracker)
    tracker.runtime_config = cfg
    tracker.keypoint_refine_sigma = cfg.keypoint_refine_sigma
    tracker.keypoint_refine_temperature = cfg.keypoint_refine_temperature

    xs = np.arange(5, dtype=np.float32) * 10.0
    ys = np.arange(5, dtype=np.float32) * 10.0

    sharp_logits = np.full((5, 5), 0.1, dtype=np.float32)
    sharp_logits[2, 2] = 1.0
    sharp_logits[2, 3] = 0.88
    _, _, sharp_conf = tracker._refine_keypoint_xy_from_connected_peak(
        center_rc=(2, 2),
        fallback_xy=(20.0, 20.0),
        refine_region=RefineRegion(logits=sharp_logits, r_min=0, c_min=0),
        patch_centers_x=xs,
        patch_centers_y=ys,
    )

    flat_logits = np.full((5, 5), 0.98, dtype=np.float32)
    _, _, flat_conf = tracker._refine_keypoint_xy_from_connected_peak(
        center_rc=(2, 2),
        fallback_xy=(20.0, 20.0),
        refine_region=RefineRegion(logits=flat_logits, r_min=0, c_min=0),
        patch_centers_x=xs,
        patch_centers_y=ys,
    )
    assert sharp_conf > flat_conf


def test_two_mice_simulation_fixture_tracks_keypoints_without_bleed_or_swap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        _LocalImageDescriptorExtractor,
    )
    manifest = json.loads(
        (BODY_PART_FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    top_frames = [f for f in manifest["frames"] if f["scenario"] == "social_huddle"]
    assert len(top_frames) >= 2

    cfg = CutieDinoTrackerConfig.from_preset(
        "rodent_30fps_occlusions",
        use_cutie_tracking=False,
        tracking_enable=True,
        tracking_matcher_algorithm="hungarian",
        dinov3_positional_debias=False,
    )
    adapter = AnnotationAdapter(
        image_height=int(manifest["height"]),
        image_width=int(manifest["width"]),
    )
    tracker = DinoKeypointTracker(model_name="local_test", runtime_config=cfg)

    vis_errors: List[float] = []
    for idx, rec in enumerate(top_frames):
        bgr = cv2.imread(
            str(BODY_PART_FIXTURE_DIR / rec["image_file"]), cv2.IMREAD_COLOR
        )
        assert bgr is not None
        pil_img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        id_bgr = cv2.imread(
            str(BODY_PART_FIXTURE_DIR / rec["id_mask_file"]), cv2.IMREAD_UNCHANGED
        )
        assert id_bgr is not None
        track_map = id_bgr[:, :, 2]
        masks: Dict[str, np.ndarray] = {
            "mouse_1": track_map == 1,
            "mouse_2": track_map == 2,
        }
        lm_path = BODY_PART_FIXTURE_DIR / rec["labelme_file"]
        gt_reg = adapter.read_annotation(lm_path)
        if idx == 0:
            tracker.start(pil_img, gt_reg, masks)
            continue

        preds = {r["id"]: r for r in tracker.update(pil_img, masks)}
        for inst in gt_reg:
            other_tid = 2 if inst.label == "mouse_1" else 1
            for kp in inst.keypoints.values():
                pr = preds.get(kp.key)
                if pr is None:
                    continue
                if pr["visible"]:
                    ix, iy = int(round(float(pr["x"]))), int(round(float(pr["y"])))
                    assert track_map[iy, ix] != other_tid
                if kp.visible:
                    vis_errors.append(
                        math.hypot(float(pr["x"]) - kp.x, float(pr["y"]) - kp.y)
                    )

    assert vis_errors
    assert float(np.median(vis_errors)) < 12.0
