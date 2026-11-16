"""Focused tests for Gaussian-weighted local-region keypoint tracking and Google vision backbones."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from annolid.features import (
    Dinov3Config,
    Dinov3FeatureExtractor,
    GoogleVisionFeatureExtractor,
)
from annolid.features import dino_models
from annolid.features.google_vision_extractor import apply_matryoshka_truncation
from annolid.tracking.configuration import CutieDinoTrackerConfig
from annolid.tracking.dino_keypoint_tracker import DinoKeypointTracker
from annolid.tracking.dino_keypoint_types import RefineRegion, SupportProbe
from annolid.tracking.domain import InstanceRegistry, KeypointState
from annolid.tracking.gaussian_region import (
    extract_gaussian_pixel_template,
    gaussian_smooth_feature_grid,
    match_gaussian_pixel_template,
    match_gaussian_pixel_template_oriented,
    pixel_to_continuous_patch,
    refine_patch_region_gaussian,
    sample_subpatch_gaussian_descriptor,
)


class _QueuedExtractor:
    patch_size = 16

    def __init__(self, cfg: Dinov3Config) -> None:
        self.cfg = cfg
        self._queue: List[torch.Tensor] = []

    def set_queue(self, tensors: List[torch.Tensor]) -> None:
        self._queue = list(tensors)

    def extract(self, _image, return_layer: str = "all", normalize: bool = True):
        assert self._queue, "No queued features left"
        return self._queue.pop(0)

    def _compute_resized_hw(self, width: int, height: int) -> Tuple[int, int]:
        return height, width


def test_mask_aware_gaussian_smoothing_prevents_background_bleed() -> None:
    # Foreground (left 3 cols) has channel 0; background (right 3 cols) has channel 1.
    feats = torch.zeros((2, 4, 6), dtype=torch.float32)
    feats[0, :, :3] = 1.0
    feats[1, :, 3:] = 10.0
    patch_mask = np.zeros((4, 6), dtype=bool)
    patch_mask[:, :3] = True

    masked_smooth = gaussian_smooth_feature_grid(
        feats,
        radius=1,
        sigma=1.0,
        patch_mask=patch_mask,
        mask_aware=True,
        blend=1.0,
        normalize=True,
    )
    unmasked_smooth = gaussian_smooth_feature_grid(
        feats,
        radius=1,
        sigma=1.0,
        patch_mask=patch_mask,
        mask_aware=False,
        blend=1.0,
        normalize=True,
    )

    # Boundary foreground column c=2 must contain zero background channel 1 when mask_aware=True.
    assert float(masked_smooth[1, 2, 2].item()) == pytest.approx(0.0, abs=1e-6)
    assert float(masked_smooth[0, 2, 2].item()) == pytest.approx(1.0, abs=1e-6)
    # Without mask gating, background bleeds heavily into the foreground boundary cell.
    assert float(unmasked_smooth[1, 2, 2].item()) > 0.3
    # Every cell remains unit L2 norm.
    norms = torch.linalg.norm(masked_smooth, dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_subpatch_continuous_gaussian_sampling_interpolates_across_patch_boundary() -> (
    None
):
    feats = torch.zeros((2, 1, 2), dtype=torch.float32)
    feats[:, 0, 0] = torch.tensor([1.0, 0.0])
    feats[:, 0, 1] = torch.tensor([0.0, 1.0])

    # Midpoint between patch 0 center (x=8) and patch 1 center (x=24) is x=16 -> cont_c = 0.5
    cont_mid = pixel_to_continuous_patch(
        16.0,
        8.0,
        roi_offset=(0.0, 0.0),
        roi_size=(32, 16),
        scale_x=1.0,
        scale_y=1.0,
        patch_size=16.0,
        grid_hw=(1, 2),
    )
    assert cont_mid == pytest.approx((0.0, 0.5), abs=1e-5)
    desc_mid = sample_subpatch_gaussian_descriptor(
        feats,
        cont_rc=cont_mid,
        radius=1,
        sigma=0.8,
    )
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    assert float(desc_mid[0].item()) == pytest.approx(inv_sqrt2, abs=1e-5)
    assert float(desc_mid[1].item()) == pytest.approx(inv_sqrt2, abs=1e-5)

    # Mask-aware sampling excludes background patch (0, 1) when keypoint straddles animal contour.
    fg_mask = np.array([[True, False]], dtype=bool)
    desc_masked = sample_subpatch_gaussian_descriptor(
        feats,
        cont_rc=cont_mid,
        radius=1,
        sigma=0.8,
        patch_mask=fg_mask,
        mask_aware=True,
    )
    assert float(desc_masked[0].item()) == pytest.approx(1.0, abs=1e-5)
    assert float(desc_masked[1].item()) == pytest.approx(0.0, abs=1e-5)


def _render_textured_landmark(
    size_hw: Tuple[int, int],
    center_xy: Tuple[float, float],
    *,
    gain: float = 1.0,
    bias: float = 0.0,
    angle_rad: float = 0.0,
) -> np.ndarray:
    h, w = size_hw
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    raw_dx = xx - float(center_xy[0])
    raw_dy = yy - float(center_xy[1])
    cos_a = float(math.cos(angle_rad))
    sin_a = float(math.sin(angle_rad))
    dx = raw_dx * cos_a + raw_dy * sin_a
    dy = -raw_dx * sin_a + raw_dy * cos_a
    # Asymmetric multi-frequency Gabor-like landmark within [0, 255] dynamic range
    # so affine lighting changes (gain * I + bias) do not saturate uint8 to 255.
    envelope = np.exp(-0.5 * (dx * dx + dy * dy) / (3.2**2))
    carrier = (
        65.0 * envelope * np.cos(0.65 * dx + 0.35 * dy)
        + 38.0 * np.exp(-0.5 * ((dx - 1.2) ** 2 + (dy + 0.8) ** 2) / (1.8**2))
        + 72.0
    )
    illuminated = np.clip(carrier * float(gain) + float(bias), 0.0, 255.0)
    return illuminated.astype(np.uint8)


def test_gaussian_pixel_template_zncc_recovers_subpixel_shift_under_lighting_change() -> (
    None
):
    blank = np.zeros((32, 32), dtype=np.uint8)
    assert extract_gaussian_pixel_template(blank, (16.0, 16.0)) is None

    seed_gray = _render_textured_landmark((64, 64), (28.0, 30.0))
    tpl = extract_gaussian_pixel_template(
        seed_gray,
        (28.0, 30.0),
        window_px=13,
        sigma_px=3.2,
    )
    assert tpl is not None

    true_xy = (30.35, 28.4)
    shifted_gray = _render_textured_landmark(
        (64, 64),
        true_xy,
        gain=1.22,
        bias=24.0,
    )
    match = match_gaussian_pixel_template(
        shifted_gray,
        tpl,
        center_xy=(28.0, 30.0),
        search_radius_px=5,
    )
    assert match is not None
    est_x, est_y, zncc = match
    assert zncc > 0.95
    assert abs(est_x - true_xy[0]) < 0.15
    assert abs(est_y - true_xy[1]) < 0.15

    # Polarity-inverted landmark at the same location has negative correlation -> rejected.
    inverted_gray = (255 - seed_gray).astype(np.uint8)
    inv_match = match_gaussian_pixel_template(
        inverted_gray,
        tpl,
        center_xy=(28.0, 30.0),
        search_radius_px=1,
    )
    assert inv_match is None


def test_pixel_region_template_prevents_multi_frame_drift_in_tracker(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        _QueuedExtractor,
    )
    # Coarse 16px patch grid (4x4 patches over 64x64 frame) with pixel flow disabled
    # so without pixel_region_refine the keypoint stays quantized to patch center (24, 24).
    cfg_with_region = CutieDinoTrackerConfig(
        region_descriptor_radius=1,
        region_descriptor_sigma=0.85,
        region_subpatch_sampling=True,
        pixel_refine_enabled=False,
        pixel_region_refine_enabled=True,
        pixel_region_window=13,
        pixel_region_sigma=3.2,
        pixel_region_search_radius_px=6,
        pixel_region_weight=1.0,
        pixel_region_min_zncc=0.5,
        keypoint_cluster_refine=False,
        keypoint_refine_radius=0,
        dinov3_positional_debias=False,
        dinov3_backward_consistency=False,
    )
    tracker = DinoKeypointTracker(
        model_name="dummy",
        runtime_config=cfg_with_region,
        search_radius=2,
        min_similarity=0.0,
    )

    num_frames = 8
    patch_feats: List[torch.Tensor] = []
    for _ in range(num_frames + 1):
        f = torch.zeros((4, 4, 4), dtype=torch.float32)
        f[:, 1, 1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        patch_feats.append(f)
    tracker.extractor.set_queue(patch_feats)

    seed_xy = (22.0, 22.0)
    seed_gray = _render_textured_landmark((64, 64), seed_xy)
    seed_img = Image.fromarray(np.stack([seed_gray] * 3, axis=-1))
    reg = InstanceRegistry()
    reg.register_keypoint(
        KeypointState(
            key="mousenose",
            instance_label="mouse",
            label="nose",
            x=seed_xy[0],
            y=seed_xy[1],
        )
    )
    tracker.start(seed_img, reg, {})

    final_true_xy = seed_xy
    last_out = None
    for step in range(1, num_frames + 1):
        # Landmark moves smoothly within the same 16px patch (from 22.0 -> 26.8).
        final_true_xy = (22.0 + 0.6 * step, 22.0 - 0.35 * step)
        gray = _render_textured_landmark((64, 64), final_true_xy)
        img = Image.fromarray(np.stack([gray] * 3, axis=-1))
        out = tracker.update(img, {})
        last_out = out[0]

    assert last_out is not None
    assert last_out["visible"] is True
    assert abs(float(last_out["x"]) - final_true_xy[0]) < 0.45
    assert abs(float(last_out["y"]) - final_true_xy[1]) < 0.45


def test_vectorized_batch_scorers_match_scalar_reference(monkeypatch) -> None:
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        _QueuedExtractor,
    )
    cfg = CutieDinoTrackerConfig(
        support_probe_weight=0.45,
        support_probe_mask_bonus=0.08,
        dinov3_backward_consistency=True,
        dinov3_backward_consistency_weight=0.25,
        dinov3_backward_consistency_tolerance=1,
        keypoint_refine_radius=1,
        keypoint_refine_sigma=1.15,
        keypoint_refine_temperature=0.25,
    )
    tracker = DinoKeypointTracker(
        model_name="dummy",
        runtime_config=cfg,
        search_radius=3,
    )

    torch.manual_seed(7)
    prev_feats = torch.randn((6, 5, 6), dtype=torch.float32)
    prev_feats = prev_feats / torch.linalg.norm(prev_feats, dim=0, keepdim=True)
    curr_feats = torch.randn((6, 5, 6), dtype=torch.float32)
    curr_feats = curr_feats / torch.linalg.norm(curr_feats, dim=0, keepdim=True)

    cand_rows = np.array([0, 1, 2, 4], dtype=np.int64)
    cand_cols = np.array([0, 2, 3, 5], dtype=np.int64)
    cand_descs = curr_feats[:, cand_rows, cand_cols]

    prev_flat = prev_feats.reshape(prev_feats.shape[0], -1)
    batch_bw = tracker._backward_consistency_scores_batch(
        cand_descs,
        previous_flat=prev_flat,
        previous_grid_hw=prev_feats.shape[1:],
        previous_rc=(2, 2),
    )
    scalar_bw = np.array(
        [
            tracker._backward_consistency_score(
                cand_descs[:, idx],
                previous_feats=prev_feats,
                previous_rc=(2, 2),
            )
            for idx in range(len(cand_rows))
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(batch_bw, scalar_bw, atol=1e-6)

    probes = [
        SupportProbe(
            offset_rc=(-1, 0),
            descriptor=curr_feats[:, 1, 2].clone(),
            weight=0.9,
        ),
        SupportProbe(
            offset_rc=(1, 2),
            descriptor=curr_feats[:, 3, 4].clone(),
            weight=0.5,
        ),
    ]
    dummy_track = SimpleNamespace(support_probes=probes)
    patch_mask = np.zeros((5, 6), dtype=bool)
    patch_mask[1:4, 1:5] = True

    batch_sup = tracker._support_scores_batch(
        dummy_track,  # type: ignore[arg-type]
        cand_rows,
        cand_cols,
        curr_feats,
        patch_mask,
    )
    scalar_sup = np.array(
        [
            tracker._support_score(
                dummy_track,  # type: ignore[arg-type]
                (int(cand_rows[idx]), int(cand_cols[idx])),
                curr_feats,
                patch_mask,
                normalized_feats=curr_feats,
            )
            for idx in range(len(cand_rows))
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(batch_sup, scalar_sup, atol=1e-6)

    # Also verify vectorized patch-region Gaussian refine parity.
    logits = np.array(
        [[0.2, 0.5, 0.4], [0.6, 0.95, 0.85], [0.1, 0.4, 0.7]], dtype=np.float32
    )
    valid_mask = np.array(
        [[True, True, False], [True, True, True], [False, True, True]], dtype=bool
    )
    region = RefineRegion(logits=logits, r_min=1, c_min=2, valid_mask=valid_mask)
    px = np.arange(8, dtype=np.float32) * 16.0 + 8.0
    py = np.arange(8, dtype=np.float32) * 16.0 + 8.0
    rx, ry, rconf = tracker._refine_keypoint_xy_from_region(
        center_rc=(2, 3),
        fallback_xy=(56.0, 40.0),
        refine_region=region,
        patch_centers_x=px,
        patch_centers_y=py,
    )
    rx2, ry2, rconf2 = refine_patch_region_gaussian(
        logits,
        center_rc=(2, 3),
        r_min=1,
        c_min=2,
        valid_mask=valid_mask,
        radius=1,
        sigma=1.15,
        temperature=0.25,
        patch_centers_x=px,
        patch_centers_y=py,
        fallback_xy=(56.0, 40.0),
    )
    assert (rx, ry, rconf) == pytest.approx((rx2, ry2, rconf2), abs=1e-6)


def test_lazy_farneback_and_batched_lk_avoid_redundant_optical_flow(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        _QueuedExtractor,
    )
    cfg = CutieDinoTrackerConfig(
        pixel_refine_enabled=True,
        pixel_refine_max_error=12.0,
        dinov3_positional_debias=False,
        dinov3_backward_consistency=False,
    )
    tracker = DinoKeypointTracker(model_name="dummy", runtime_config=cfg)

    feats = [torch.ones((3, 4, 4), dtype=torch.float32) for _ in range(4)]
    tracker.extractor.set_queue(feats)

    farneback_calls: List[int] = []
    lk_batch_sizes: List[int] = []
    real_lk = cv2.calcOpticalFlowPyrLK
    real_farneback = cv2.calcOpticalFlowFarneback

    def spy_farneback(*args, **kwargs):
        farneback_calls.append(1)
        return real_farneback(*args, **kwargs)

    def spy_lk(prev_img, next_img, prev_pts, *args, **kwargs):
        lk_batch_sizes.append(int(prev_pts.shape[0]))
        return real_lk(prev_img, next_img, prev_pts, *args, **kwargs)

    monkeypatch.setattr(cv2, "calcOpticalFlowFarneback", spy_farneback)
    monkeypatch.setattr(cv2, "calcOpticalFlowPyrLK", spy_lk)

    g0 = _render_textured_landmark((64, 64), (20.0, 20.0))
    g1 = _render_textured_landmark((64, 64), (22.0, 20.0))
    img0 = Image.fromarray(np.stack([g0] * 3, axis=-1))
    img1 = Image.fromarray(np.stack([g1] * 3, axis=-1))

    reg = InstanceRegistry()
    reg.register_keypoint(
        KeypointState(key="m_nose", instance_label="m", label="nose", x=20.0, y=20.0)
    )
    reg.register_keypoint(
        KeypointState(key="m_ear", instance_label="m", label="ear", x=24.0, y=22.0)
    )
    tracker.start(img0, reg, {})
    tracker.update(img1, {})

    # Both tracks succeeded in one batched LK call -> zero Farneback invocations!
    assert lk_batch_sizes == [2]
    assert len(farneback_calls) == 0

    # Transition onto uniform blank frames so prev_gray has zero spatial gradient ->
    # both tracks fail LK simultaneously, and lazy Farneback runs once (not per track).
    blank_img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    tracker.update(blank_img, {})
    farneback_calls.clear()
    tracker.update(blank_img, {})
    assert len(farneback_calls) == 1


def test_google_model_catalog_and_factory_dispatch() -> None:
    assert (
        dino_models.resolve_dino_model_id("embeddinggemma-2")
        == "google/embeddinggemma-2"
    )
    assert (
        dino_models.resolve_dino_model_id("siglip2")
        == "google/siglip2-base-patch16-224"
    )
    assert (
        dino_models.resolve_dino_model_id("mobilenetv4")
        == "timm/mobilenetv4_conv_small.e2400_r224_in1k"
    )
    assert (
        dino_models.resolve_dino_model_id("mobilenetv3_small")
        == "torchvision/mobilenet_v3_small"
    )
    assert dino_models.is_google_vision_model("embeddinggemma-2") is True
    assert dino_models.is_google_vision_model("dinov3_vits16") is False
    # --dinov3-only catalog must remain strictly DINOv3 models.
    assert all(
        item.family == "dinov3"
        for item in dino_models.iter_dino_feature_models(dinov3_only=True)
    )


def test_embeddinggemma2_selective_loading_precision_and_matryoshka(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class _MockEmbeddingGemmaVisionTower(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                patch_size=16,
                pooling_kernel_size=3,
                num_hidden_layers=2,
                use_cls_token=False,
                hidden_size=768,
            )
            self.proj = nn.Conv2d(3, 768, kernel_size=16, stride=16, bias=False)
            nn.init.orthogonal_(self.proj.weight)

        def forward(
            self,
            pixel_values: torch.Tensor,
            pixel_position_ids: Optional[torch.Tensor] = None,
            output_hidden_states: bool = False,
        ):
            captured["input_dtype"] = pixel_values.dtype
            captured["pixel_position_ids_shape"] = (
                tuple(pixel_position_ids.shape)
                if pixel_position_ids is not None
                else None
            )
            feat_map = self.proj(pixel_values)  # [B, 768, H/16, W/16]
            b, c, gh, gw = feat_map.shape
            unpooled_tokens = feat_map.flatten(2).transpose(1, 2)  # [B, gh*gw, 768]
            pooled_map = nn.functional.avg_pool2d(feat_map, kernel_size=3, stride=3)
            pooled_tokens = pooled_map.flatten(2).transpose(1, 2)
            hidden_states = (
                (unpooled_tokens, unpooled_tokens * 0.9, unpooled_tokens)
                if output_hidden_states
                else None
            )
            return SimpleNamespace(
                last_hidden_state=pooled_tokens,
                hidden_states=hidden_states,
            )

    class _MockEmbeddingGemma2Full(nn.Module):
        def __init__(self, config) -> None:
            super().__init__()
            self.config = config
            self.vision_tower = _MockEmbeddingGemmaVisionTower()

        def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
            out = self.vision_tower(pixel_values)
            return out.last_hidden_state.mean(dim=1)

    mock_cfg = SimpleNamespace(
        audio_config={"hidden_size": 1024},
        vision_config=SimpleNamespace(
            patch_size=16,
            pooling_kernel_size=3,
            num_hidden_layers=2,
            use_cls_token=False,
            hidden_size=768,
        ),
    )

    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig,
        "from_pretrained",
        lambda *_a, **_kw: mock_cfg,
    )
    monkeypatch.setattr(
        transformers.AutoImageProcessor,
        "from_pretrained",
        lambda *_a, **_kw: SimpleNamespace(
            image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5)
        ),
    )

    def fake_from_pretrained(_repo_id, config=None, **kwargs):
        captured["audio_config_after_selective_load"] = getattr(
            config, "audio_config", "missing"
        )
        captured["load_kwargs"] = kwargs
        return _MockEmbeddingGemma2Full(config)

    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", fake_from_pretrained)

    extractor = Dinov3FeatureExtractor(
        Dinov3Config(
            model_name="embeddinggemma-2",
            short_side=96,
            device="cpu",
            matryoshka_dim=256,
        )
    )
    assert isinstance(extractor, GoogleVisionFeatureExtractor)
    assert extractor.backend_kind == "embeddinggemma"
    assert captured["audio_config_after_selective_load"] is None
    assert extractor._model_dtype in (torch.bfloat16, torch.float32)
    assert extractor._model_dtype != torch.float16
    assert extractor.patch_size == 16

    rgb = np.random.default_rng(42).integers(0, 255, size=(96, 144, 3), dtype=np.uint8)
    dense = extractor.extract(rgb, return_layer="last", normalize=True)
    # 96x144 snapped to multiple of 16*3=48 -> 96x144 -> stride 16 grid is 6x9, truncated to 256d.
    assert tuple(dense.shape) == (256, 6, 9)
    assert captured["pixel_position_ids_shape"] == (1, 54, 2)
    assert captured["input_dtype"] != torch.float16
    norms = torch.linalg.norm(dense, dim=0)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    crop_embs = extractor.embed_keypoint_regions(
        rgb,
        [(32.5, 40.2), (90.0, 52.0)],
        crop_size_px=48,
        matryoshka_dim=128,
    )
    assert tuple(crop_embs.shape) == (2, 128)
    crop_norms = torch.linalg.norm(crop_embs, dim=1)
    assert torch.allclose(crop_norms, torch.ones_like(crop_norms), atol=1e-5)


def test_embeddinggemma2_reports_clear_upgrade_hint_on_old_transformers(
    monkeypatch,
) -> None:
    import transformers

    def fail_unrecognized(*_a, **_kw):
        raise ValueError(
            "The checkpoint you are trying to load has model type `embedding_gemma2` "
            "but Transformers does not recognize this architecture."
        )

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", fail_unrecognized)
    with pytest.raises(RuntimeError, match="transformers>=5.19"):
        Dinov3FeatureExtractor(
            Dinov3Config(model_name="google/embeddinggemma-2", device="cpu")
        )


def test_siglip2_vision_model_extraction_and_pos_interpolation(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _MockSiglip2Vision(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                patch_size=16,
                num_hidden_layers=2,
                use_cls_token=False,
                hidden_size=64,
            )
            self.proj = nn.Conv2d(3, 64, kernel_size=16, stride=16, bias=False)

        def forward(
            self,
            pixel_values: torch.Tensor,
            interpolate_pos_encoding: bool = False,
            output_hidden_states: bool = False,
        ):
            captured["interpolate_pos_encoding"] = interpolate_pos_encoding
            tokens = self.proj(pixel_values).flatten(2).transpose(1, 2)
            hs = (tokens, tokens, tokens) if output_hidden_states else None
            return SimpleNamespace(last_hidden_state=tokens, hidden_states=hs)

    import transformers

    monkeypatch.setattr(
        transformers.AutoImageProcessor,
        "from_pretrained",
        lambda *_a, **_kw: SimpleNamespace(
            image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5)
        ),
    )
    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda *_a, **_kw: SimpleNamespace(
            vision_model=_MockSiglip2Vision(),
            to=lambda *_ta, **_tkw: None,
            eval=lambda: None,
        ),
    )

    extractor = Dinov3FeatureExtractor(
        Dinov3Config(model_name="siglip2", short_side=64, device="cpu")
    )
    assert isinstance(extractor, GoogleVisionFeatureExtractor)
    assert extractor.backend_kind == "siglip"
    img = np.zeros((64, 96, 3), dtype=np.uint8)
    grid = extractor.extract(img, return_layer="last", normalize=True)
    assert tuple(grid.shape) == (64, 4, 6)
    assert captured["interpolate_pos_encoding"] is True


@pytest.mark.parametrize(
    "model_alias",
    ["mobilenetv3_small", "mobilenetv4"],
)
def test_ondevice_mobilenet_pyramids_extract_stride8_and_track_offline(
    model_alias: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = Dinov3Config(
        model_name=model_alias,
        short_side=64,
        device="cpu",
        matryoshka_dim=64,
    )
    # Avoid any network weight download in offline CI.
    cfg._disable_pretrained = True  # type: ignore[attr-defined]
    if model_alias == "mobilenetv4" and importlib.util.find_spec("timm") is None:
        with pytest.raises(ImportError, match="timm>=1.0.0"):
            Dinov3FeatureExtractor(cfg)

        class _StubTimmPyramid(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.pretrained_cfg = {}
                self.s8 = nn.Conv2d(3, 40, kernel_size=3, stride=8, padding=1)
                self.s16 = nn.Conv2d(3, 48, kernel_size=3, stride=16, padding=1)

            def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
                return [self.s8(x), self.s16(x)]

        fake_timm = SimpleNamespace(
            create_model=lambda _arch, **_kw: _StubTimmPyramid()
        )
        monkeypatch.setitem(sys.modules, "timm", fake_timm)

    extractor = Dinov3FeatureExtractor(cfg)
    assert isinstance(extractor, GoogleVisionFeatureExtractor)
    assert extractor.patch_size == 8

    frame0 = np.stack([_render_textured_landmark((64, 96), (30.0, 28.0))] * 3, axis=-1)
    grid = extractor.extract(frame0, return_layer="last", normalize=True)
    assert tuple(grid.shape) == (64, 8, 12)
    crops = extractor.embed_keypoint_regions(frame0, [(30.0, 28.0)], crop_size_px=32)
    assert tuple(crops.shape) == (1, 64)


def test_ondevice_mobilenetv4_handles_missing_timm_gracefully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orig_find_spec = importlib.util.find_spec
    monkeypatch.setitem(sys.modules, "timm", None)  # type: ignore[arg-type]
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *args, **kwargs: None
        if name == "timm"
        else orig_find_spec(name, *args, **kwargs),
    )
    test_ondevice_mobilenet_pyramids_extract_stride8_and_track_offline(
        "mobilenetv4", monkeypatch
    )


def test_advanced_parameters_dialog_roundtrips_gaussian_region_and_matryoshka() -> None:
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    from annolid.gui.widgets.advanced_parameters_dialog import AdvancedParametersDialog

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None

    cfg = CutieDinoTrackerConfig(
        patch_model_name="google/embeddinggemma-2",
        embedding_matryoshka_dim=256,
        region_descriptor_radius=2,
        region_descriptor_sigma=1.1,
        region_subpatch_sampling=True,
        pixel_region_refine_enabled=True,
    )
    dialog = AdvancedParametersDialog(tracker_config=cfg)
    assert dialog.embedding_matryoshka_dim_combo.currentData() == 256
    assert dialog.region_descriptor_radius_spinbox.value() == 2
    assert dialog.region_descriptor_sigma_spinbox.value() == pytest.approx(1.1)
    assert dialog.region_subpatch_sampling_checkbox.isChecked() is True
    assert dialog.pixel_region_refine_checkbox.isChecked() is True

    dialog.embedding_matryoshka_dim_combo.setCurrentIndex(
        dialog.embedding_matryoshka_dim_combo.findData(128)
    )
    dialog.region_descriptor_radius_spinbox.setValue(1)
    dialog._collect_values()
    settings = dialog.get_tracker_settings()
    assert settings["patch_model_name"] == "google/embeddinggemma-2"
    assert settings["embedding_matryoshka_dim"] == 128
    assert settings["region_descriptor_radius"] == 1
    assert settings["region_subpatch_sampling"] is True
    assert settings["pixel_region_refine_enabled"] is True


def test_multi_angle_gaussian_pixel_template_recovers_large_in_plane_rotations() -> (
    None
):
    seed_gray = _render_textured_landmark((72, 72), (32.0, 32.0), angle_rad=0.0)
    tpl_multi = extract_gaussian_pixel_template(
        seed_gray,
        (32.0, 32.0),
        window_px=13,
        sigma_px=3.2,
        num_orientations=12,
    )
    tpl_single = extract_gaussian_pixel_template(
        seed_gray,
        (32.0, 32.0),
        window_px=13,
        sigma_px=3.2,
        num_orientations=1,
    )
    assert tpl_multi is not None
    assert tpl_single is not None
    assert len(tpl_multi.angles_rad) == 12

    true_xy = (34.3, 30.65)
    for deg in (90.0, 135.0, -150.0, 180.0):
        theta = math.radians(deg)
        rot_gray = _render_textured_landmark(
            (72, 72),
            true_xy,
            gain=1.18,
            bias=18.0,
            angle_rad=theta,
        )
        match = match_gaussian_pixel_template_oriented(
            rot_gray,
            tpl_multi,
            center_xy=(32.0, 32.0),
            search_radius_px=5,
        )
        assert match is not None
        est_x, est_y, zncc, est_angle = match
        assert zncc > 0.85
        assert abs(est_x - true_xy[0]) < 0.25
        assert abs(est_y - true_xy[1]) < 0.25
        ang_err = abs((est_angle - theta + math.pi) % (2.0 * math.pi) - math.pi)
        assert ang_err < math.radians(15.0)

        if deg == 135.0:
            single_match = match_gaussian_pixel_template(
                rot_gray,
                tpl_single,
                center_xy=(32.0, 32.0),
                search_radius_px=5,
            )
            single_zncc = 0.0 if single_match is None else single_match[2]
            assert zncc > single_zncc + 0.15
            if single_match is not None:
                single_err = math.hypot(
                    single_match[0] - true_xy[0], single_match[1] - true_xy[1]
                )
                multi_err = math.hypot(est_x - true_xy[0], est_y - true_xy[1])
                assert multi_err < single_err


def _render_two_landmark_frame(
    size_hw: Tuple[int, int],
    nose_xy: Tuple[float, float],
    tail_xy: Tuple[float, float],
    angle_rad: float,
) -> np.ndarray:
    h, w = size_hw
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    cos_a = float(math.cos(angle_rad))
    sin_a = float(math.sin(angle_rad))

    def _spot(cx: float, cy: float, freq_x: float, freq_y: float) -> np.ndarray:
        rx = xx - cx
        ry = yy - cy
        dx = rx * cos_a + ry * sin_a
        dy = -rx * sin_a + ry * cos_a
        env = np.exp(-0.5 * (dx * dx + dy * dy) / (3.1**2))
        return 62.0 * env * np.cos(freq_x * dx + freq_y * dy) + 34.0 * np.exp(
            -0.5 * ((dx - 1.1) ** 2 + (dy + 0.7) ** 2) / (1.7**2)
        )

    canvas = (
        78.0
        + _spot(float(nose_xy[0]), float(nose_xy[1]), 0.7, 0.3)
        + _spot(float(tail_xy[0]), float(tail_xy[1]), 0.35, -0.65)
    )
    return np.clip(canvas, 0.0, 255.0).astype(np.uint8)


def test_tracker_constellation_procrustes_rotation_and_partial_correction(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        _QueuedExtractor,
    )
    cfg = CutieDinoTrackerConfig(
        pixel_refine_enabled=False,
        pixel_region_refine_enabled=True,
        pixel_region_window=13,
        pixel_region_sigma=3.1,
        pixel_region_search_radius_px=6,
        pixel_region_weight=1.0,
        pixel_region_min_zncc=0.5,
        pixel_region_num_orientations=12,
        keypoint_cluster_refine=False,
        keypoint_refine_radius=0,
        dinov3_positional_debias=False,
        dinov3_backward_consistency=False,
        structural_consistency_weight=0.0,
        motion_prior_penalty_weight=0.0,
    )
    tracker = DinoKeypointTracker(
        model_name="dummy",
        runtime_config=cfg,
        search_radius=6,
        min_similarity=0.0,
    )

    # 96x96 image -> 6x6 grid of 16px patches (patch centers at 8, 24, 40, 56, 72, 88).
    def _grid_with_two_peaks(
        rc_nose: Tuple[int, int], rc_tail: Tuple[int, int]
    ) -> torch.Tensor:
        f = torch.zeros((4, 6, 6), dtype=torch.float32)
        f[:, rc_nose[0], rc_nose[1]] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        f[:, rc_tail[0], rc_tail[1]] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        return f

    # Seed at angle 0: nose=(38.0, 42.0) [rc=(2,2)], tail=(70.0, 42.0) [rc=(2,4)].
    seed_nose = (38.0, 42.0)
    seed_tail = (70.0, 42.0)

    # Frame 1 rotated +120 deg in image coords (tail - nose = (-16.0, +27.7)):
    # nose near patch (1, 3) -> center (56, 24); true=(59.4, 26.8)
    # tail near patch (3, 2) -> center (40, 56); true=(43.4, 54.5)
    theta_1 = math.radians(120.0)
    f1_nose = (59.4, 26.8)
    f1_tail = (43.4, 54.5)

    # Frame 2 rotated +150 deg in image coords (tail - nose = (-27.7, +16.0)):
    # nose near patch (2, 4) -> center (72, 40); true=(69.4, 37.2)
    # tail near patch (3, 2) -> center (40, 56); true=(41.7, 53.2)
    theta_2 = math.radians(150.0)
    f2_nose = (69.4, 37.2)
    f2_tail = (41.7, 53.2)

    tracker.extractor.set_queue(
        [
            _grid_with_two_peaks((2, 2), (2, 4)),
            _grid_with_two_peaks((1, 3), (3, 2)),
            _grid_with_two_peaks((2, 4), (3, 2)),
        ]
    )

    g0 = _render_two_landmark_frame((96, 96), seed_nose, seed_tail, 0.0)
    g1 = _render_two_landmark_frame((96, 96), f1_nose, f1_tail, theta_1)
    g2 = _render_two_landmark_frame((96, 96), f2_nose, f2_tail, theta_2)

    reg = InstanceRegistry()
    reg.register_keypoint(
        KeypointState(
            key="mousenose",
            instance_label="mouse",
            label="nose",
            x=seed_nose[0],
            y=seed_nose[1],
        )
    )
    reg.register_keypoint(
        KeypointState(
            key="mousetail",
            instance_label="mouse",
            label="tail",
            x=seed_tail[0],
            y=seed_tail[1],
        )
    )
    tracker.start(Image.fromarray(np.stack([g0] * 3, axis=-1)), reg, {})

    out1 = {
        item["id"]: item
        for item in tracker.update(Image.fromarray(np.stack([g1] * 3, axis=-1)), {})
    }
    assert abs(float(out1["mousenose"]["x"]) - f1_nose[0]) < 0.45
    assert abs(float(out1["mousenose"]["y"]) - f1_nose[1]) < 0.45
    assert abs(float(out1["mousetail"]["x"]) - f1_tail[0]) < 0.45
    assert abs(float(out1["mousetail"]["y"]) - f1_tail[1]) < 0.45

    # Apply a partial manual correction to ONLY nose while the animal is rotated ~120 deg.
    tracker.apply_external_corrections(
        {"mousenose": (f1_nose[0] + 0.05, f1_nose[1] - 0.05)},
        clear_cache=False,
    )
    nose_heading = tracker.tracks["mousenose"].template_heading_rad
    # Constellation vector (32, 0) -> (-15.95, 27.75) has heading ~ +120 deg (~2.09 rad).
    assert abs(nose_heading - theta_1) < math.radians(10.0)
    assert tracker.tracks["mousetail"].template_heading_rad == pytest.approx(0.0)

    out2 = {
        item["id"]: item
        for item in tracker.update(Image.fromarray(np.stack([g2] * 3, axis=-1)), {})
    }
    assert abs(float(out2["mousenose"]["x"]) - f2_nose[0]) < 0.45
    assert abs(float(out2["mousenose"]["y"]) - f2_nose[1]) < 0.45
    assert abs(float(out2["mousetail"]["x"]) - f2_tail[0]) < 0.45
    assert abs(float(out2["mousetail"]["y"]) - f2_tail[1]) < 0.45


def test_strict_mrl_guard_and_non_mrl_orthonormal_compression(monkeypatch) -> None:
    assert dino_models.supports_matryoshka_truncation("embeddinggemma-2") is True
    assert dino_models.supports_matryoshka_truncation("google/embeddinggemma-2") is True
    assert dino_models.supports_matryoshka_truncation("dinov3_vits16") is False
    assert dino_models.supports_matryoshka_truncation("siglip2") is False
    assert dino_models.supports_matryoshka_truncation("mobilenetv4") is False

    # Two multi-stage descriptors identical in early channels and orthogonal in late channels.
    v_a = torch.tensor([1.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    v_b = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0], dtype=torch.float32)
    naive_a = apply_matryoshka_truncation(v_a, 4, channel_dim=0, mrl_native=True)
    naive_b = apply_matryoshka_truncation(v_b, 4, channel_dim=0, mrl_native=True)
    # Prefix slicing drops channels 4..7 completely -> identical vectors!
    assert float(torch.dot(naive_a, naive_b).item()) == pytest.approx(1.0, abs=1e-6)

    ortho_a = apply_matryoshka_truncation(v_a, 4, channel_dim=0, mrl_native=False)
    ortho_b = apply_matryoshka_truncation(v_b, 4, channel_dim=0, mrl_native=False)
    ortho_a_repeat = apply_matryoshka_truncation(
        v_a, 4, channel_dim=0, mrl_native=False
    )
    assert torch.allclose(ortho_a, ortho_a_repeat, atol=1e-7)
    assert float(ortho_a.norm().item()) == pytest.approx(1.0, abs=1e-6)
    # Orthonormal projection retains signal from trailing stages -> clearly distinct!
    assert float(torch.dot(ortho_a, ortho_b).item()) < 0.85

    recorded_cfgs: List[Dinov3Config] = []

    class _RecordingExtractor(_QueuedExtractor):
        def __init__(self, cfg: Dinov3Config) -> None:
            super().__init__(cfg)
            recorded_cfgs.append(cfg)

    monkeypatch.setattr(
        "annolid.tracking.dino_keypoint_tracker.Dinov3FeatureExtractor",
        _RecordingExtractor,
    )

    # 1) Strict MRL (default True) ignores leftover embedding_matryoshka_dim on non-MRL model.
    DinoKeypointTracker(
        model_name="facebook/dinov2-base",
        runtime_config=CutieDinoTrackerConfig(embedding_matryoshka_dim=128),
    )
    assert recorded_cfgs[-1].matryoshka_dim is None

    # 2) Strict MRL honors embedding_matryoshka_dim on EmbeddingGemma 2.
    DinoKeypointTracker(
        model_name="google/embeddinggemma-2",
        runtime_config=CutieDinoTrackerConfig(embedding_matryoshka_dim=128),
    )
    assert recorded_cfgs[-1].matryoshka_dim == 128

    # 3) Explicit opt-in (strict_mrl=False) forwards matryoshka_dim on non-MRL model.
    DinoKeypointTracker(
        model_name="facebook/dinov2-base",
        runtime_config=CutieDinoTrackerConfig(
            embedding_matryoshka_dim=128,
            embedding_matryoshka_strict_mrl=False,
        ),
    )
    assert recorded_cfgs[-1].matryoshka_dim == 128


def test_embeddinggemma2_projector_alignment_and_direct_safetensors_vision_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    # Part A: Verify 1152 -> 768 multimodal projection happens BEFORE blending & MRL slicing,
    # and get_image_features receives 2D pixel_position_ids when supported.
    captured: dict[str, object] = {}

    class _Tower1152(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                patch_size=16,
                pooling_kernel_size=3,
                num_hidden_layers=1,
                use_cls_token=False,
                hidden_size=1152,
            )
            self.proj = nn.Conv2d(3, 1152, kernel_size=16, stride=16, bias=False)

        def forward(
            self,
            pixel_values: torch.Tensor,
            pixel_position_ids: Optional[torch.Tensor] = None,
            output_hidden_states: bool = False,
        ):
            feat_map = self.proj(pixel_values)
            unpooled = feat_map.flatten(2).transpose(1, 2)
            pooled = (
                nn.functional.avg_pool2d(feat_map, kernel_size=3, stride=3)
                .flatten(2)
                .transpose(1, 2)
            )
            hs = (unpooled, unpooled) if output_hidden_states else None
            return SimpleNamespace(last_hidden_state=pooled, hidden_states=hs)

    class _FullModelWithProjector(nn.Module):
        def __init__(self, config) -> None:
            super().__init__()
            self.config = config
            self.vision_tower = _Tower1152()
            self.embed_vision = nn.Linear(1152, 768, bias=False)
            with torch.no_grad():
                self.embed_vision.weight.zero_()
                # Map trailing vision channel 1151 -> MRL channel 0 so naive pre-projection
                # truncation [:256] would completely miss it!
                self.embed_vision.weight[0, 1151] = 1.0

        def get_image_features(
            self,
            pixel_values: torch.Tensor,
            pixel_position_ids: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            captured["gif_pos_ids_shape"] = (
                tuple(pixel_position_ids.shape)
                if pixel_position_ids is not None
                else None
            )
            out = self.vision_tower(pixel_values, pixel_position_ids=pixel_position_ids)
            return self.embed_vision(out.last_hidden_state.mean(dim=1))

    import transformers

    with monkeypatch.context() as mp:
        mp.setattr(
            transformers.AutoConfig,
            "from_pretrained",
            lambda *_a, **_kw: SimpleNamespace(
                audio_config=None,
                vision_config=SimpleNamespace(
                    patch_size=16,
                    pooling_kernel_size=3,
                    num_hidden_layers=1,
                    use_cls_token=False,
                    hidden_size=1152,
                ),
            ),
        )
        mp.setattr(
            transformers.AutoImageProcessor,
            "from_pretrained",
            lambda *_a, **_kw: SimpleNamespace(
                image_mean=(0.5, 0.5, 0.5), image_std=(0.5, 0.5, 0.5)
            ),
        )
        mp.setattr(
            transformers.AutoModel,
            "from_pretrained",
            lambda _repo, config=None, **_kw: _FullModelWithProjector(config),
        )

        ext = Dinov3FeatureExtractor(
            Dinov3Config(
                model_name="google/embeddinggemma-2",
                short_side=48,
                device="cpu",
                matryoshka_dim=256,
            )
        )
        with torch.no_grad():
            ext.model.proj.weight.zero_()
            ext.model.proj.weight[1151] = 1.0

        ones_img = np.full((48, 48, 3), 255, dtype=np.uint8)
        dense = ext.extract(ones_img, return_layer="last", normalize=True)
        assert tuple(dense.shape) == (256, 3, 3)
        # Channel 1151 in vision_tower was projected to MRL channel 0 before [:256] slice!
        assert torch.allclose(dense[0], torch.ones((3, 3)), atol=1e-5)

        crops = ext.embed_keypoint_regions(
            ones_img, [(24.0, 24.0), (16.0, 16.0)], crop_size_px=48, matryoshka_dim=128
        )
        assert tuple(crops.shape) == (2, 128)
        assert captured["gif_pos_ids_shape"] == (2, 9, 2)

    # Part B: Unpatched transformers offline safetensors fallback when top-level
    # "embedding_gemma2" is unrecognized by AutoConfig, while nested vision_config is valid.
    from safetensors.torch import save_file
    from transformers import SiglipVisionConfig, SiglipVisionModel

    ckpt_dir = tmp_path / "local_multimodal_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_cfg = SiglipVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        image_size=32,
        patch_size=16,
    )
    ref_tower = SiglipVisionModel(vis_cfg)
    ref_proj = nn.Linear(32, 24, bias=False)
    nn.init.orthogonal_(ref_proj.weight)

    tensors: dict[str, torch.Tensor] = {
        f"vision_tower.{k}": v.contiguous() for k, v in ref_tower.state_dict().items()
    }
    tensors["embed_vision.weight"] = ref_proj.weight.detach().contiguous()
    # Include unused modality weights to verify selective loading ignores them cleanly.
    tensors["audio_tower.dummy_weight"] = torch.randn((8, 8))
    tensors["language_model.dummy_weight"] = torch.randn((8, 8))
    save_file(tensors, str(ckpt_dir / "model.safetensors"))

    vis_dict = vis_cfg.to_dict()
    vis_dict["model_type"] = "siglip_vision_model"
    (ckpt_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "embedding_gemma2",
                "embedding_dim": 24,
                "vision_config": vis_dict,
                "audio_config": {"model_type": "gemma4_audio", "hidden_size": 1024},
            }
        ),
        encoding="utf-8",
    )

    ok, status_msg = dino_models.check_dino_model_runtime_support(str(ckpt_dir))
    assert ok is True
    assert "Ready" in status_msg

    direct_ext = Dinov3FeatureExtractor(
        Dinov3Config(
            model_name=str(ckpt_dir),
            short_side=32,
            device="cpu",
            matryoshka_dim=16,
        )
    )
    assert isinstance(direct_ext, GoogleVisionFeatureExtractor)
    assert direct_ext.backend_kind == "embeddinggemma"
    assert direct_ext._vision_projector is not None

    rgb_small = np.random.default_rng(9).integers(
        0, 255, size=(32, 48, 3), dtype=np.uint8
    )
    grid_out = direct_ext.extract(rgb_small, return_layer="last", normalize=True)
    assert tuple(grid_out.shape) == (16, 2, 3)
    crop_out = direct_ext.embed_keypoint_regions(
        rgb_small, [(16.0, 16.0)], crop_size_px=32, matryoshka_dim=12
    )
    assert tuple(crop_out.shape) == (1, 12)


def test_advanced_parameters_dialog_preset_sync_and_runtime_status_label() -> None:
    QtWidgets = pytest.importorskip("qtpy.QtWidgets")
    from annolid.gui.widgets.advanced_parameters_dialog import AdvancedParametersDialog

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None

    cfg = CutieDinoTrackerConfig(
        patch_model_name="torchvision/mobilenet_v3_small",
        region_descriptor_radius=0,
        region_descriptor_sigma=1.5,
        region_subpatch_sampling=False,
        pixel_region_refine_enabled=False,
    )
    dialog = AdvancedParametersDialog(tracker_config=cfg)
    assert "Ready" in dialog.dino_model_status_label.text()
    assert dialog.region_descriptor_radius_spinbox.value() == 0
    assert dialog.region_subpatch_sampling_checkbox.isChecked() is False
    assert dialog.pixel_region_refine_checkbox.isChecked() is False

    # Switching preset must synchronize Gaussian region & pixel template controls.
    dialog.tracker_preset_combo.setCurrentText("fly_70fps_keypoints")
    assert dialog.region_descriptor_radius_spinbox.value() == 1
    assert dialog.region_descriptor_sigma_spinbox.value() == pytest.approx(0.7)
    assert dialog.region_subpatch_sampling_checkbox.isChecked() is True
    assert dialog.pixel_region_refine_checkbox.isChecked() is True

    # Switching model updates runtime readiness status text immediately.
    dialog.dino_model_combo.setEditText("google/embeddinggemma-2")
    _, expected_msg = dino_models.check_dino_model_runtime_support(
        "google/embeddinggemma-2"
    )
    assert dialog.dino_model_status_label.text() == expected_msg
