"""Focused tests for Gaussian-weighted local-region keypoint tracking and Google vision backbones."""

from __future__ import annotations

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
from annolid.tracking.configuration import CutieDinoTrackerConfig
from annolid.tracking.dino_keypoint_tracker import DinoKeypointTracker
from annolid.tracking.dino_keypoint_types import RefineRegion, SupportProbe
from annolid.tracking.domain import InstanceRegistry, KeypointState
from annolid.tracking.gaussian_region import (
    extract_gaussian_pixel_template,
    gaussian_smooth_feature_grid,
    match_gaussian_pixel_template,
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
) -> np.ndarray:
    h, w = size_hw
    yy, xx = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    dx = xx - float(center_xy[0])
    dy = yy - float(center_xy[1])
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
) -> None:
    cfg = Dinov3Config(
        model_name=model_alias,
        short_side=64,
        device="cpu",
        matryoshka_dim=64,
    )
    # Avoid any network weight download in offline CI.
    cfg._disable_pretrained = True  # type: ignore[attr-defined]
    extractor = Dinov3FeatureExtractor(cfg)
    assert isinstance(extractor, GoogleVisionFeatureExtractor)
    assert extractor.patch_size == 8

    frame0 = np.stack([_render_textured_landmark((64, 96), (30.0, 28.0))] * 3, axis=-1)
    grid = extractor.extract(frame0, return_layer="last", normalize=True)
    assert tuple(grid.shape) == (64, 8, 12)
    crops = extractor.embed_keypoint_regions(frame0, [(30.0, 28.0)], crop_size_px=32)
    assert tuple(crops.shape) == (1, 64)


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
