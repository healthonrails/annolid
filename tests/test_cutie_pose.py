"""Focused unit tests for Compressed Cutie-Pose keypoint head and feature tap."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from annolid.segmentation.cutie_vos.model.cutie_pose import (
    CutieDecoderFeatureTap,
    CutiePoseHeadConfig,
    CutiePoseKeypointHead,
    compute_cutie_pose_loss,
    compute_square_roi_from_mask,
    load_cutie_pose_checkpoint,
    render_gaussian_heatmaps,
    sample_roi_tensor,
    save_cutie_pose_checkpoint,
)


def test_cutie_pose_head_forward_dsnt_loss_and_fp16_roundtrip(tmp_path: Path) -> None:
    cfg = CutiePoseHeadConfig(
        num_keypoints=7, hidden_dim=64, tucker_rank=32, roi_size=32, feat_size=16
    )
    head = CutiePoseKeypointHead(cfg)
    assert sum(p.numel() for p in head.parameters()) < 300_000

    p4 = torch.randn(2, cfg.c_p4, cfg.feat_size, cfg.feat_size)
    p8 = torch.randn(2, cfg.c_p8, cfg.feat_size // 2, cfg.feat_size // 2)
    msk = torch.rand(2, 1, cfg.roi_size, cfg.roi_size)
    img = torch.rand(2, 3, cfg.roi_size, cfg.roi_size)

    # Cold-start (prev_heatmaps=None)
    hm_logits, coords, vis_logits = head(p4, p8, msk, img, None)
    assert hm_logits.shape == (2, 7, cfg.roi_size, cfg.roi_size)
    assert coords.shape == (2, 7, 2)
    assert vis_logits.shape == (2, 7)
    assert torch.all((coords >= 0.0) & (coords <= 1.0))

    # Temporal prior forward + backward
    tgt_xy = torch.tensor(
        [
            [
                [0.70, 0.50],
                [0.58, 0.62],
                [0.58, 0.38],
                [0.52, 0.50],
                [0.34, 0.64],
                [0.34, 0.36],
                [0.20, 0.50],
            ]
        ],
        dtype=torch.float32,
    ).repeat(2, 1, 1)
    tgt_valid = torch.ones((2, 7), dtype=torch.bool)
    prev_hm = render_gaussian_heatmaps(tgt_xy, tgt_valid, size=cfg.roi_size, sigma=2.0)

    hm_t, coords_t, vis_t = head(p4, p8, msk, img, prev_hm)
    loss, metrics = compute_cutie_pose_loss(
        hm_t, coords_t, vis_t, tgt_xy, tgt_valid, enforce_calms_chirality=True
    )
    assert torch.isfinite(loss)
    assert set(metrics.keys()) == {"total", "coord", "kl", "vis", "chiral"}
    loss.backward()

    # Compact fp16 checkpoint round-trip
    ckpt = save_cutie_pose_checkpoint(
        head, tmp_path / "cutie_pose_head.pt", metadata={"tag": "unit"}, fp16=True
    )
    assert ckpt.is_file()
    loaded, meta = load_cutie_pose_checkpoint(ckpt, device="cpu")
    assert meta["tag"] == "unit"
    assert loaded.cfg.num_keypoints == 7
    with torch.no_grad():
        _, xy_loaded, _ = loaded(p4, p8, msk, img, prev_hm)
    assert xy_loaded.shape == (2, 7, 2)


def test_cutie_pose_roi_padding_alignment_and_decoder_tap() -> None:
    mask = np.zeros((428, 768), dtype=np.float32)
    mask[180:260, 300:400] = 1.0
    roi = compute_square_roi_from_mask(mask, pad_scale=1.4, min_size=64.0)
    assert roi is not None
    x0, y0, x1, y1 = roi
    assert (x1 - x0) == pytest.approx(y1 - y0, abs=1e-4)
    assert x0 < 300.0 and x1 > 400.0 and y0 < 180.0 and y1 > 260.0

    # Stride-4 padded Cutie feature map: 428x768 -> padded 432x768 -> stride-4 108x192
    # Place a linear x-ramp in unpadded internal pixel space and verify exact coordinate recovery
    pad_h = 432
    lh = (pad_h - 428) // 2  # 2 px top pad
    ys_s4 = (torch.arange(108, dtype=torch.float32) + 0.5) * 4.0 - lh
    xs_s4 = (torch.arange(192, dtype=torch.float32) + 0.5) * 4.0
    gy, gx = torch.meshgrid(ys_s4, xs_s4, indexing="ij")
    feat = torch.stack([gx, gy], dim=0).unsqueeze(0)  # (1, 2, 108, 192)
    sampled = sample_roi_tensor(
        feat,
        (200.0, 100.0, 328.0, 228.0),
        frame_hw=(428, 768),
        out_size=16,
        max_internal_size=480,
        feat_stride=4,
    )
    # Center of ROI (264, 164) must match sampled center within <1 px
    center_x = float(sampled[0, 0, 7:9, 7:9].mean().item())
    center_y = float(sampled[0, 1, 7:9, 7:9].mean().item())
    assert center_x == pytest.approx(264.0, abs=1.0)
    assert center_y == pytest.approx(164.0, abs=1.0)

    # Non-invasive CutieDecoderFeatureTap captures p8, p4, logits_s4
    class _DummyDec(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up_16_8 = nn.Identity()
            self.up_8_4 = nn.Identity()
            self.pred = nn.Identity()

    class _DummyCutie(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mask_decoder = _DummyDec()

    dummy = _DummyCutie()
    with CutieDecoderFeatureTap(dummy) as tap:
        dummy.mask_decoder.up_16_8(torch.ones(1, 2, 128, 16, 16))
        dummy.mask_decoder.up_8_4(torch.full((1, 2, 128, 32, 32), 2.0))
        dummy.mask_decoder.pred(torch.full((2, 1, 32, 32), 3.0))
        assert tap.p8 is not None and tap.p8.shape == (1, 2, 128, 16, 16)
        assert tap.p4 is not None and tap.p4.shape == (1, 2, 128, 32, 32)
        assert tap.logits_s4 is not None and tap.logits_s4.shape == (2, 1, 32, 32)
    assert tap._handles == []
