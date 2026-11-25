"""Lightweight temporal pose head branching from Compressed CUTIE VOS features.

Attaches a Tucker-2 factorized keypoint decoder (~0.35M parameters) to the
per-object multi-scale decoder features (``p8`` at stride 8, ``p4`` at stride 4)
and tracked silhouette probabilities produced by Annolid's compressed CUTIE VOS
backbone (``tucker_bn_folded_v2``).

Key design properties:
- Zero modification to ``CUTIE`` or ``MaskDecoder`` forward contracts: uses
  non-invasive forward hooks (``CutieDecoderFeatureTap``) to read per-object
  ``p8``, ``p4``, and mask logits during standard ``InferenceCore.step`` calls.
- Mask-centered canonical ROI cropping normalizes animal scale and position
  (e.g., center vs. dark cage corners) onto a fixed ``64x64`` spatial grid.
- Recurrent temporal keypoint heatmap memory (``prev_heatmaps`` projected into
  the current frame's ROI) stabilizes orientation across headstage occlusions,
  curled postures, and dark corners while supporting single-frame cold start
  (``prev_heatmaps=None`` or zeros).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from annolid.segmentation.cutie_vos.model.weight_compression import TuckerConv2d


@dataclass(frozen=True)
class CutiePoseHeadConfig:
    """Configuration for :class:`CutiePoseKeypointHead`."""

    num_keypoints: int = 7
    keypoint_names: Tuple[str, ...] = (
        "nose",
        "left_ear",
        "right_ear",
        "neck",
        "left_hip",
        "right_hip",
        "tail_base",
    )
    flip_idx: Tuple[int, ...] = (0, 2, 1, 3, 5, 4, 6)
    c_p4: int = 128
    c_p8: int = 128
    hidden_dim: int = 96
    tucker_rank: int = 48
    roi_size: int = 64
    feat_size: int = 32
    roi_pad_scale: float = 1.45
    min_roi_px: float = 64.0
    img_dropout_p: float = 0.20


class _TuckerResBlock(nn.Module):
    """Residual block using Tucker-2 factorized 3x3 convolutions + GroupNorm."""

    def __init__(self, channels: int, rank: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = TuckerConv2d(
            channels,
            channels,
            rank,
            rank,
            padding=dilation,
            dilation=dilation,
            has_bias=True,
        )
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv2 = TuckerConv2d(
            channels,
            channels,
            rank,
            rank,
            padding=1,
            dilation=1,
            has_bias=True,
        )
        self.norm2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(x + h)


class CutiePoseKeypointHead(nn.Module):
    """Tucker-2 factorized temporal pose head for Compressed CUTIE VOS."""

    def __init__(self, cfg: Optional[CutiePoseHeadConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or CutiePoseHeadConfig()
        c_hid = int(self.cfg.hidden_dim)
        rank = int(self.cfg.tucker_rank)
        k = int(self.cfg.num_keypoints)

        # Branch 1: Cutie per-object decoder features (p4 + p8) @ feat_size (32x32)
        self.cutie_proj = nn.Sequential(
            nn.Conv2d(self.cfg.c_p4 + self.cfg.c_p8, c_hid, kernel_size=1, bias=False),
            nn.GroupNorm(8, c_hid),
            nn.SiLU(inplace=True),
        )
        self.cutie_blocks = nn.Sequential(
            _TuckerResBlock(c_hid, rank, dilation=1),
            _TuckerResBlock(c_hid, rank, dilation=2),
        )

        # Branch 2: High-res grayscale/RGB ROI + tracked object mask + temporal keypoint prior @ roi_size (64x64)
        self.img_stem = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 24),
            nn.SiLU(inplace=True),
        )
        self.prior_stem = nn.Sequential(
            nn.Conv2d(1 + k, 24, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 24),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(c_hid + 48, c_hid, kernel_size=1, bias=False),
            nn.GroupNorm(8, c_hid),
            nn.SiLU(inplace=True),
        )
        self.refine = nn.Sequential(
            _TuckerResBlock(c_hid, rank, dilation=1),
            _TuckerResBlock(c_hid, rank, dilation=2),
        )
        self.heatmap_head = nn.Conv2d(c_hid, k, kernel_size=1)
        self.vis_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c_hid, k),
        )

        # Precomputed normalized coordinate grid for DSNT soft-argmax in [0, 1]
        grid_1d = torch.linspace(
            0.5 / self.cfg.roi_size, 1.0 - 0.5 / self.cfg.roi_size, self.cfg.roi_size
        )
        gy, gx = torch.meshgrid(grid_1d, grid_1d, indexing="ij")
        self.register_buffer(
            "grid_x",
            gx.view(1, 1, self.cfg.roi_size, self.cfg.roi_size),
            persistent=False,
        )
        self.register_buffer(
            "grid_y",
            gy.view(1, 1, self.cfg.roi_size, self.cfg.roi_size),
            persistent=False,
        )

    def forward(
        self,
        p4_roi: torch.Tensor,
        p8_roi: torch.Tensor,
        mask_roi: torch.Tensor,
        img_roi: torch.Tensor,
        prev_heatmaps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass on object-centered ROI crops.

        Returns
        -------
        heatmaps_logits : ``(B, K, R, R)`` spatial logits
        coords_norm : ``(B, K, 2)`` DSNT soft-argmax coordinates in ``[0, 1]`` (x, y)
        vis_logits : ``(B, K)`` keypoint presence/visibility logits
        """
        bsz = p4_roi.shape[0]
        r = int(self.cfg.roi_size)
        k = int(self.cfg.num_keypoints)

        if p8_roi.shape[-2:] != p4_roi.shape[-2:]:
            p8_roi = F.interpolate(
                p8_roi, size=p4_roi.shape[-2:], mode="bilinear", align_corners=False
            )

        c_feat = self.cutie_proj(torch.cat([p4_roi, p8_roi], dim=1))
        c_feat = self.cutie_blocks(c_feat)
        if c_feat.shape[-2:] != (r, r):
            c_feat = F.interpolate(
                c_feat, size=(r, r), mode="bilinear", align_corners=False
            )

        if mask_roi.shape[-2:] != (r, r):
            mask_roi = F.interpolate(
                mask_roi, size=(r, r), mode="bilinear", align_corners=False
            )
        if img_roi.shape[-2:] != (r, r):
            img_roi = F.interpolate(
                img_roi, size=(r, r), mode="bilinear", align_corners=False
            )
        if prev_heatmaps is None:
            prev_heatmaps = torch.zeros(
                (bsz, k, r, r), device=p4_roi.device, dtype=p4_roi.dtype
            )
        elif prev_heatmaps.shape[-2:] != (r, r):
            prev_heatmaps = F.interpolate(
                prev_heatmaps, size=(r, r), mode="bilinear", align_corners=False
            )

        img_feat = self.img_stem(img_roi)
        if self.training and self.cfg.img_dropout_p > 0.0:
            # Feature-level dropout encourages reliance on real-world-pretrained Cutie features
            keep = (
                torch.rand((bsz, 1, 1, 1), device=img_feat.device)
                >= self.cfg.img_dropout_p
            ).to(img_feat.dtype)
            img_feat = img_feat * keep / max(1e-6, 1.0 - self.cfg.img_dropout_p)

        prior_feat = self.prior_stem(torch.cat([mask_roi, prev_heatmaps], dim=1))
        fused = self.fuse(torch.cat([c_feat, img_feat, prior_feat], dim=1))
        refined = self.refine(fused)

        hm_logits = self.heatmap_head(refined)
        vis_logits = self.vis_head(refined)
        coords_norm = self.dsnt_decode(hm_logits)
        return hm_logits, coords_norm, vis_logits

    def dsnt_decode(self, hm_logits: torch.Tensor) -> torch.Tensor:
        """Differentiable Spatial-to-Numerical Transform (sub-pixel soft-argmax)."""
        bsz, k, h, w = hm_logits.shape
        probs = F.softmax(hm_logits.view(bsz, k, h * w), dim=-1).view(bsz, k, h, w)
        exp_x = (probs * self.grid_x.to(probs.dtype)).sum(dim=(-2, -1))
        exp_y = (probs * self.grid_y.to(probs.dtype)).sum(dim=(-2, -1))
        return torch.stack([exp_x, exp_y], dim=-1)


class CutieDecoderFeatureTap:
    """Non-invasive forward hook context capturing Cutie MaskDecoder per-object features."""

    def __init__(self, cutie_model: nn.Module) -> None:
        self.cutie_model = cutie_model
        self.p8: Optional[torch.Tensor] = None
        self.p4: Optional[torch.Tensor] = None
        self.logits_s4: Optional[torch.Tensor] = None
        self._handles: List[Any] = []

    def __enter__(self) -> "CutieDecoderFeatureTap":
        dec = self.cutie_model.mask_decoder

        def _hook_p8(_mod: nn.Module, _inp: Any, out: torch.Tensor) -> None:
            self.p8 = out.detach()

        def _hook_p4(_mod: nn.Module, _inp: Any, out: torch.Tensor) -> None:
            self.p4 = out.detach()

        def _hook_pred(_mod: nn.Module, _inp: Any, out: torch.Tensor) -> None:
            self.logits_s4 = out.detach()

        self._handles = [
            dec.up_16_8.register_forward_hook(_hook_p8),
            dec.up_8_4.register_forward_hook(_hook_p4),
            dec.pred.register_forward_hook(_hook_pred),
        ]
        return self

    def __exit__(self, *_exc: Any) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def clear(self) -> None:
        self.p8 = None
        self.p4 = None
        self.logits_s4 = None


def compute_square_roi_from_mask(
    mask_2d: np.ndarray,
    *,
    pad_scale: float = 1.45,
    min_size: float = 64.0,
) -> Optional[Tuple[float, float, float, float]]:
    """Compute a square ``(x0, y0, x1, y1)`` ROI centered on a binary/probability mask."""
    ys, xs = np.nonzero(mask_2d > 0.35)
    if xs.size < 8:
        return None
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    side = max(min_size, max(x1 - x0 + 1.0, y1 - y0 + 1.0) * pad_scale)
    half = 0.5 * side
    return (cx - half, cy - half, cx + half, cy + half)


def sample_roi_tensor(
    feat_map: torch.Tensor,
    roi_xyxy: Tuple[float, float, float, float],
    *,
    frame_hw: Tuple[int, int],
    out_size: int,
    max_internal_size: int = -1,
    feat_stride: int = 1,
) -> torch.Tensor:
    """Bilinearly sample ``feat_map`` ``(1, C, Hf, Wf)`` on ``roi_xyxy`` in frame pixel coords.

    When ``feat_stride > 1`` (e.g. Cutie ``p4`` stride 4 or ``p8`` stride 8), accounts for
    Cutie's internal ``max_internal_size`` resize and symmetric ``pad_divide_by(..., 16)``
    padding so ROI sampling is sub-pixel aligned at arbitrary input resolutions.
    """
    frame_h, frame_w = frame_hw
    x0, y0, x1, y1 = roi_xyxy
    xs = torch.linspace(x0, x1, out_size, device=feat_map.device, dtype=torch.float32)
    ys = torch.linspace(y0, y1, out_size, device=feat_map.device, dtype=torch.float32)

    if feat_stride > 1:
        min_side = min(frame_h, frame_w)
        if max_internal_size > 0 and min_side > max_internal_size:
            int_h = int(frame_h / min_side * max_internal_size)
            int_w = int(frame_w / min_side * max_internal_size)
        else:
            int_h, int_w = frame_h, frame_w
        pad_h = int(feat_map.shape[-2]) * int(feat_stride)
        pad_w = int(feat_map.shape[-1]) * int(feat_stride)
        lw = int((pad_w - int_w) / 2)
        lh = int((pad_h - int_h) / 2)
        xs_pad = xs * (float(int_w) / float(frame_w)) + float(lw)
        ys_pad = ys * (float(int_h) / float(frame_h)) + float(lh)
        gx = ((xs_pad + 0.5) / float(pad_w)) * 2.0 - 1.0
        gy = ((ys_pad + 0.5) / float(pad_h)) * 2.0 - 1.0
    else:
        gx = ((xs + 0.5) / float(frame_w)) * 2.0 - 1.0
        gy = ((ys + 0.5) / float(frame_h)) * 2.0 - 1.0

    grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, out, out, 2)
    return F.grid_sample(
        feat_map.float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


def render_gaussian_heatmaps(
    coords_norm: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    size: int = 64,
    sigma: float = 2.5,
) -> torch.Tensor:
    """Render Gaussian heatmaps ``(B, K, size, size)`` for normalized ``(x, y) in [0, 1]``."""
    bsz, k, _ = coords_norm.shape
    device = coords_norm.device
    coords_px = coords_norm * float(size) - 0.5
    grid = torch.arange(size, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(grid, grid, indexing="ij")
    dx = xx.view(1, 1, size, size) - coords_px[:, :, 0].view(bsz, k, 1, 1)
    dy = yy.view(1, 1, size, size) - coords_px[:, :, 1].view(bsz, k, 1, 1)
    hm = torch.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
    hm = hm * valid_mask.float().view(bsz, k, 1, 1)
    return hm


def compute_cutie_pose_loss(
    hm_logits: torch.Tensor,
    coords_pred: torch.Tensor,
    vis_logits: torch.Tensor,
    target_coords_norm: torch.Tensor,
    target_valid: torch.Tensor,
    *,
    sigma: float = 2.5,
    enforce_calms_chirality: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Joint DSNT Smooth-L1 + spatial KL heatmap + visibility + optional chirality loss."""
    bsz, k, h, w = hm_logits.shape
    valid_f = target_valid.float()  # (B, K)
    denom = valid_f.sum().clamp(min=1.0)

    # 1. DSNT coordinate Smooth-L1 loss (scaled to ROI pixels)
    coord_err = F.smooth_l1_loss(
        coords_pred * float(h),
        target_coords_norm * float(h),
        beta=1.5,
        reduction="none",
    ).sum(dim=-1)  # (B, K)
    loss_coord = (coord_err * valid_f).sum() / denom

    # 2. Spatial KL divergence against normalized target Gaussian heatmaps
    tgt_hm = render_gaussian_heatmaps(
        target_coords_norm, target_valid, size=h, sigma=sigma
    )
    tgt_dist = tgt_hm.view(bsz, k, h * w)
    tgt_dist = tgt_dist / tgt_dist.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    log_pred = F.log_softmax(hm_logits.view(bsz, k, h * w), dim=-1)
    kl_per_kp = F.kl_div(log_pred, tgt_dist, reduction="none").sum(dim=-1)
    loss_kl = (kl_per_kp * valid_f).sum() / denom

    # 3. Visibility BCE loss
    loss_vis = F.binary_cross_entropy_with_logits(vis_logits, valid_f)

    # 4. CalMS21 chiral orientation regularizer (penalizes inverted ear/hip chirality)
    loss_chiral = hm_logits.new_tensor(0.0)
    if enforce_calms_chirality and k >= 7:
        both_ears = valid_f[:, 0] * valid_f[:, 1] * valid_f[:, 2] * valid_f[:, 3]
        fwd = coords_pred[:, 0] - coords_pred[:, 3]
        el = coords_pred[:, 1] - coords_pred[:, 2]
        cross_ear = fwd[:, 0] * el[:, 1] - fwd[:, 1] * el[:, 0]
        pen_ear = F.relu(0.004 - cross_ear) * both_ears
        both_hips = valid_f[:, 0] * valid_f[:, 3] * valid_f[:, 4] * valid_f[:, 5]
        hl = coords_pred[:, 4] - coords_pred[:, 5]
        cross_hip = fwd[:, 0] * hl[:, 1] - fwd[:, 1] * hl[:, 0]
        pen_hip = F.relu(0.004 - cross_hip) * both_hips
        loss_chiral = (pen_ear.sum() + pen_hip.sum()) / (
            both_ears.sum() + both_hips.sum()
        ).clamp(min=1.0)

    total = loss_coord + 0.8 * loss_kl + 0.15 * loss_vis + 25.0 * loss_chiral
    return total, {
        "total": float(total.detach().item()),
        "coord": float(loss_coord.detach().item()),
        "kl": float(loss_kl.detach().item()),
        "vis": float(loss_vis.detach().item()),
        "chiral": float(loss_chiral.detach().item()),
    }


def save_cutie_pose_checkpoint(
    head: CutiePoseKeypointHead,
    path: str | Path,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    fp16: bool = True,
) -> Path:
    """Save compact Cutie-Pose head checkpoint (fp16 weights + metadata)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state: Dict[str, torch.Tensor] = {}
    for k, v in head.state_dict().items():
        t = v.detach().cpu()
        if fp16 and t.is_floating_point() and t.ndim >= 2:
            t = t.half()
        state[k] = t
    payload = {
        "format": "annolid_cutie_pose_head_v1",
        "config": asdict(head.cfg),
        "state_dict": state,
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, str(out_path))
    return out_path


def load_cutie_pose_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> Tuple[CutiePoseKeypointHead, Dict[str, Any]]:
    """Load a saved :class:`CutiePoseKeypointHead` checkpoint."""
    from annolid.utils.safe_torch_load import safe_torch_load

    raw = safe_torch_load(str(path), map_location="cpu")
    cfg_dict = dict(raw.get("config") or {})
    if "keypoint_names" in cfg_dict:
        cfg_dict["keypoint_names"] = tuple(cfg_dict["keypoint_names"])
    if "flip_idx" in cfg_dict:
        cfg_dict["flip_idx"] = tuple(int(x) for x in cfg_dict["flip_idx"])
    cfg = CutiePoseHeadConfig(**cfg_dict)
    head = CutiePoseKeypointHead(cfg)
    sd = {
        k: v.float() if v.is_floating_point() else v
        for k, v in raw["state_dict"].items()
    }
    head.load_state_dict(sd, strict=True)
    head.to(device)
    head.eval()
    return head, dict(raw.get("metadata") or {})
