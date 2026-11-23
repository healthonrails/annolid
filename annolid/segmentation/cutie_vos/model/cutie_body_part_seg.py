"""Lightweight Tucker-2 temporal semantic body-part segmentation head for Compressed CUTIE.

Branches off Compressed CUTIE's per-object decoder features (``p4`` stride 4,
``p8`` stride 8, tracked object mask) + canonical object-centered ROI crop and
recurrent previous-frame body-part probability memory to predict dense 6-class
anatomical part masks: ``{0: background, 1: head, 2: snout, 3: ear, 4: torso, 5: tail}``.

Predicted part probability maps feed directly into
:func:`annolid.postprocessing.body_part_to_pose.extract_pose_from_part_masks`
for segmentation-driven pose estimation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from annolid.data.body_part_dataset import SEMANTIC_PART_NAMES
from annolid.segmentation.cutie_vos.model.cutie_pose import _TuckerResBlock


@dataclass(frozen=True)
class CutieBodyPartSegConfig:
    """Configuration for :class:`CutieBodyPartSegHead`."""

    num_classes: int = len(SEMANTIC_PART_NAMES)  # 6: bg, head, snout, ear, torso, tail
    class_names: Tuple[str, ...] = SEMANTIC_PART_NAMES
    c_p4: int = 128
    c_p8: int = 128
    hidden_dim: int = 96
    tucker_rank: int = 48
    roi_size: int = 64
    feat_size: int = 32
    roi_pad_scale: float = 1.50
    min_roi_px: float = 64.0
    img_dropout_p: float = 0.20


class CutieBodyPartSegHead(nn.Module):
    """Tucker-2 factorized temporal semantic body-part decoder (~285K params)."""

    def __init__(self, cfg: Optional[CutieBodyPartSegConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or CutieBodyPartSegConfig()
        c_hid = int(self.cfg.hidden_dim)
        rank = int(self.cfg.tucker_rank)
        n_cls = int(self.cfg.num_classes)

        self.cutie_proj = nn.Sequential(
            nn.Conv2d(self.cfg.c_p4 + self.cfg.c_p8, c_hid, kernel_size=1, bias=False),
            nn.GroupNorm(8, c_hid),
            nn.SiLU(inplace=True),
        )
        self.cutie_blocks = nn.Sequential(
            _TuckerResBlock(c_hid, rank, dilation=1),
            _TuckerResBlock(c_hid, rank, dilation=2),
        )
        self.img_stem = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(4, 24),
            nn.SiLU(inplace=True),
        )
        self.prior_stem = nn.Sequential(
            nn.Conv2d(1 + n_cls, 24, kernel_size=3, padding=1, bias=False),
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
        self.seg_head = nn.Conv2d(c_hid, n_cls, kernel_size=1)

    def forward(
        self,
        p4_roi: torch.Tensor,
        p8_roi: torch.Tensor,
        mask_roi: torch.Tensor,
        img_roi: torch.Tensor,
        prev_part_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return per-pixel class logits ``(B, num_classes, roi_size, roi_size)``."""
        bsz = p4_roi.shape[0]
        r = int(self.cfg.roi_size)
        n_cls = int(self.cfg.num_classes)

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
        if prev_part_probs is None:
            prev_part_probs = torch.zeros(
                (bsz, n_cls, r, r), device=p4_roi.device, dtype=p4_roi.dtype
            )
        elif prev_part_probs.shape[-2:] != (r, r):
            prev_part_probs = F.interpolate(
                prev_part_probs, size=(r, r), mode="bilinear", align_corners=False
            )

        img_feat = self.img_stem(img_roi)
        if self.training and self.cfg.img_dropout_p > 0.0:
            keep = (
                torch.rand((bsz, 1, 1, 1), device=img_feat.device)
                >= self.cfg.img_dropout_p
            ).to(img_feat.dtype)
            img_feat = img_feat * keep / max(1e-6, 1.0 - self.cfg.img_dropout_p)

        prior_feat = self.prior_stem(torch.cat([mask_roi, prev_part_probs], dim=1))
        fused = self.fuse(torch.cat([c_feat, img_feat, prior_feat], dim=1))
        return self.seg_head(self.refine(fused))


def compute_body_part_seg_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    class_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Weighted Cross-Entropy + Multi-Class Soft Dice Loss (upweighting small snout/ear/tail parts)."""
    n_cls = logits.shape[1]
    if class_weights is None:
        # Upweight small high-value pose landmarks: snout (2), ear (3), tail (5)
        w = torch.tensor(
            [0.35, 1.2, 2.6, 2.8, 1.0, 2.2], device=logits.device, dtype=torch.float32
        )
        if n_cls != w.numel():
            w = torch.ones(n_cls, device=logits.device, dtype=torch.float32)
        class_weights = w

    loss_ce = F.cross_entropy(logits, target_ids.long(), weight=class_weights)

    probs = F.softmax(logits, dim=1)
    tgt_oh = (
        F.one_hot(target_ids.long().clamp(0, n_cls - 1), num_classes=n_cls)
        .permute(0, 3, 1, 2)
        .float()
    )
    # Soft Dice over foreground body-part classes (1..n_cls-1)
    inter = (probs[:, 1:] * tgt_oh[:, 1:]).sum(dim=(0, 2, 3))
    denom = probs[:, 1:].sum(dim=(0, 2, 3)) + tgt_oh[:, 1:].sum(dim=(0, 2, 3))
    dice = (2.0 * inter + 1.0) / (denom + 1.0)
    loss_dice = (1.0 - dice).mean()

    total = loss_ce + 1.2 * loss_dice
    return total, {
        "total": float(total.detach().item()),
        "ce": float(loss_ce.detach().item()),
        "dice": float(loss_dice.detach().item()),
        "fg_dice": float(dice.mean().detach().item()),
    }


def save_cutie_body_part_seg_checkpoint(
    head: CutieBodyPartSegHead,
    path: str | Path,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    fp16: bool = True,
) -> Path:
    """Save compact fp16 Cutie-BodyPartSeg head checkpoint."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    state: Dict[str, torch.Tensor] = {}
    for k, v in head.state_dict().items():
        t = v.detach().cpu()
        if fp16 and t.is_floating_point() and t.ndim >= 2:
            t = t.half()
        state[k] = t
    torch.save(
        {
            "format": "annolid_cutie_body_part_seg_v1",
            "config": asdict(head.cfg),
            "state_dict": state,
            "metadata": dict(metadata or {}),
        },
        str(out_path),
    )
    return out_path


def load_cutie_body_part_seg_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> Tuple[CutieBodyPartSegHead, Dict[str, Any]]:
    """Load a saved :class:`CutieBodyPartSegHead` checkpoint."""
    raw = torch.load(str(path), map_location="cpu", weights_only=False)
    cfg_dict = dict(raw.get("config") or {})
    if "class_names" in cfg_dict:
        cfg_dict["class_names"] = tuple(cfg_dict["class_names"])
    cfg = CutieBodyPartSegConfig(**cfg_dict)
    head = CutieBodyPartSegHead(cfg)
    sd = {
        k: v.float() if v.is_floating_point() else v
        for k, v in raw["state_dict"].items()
    }
    head.load_state_dict(sd, strict=True)
    head.to(device)
    head.eval()
    return head, dict(raw.get("metadata") or {})
