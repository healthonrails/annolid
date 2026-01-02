from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn


@dataclass(frozen=True)
class DinoKPSEGCheckpointMeta:
    model_name: str
    short_side: int
    layers: tuple[int, ...]
    num_parts: int
    radius_px: float
    threshold: float
    in_dim: int
    hidden_dim: int
    keypoint_names: Optional[list[str]] = None
    flip_idx: Optional[list[int]] = None


class DinoKPSEGHead(nn.Module):
    """Small conv head mapping frozen DINO patch features -> per-keypoint logits."""

    def __init__(self, *, in_dim: int, hidden_dim: int, num_parts: int) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_parts = int(num_parts)

        self.net = nn.Sequential(
            nn.Conv2d(self.in_dim, self.hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.GroupNorm(num_groups=min(8, self.hidden_dim),
                         num_channels=self.hidden_dim),
            nn.Conv2d(self.hidden_dim, self.hidden_dim,
                      kernel_size=3, padding=1, groups=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.num_parts, kernel_size=1),
        )

    def forward(self, feats_bchw: torch.Tensor) -> torch.Tensor:
        if feats_bchw.ndim != 4:
            raise ValueError("Expected feats in BCHW format")
        return self.net(feats_bchw)


def checkpoint_pack(
    *,
    head: DinoKPSEGHead,
    meta: DinoKPSEGCheckpointMeta,
) -> Dict[str, object]:
    return {
        "format": "annolid.dino_kpseg.v1",
        "meta": {
            "model_name": meta.model_name,
            "short_side": meta.short_side,
            "layers": list(meta.layers),
            "num_parts": meta.num_parts,
            "radius_px": meta.radius_px,
            "threshold": meta.threshold,
            "in_dim": meta.in_dim,
            "hidden_dim": meta.hidden_dim,
            "keypoint_names": meta.keypoint_names,
            "flip_idx": meta.flip_idx,
        },
        "state_dict": head.state_dict(),
    }


def checkpoint_unpack(payload: Dict[str, object]) -> tuple[DinoKPSEGHead, DinoKPSEGCheckpointMeta]:
    fmt = payload.get("format")
    if fmt != "annolid.dino_kpseg.v1":
        raise ValueError(f"Unsupported checkpoint format: {fmt!r}")
    meta_raw = payload.get("meta") or {}
    if not isinstance(meta_raw, dict):
        raise ValueError("Invalid checkpoint meta")

    layers = meta_raw.get("layers") or []
    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    keypoint_names = meta_raw.get("keypoint_names")
    if keypoint_names is not None and not isinstance(keypoint_names, list):
        keypoint_names = None

    flip_idx = meta_raw.get("flip_idx")
    if flip_idx is not None:
        if not isinstance(flip_idx, list):
            flip_idx = None
        else:
            try:
                flip_idx = [int(v) for v in flip_idx]
            except Exception:
                flip_idx = None

    meta = DinoKPSEGCheckpointMeta(
        model_name=str(meta_raw.get("model_name")),
        short_side=int(meta_raw.get("short_side")),
        layers=tuple(int(x) for x in layers),
        num_parts=int(meta_raw.get("num_parts")),
        radius_px=float(meta_raw.get("radius_px")),
        threshold=float(meta_raw.get("threshold")),
        in_dim=int(meta_raw.get("in_dim")),
        hidden_dim=int(meta_raw.get("hidden_dim")),
        keypoint_names=keypoint_names,
        flip_idx=flip_idx,
    )

    head = DinoKPSEGHead(in_dim=meta.in_dim,
                         hidden_dim=meta.hidden_dim, num_parts=meta.num_parts)
    state = payload.get("state_dict") or {}
    if not isinstance(state, dict):
        raise ValueError("Invalid checkpoint state_dict")
    head.load_state_dict(state)
    return head, meta
