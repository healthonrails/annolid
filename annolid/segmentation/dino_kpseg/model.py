from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


def _groupnorm_groups(num_channels: int, *, max_groups: int = 8) -> int:
    num_channels = int(num_channels)
    max_groups = max(1, int(max_groups))
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


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
    head_type: str = "conv"
    attn_heads: int = 4
    attn_layers: int = 1
    attn_dropout: float = 0.0
    attn_pos_scale: float = 0.2
    attn_token_norm: bool = True
    attn_proj_norm: bool = True
    attn_anchor_kv_norm: bool = True
    orientation_anchor_idx: Optional[list[int]] = None


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
            nn.GroupNorm(
                num_groups=_groupnorm_groups(self.hidden_dim, max_groups=8),
                num_channels=self.hidden_dim,
            ),
            nn.Conv2d(
                self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, groups=1
            ),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.num_parts, kernel_size=1),
        )

    def forward(self, feats_bchw: torch.Tensor) -> torch.Tensor:
        if feats_bchw.ndim != 4:
            raise ValueError("Expected feats in BCHW format")
        return self.net(feats_bchw)


def _positional_encoding_2d_sincos(
    *,
    h: int,
    w: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return [1, HW, dim] 2D sine/cosine positional embedding."""
    if int(dim) % 4 != 0:
        raise ValueError("positional embedding dim must be divisible by 4")
    h = int(h)
    w = int(w)
    dim = int(dim)

    y = torch.arange(h, device=device, dtype=dtype)
    x = torch.arange(w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    yy = yy.reshape(-1)  # [HW]
    xx = xx.reshape(-1)  # [HW]

    half = dim // 2
    quarter = half // 2
    omega = torch.arange(quarter, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / max(1.0, float(quarter))))

    y_out = yy[:, None] * omega[None, :]  # [HW, quarter]
    x_out = xx[:, None] * omega[None, :]  # [HW, quarter]

    pos = torch.cat(
        [torch.sin(y_out), torch.cos(y_out), torch.sin(x_out), torch.cos(x_out)],
        dim=1,
    )
    return pos[None, :, :].to(dtype=dtype)


class _FFN(nn.Module):
    def __init__(
        self, dim: int, *, mlp_ratio: float = 4.0, dropout: float = 0.0
    ) -> None:
        super().__init__()
        hidden = max(1, int(float(dim) * float(mlp_ratio)))
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, dim),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DinoKPSEGAttentionHead(nn.Module):
    """Attention-based head that models keypoint relationships."""

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        num_parts: int,
        num_heads: int = 4,
        num_layers: int = 1,
        orientation_anchor_idx: Optional[list[int]] = None,
        dropout: float = 0.0,
        pos_scale: float = 0.2,
        token_norm: bool = True,
        proj_norm: bool = True,
        anchor_kv_norm: bool = True,
        anchor_gate_init: float = -4.0,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_parts = int(num_parts)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.pos_scale = float(pos_scale)
        self.token_norm = bool(token_norm)
        self.proj_norm = bool(proj_norm)
        self.anchor_kv_norm = bool(anchor_kv_norm)

        if self.hidden_dim % max(1, self.num_heads) != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.hidden_dim % 4 != 0:
            raise ValueError(
                "hidden_dim must be divisible by 4 for sin/cos positional encoding"
            )

        self.proj = nn.Conv2d(self.in_dim, self.hidden_dim, kernel_size=1)
        self.proj_act = nn.GELU()
        self.proj_gn = nn.GroupNorm(
            num_groups=_groupnorm_groups(self.hidden_dim, max_groups=8),
            num_channels=self.hidden_dim,
            affine=False,
        )
        self.query_embed = nn.Parameter(
            torch.randn(self.num_parts, self.hidden_dim) * 0.02
        )

        self.cross_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    self.hidden_dim,
                    self.num_heads,
                    dropout=float(dropout),
                    batch_first=True,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.self_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    self.hidden_dim,
                    self.num_heads,
                    dropout=float(dropout),
                    batch_first=True,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm_q1 = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)]
        )
        self.norm_q2 = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)]
        )
        self.norm_q3 = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)]
        )
        self.ffn = nn.ModuleList(
            [
                _FFN(self.hidden_dim, dropout=float(dropout))
                for _ in range(self.num_layers)
            ]
        )

        self.logit_scale = nn.Parameter(torch.tensor(0.0))

        anchors: list[int] = []
        if orientation_anchor_idx:
            for idx in orientation_anchor_idx:
                try:
                    idx_i = int(idx)
                except Exception:
                    continue
                if 0 <= idx_i < self.num_parts and idx_i not in anchors:
                    anchors.append(idx_i)
        self.orientation_anchor_idx = anchors

        self.anchor_attn = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    self.hidden_dim,
                    self.num_heads,
                    dropout=float(dropout),
                    batch_first=True,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.norm_q_anchor = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)]
        )
        self.anchor_gate = nn.Parameter(
            torch.full((self.num_layers,), float(anchor_gate_init), dtype=torch.float32)
        )

    def forward(
        self,
        feats_bchw: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if feats_bchw.ndim != 4:
            raise ValueError("Expected feats in BCHW format")
        b, _, h, w = feats_bchw.shape
        if key_padding_mask is not None:
            if not isinstance(key_padding_mask, torch.Tensor):
                raise ValueError("key_padding_mask must be a torch.Tensor")
            key_padding_mask = key_padding_mask.to(dtype=torch.bool)
            if (
                key_padding_mask.ndim != 2
                or key_padding_mask.shape[0] != b
                or key_padding_mask.shape[1] != (h * w)
            ):
                raise ValueError("key_padding_mask must be shaped [B, H*W]")

        tokens = self.proj(feats_bchw)  # [B, D, H, W]
        if self.proj_norm:
            tokens = self.proj_act(tokens)
            tokens = self.proj_gn(tokens)
        tokens = tokens.flatten(2).transpose(1, 2)  # [B, HW, D]
        if self.pos_scale != 0.0:
            tokens = tokens + _positional_encoding_2d_sincos(
                h=h, w=w, dim=self.hidden_dim, device=tokens.device, dtype=tokens.dtype
            ) * float(self.pos_scale)
        if self.token_norm:
            tokens = F.layer_norm(tokens, (self.hidden_dim,))

        q = self.query_embed[None, :, :].expand(b, -1, -1)  # [B, K, D]

        for layer in range(self.num_layers):
            q_norm = self.norm_q1[layer](q)
            attn_out, _ = self.cross_attn[layer](
                q_norm,
                tokens,
                tokens,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )
            q = q + attn_out

            # Cross-attention from all keypoints to orientation anchors (if available).
            if self.orientation_anchor_idx:
                q_anchor = self.norm_q_anchor[layer](q)
                if self.anchor_kv_norm:
                    # [B, A, D]
                    anchors = q_anchor[:, self.orientation_anchor_idx, :]
                else:
                    anchors = q[:, self.orientation_anchor_idx, :]  # [B, A, D]
                anchor_out, _ = self.anchor_attn[layer](
                    q_anchor, anchors, anchors, need_weights=False
                )
                gate = torch.sigmoid(self.anchor_gate[layer]).to(dtype=q.dtype)
                q = q + anchor_out * gate

            q_norm = self.norm_q2[layer](q)
            attn_out, _ = self.self_attn[layer](
                q_norm, q_norm, q_norm, need_weights=False
            )
            q = q + attn_out

            q_norm = self.norm_q3[layer](q)
            q = q + self.ffn[layer](q_norm)

        q = F.layer_norm(q, (self.hidden_dim,))
        scale = torch.clamp(self.logit_scale.exp(), 0.1, 10.0)
        logits_flat = torch.matmul(q, tokens.transpose(1, 2)) * (
            scale / (self.hidden_dim**0.5)
        )
        logits = logits_flat.view(b, self.num_parts, h, w)
        if key_padding_mask is not None:
            mask_hw = key_padding_mask.view(b, h, w)
            logits = logits.masked_fill(mask_hw[:, None, :, :], -20.0)
        return logits


class DinoKPSEGHybridHead(nn.Module):
    """Hybrid head: conv locality + attention relations.

    The conv branch provides strong local inductive bias for tight keypoint masks,
    while the attention branch can learn cross-keypoint constraints (e.g., symmetry).
    Outputs are combined by a learnable per-keypoint mixing weight (initialized to
    favor the conv branch).
    """

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        num_parts: int,
        num_heads: int = 4,
        num_layers: int = 1,
        orientation_anchor_idx: Optional[list[int]] = None,
        dropout: float = 0.0,
        pos_scale: float = 0.2,
        token_norm: bool = True,
        proj_norm: bool = True,
        anchor_kv_norm: bool = True,
        mix_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_parts = int(num_parts)
        self.num_heads = int(num_heads)
        self.num_layers = int(num_layers)

        self.conv = DinoKPSEGHead(
            in_dim=self.in_dim, hidden_dim=self.hidden_dim, num_parts=self.num_parts
        )
        self.attn = DinoKPSEGAttentionHead(
            in_dim=self.in_dim,
            hidden_dim=self.hidden_dim,
            num_parts=self.num_parts,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            orientation_anchor_idx=orientation_anchor_idx,
            dropout=float(dropout),
            pos_scale=float(pos_scale),
            token_norm=bool(token_norm),
            proj_norm=bool(proj_norm),
            anchor_kv_norm=bool(anchor_kv_norm),
        )
        self.mix_logit = nn.Parameter(
            torch.full((self.num_parts,), float(mix_init), dtype=torch.float32)
        )

    def forward(
        self,
        feats_bchw: torch.Tensor,
        *,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits_conv = self.conv(feats_bchw)
        logits_attn = self.attn(feats_bchw, key_padding_mask=key_padding_mask)
        mix = torch.sigmoid(self.mix_logit).to(dtype=logits_conv.dtype)[
            None, :, None, None
        ]
        logits = logits_conv * (1.0 - mix) + logits_attn * mix
        if key_padding_mask is not None:
            if feats_bchw.ndim != 4:
                return logits
            b, _, h, w = feats_bchw.shape
            mask_hw = key_padding_mask.to(dtype=torch.bool).view(b, h, w)
            logits = logits.masked_fill(mask_hw[:, None, :, :], -20.0)
        return logits


def checkpoint_pack(
    *,
    head: nn.Module,
    meta: DinoKPSEGCheckpointMeta,
) -> Dict[str, object]:
    return {
        "format": "annolid.dino_kpseg.v2",
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
            "head_type": meta.head_type,
            "attn_heads": int(meta.attn_heads),
            "attn_layers": int(meta.attn_layers),
            "attn_dropout": float(meta.attn_dropout),
            "attn_pos_scale": float(meta.attn_pos_scale),
            "attn_token_norm": bool(meta.attn_token_norm),
            "attn_proj_norm": bool(meta.attn_proj_norm),
            "attn_anchor_kv_norm": bool(meta.attn_anchor_kv_norm),
            "orientation_anchor_idx": meta.orientation_anchor_idx,
        },
        "state_dict": head.state_dict(),
    }


def checkpoint_unpack(
    payload: Dict[str, object],
) -> tuple[nn.Module, DinoKPSEGCheckpointMeta]:
    fmt = payload.get("format")
    if fmt not in ("annolid.dino_kpseg.v1", "annolid.dino_kpseg.v2"):
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

    head_type = (
        str(
            meta_raw.get("head_type")
            or ("conv" if fmt == "annolid.dino_kpseg.v1" else "conv")
        )
        .strip()
        .lower()
    )
    if head_type not in ("conv", "attn", "hybrid"):
        head_type = "conv"

    try:
        attn_heads = int(meta_raw.get("attn_heads") or 4)
    except Exception:
        attn_heads = 4
    try:
        attn_layers = int(meta_raw.get("attn_layers") or 1)
    except Exception:
        attn_layers = 1

    attn_dropout = meta_raw.get("attn_dropout")
    if attn_dropout is None:
        attn_dropout = 0.0 if fmt == "annolid.dino_kpseg.v1" else 0.0
    try:
        attn_dropout = float(attn_dropout)
    except Exception:
        attn_dropout = 0.0

    attn_pos_scale = meta_raw.get("attn_pos_scale")
    if attn_pos_scale is None:
        attn_pos_scale = 1.0
    try:
        attn_pos_scale = float(attn_pos_scale)
    except Exception:
        attn_pos_scale = 1.0

    def _parse_bool(value: object, *, default: bool) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return bool(default)
        if isinstance(value, (int, float)):
            return bool(int(value) != 0)
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return bool(default)

    # Newer (v2+) optional attention head behavior flags; default to the legacy behavior
    # when missing to preserve old checkpoint outputs.
    attn_token_norm = _parse_bool(meta_raw.get("attn_token_norm"), default=False)
    attn_proj_norm = _parse_bool(meta_raw.get("attn_proj_norm"), default=False)
    attn_anchor_kv_norm = _parse_bool(
        meta_raw.get("attn_anchor_kv_norm"), default=False
    )

    orientation_anchor_idx = meta_raw.get("orientation_anchor_idx")
    if orientation_anchor_idx is not None:
        if not isinstance(orientation_anchor_idx, list):
            orientation_anchor_idx = None
        else:
            try:
                orientation_anchor_idx = [int(v) for v in orientation_anchor_idx]
            except Exception:
                orientation_anchor_idx = None

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
        head_type=head_type,
        attn_heads=int(attn_heads),
        attn_layers=int(attn_layers),
        attn_dropout=float(attn_dropout),
        attn_pos_scale=float(attn_pos_scale),
        attn_token_norm=bool(attn_token_norm),
        attn_proj_norm=bool(attn_proj_norm),
        attn_anchor_kv_norm=bool(attn_anchor_kv_norm),
        orientation_anchor_idx=orientation_anchor_idx,
    )

    if meta.head_type == "attn":
        head: nn.Module = DinoKPSEGAttentionHead(
            in_dim=meta.in_dim,
            hidden_dim=meta.hidden_dim,
            num_parts=meta.num_parts,
            num_heads=int(meta.attn_heads),
            num_layers=int(meta.attn_layers),
            orientation_anchor_idx=meta.orientation_anchor_idx,
            dropout=float(meta.attn_dropout),
            pos_scale=float(meta.attn_pos_scale),
            token_norm=bool(meta.attn_token_norm),
            proj_norm=bool(meta.attn_proj_norm),
            anchor_kv_norm=bool(meta.attn_anchor_kv_norm),
        )
    elif meta.head_type == "hybrid":
        head = DinoKPSEGHybridHead(
            in_dim=meta.in_dim,
            hidden_dim=meta.hidden_dim,
            num_parts=meta.num_parts,
            num_heads=int(meta.attn_heads),
            num_layers=int(meta.attn_layers),
            orientation_anchor_idx=meta.orientation_anchor_idx,
            dropout=float(meta.attn_dropout),
            pos_scale=float(meta.attn_pos_scale),
            token_norm=bool(meta.attn_token_norm),
            proj_norm=bool(meta.attn_proj_norm),
            anchor_kv_norm=bool(meta.attn_anchor_kv_norm),
        )
    else:
        head = DinoKPSEGHead(
            in_dim=meta.in_dim, hidden_dim=meta.hidden_dim, num_parts=meta.num_parts
        )

    state = payload.get("state_dict") or {}
    if not isinstance(state, dict):
        raise ValueError("Invalid checkpoint state_dict")
    head.load_state_dict(state)
    return head, meta
