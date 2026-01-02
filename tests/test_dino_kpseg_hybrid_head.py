from __future__ import annotations

import torch

from annolid.segmentation.dino_kpseg.model import (
    DinoKPSEGCheckpointMeta,
    DinoKPSEGHybridHead,
    checkpoint_pack,
    checkpoint_unpack,
)


def test_hybrid_head_forward_shape() -> None:
    head = DinoKPSEGHybridHead(
        in_dim=64, hidden_dim=128, num_parts=5, num_heads=4, num_layers=2)
    x = torch.randn(2, 64, 7, 9)
    y = head(x)
    assert y.shape == (2, 5, 7, 9)


def test_hybrid_checkpoint_roundtrip() -> None:
    head = DinoKPSEGHybridHead(
        in_dim=32,
        hidden_dim=64,
        num_parts=3,
        num_heads=4,
        num_layers=1,
        orientation_anchor_idx=[0, 2],
    )
    meta = DinoKPSEGCheckpointMeta(
        model_name="dummy",
        short_side=128,
        layers=(-1,),
        num_parts=3,
        radius_px=6.0,
        threshold=0.4,
        in_dim=32,
        hidden_dim=64,
        keypoint_names=["l", "c", "r"],
        flip_idx=[2, 1, 0],
        head_type="hybrid",
        attn_heads=4,
        attn_layers=1,
        attn_dropout=0.0,
        attn_pos_scale=0.2,
        attn_token_norm=True,
        attn_proj_norm=True,
        attn_anchor_kv_norm=True,
        orientation_anchor_idx=[0, 2],
    )
    payload = checkpoint_pack(head=head, meta=meta)
    head2, meta2 = checkpoint_unpack(payload)
    assert meta2.head_type == "hybrid"
    y = head2(torch.randn(1, 32, 4, 4))
    assert y.shape == (1, 3, 4, 4)
