from __future__ import annotations

import torch

from annolid.segmentation.dino_kpseg.model import (
    DinoKPSEGAttentionHead,
    DinoKPSEGCheckpointMeta,
    checkpoint_pack,
    checkpoint_unpack,
)
from annolid.segmentation.dino_kpseg.keypoints import infer_orientation_anchor_indices


def test_attention_head_forward_shape() -> None:
    head = DinoKPSEGAttentionHead(
        in_dim=64, hidden_dim=128, num_parts=5, num_heads=4, num_layers=2
    )
    x = torch.randn(2, 64, 7, 9)
    y = head(x)
    assert y.shape == (2, 5, 7, 9)


def test_attention_checkpoint_roundtrip() -> None:
    head = DinoKPSEGAttentionHead(
        in_dim=32, hidden_dim=64, num_parts=3, num_heads=4, num_layers=1
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
        head_type="attn",
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
    assert meta2.head_type == "attn"
    assert meta2.attn_heads == 4
    assert meta2.attn_layers == 1
    assert meta2.attn_dropout == 0.0
    assert abs(meta2.attn_pos_scale - 0.2) < 1e-6
    assert meta2.attn_token_norm is True
    assert meta2.attn_proj_norm is True
    assert meta2.attn_anchor_kv_norm is True
    assert meta2.orientation_anchor_idx == [0, 2]
    y = head2(torch.randn(1, 32, 4, 4))
    assert y.shape == (1, 3, 4, 4)


def test_infer_orientation_anchor_indices() -> None:
    names = ["left_ear", "nose", "right_ear", "tailbase"]
    anchors = infer_orientation_anchor_indices(names, max_anchors=4)
    assert anchors == [1, 3]


def test_attention_head_accepts_orientation_anchors() -> None:
    head = DinoKPSEGAttentionHead(
        in_dim=64,
        hidden_dim=128,
        num_parts=6,
        num_heads=4,
        num_layers=2,
        orientation_anchor_idx=[1, 4],
    )
    y = head(torch.randn(1, 64, 5, 5))
    assert y.shape == (1, 6, 5, 5)
