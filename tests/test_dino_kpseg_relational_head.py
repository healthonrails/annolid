from __future__ import annotations

import pytest
import torch

from annolid.segmentation.dino_kpseg.keypoints import infer_orientation_anchor_indices
from annolid.segmentation.dino_kpseg.model import (
    DinoKPSEGCheckpointMeta,
    DinoKPSEGRelationalHead,
    checkpoint_pack,
    checkpoint_unpack,
)


def test_relational_head_forward_shape() -> None:
    head = DinoKPSEGRelationalHead(
        in_dim=64,
        hidden_dim=128,
        num_parts=5,
        num_heads=4,
        num_layers=2,
    )
    x = torch.randn(2, 64, 7, 9)
    y = head(x)
    assert y.shape == (2, 5, 7, 9)
    assert bool(torch.isfinite(y).all().item())


def test_relational_head_key_padding_mask() -> None:
    head = DinoKPSEGRelationalHead(
        in_dim=32,
        hidden_dim=64,
        num_parts=3,
        num_heads=4,
        num_layers=1,
    )
    x = torch.randn(1, 32, 4, 4)
    kpm = torch.zeros((1, 16), dtype=torch.bool)
    kpm[:, :4] = True
    y = head(x, key_padding_mask=kpm)
    assert y.shape == (1, 3, 4, 4)
    assert torch.all(y[:, :, 0, :] <= -19.0)


def test_relational_checkpoint_roundtrip() -> None:
    head = DinoKPSEGRelationalHead(
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
        head_type="relational",
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
    assert meta2.head_type == "relational"
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


@pytest.mark.parametrize("legacy_head_type", ["attn", "hybrid", "videomt"])
def test_legacy_checkpoint_head_aliases_normalize_to_relational(
    legacy_head_type: str,
) -> None:
    head = DinoKPSEGRelationalHead(
        in_dim=16,
        hidden_dim=32,
        num_parts=2,
        num_heads=4,
        num_layers=1,
    )
    meta = DinoKPSEGCheckpointMeta(
        model_name="dummy",
        short_side=128,
        layers=(-1,),
        num_parts=2,
        radius_px=6.0,
        threshold=0.4,
        in_dim=16,
        hidden_dim=32,
        head_type=legacy_head_type,
    )
    payload = checkpoint_pack(head=head, meta=meta)
    head2, meta2 = checkpoint_unpack(payload)
    assert meta2.head_type == "relational"
    assert head2(torch.randn(1, 16, 4, 4)).shape == (1, 2, 4, 4)


def test_infer_orientation_anchor_indices() -> None:
    names = ["left_ear", "nose", "right_ear", "tailbase"]
    anchors = infer_orientation_anchor_indices(names, max_anchors=4)
    assert anchors == [1, 3]


def test_relational_head_accepts_orientation_anchors() -> None:
    head = DinoKPSEGRelationalHead(
        in_dim=64,
        hidden_dim=128,
        num_parts=6,
        num_heads=4,
        num_layers=2,
        orientation_anchor_idx=[1, 4],
    )
    y = head(torch.randn(1, 64, 5, 5))
    assert y.shape == (1, 6, 5, 5)
