from __future__ import annotations

import torch

from annolid.segmentation.dino_kpseg.data import merge_feature_layers
from annolid.segmentation.dino_kpseg.model import (
    DinoKPSEGAlignedHead,
    DinoKPSEGCheckpointMeta,
    DinoKPSEGHead,
    DinoKPSEGMultiTaskHead,
    checkpoint_pack,
    checkpoint_unpack,
)


def test_merge_feature_layers_modes() -> None:
    feats = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).view(2, 3, 2, 2)
    out_concat = merge_feature_layers(feats, mode="concat")
    out_mean = merge_feature_layers(feats, mode="mean")
    out_max = merge_feature_layers(feats, mode="max")

    assert out_concat.shape == (6, 2, 2)
    assert out_mean.shape == (3, 2, 2)
    assert out_max.shape == (3, 2, 2)
    assert torch.allclose(out_mean, feats.mean(dim=0))
    assert torch.allclose(out_max, feats.max(dim=0).values)


def test_checkpoint_roundtrip_preserves_feature_merge() -> None:
    head = DinoKPSEGHead(in_dim=8, hidden_dim=16, num_parts=3)
    meta = DinoKPSEGCheckpointMeta(
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        short_side=768,
        layers=(-2, -1),
        num_parts=3,
        radius_px=5.0,
        threshold=0.2,
        in_dim=8,
        hidden_dim=16,
        feature_merge="mean",
    )
    payload = checkpoint_pack(head=head, meta=meta)
    _, meta_out = checkpoint_unpack(payload)
    assert meta_out.feature_merge == "mean"


def test_aligned_head_checkpoint_roundtrip() -> None:
    base = DinoKPSEGHead(in_dim=8, hidden_dim=16, num_parts=3)
    head = DinoKPSEGAlignedHead(base_head=base, in_dim=12, feature_dim=8)
    x = torch.randn(2, 12, 4, 5)
    y = head(x)
    assert y.shape == (2, 3, 4, 5)

    meta = DinoKPSEGCheckpointMeta(
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        short_side=768,
        layers=(-2, -1),
        num_parts=3,
        radius_px=5.0,
        threshold=0.2,
        in_dim=12,
        hidden_dim=16,
        feature_align_dim=8,
    )
    payload = checkpoint_pack(head=head, meta=meta)
    restored_head, restored_meta = checkpoint_unpack(payload)
    assert restored_meta.feature_align_dim == 8
    out = restored_head(x)
    assert out.shape == (2, 3, 4, 5)


def test_multitask_head_outputs_and_checkpoint_roundtrip() -> None:
    head = DinoKPSEGMultiTaskHead(in_dim=10, hidden_dim=16, num_parts=4)
    x = torch.randn(2, 10, 6, 7)
    out = head.forward_all(x)
    assert out["kpt_logits"].shape == (2, 4, 6, 7)
    assert out["obj_logits"].shape == (2, 1, 6, 7)
    assert out["box_logits"].shape == (2, 4, 6, 7)
    assert out["inst_logits"].shape == (2, 1, 6, 7)
    assert head(x).shape == (2, 4, 6, 7)

    meta = DinoKPSEGCheckpointMeta(
        model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
        short_side=768,
        layers=(-1,),
        num_parts=4,
        radius_px=5.0,
        threshold=0.2,
        in_dim=10,
        hidden_dim=16,
        head_type="multitask",
        multitask=True,
    )
    payload = checkpoint_pack(head=head, meta=meta)
    restored_head, restored_meta = checkpoint_unpack(payload)
    assert restored_meta.head_type == "multitask"
    out2 = restored_head.forward_all(x)
    assert out2["kpt_logits"].shape == (2, 4, 6, 7)
