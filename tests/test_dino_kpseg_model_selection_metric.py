import pytest
import argparse
import torch
import numpy as np


from annolid.segmentation.dino_kpseg.train import (
    _canonicalize_lr_gt_instances,
    _canonicalize_lr_supervision_tensors,
    _should_enable_lr_canonicalize,
    _apply_data_aware_augmentation_defaults,
    _apply_data_aware_small_data_profile,
    _apply_schedule_profile_defaults,
    _compute_no_aug_start_epoch,
    _ema_update_,
    _is_epoch_augment_enabled,
    _metric_higher_is_better,
    _parse_weight_list,
    _resolve_feature_align_dim,
    _resolve_selection_metric,
    _canonicalize_head_type,
    _soft_argmax_coords_from_logits_batched,
    _argmax_coords_batched,
    _coord_loss,
)
from annolid.segmentation.dino_kpseg.data import DinoKPSEGAugmentConfig
from annolid.segmentation.dino_kpseg import defaults as dino_defaults


def test_metric_direction_is_inferred() -> None:
    assert _metric_higher_is_better("pck@8px") is True
    assert _metric_higher_is_better("pck_weighted") is True
    assert _metric_higher_is_better("val_loss") is False
    assert _metric_higher_is_better("train_loss") is False


def test_resolve_metric_pck8px() -> None:
    pck_vals = {"8.0": 0.5}
    out = _resolve_selection_metric(
        "pck@8px",
        train_loss=3.0,
        val_loss=9.0,
        pck_vals=pck_vals,
        pck_weighted_weights=(1.0, 1.0, 1.0, 1.0),
    )
    assert out == pytest.approx(0.5)


def test_resolve_metric_weighted_pck() -> None:
    pck_vals = {"2.0": 0.0, "4.0": 0.5, "8.0": 1.0, "16.0": 1.0}
    out = _resolve_selection_metric(
        "pck_weighted",
        train_loss=3.0,
        val_loss=9.0,
        pck_vals=pck_vals,
        pck_weighted_weights=(1.0, 1.0, 1.0, 1.0),
    )
    assert out == pytest.approx((0.0 + 0.5 + 1.0 + 1.0) / 4.0)

    out2 = _resolve_selection_metric(
        "pck_weighted",
        train_loss=3.0,
        val_loss=9.0,
        pck_vals=pck_vals,
        pck_weighted_weights=(0.0, 0.0, 1.0, 0.0),
    )
    assert out2 == pytest.approx(1.0)


def test_parse_pck_weights() -> None:
    assert _parse_weight_list("0,0,1,0", n=4) == (0.0, 0.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        _parse_weight_list("1,2,3", n=4)


def test_compute_no_aug_start_epoch() -> None:
    assert _compute_no_aug_start_epoch(epochs=132, no_aug_epoch=12) == 121
    assert _compute_no_aug_start_epoch(epochs=10, no_aug_epoch=0) is None


def test_epoch_augment_enabled_with_windows_and_tail() -> None:
    assert (
        _is_epoch_augment_enabled(
            epoch=3,
            epochs=132,
            augment_enabled=True,
            aug_start_epoch=4,
            aug_stop_epoch=120,
            no_aug_epoch=12,
        )
        is False
    )
    assert (
        _is_epoch_augment_enabled(
            epoch=64,
            epochs=132,
            augment_enabled=True,
            aug_start_epoch=4,
            aug_stop_epoch=120,
            no_aug_epoch=12,
        )
        is True
    )
    assert (
        _is_epoch_augment_enabled(
            epoch=121,
            epochs=132,
            augment_enabled=True,
            aug_start_epoch=4,
            aug_stop_epoch=132,
            no_aug_epoch=12,
        )
        is False
    )


def test_apply_schedule_profile_defaults_aggressive_s() -> None:
    args = argparse.Namespace(
        schedule_profile="aggressive_s",
        epochs=120,
        warmup_epochs=3,
        flat_epoch=0,
        aug_start_epoch=1,
        aug_stop_epoch=0,
        no_aug_epoch=0,
        change_matcher=False,
        matcher_change_epoch=0,
        iou_order_alpha=0.0,
    )
    _apply_schedule_profile_defaults(args)
    assert args.epochs == 132
    assert args.warmup_epochs == 4
    assert args.flat_epoch == 64
    assert args.aug_start_epoch == 4
    assert args.aug_stop_epoch == 120
    assert args.no_aug_epoch == 12
    assert args.change_matcher is True
    assert args.matcher_change_epoch == 100
    assert args.iou_order_alpha == pytest.approx(4.0)


def test_resolve_feature_align_dim_auto_and_numeric() -> None:
    assert _resolve_feature_align_dim("auto", in_dim=768, hidden_dim=192) == 192
    assert _resolve_feature_align_dim("auto", in_dim=128, hidden_dim=192) == 128
    assert _resolve_feature_align_dim("256", in_dim=768, hidden_dim=192) == 256
    assert _resolve_feature_align_dim(0, in_dim=768, hidden_dim=192) == 0
    assert _resolve_feature_align_dim("off", in_dim=768, hidden_dim=192) == 0


def test_resolve_feature_align_dim_invalid_raises() -> None:
    with pytest.raises(ValueError):
        _resolve_feature_align_dim("abc", in_dim=768, hidden_dim=192)


def test_ema_update_moves_toward_model_weights() -> None:
    model = torch.nn.Linear(4, 2, bias=False)
    ema = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(2.0)
        ema.weight.fill_(0.0)
    _ema_update_(ema, model, decay=0.5)
    assert torch.allclose(ema.weight, torch.full_like(ema.weight, 1.0))


def test_data_aware_augmentation_small_ambiguous_dataset_disables_geometry() -> None:
    cfg = DinoKPSEGAugmentConfig(
        enabled=True,
        hflip_prob=0.25,
        degrees=3.0,
        translate=0.01,
        scale=0.03,
        brightness=0.03,
        contrast=0.03,
        saturation=0.02,
    )
    out = _apply_data_aware_augmentation_defaults(
        ambiguous_frac=0.59,
        augment_cfg=cfg,
        num_train_images=42,
    )
    # hflip is suppressed for ambiguous datasets
    assert out.hflip_prob == pytest.approx(0.0)
    # Geometric augmentation is preserved (critical for small datasets)
    assert out.degrees == pytest.approx(cfg.degrees)
    assert out.translate == pytest.approx(cfg.translate)
    assert out.scale == pytest.approx(cfg.scale)
    assert out.brightness == pytest.approx(cfg.brightness)
    assert out.contrast == pytest.approx(cfg.contrast)
    assert out.saturation == pytest.approx(cfg.saturation)


def test_small_data_profile_adjusts_only_defaults() -> None:
    head, feat_align, lr, patience = _apply_data_aware_small_data_profile(
        num_train_images=42,
        ambiguous_frac=0.59,
        head_type=str(dino_defaults.HEAD_TYPE),
        feature_align_dim=0,
        lr=3e-4,
        early_stop_patience=20,
    )
    # Head type is preserved (relational head kept for small datasets)
    assert head == str(dino_defaults.HEAD_TYPE).strip().lower()
    assert str(feat_align).lower() == "auto"
    # LR is preserved (no longer reduced for small datasets)
    assert lr == pytest.approx(3e-4)
    assert patience == 12

    custom_head, custom_feat_align, custom_lr, custom_patience = (
        _apply_data_aware_small_data_profile(
            num_train_images=42,
            ambiguous_frac=0.59,
            head_type="multitask",
            feature_align_dim=128,
            lr=2e-4,
            early_stop_patience=8,
        )
    )
    assert custom_head == "multitask"
    assert int(custom_feat_align) == 128
    assert custom_lr == pytest.approx(2e-4)
    assert custom_patience == 8


def test_should_enable_lr_canonicalize_auto_and_overrides() -> None:
    assert (
        _should_enable_lr_canonicalize(
            requested=None,
            swap_ambiguity_frac=0.60,
            num_train_images=42,
            lr_pairs=[(0, 2)],
            orientation_anchor_idx=[1, 3],
        )
        is True
    )
    assert (
        _should_enable_lr_canonicalize(
            requested=None,
            swap_ambiguity_frac=0.30,
            num_train_images=42,
            lr_pairs=[(0, 2)],
            orientation_anchor_idx=[1, 3],
        )
        is False
    )


def test_canonicalize_head_type_normalizes_text() -> None:
    assert _canonicalize_head_type("  RELATIONAL ") == "relational"
    assert _canonicalize_head_type("relational") == "relational"
    assert _canonicalize_head_type("conv") == "conv"
    assert _canonicalize_head_type("multitask") == "multitask"
    assert (
        _should_enable_lr_canonicalize(
            requested=True,
            swap_ambiguity_frac=0.0,
            num_train_images=999,
            lr_pairs=[],
            orientation_anchor_idx=[],
        )
        is True
    )
    assert (
        _should_enable_lr_canonicalize(
            requested=False,
            swap_ambiguity_frac=0.99,
            num_train_images=1,
            lr_pairs=[(0, 2)],
            orientation_anchor_idx=[1, 3],
        )
        is False
    )


def test_canonicalize_lr_supervision_swaps_inverted_pair() -> None:
    masks = torch.zeros((1, 4, 2, 2), dtype=torch.float32)
    masks[0, 0, 0, 0] = 1.0
    masks[0, 2, 1, 1] = 1.0
    coords = torch.tensor(
        [[[9.0, 1.0], [5.0, 5.0], [1.0, 1.0], [5.0, 9.0]]], dtype=torch.float32
    )
    coord_mask = torch.ones((1, 4), dtype=torch.float32)

    out_masks, out_coords, out_coord_mask, swaps = _canonicalize_lr_supervision_tensors(
        masks=masks,
        coords=coords,
        coord_mask=coord_mask,
        lr_pairs=[(0, 2)],
        orientation_anchor_idx=[1, 3],
        min_side_diff_px=0.0,
    )
    assert swaps == 1
    assert torch.allclose(out_masks[0, 0], masks[0, 2])
    assert torch.allclose(out_masks[0, 2], masks[0, 0])
    assert torch.allclose(out_coords[0, 0], coords[0, 2])
    assert torch.allclose(out_coords[0, 2], coords[0, 0])
    assert torch.allclose(out_coord_mask, coord_mask)


def test_canonicalize_lr_gt_instances_swaps_inverted_pair() -> None:
    gt = np.asarray(
        [
            [0.9, 0.1, 2.0],  # left (inverted)
            [0.5, 0.5, 2.0],  # anchor 0
            [0.1, 0.1, 2.0],  # right (inverted)
            [0.5, 0.9, 2.0],  # anchor 1
        ],
        dtype=np.float32,
    )
    out, swaps = _canonicalize_lr_gt_instances(
        gt_instances=[gt],
        image_hw=(100, 100),
        lr_pairs=[(0, 2)],
        orientation_anchor_idx=[1, 3],
        min_side_diff_px=0.0,
    )
    assert swaps == 1
    assert len(out) == 1
    assert np.allclose(out[0][0], gt[2])
    assert np.allclose(out[0][2], gt[0])


def test_soft_argmax_respects_valid_mask() -> None:
    # Best logit is in an invalid padded column; decode should stay in valid area.
    logits = torch.zeros((1, 1, 2, 3), dtype=torch.float32)
    logits[:, :, :, 2] = 20.0
    valid_mask = torch.ones((1, 1, 2, 3), dtype=torch.bool)
    valid_mask[:, :, :, 2] = False

    xy = _soft_argmax_coords_from_logits_batched(
        logits,
        patch_size=1,
        valid_mask_b1hw=valid_mask,
    )
    x = float(xy[0, 0, 0].item())
    assert x < 2.0


def test_argmax_respects_valid_mask() -> None:
    probs = torch.zeros((1, 1, 2, 3), dtype=torch.float32)
    probs[:, :, :, 2] = 1.0
    valid_mask = torch.ones((1, 1, 2, 3), dtype=torch.bool)
    valid_mask[:, :, :, 2] = False

    xy = _argmax_coords_batched(probs, patch_size=1, valid_mask_b1hw=valid_mask)
    x = float(xy[0, 0, 0].item())
    assert x < 2.0


def test_coord_loss_normalizer_scales_loss_magnitude() -> None:
    pred = torch.tensor([[16.0, 16.0]], dtype=torch.float32)
    target = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    mask = torch.tensor([1.0], dtype=torch.float32)
    raw = _coord_loss(pred, target, mask, mode="l1", normalizer_px=1.0)
    norm = _coord_loss(pred, target, mask, mode="l1", normalizer_px=16.0)
    assert float(norm.item()) < float(raw.item())
    assert float(norm.item()) == pytest.approx(float(raw.item()) / 16.0)


def test_coord_loss_clip_delta_limits_outlier_impact() -> None:
    pred = torch.tensor([[1000.0, 1000.0]], dtype=torch.float32)
    target = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    mask = torch.tensor([1.0], dtype=torch.float32)
    unclipped = _coord_loss(
        pred, target, mask, mode="smooth_l1", normalizer_px=1.0, clip_delta_px=0.0
    )
    clipped = _coord_loss(
        pred, target, mask, mode="smooth_l1", normalizer_px=1.0, clip_delta_px=16.0
    )
    assert float(clipped.item()) < float(unclipped.item())
