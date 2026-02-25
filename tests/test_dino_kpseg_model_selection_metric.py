import pytest
import argparse
import torch


from annolid.segmentation.dino_kpseg.train import (
    _apply_schedule_profile_defaults,
    _compute_no_aug_start_epoch,
    _ema_update_,
    _is_epoch_augment_enabled,
    _metric_higher_is_better,
    _parse_weight_list,
    _resolve_feature_align_dim,
    _resolve_selection_metric,
)


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
