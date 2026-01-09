import pytest


from annolid.segmentation.dino_kpseg.train import (
    _metric_higher_is_better,
    _parse_weight_list,
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
