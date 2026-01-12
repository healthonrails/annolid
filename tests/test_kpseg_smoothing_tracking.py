from __future__ import annotations

from annolid.tracking.kpseg_smoothing import KeypointSmoother


def test_kalman_predicts_forward_when_measurement_missing() -> None:
    smoother = KeypointSmoother(
        mode="kalman",
        fps=30.0,
        min_score=0.25,
        kalman_process_noise=1e-3,
        kalman_measurement_noise=1e-2,
    )

    p1 = smoother.smooth("inst0:k0", (0.0, 0.0), score=1.0, mask_ok=True)
    p2 = smoother.smooth("inst0:k0", (10.0, 0.0), score=1.0, mask_ok=True)
    p3 = smoother.smooth("inst0:k0", (999.0, 0.0), score=0.0, mask_ok=False)

    assert p1[0] <= p2[0]
    assert p3[0] > p2[0]
    assert p3[0] < 100.0


def test_ema_holds_last_value_when_measurement_missing() -> None:
    smoother = KeypointSmoother(mode="ema", fps=30.0, ema_alpha=0.5, min_score=0.25)

    p1 = smoother.smooth("inst0:k0", (5.0, 5.0), score=1.0, mask_ok=True)
    p2 = smoother.smooth("inst0:k0", (100.0, 100.0), score=0.0, mask_ok=False)

    assert p2 == p1


def test_per_key_state_is_isolated() -> None:
    smoother = KeypointSmoother(mode="kalman", fps=30.0)

    a1 = smoother.smooth("inst0:k0", (0.0, 0.0), score=1.0, mask_ok=True)
    b1 = smoother.smooth("inst1:k0", (100.0, 0.0), score=1.0, mask_ok=True)
    a2 = smoother.smooth("inst0:k0", (10.0, 0.0), score=1.0, mask_ok=True)

    assert a1[0] != b1[0]
    assert a2[0] > a1[0]
