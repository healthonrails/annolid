import numpy as np
import cv2
import pytest

from annolid.motion.farneback_torch import calc_optical_flow_farneback_torch


def _synthetic_pair_translation(H=128, W=160, dx=0.25, dy=-0.35, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    base = np.zeros((H, W), np.float32)
    base[rng.integers(0, H, 400), rng.integers(0, W, 400)] = 255.0
    kernel = cv2.getGaussianKernel(9, 1.5)
    base = cv2.sepFilter2D(base, -1, kernel, kernel)

    nxt = cv2.warpAffine(
        base,
        np.float32([[1, 0, dx], [0, 1, dy]]),
        (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    if noise > 0:
        nxt = nxt + rng.normal(0, noise, size=nxt.shape).astype(np.float32)

    return base, nxt


@pytest.mark.parametrize(
    "case",
    [
        dict(dx=0.25, dy=-0.35, noise=0.0, seed=0),
        dict(dx=-0.6, dy=0.4, noise=1.5, seed=1),
        dict(dx=2.0, dy=-1.5, noise=0.0, seed=2),
    ],
)
def test_farneback_torch_matches_opencv(case):
    """
    Torch Farneback should closely match OpenCV's CPU Farneback
    for small-to-moderate translations, excluding border attenuation region.
    """

    params = dict(
        pyr_scale=0.5,
        levels=1,
        winsize=15,  # IMPORTANT: winsize=1 is ill-conditioned
        iterations=3,
        poly_n=5,
        poly_sigma=1.1,
        flags=0,
    )

    prev, nxt = _synthetic_pair_translation(**case)

    flow_cv = cv2.calcOpticalFlowFarneback(
        prev,
        nxt,
        None,
        pyr_scale=params["pyr_scale"],
        levels=params["levels"],
        winsize=params["winsize"],
        iterations=params["iterations"],
        poly_n=params["poly_n"],
        poly_sigma=params["poly_sigma"],
        flags=params["flags"],
    ).astype(np.float32)

    flow_th = calc_optical_flow_farneback_torch(
        prev,
        nxt,
        device="cpu",
        clip_percentile=None,  # disable post-processing
        max_magnitude=None,
        outlier_ksize=0,
        **params,
    )

    assert flow_th.shape == flow_cv.shape

    # --- exclude OpenCV Farnebäck border attenuation region ---
    BORDER = 5
    flow_cv_i = flow_cv[BORDER:-BORDER, BORDER:-BORDER]
    flow_th_i = flow_th[BORDER:-BORDER, BORDER:-BORDER]

    diff = flow_th_i - flow_cv_i
    epe = np.linalg.norm(diff, axis=2)

    mean_epe = float(epe.mean())
    max_epe = float(epe.max())

    # --- assertions ---
    # Mean error should be very small for translation
    assert mean_epe < 0.15, f"Mean EPE too high: {mean_epe}"

    # Max error can spike locally but should stay bounded
    assert max_epe < 3.0, f"Max EPE too high: {max_epe}"

    # --- sanity: flow direction should match translation ---
    mean_flow = flow_th_i.mean(axis=(0, 1))
    assert np.sign(mean_flow[0]) == np.sign(case["dx"])
    assert np.sign(mean_flow[1]) == np.sign(case["dy"])
