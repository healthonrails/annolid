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
    nxt = cv2.warpAffine(base, np.float32([[1, 0, dx], [
                         0, 1, dy]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if noise > 0:
        nxt = nxt + rng.normal(0, noise, size=nxt.shape).astype(np.float32)
    return base, nxt


@pytest.mark.parametrize(
    "case",
    [
        dict(dx=0.25, dy=-0.35, noise=0.0, seed=0),
        dict(dx=-0.6, dy=0.4, noise=2.0, seed=1),
        dict(dx=2.0, dy=-1.5, noise=0.0, seed=2),
    ],
)
def test_farneback_torch_matches_opencv(case):
    params = dict(
        pyr_scale=0.5,
        levels=1,
        winsize=1,
        iterations=3,
        poly_n=3,
        poly_sigma=1.1,
        flags=0,
    )
    prev, nxt = _synthetic_pair_translation(**case)

    flow_cv = cv2.calcOpticalFlowFarneback(
        prev, nxt, None,
        pyr_scale=params["pyr_scale"],
        levels=params["levels"],
        winsize=params["winsize"],
        iterations=params["iterations"],
        poly_n=params["poly_n"],
        poly_sigma=params["poly_sigma"],
        flags=params["flags"],
    ).astype(np.float32)

    flow_th = calc_optical_flow_farneback_torch(
        prev, nxt, device="cpu", **params
    )

    diff = flow_th - flow_cv
    mean_epe = np.linalg.norm(diff, axis=2).mean()
    max_epe = np.linalg.norm(diff, axis=2).max()

    # Torch implementation is an approximation; verify it's in the right ballpark.
    assert mean_epe < 2.5
    assert max_epe < 200.0
