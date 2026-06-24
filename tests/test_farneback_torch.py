import numpy as np
import cv2
import pytest
import torch

from annolid.motion.farneback_torch import (
    _acc_dtype_for_device,
    _farneback_polyexp_batched,
    _grid_sample_R1,
    calc_optical_flow_farneback_torch,
    farneback_polyexp,
    farneback_prepare_gaussian,
    resolve_torch_optical_flow_device,
)


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


def test_accumulator_precision_preserves_cpu_parity_and_uses_fast_gpu_dtype():
    assert _acc_dtype_for_device(torch.device("cpu")) == torch.float64
    assert _acc_dtype_for_device(torch.device("cuda")) == torch.float32
    assert _acc_dtype_for_device(torch.device("mps")) == torch.float32


def test_torch_flow_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    assert resolve_torch_optical_flow_device() == torch.device("cuda")
    assert resolve_torch_optical_flow_device("auto") == torch.device("cuda")
    assert resolve_torch_optical_flow_device("cuda:0") == torch.device("cuda:0")


def test_unavailable_torch_flow_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)

    assert resolve_torch_optical_flow_device("cuda") == torch.device("cpu")


def test_batched_polynomial_expansion_matches_sequential_results():
    device = torch.device("cpu")
    kernels = farneback_prepare_gaussian(5, 1.1, device)
    pair = torch.arange(2 * 16 * 20, dtype=torch.float32).reshape(2, 1, 16, 20)

    batched = _farneback_polyexp_batched(pair, kernels)
    sequential = torch.stack(
        [
            farneback_polyexp(pair[0:1], kernels),
            farneback_polyexp(pair[1:2], kernels),
        ]
    )

    torch.testing.assert_close(batched, sequential)


def test_grid_sample_warp_matches_manual_bilinear_interpolation():
    height, width = 5, 6
    source = torch.arange(
        height * width * 5,
        dtype=torch.float32,
    ).reshape(height, width, 5)
    fx = torch.tensor([[1.25, 2.50], [3.10, 0.75]], dtype=torch.float32)
    fy = torch.tensor([[0.50, 1.25], [2.75, 3.10]], dtype=torch.float32)

    actual = _grid_sample_R1(source, fx, fy)
    expected = torch.empty_like(actual)
    for y in range(fy.shape[0]):
        for x in range(fx.shape[1]):
            sample_x = float(fx[y, x])
            sample_y = float(fy[y, x])
            x0 = int(np.floor(sample_x))
            y0 = int(np.floor(sample_y))
            wx = sample_x - x0
            wy = sample_y - y0
            expected[y, x] = (
                source[y0, x0] * (1.0 - wx) * (1.0 - wy)
                + source[y0, x0 + 1] * wx * (1.0 - wy)
                + source[y0 + 1, x0] * (1.0 - wx) * wy
                + source[y0 + 1, x0 + 1] * wx * wy
            )

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


def test_default_torch_postprocessing_preserves_raw_farneback_output():
    prev, nxt = _synthetic_pair_translation(dx=1.0, dy=-0.5)
    params = dict(
        pyr_scale=0.5,
        levels=1,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.1,
        flags=0,
        device="cpu",
    )

    default_flow = calc_optical_flow_farneback_torch(prev, nxt, **params)
    strict_flow = calc_optical_flow_farneback_torch(
        prev,
        nxt,
        clip_percentile=None,
        max_magnitude=None,
        outlier_ksize=0,
        **params,
    )

    np.testing.assert_array_equal(default_flow, strict_flow)
