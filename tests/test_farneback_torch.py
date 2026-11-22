import numpy as np
import cv2
import pytest
import torch

import annolid.motion.farneback_torch as ft
from annolid.motion.farneback_torch import (
    OPTFLOW_FARNEBACK_GAUSSIAN,
    _acc_dtype_for_device,
    _box_blur5,
    _farneback_polyexp_batched,
    _gaussian_blur5,
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


def test_farneback_torch_gaussian_flag_matches_opencv():
    prev, nxt = _synthetic_pair_translation(dx=1.5, dy=-1.0, seed=3)
    params = dict(
        pyr_scale=0.5,
        levels=1,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.1,
        flags=OPTFLOW_FARNEBACK_GAUSSIAN,
    )

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
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    ).astype(np.float32)

    flow_th = calc_optical_flow_farneback_torch(
        prev,
        nxt,
        device="cpu",
        **params,
    )

    border = 5
    epe = np.linalg.norm(
        flow_th[border:-border, border:-border]
        - flow_cv[border:-border, border:-border],
        axis=2,
    )
    assert float(epe.mean()) < 1e-3
    assert float(epe.max()) < 1e-2


def test_farneback_torch_handles_even_winsize_and_color_inputs():
    prev, nxt = _synthetic_pair_translation(dx=0.75, dy=-0.5, seed=5)
    prev_u8 = np.clip(prev, 0, 255).astype(np.uint8)
    nxt_u8 = np.clip(nxt, 0, 255).astype(np.uint8)
    prev_bgr = cv2.cvtColor(prev_u8, cv2.COLOR_GRAY2BGR)
    nxt_bgr = cv2.cvtColor(nxt_u8, cv2.COLOR_GRAY2BGR)
    prev_bgra = cv2.cvtColor(prev_u8, cv2.COLOR_GRAY2BGRA)
    nxt_bgra = cv2.cvtColor(nxt_u8, cv2.COLOR_GRAY2BGRA)

    flow_gray = calc_optical_flow_farneback_torch(
        prev_u8, nxt_u8, winsize=14, device="cpu"
    )
    flow_bgr = calc_optical_flow_farneback_torch(
        prev_bgr, nxt_bgr, winsize=14, device="cpu"
    )
    flow_bgra = calc_optical_flow_farneback_torch(
        prev_bgra, nxt_bgra, winsize=14, device="cpu"
    )

    assert flow_gray.shape == (prev.shape[0], prev.shape[1], 2)
    assert np.isfinite(flow_gray).all()
    np.testing.assert_allclose(flow_bgr, flow_gray, atol=1e-4)
    np.testing.assert_allclose(flow_bgra, flow_gray, atol=1e-4)


@pytest.mark.parametrize("winsize", [14, 15])
def test_gpu_float32_separable_blur_and_polyexp_match_float64(monkeypatch, winsize):
    """Exercise the float32 accelerator branches (separable avg_pool2d, Gaussian, sparse polyexp)."""
    rng = np.random.default_rng(13)
    mat_m = torch.from_numpy(
        rng.uniform(-5.0, 5.0, size=(32, 40, 5)).astype(np.float32)
    )
    pair = torch.from_numpy(
        rng.uniform(0.0, 255.0, size=(2, 1, 32, 40)).astype(np.float32)
    )
    kernels_f64 = farneback_prepare_gaussian(5, 1.1, torch.device("cpu"))

    box_f64 = _box_blur5(mat_m, winsize)
    gauss_f64 = _gaussian_blur5(mat_m, winsize)
    poly_f64 = _farneback_polyexp_batched(pair, kernels_f64)

    # Force float32 accumulation path used by CUDA/MPS accelerators.
    monkeypatch.setattr(ft, "_acc_dtype_for_device", lambda _dev: torch.float32)

    box_f32 = _box_blur5(mat_m, winsize)
    gauss_f32 = _gaussian_blur5(mat_m, winsize)
    poly_f32 = _farneback_polyexp_batched(pair, kernels_f64)

    assert box_f32.shape == box_f64.shape
    assert gauss_f32.shape == gauss_f64.shape
    torch.testing.assert_close(box_f32, box_f64, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(gauss_f32, gauss_f64, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(poly_f32, poly_f64, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("flags", [0, OPTFLOW_FARNEBACK_GAUSSIAN])
def test_farneback_torch_float32_accelerator_path_matches_opencv(monkeypatch, flags):
    prev, nxt = _synthetic_pair_translation(dx=1.25, dy=-0.75, seed=9)
    flow_cv = cv2.calcOpticalFlowFarneback(
        prev, nxt, None, 0.5, 1, 15, 3, 5, 1.1, flags
    ).astype(np.float32)

    monkeypatch.setattr(ft, "_acc_dtype_for_device", lambda _dev: torch.float32)
    flow_f32 = calc_optical_flow_farneback_torch(
        prev,
        nxt,
        pyr_scale=0.5,
        levels=1,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.1,
        flags=flags,
        device="cpu",
    )

    border = 5
    epe = np.linalg.norm(
        flow_f32[border:-border, border:-border]
        - flow_cv[border:-border, border:-border],
        axis=2,
    )
    assert float(epe.mean()) < 1e-2
    assert float(epe.max()) < 0.25


def test_mps_warmup_failure_clears_cache_and_falls_back_to_cpu(monkeypatch):
    prev, nxt = _synthetic_pair_translation(H=48, W=64, dx=1.0, dy=-0.5, seed=4)
    cleaned_devices: list[str] = []

    monkeypatch.setattr(
        ft, "resolve_torch_optical_flow_device", lambda _d=None: torch.device("mps")
    )
    monkeypatch.setattr(ft, "_WARMED_MPS_DEVICES", set())
    monkeypatch.setattr(
        ft,
        "clear_device_cache",
        lambda **kwargs: cleaned_devices.append(str(kwargs.get("device"))),
    )

    orig_zeros = torch.zeros

    def _fail_mps_zeros(*args, **kwargs):
        dev = kwargs.get("device")
        if dev is not None and torch.device(dev).type == "mps":
            raise RuntimeError("synthetic MPS warmup failure")
        return orig_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", _fail_mps_zeros)

    with pytest.warns(RuntimeWarning, match="MPS Farneback failed"):
        flow = calc_optical_flow_farneback_torch(prev, nxt, device="mps")

    assert cleaned_devices == ["mps"]
    assert flow.shape == (48, 64, 2)
    assert np.isfinite(flow).all()
