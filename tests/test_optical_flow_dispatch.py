import numpy as np

import annolid.motion.farneback_torch as farneback_torch
import annolid.motion.optical_flow as optical_flow


def test_torch_farneback_dispatch_uses_runtime_gpu_device(monkeypatch):
    captured = {}

    def _fake_torch_farneback(prev, nxt, **kwargs):
        captured["shape"] = (prev.shape, nxt.shape)
        captured["device"] = kwargs["device"]
        return np.zeros((*prev.shape[:2], 2), dtype=np.float32)

    monkeypatch.setattr(optical_flow, "get_device", lambda: "cuda")
    monkeypatch.setattr(
        farneback_torch,
        "calc_optical_flow_farneback_torch",
        _fake_torch_farneback,
    )

    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    flow_hsv, flow = optical_flow.compute_optical_flow(
        frame,
        frame.copy(),
        use_torch_farneback=True,
        prefer_cuda=False,
        use_umat=False,
    )

    assert captured == {
        "shape": ((24, 32), (24, 32)),
        "device": "cuda",
    }
    assert flow.shape == (24, 32, 2)
    assert flow_hsv.shape == (24, 32, 3)


def test_optical_flow_settings_accept_explicit_torch_device():
    settings = optical_flow.optical_flow_settings_from(
        {"optical_flow_torch_device": "cuda:1"}
    )
    kwargs = optical_flow.optical_flow_compute_kwargs(settings)

    assert settings["optical_flow_torch_device"] == "cuda:1"
    assert kwargs["torch_device"] == "cuda:1"


def test_optical_flow_defaults_use_stable_farneback_parameters():
    settings = optical_flow.optical_flow_settings_from({})

    assert settings["flow_farneback_winsize"] == 15
    assert settings["flow_farneback_poly_n"] == 5


def test_explicit_legacy_farneback_parameters_remain_supported():
    settings = optical_flow.optical_flow_settings_from(
        {
            "flow_farneback_winsize": 1,
            "flow_farneback_poly_n": 3,
        }
    )

    assert settings["flow_farneback_winsize"] == 1
    assert settings["flow_farneback_poly_n"] == 3


def test_legacy_farneback_parameters_bypass_torch_and_use_opencv(monkeypatch):
    calls = {"torch": 0, "opencv": []}

    def _unexpected_torch_call(*_args, **_kwargs):
        calls["torch"] += 1
        raise AssertionError("legacy parameters must not use Torch Farneback")

    def _capture_opencv(_prev, _curr, _flow, **kwargs):
        calls["opencv"].append(kwargs)
        return np.zeros((24, 32, 2), dtype=np.float32)

    monkeypatch.setattr(
        farneback_torch,
        "calc_optical_flow_farneback_torch",
        _unexpected_torch_call,
    )
    monkeypatch.setattr(
        optical_flow.cv2,
        "calcOpticalFlowFarneback",
        _capture_opencv,
    )

    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    _, flow = optical_flow.compute_optical_flow(
        frame,
        frame.copy(),
        use_torch_farneback=True,
        prefer_cuda=False,
        use_umat=False,
        farneback_winsize=1,
        farneback_poly_n=3,
    )

    assert calls["torch"] == 0
    assert calls["opencv"] == [
        {
            "pyr_scale": 0.5,
            "levels": 1,
            "winsize": 1,
            "iterations": 3,
            "poly_n": 3,
            "poly_sigma": 1.1,
            "flags": 0,
        }
    ]
    assert flow.shape == (24, 32, 2)


def test_torch_backend_failure_clears_device_cache_before_opencv_fallback(
    monkeypatch,
):
    cache_cleanup = []

    def _raise_torch_failure(*_args, **_kwargs):
        raise RuntimeError("synthetic CUDA OOM")

    monkeypatch.setattr(optical_flow, "get_device", lambda: "cuda")
    monkeypatch.setattr(
        farneback_torch,
        "calc_optical_flow_farneback_torch",
        _raise_torch_failure,
    )
    monkeypatch.setattr(
        optical_flow,
        "clear_device_cache",
        lambda **kwargs: cache_cleanup.append(kwargs["device"]),
    )

    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    _, flow = optical_flow.compute_optical_flow(
        frame,
        frame.copy(),
        use_torch_farneback=True,
        prefer_cuda=False,
        use_umat=False,
    )

    assert cache_cleanup == ["cuda"]
    assert flow.shape == (24, 32, 2)
