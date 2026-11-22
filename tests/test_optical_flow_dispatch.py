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


def test_compute_optical_flow_accepts_grayscale_and_bgra_inputs():
    rng = np.random.default_rng(11)
    gray0 = rng.integers(0, 255, size=(48, 64), dtype=np.uint8)
    gray1 = np.roll(gray0, 2, axis=1)
    bgra0 = np.dstack([gray0, gray0, gray0, np.full_like(gray0, 255)])
    bgra1 = np.dstack([gray1, gray1, gray1, np.full_like(gray1, 255)])

    hsv_g, flow_g = optical_flow.compute_optical_flow(
        gray0, gray1, prefer_cuda=False, use_umat=False
    )
    hsv_a, flow_a = optical_flow.compute_optical_flow(
        bgra0, bgra1, prefer_cuda=False, use_umat=False
    )
    field_g = optical_flow.compute_optical_flow_field(
        gray0, gray1, prefer_cuda=False, use_umat=False
    )

    assert flow_g.shape == (48, 64, 2)
    assert hsv_g.shape == (48, 64, 3)
    np.testing.assert_allclose(flow_g, flow_a, atol=1e-5)
    np.testing.assert_array_equal(flow_g, field_g)


def test_compute_optical_flow_downscale_uses_exact_per_axis_scale(monkeypatch):
    def _constant_flow(prev, _curr, _flow, **_kwargs):
        out = np.empty((*prev.shape[:2], 2), dtype=np.float32)
        out[..., 0] = 2.0
        out[..., 1] = -3.0
        return out

    monkeypatch.setattr(optical_flow.cv2, "calcOpticalFlowFarneback", _constant_flow)

    # 101x51 with scale=0.5 rounds to 50x26 -> scale_x=50/101, scale_y=26/51
    frame = np.zeros((51, 101, 3), dtype=np.uint8)
    flow = optical_flow.compute_optical_flow_field(
        frame, frame.copy(), scale=0.5, prefer_cuda=False, use_umat=False
    )

    assert flow.shape == (51, 101, 2)
    np.testing.assert_allclose(flow[..., 0], 2.0 / (50.0 / 101.0), rtol=1e-5)
    np.testing.assert_allclose(flow[..., 1], -3.0 / (26.0 / 51.0), rtol=1e-5)


def test_flow_hsv_to_bgr_compatible_with_uint8_addweighted():
    import cv2

    rng = np.random.default_rng(19)
    frame0 = rng.integers(0, 255, size=(32, 48, 3), dtype=np.uint8)
    frame1 = np.roll(frame0, 1, axis=1)
    flow_hsv, _ = optical_flow.compute_optical_flow(
        frame0, frame1, prefer_cuda=False, use_umat=False
    )

    flow_bgr = optical_flow.flow_hsv_to_bgr(flow_hsv)
    assert flow_bgr.dtype == np.uint8
    assert flow_bgr.shape == (32, 48, 3)
    blended = cv2.addWeighted(frame0, 1.0, flow_bgr, 0.5, 0)
    assert blended.dtype == np.uint8
    assert blended.shape == (32, 48, 3)


def test_optical_flow_settings_and_preferences_coerce_qsettings_bool_strings():
    settings = optical_flow.optical_flow_settings_from(
        {
            "compute_optical_flow": "false",
            "optical_flow_backend": "farneback (torch)",
        }
    )
    assert settings["compute_optical_flow"] is False
    assert settings["optical_flow_backend"] == "farneback_torch"

    prefs = optical_flow.OpticalFlowPreferences.from_source(
        {
            "optical_flow/stable_hsv": "false",
            "optical_flow/backend": "raft (torchvision)",
            "optical_flow/farneback_pyr_scale": 1.5,  # invalid -> fallback to default
        }
    )
    assert prefs.stable_hsv is False
    assert prefs.backend == "raft"
    assert prefs.farneback_pyr_scale == optical_flow.DEFAULT_FARNEBACK_PYR_SCALE


def test_cuda_farneback_cache_reuse_eviction_and_fallback(monkeypatch):
    from collections import OrderedDict
    from types import SimpleNamespace

    created: list[dict] = []
    fail_next = {"value": False}

    class _FakeGpuMat:
        def __init__(self, data=None):
            self.data = data

        def upload(self, arr):
            self.data = arr

        def download(self):
            return self.data

    class _FakeCudaFlow:
        def __init__(self, params):
            self.params = params
            self.calls = 0

        def calc(self, prev_gpu, curr_gpu, _flow):
            self.calls += 1
            if fail_next["value"]:
                raise RuntimeError("synthetic OpenCV CUDA failure")
            assert prev_gpu.data.ndim == 2
            assert curr_gpu.data.ndim == 2
            out = np.full((*prev_gpu.data.shape, 2), 1.25, dtype=np.float32)
            return _FakeGpuMat(out)

    def _create(**kwargs):
        created.append(kwargs)
        return _FakeCudaFlow(kwargs)

    fake_cuda = SimpleNamespace(
        getCudaEnabledDeviceCount=lambda: 1,
        FarnebackOpticalFlow_create=_create,
    )

    monkeypatch.setattr(optical_flow.cv2, "cuda", fake_cuda, raising=False)
    monkeypatch.setattr(optical_flow.cv2, "cuda_GpuMat", _FakeGpuMat, raising=False)
    monkeypatch.setattr(optical_flow, "_CUDA_FLOW_CACHE", OrderedDict())
    monkeypatch.setattr(optical_flow, "_MAX_CUDA_FLOW_CACHE", 2)

    gray = np.zeros((24, 32), dtype=np.uint8)
    bgr = np.zeros((24, 32, 3), dtype=np.uint8)

    # Identical params across grayscale and BGR reuse the cached CUDA flow instance.
    f1 = optical_flow.compute_optical_flow_field(
        gray, gray.copy(), prefer_cuda=True, use_umat=False
    )
    f2 = optical_flow.compute_optical_flow_field(
        bgr, bgr.copy(), prefer_cuda=True, use_umat=False
    )
    assert len(created) == 1
    np.testing.assert_allclose(f1, 1.25)
    np.testing.assert_allclose(f2, 1.25)

    # Distinct parameter sets evict LRU entries beyond _MAX_CUDA_FLOW_CACHE=2.
    optical_flow.compute_optical_flow_field(
        gray, gray.copy(), prefer_cuda=True, use_umat=False, farneback_winsize=11
    )
    optical_flow.compute_optical_flow_field(
        gray, gray.copy(), prefer_cuda=True, use_umat=False, farneback_winsize=19
    )
    assert len(optical_flow._CUDA_FLOW_CACHE) == 2

    # Failure in CUDA calc falls back cleanly to CPU OpenCV Farneback.
    fail_next["value"] = True
    fallback = optical_flow.compute_optical_flow_field(
        gray, gray.copy(), prefer_cuda=True, use_umat=False, farneback_winsize=19
    )
    assert fallback.shape == (24, 32, 2)
    assert np.isfinite(fallback).all()


def test_raft_model_cache_eviction_and_clear(monkeypatch):
    from collections import OrderedDict
    import torch
    import annolid.motion.raft_wrapper as rw

    created_keys: list[tuple[str, str]] = []
    cleared_devices: list[str] = []

    class _FakeRaft:
        def __init__(self, model_type="small", device=None):
            self.model_type = model_type
            self.device = torch.device(device or "cpu")
            created_keys.append((self.model_type, str(self.device)))

    monkeypatch.setattr(rw, "RAFTOpticalFlow", _FakeRaft)
    monkeypatch.setattr(rw, "_RAFT_MODEL_CACHE", OrderedDict())
    monkeypatch.setattr(rw, "_MAX_CACHED_RAFT_MODELS", 1)
    monkeypatch.setattr(
        rw,
        "clear_device_cache",
        lambda **kwargs: cleared_devices.append(str(kwargs.get("device"))),
    )

    m_small = rw.get_raft_model("small", device="cpu")
    assert rw.get_raft_model("small", device="cpu") is m_small
    assert len(created_keys) == 1

    # Adding a second model with cap=1 evicts the first and clears its device cache.
    m_large = rw.get_raft_model("large", device="cpu")
    assert m_large is not m_small
    assert len(rw._RAFT_MODEL_CACHE) == 1
    assert cleared_devices == ["cpu"]

    rw.clear_raft_model_cache()
    assert len(rw._RAFT_MODEL_CACHE) == 0
    assert cleared_devices == ["cpu", "cpu"]
