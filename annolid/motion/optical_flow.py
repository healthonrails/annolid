from typing import Any, Mapping, Optional, Tuple, Dict

import cv2
import numpy as np
from annolid.utils.devices import get_device

AVAILABLE_DEVICE = get_device()


def _coerce_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return float(out)


def _coerce_int(value: Any, default: int) -> int:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return int(default)
    return int(out)


def _coerce_pos_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    if out <= 0:
        return None
    return out


def _read_setting(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def optical_flow_settings_from(source: Any) -> Dict[str, Any]:
    """
    Return a normalized optical-flow settings mapping using the same keys
    the Settings menu exposes (mirrored onto the window/manager).

    This is the preferred way to thread optical flow preferences into
    prediction/tracking code without duplicating per-key plumbing.
    """
    compute_enabled = bool(
        _read_setting(
            source,
            "compute_optical_flow",
            _read_setting(source, "optical_flow/compute", False),
        )
    )
    backend = (
        str(
            _read_setting(
                source,
                "optical_flow_backend",
                _read_setting(source, "optical_flow/backend", "farneback"),
            )
        )
        .strip()
        .lower()
    )
    raft_model = (
        str(
            _read_setting(
                source,
                "optical_flow_raft_model",
                _read_setting(source, "optical_flow/raft_model", "small"),
            )
        )
        .strip()
        .lower()
    )
    if raft_model not in {"small", "large"}:
        raft_model = "small"

    # Prefer the GUI-exposed names, fall back to internal names used elsewhere.
    farneback_pyr_scale = _coerce_float(
        _read_setting(
            source,
            "flow_farneback_pyr_scale",
            _read_setting(source, "farneback_pyr_scale", 0.5),
        ),
        0.5,
    )
    farneback_levels = max(
        1,
        _coerce_int(
            _read_setting(
                source,
                "flow_farneback_levels",
                _read_setting(source, "farneback_levels", 1),
            ),
            1,
        ),
    )
    farneback_winsize = max(
        1,
        _coerce_int(
            _read_setting(
                source,
                "flow_farneback_winsize",
                _read_setting(source, "farneback_winsize", 1),
            ),
            1,
        ),
    )
    farneback_iterations = max(
        1,
        _coerce_int(
            _read_setting(
                source,
                "flow_farneback_iterations",
                _read_setting(source, "farneback_iterations", 3),
            ),
            3,
        ),
    )
    farneback_poly_n = max(
        3,
        _coerce_int(
            _read_setting(
                source,
                "flow_farneback_poly_n",
                _read_setting(source, "farneback_poly_n", 3),
            ),
            3,
        ),
    )
    farneback_poly_sigma = _coerce_float(
        _read_setting(
            source,
            "flow_farneback_poly_sigma",
            _read_setting(source, "farneback_poly_sigma", 1.1),
        ),
        1.1,
    )

    scale = _coerce_float(_read_setting(source, "optical_flow_scale", 1.0), 1.0)
    if scale <= 0:
        scale = 1.0
    scale = min(1.0, scale)
    max_dim = _coerce_pos_int_or_none(_read_setting(source, "optical_flow_max_dim"))

    return {
        "compute_optical_flow": compute_enabled,
        "optical_flow_backend": backend,
        "optical_flow_raft_model": raft_model,
        "flow_farneback_pyr_scale": farneback_pyr_scale,
        "flow_farneback_levels": farneback_levels,
        "flow_farneback_winsize": farneback_winsize,
        "flow_farneback_iterations": farneback_iterations,
        "flow_farneback_poly_n": farneback_poly_n,
        "flow_farneback_poly_sigma": farneback_poly_sigma,
        "optical_flow_scale": scale,
        "optical_flow_max_dim": max_dim,
    }


def optical_flow_compute_kwargs(settings: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Extract `compute_optical_flow()` keyword arguments from Annolid settings/config.

    Supports GUI settings-menu keys (mirrored onto the window):
      - `optical_flow_raft_model`
      - `flow_farneback_pyr_scale`, `flow_farneback_levels`, `flow_farneback_winsize`,
        `flow_farneback_iterations`, `flow_farneback_poly_n`, `flow_farneback_poly_sigma`

    Also supports backend-agnostic speed knobs:
      - `optical_flow_scale`
      - `optical_flow_max_dim`
    """
    normalized = optical_flow_settings_from(settings)

    return {
        "raft_model": normalized["optical_flow_raft_model"],
        "farneback_pyr_scale": normalized["flow_farneback_pyr_scale"],
        "farneback_levels": normalized["flow_farneback_levels"],
        "farneback_winsize": normalized["flow_farneback_winsize"],
        "farneback_iterations": normalized["flow_farneback_iterations"],
        "farneback_poly_n": normalized["flow_farneback_poly_n"],
        "farneback_poly_sigma": normalized["flow_farneback_poly_sigma"],
        "scale": normalized["optical_flow_scale"],
        "max_dim": normalized["optical_flow_max_dim"],
    }


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert to single-channel grayscale."""
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 1:
        return frame[..., 0]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _cuda_flow_available() -> bool:
    """Check whether CUDA Farneback is available."""
    return bool(
        getattr(cv2, "cuda", None)
        and hasattr(cv2.cuda, "getCudaEnabledDeviceCount")
        and cv2.cuda.getCudaEnabledDeviceCount() > 0
        and hasattr(cv2.cuda, "FarnebackOpticalFlow_create")
    )


def compute_optical_flow(
    prev_frame: np.ndarray,
    current_frame: np.ndarray,
    scale: float = 1.0,
    max_dim: Optional[int] = None,
    use_umat: Optional[bool] = None,
    prefer_cuda: Optional[bool] = None,
    use_raft: bool = False,
    raft_model: str = "small",
    use_torch_farneback: bool = False,
    farneback_pyr_scale: float = 0.5,
    farneback_levels: int = 1,
    farneback_winsize: int = 1,
    farneback_iterations: int = 3,
    farneback_poly_n: int = 3,
    farneback_poly_sigma: float = 1.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute dense Farneback optical flow with automatic device selection.

    When use_raft=True and torch/torchvision RAFT are available, uses RAFT first;
    otherwise falls back to CUDA Farneback, then OpenCL/UMat, then CPU.
    Frames can be optionally downscaled for speed via `scale` and/or `max_dim`.
    The returned flow is always resized back to the original resolution and
    scaled so downstream consumers do not need to adjust.
    """
    if prev_frame is None or current_frame is None:
        raise ValueError("prev_frame and current_frame must be non-None arrays.")
    if prev_frame.shape != current_frame.shape:
        raise ValueError(
            f"prev_frame and current_frame must share the same shape, "
            f"got {prev_frame.shape} vs {current_frame.shape}"
        )

    orig_h, orig_w = prev_frame.shape[:2]

    # Combine scale and max_dim into an effective scale
    scale = float(max(scale, 1e-3))
    if max_dim is not None:
        dynamic_scale = min(1.0, float(max_dim) / float(max(orig_h, orig_w)))
        scale = min(scale, dynamic_scale)

    if scale != 1.0:
        prev_frame = cv2.resize(prev_frame, (0, 0), fx=scale, fy=scale)
        current_frame = cv2.resize(current_frame, (0, 0), fx=scale, fy=scale)

    prefer_cuda = True if prefer_cuda is None else bool(prefer_cuda)
    opencl_available = cv2.ocl.haveOpenCL()
    if use_umat is None:
        use_umat = opencl_available and cv2.ocl.useOpenCL()
    elif use_umat and opencl_available:
        cv2.ocl.setUseOpenCL(True)
        use_umat = cv2.ocl.useOpenCL()
    else:
        use_umat = False

    flow: Optional[np.ndarray] = None

    # Try RAFT (torchvision) first if requested and available
    if use_raft:
        try:
            flow = compute_optical_flow_raft(
                prev_frame, current_frame, model=raft_model, device=AVAILABLE_DEVICE
            )
        except Exception:
            flow = None  # fall back to Farneback paths

    # Torch Farneback (optional)
    if flow is None and use_torch_farneback:
        try:
            from annolid.motion.farneback_torch import calc_optical_flow_farneback_torch

            prev_gray = _to_gray(prev_frame)
            current_gray = _to_gray(current_frame)
            flow = calc_optical_flow_farneback_torch(
                prev_gray,
                current_gray,
                pyr_scale=float(farneback_pyr_scale),
                levels=int(farneback_levels),
                winsize=int(farneback_winsize),
                iterations=int(farneback_iterations),
                poly_n=int(farneback_poly_n),
                poly_sigma=float(farneback_poly_sigma),
                flags=0,
                device=AVAILABLE_DEVICE,
            )
        except Exception:
            flow = None

    # Fastest path: CUDA Farneback
    if flow is None and prefer_cuda and _cuda_flow_available():
        try:
            cuda_flow = cv2.cuda.FarnebackOpticalFlow_create(
                numLevels=int(farneback_levels),
                pyrScale=float(farneback_pyr_scale),
                fastPyramids=False,
                winSize=int(farneback_winsize),
                numIters=int(farneback_iterations),
                polyN=int(farneback_poly_n),
                polySigma=float(farneback_poly_sigma),
                flags=0,
            )
            prev_gpu = cv2.cuda_GpuMat()
            curr_gpu = cv2.cuda_GpuMat()
            prev_gpu.upload(prev_frame)
            curr_gpu.upload(current_frame)
            prev_gray_gpu = cv2.cuda.cvtColor(prev_gpu, cv2.COLOR_BGR2GRAY)
            curr_gray_gpu = cv2.cuda.cvtColor(curr_gpu, cv2.COLOR_BGR2GRAY)
            flow_gpu = cuda_flow.calc(prev_gray_gpu, curr_gray_gpu, None)
            flow = flow_gpu.download()
        except Exception:
            flow = None  # Fall back to OpenCL/CPU

    # Next: OpenCL via UMat
    if flow is None:
        used_umat = False
        if use_umat and opencl_available:
            try:
                prev_gray = cv2.cvtColor(cv2.UMat(prev_frame), cv2.COLOR_BGR2GRAY)
                current_gray = cv2.cvtColor(cv2.UMat(current_frame), cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray,
                    current_gray,
                    None,
                    pyr_scale=float(farneback_pyr_scale),
                    levels=int(farneback_levels),
                    winsize=int(farneback_winsize),
                    iterations=int(farneback_iterations),
                    poly_n=int(farneback_poly_n),
                    poly_sigma=float(farneback_poly_sigma),
                    flags=0,
                )
                used_umat = True
            except Exception:
                flow = None
        if flow is None and not used_umat:
            # CPU fallback
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                current_gray,
                None,
                pyr_scale=float(farneback_pyr_scale),
                levels=int(farneback_levels),
                winsize=int(farneback_winsize),
                iterations=int(farneback_iterations),
                poly_n=int(farneback_poly_n),
                poly_sigma=float(farneback_poly_sigma),
                flags=0,
            )

    # Ensure ndarray regardless of UMat path
    if hasattr(flow, "get"):
        flow = flow.get()

    # Upscale flow to original resolution if downscaled
    if scale != 1.0:
        flow = cv2.resize(flow, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        flow /= scale  # compensate for downscaling

    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    magnitude = np.nan_to_num(magnitude)
    magnitude_normalized = np.clip(
        cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX), 0, 255
    ).astype(np.uint8)
    angle_degrees = (angle * 180 / np.pi / 2).astype(np.uint8)

    # Store both the display-friendly normalized magnitude (channel 1) and the raw
    # float magnitude (channel 2) so motion index computations can use unscaled values.
    flow_hsv = np.empty((orig_h, orig_w, 3), dtype=np.float32)
    flow_hsv[..., 0] = angle_degrees
    flow_hsv[..., 1] = magnitude_normalized.astype(np.float32)
    flow_hsv[..., 2] = magnitude

    return flow_hsv, flow


def compute_optical_flow_raft(
    prev_frame: np.ndarray,
    current_frame: np.ndarray,
    model: str = "small",
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Compute RAFT optical flow via torchvision and return (H, W, 2) numpy array.
    This is optional and requires torch + torchvision with optical_flow models.
    """
    try:
        import torch
        from annolid.motion.raft_wrapper import compute_raft_flow
    except Exception as exc:  # pragma: no cover - optional
        raise ImportError("RAFT flow requires torch + torchvision") from exc

    prev_t = torch.from_numpy(prev_frame).permute(2, 0, 1).unsqueeze(0).float()
    curr_t = torch.from_numpy(current_frame).permute(2, 0, 1).unsqueeze(0).float()
    flow_t = compute_raft_flow(prev_t, curr_t, model_type=model, device=device)
    return flow_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
