from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading
from typing import Any, Dict, Mapping, Optional, Tuple

import cv2
import numpy as np
from annolid.utils.devices import clear_device_cache, get_device
from annolid.utils.logger import logger

_LOGGED_TORCH_BACKEND_FAILURES: set[tuple[str, str]] = set()
_LOGGED_TORCH_PARAMETER_FALLBACKS: set[tuple[int, int]] = set()
_CUDA_FLOW_CACHE: OrderedDict[tuple[int, float, int, int, int, float, int], Any] = (
    OrderedDict()
)
_CUDA_FLOW_LOCK = threading.Lock()
_MAX_CUDA_FLOW_CACHE = 8

DEFAULT_FARNEBACK_PYR_SCALE = 0.5
DEFAULT_FARNEBACK_LEVELS = 1
DEFAULT_FARNEBACK_WINSIZE = 15
DEFAULT_FARNEBACK_ITERATIONS = 3
DEFAULT_FARNEBACK_POLY_N = 5
DEFAULT_FARNEBACK_POLY_SIGMA = 1.1
LEGACY_FARNEBACK_WINSIZE = 1
LEGACY_FARNEBACK_POLY_N = 3

OPTICAL_FLOW_BACKENDS: tuple[str, ...] = ("farneback", "farneback_torch", "raft")
OPTICAL_FLOW_BACKEND_LABELS: tuple[str, ...] = (
    "farneback (default)",
    "farneback (torch)",
    "raft (torchvision)",
)


def coerce_bool(value: Any, default: bool = False) -> bool:
    """Coerce Python/QSettings values (including 'false'/'0'/'no') safely to bool."""
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "on", "y", "t"}:
            return True
        if token in {"false", "0", "no", "off", "n", "f", ""}:
            return False
        return bool(default)
    return bool(value)


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
        try:
            out_f = float(value)
            if not np.isfinite(out_f):
                return int(default)
            return int(round(out_f))
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


def normalize_optical_flow_backend(backend: Any) -> str:
    """Normalize backend names or UI combo labels to a canonical backend key."""
    token = str(backend or "").strip().lower()
    if "raft" in token:
        return "raft"
    if "torch" in token:
        return "farneback_torch"
    return "farneback"


def optical_flow_backend_combo_index(backend: Any) -> int:
    """Map a backend name/label to its index in OPTICAL_FLOW_BACKEND_LABELS."""
    normalized = normalize_optical_flow_backend(backend)
    if normalized == "raft":
        return 2
    if normalized == "farneback_torch":
        return 1
    return 0


def resolve_optical_flow_backend_flags(
    backend: Any = None,
    *,
    use_raft: bool = False,
    use_torch_farneback: bool = False,
) -> Tuple[bool, bool]:
    """Resolve `(use_raft, use_torch_farneback)` flags from backend name + overrides."""
    normalized = (
        normalize_optical_flow_backend(backend) if backend is not None else None
    )
    resolved_raft = bool(use_raft) or (normalized == "raft")
    resolved_torch = (
        bool(use_torch_farneback) or (normalized == "farneback_torch")
    ) and not resolved_raft
    return resolved_raft, resolved_torch


def _read_setting(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


@dataclass
class OpticalFlowPreferences:
    """Canonical optical-flow UI and algorithm preferences."""

    backend: str = "farneback"
    raft_model: str = "small"
    visualization: str = "hsv"
    opacity: int = 70
    quiver_step: int = 16
    quiver_gain: float = 1.0
    stable_hsv: bool = True
    farneback_pyr_scale: float = DEFAULT_FARNEBACK_PYR_SCALE
    farneback_levels: int = DEFAULT_FARNEBACK_LEVELS
    farneback_winsize: int = DEFAULT_FARNEBACK_WINSIZE
    farneback_iterations: int = DEFAULT_FARNEBACK_ITERATIONS
    farneback_poly_n: int = DEFAULT_FARNEBACK_POLY_N
    farneback_poly_sigma: float = DEFAULT_FARNEBACK_POLY_SIGMA

    @classmethod
    def from_source(
        cls,
        source: Any,
        *,
        defaults: Optional["OpticalFlowPreferences"] = None,
    ) -> "OpticalFlowPreferences":
        base = defaults or cls()

        def _get(*keys: str, default: Any) -> Any:
            sentinel = object()
            for key in keys:
                val = _read_setting(source, key, sentinel)
                if val is not sentinel and val is not None:
                    return val
            return default

        backend = normalize_optical_flow_backend(
            _get(
                "optical_flow_backend",
                "backend",
                "optical_flow/backend",
                default=base.backend,
            )
        )
        raft_model = (
            str(
                _get(
                    "optical_flow_raft_model",
                    "raft_model",
                    "optical_flow/raft_model",
                    default=base.raft_model,
                )
            )
            .strip()
            .lower()
        )
        if raft_model not in {"small", "large"}:
            raft_model = "small"

        viz = (
            str(
                _get(
                    "flow_visualization",
                    "visualization",
                    "optical_flow/visualization",
                    default=base.visualization,
                )
            )
            .strip()
            .lower()
        )
        if viz not in {"hsv", "quiver"}:
            viz = base.visualization

        opacity = max(
            0,
            min(
                100,
                _coerce_int(
                    _get(
                        "flow_opacity",
                        "opacity",
                        "optical_flow/opacity",
                        default=base.opacity,
                    ),
                    base.opacity,
                ),
            ),
        )
        quiver_step = max(
            1,
            _coerce_int(
                _get(
                    "flow_quiver_step",
                    "quiver_step",
                    "optical_flow/quiver_step",
                    default=base.quiver_step,
                ),
                base.quiver_step,
            ),
        )
        quiver_gain = max(
            0.05,
            _coerce_float(
                _get(
                    "flow_quiver_gain",
                    "quiver_gain",
                    "optical_flow/quiver_gain",
                    default=base.quiver_gain,
                ),
                base.quiver_gain,
            ),
        )
        stable_hsv = coerce_bool(
            _get(
                "flow_stable_hsv",
                "stable_hsv",
                "optical_flow/stable_hsv",
                default=base.stable_hsv,
            ),
            base.stable_hsv,
        )

        pyr_scale = _coerce_float(
            _get(
                "flow_farneback_pyr_scale",
                "farneback_pyr_scale",
                "optical_flow/farneback_pyr_scale",
                default=base.farneback_pyr_scale,
            ),
            base.farneback_pyr_scale,
        )
        if not (0.0 < pyr_scale < 1.0):
            pyr_scale = DEFAULT_FARNEBACK_PYR_SCALE

        levels = max(
            1,
            _coerce_int(
                _get(
                    "flow_farneback_levels",
                    "farneback_levels",
                    "optical_flow/farneback_levels",
                    default=base.farneback_levels,
                ),
                base.farneback_levels,
            ),
        )
        winsize = max(
            1,
            _coerce_int(
                _get(
                    "flow_farneback_winsize",
                    "farneback_winsize",
                    "optical_flow/farneback_winsize",
                    default=base.farneback_winsize,
                ),
                base.farneback_winsize,
            ),
        )
        iterations = max(
            1,
            _coerce_int(
                _get(
                    "flow_farneback_iterations",
                    "farneback_iterations",
                    "optical_flow/farneback_iterations",
                    default=base.farneback_iterations,
                ),
                base.farneback_iterations,
            ),
        )
        poly_n = max(
            3,
            _coerce_int(
                _get(
                    "flow_farneback_poly_n",
                    "farneback_poly_n",
                    "optical_flow/farneback_poly_n",
                    default=base.farneback_poly_n,
                ),
                base.farneback_poly_n,
            ),
        )
        poly_sigma = _coerce_float(
            _get(
                "flow_farneback_poly_sigma",
                "farneback_poly_sigma",
                "optical_flow/farneback_poly_sigma",
                default=base.farneback_poly_sigma,
            ),
            base.farneback_poly_sigma,
        )
        if poly_sigma <= 0.0:
            poly_sigma = DEFAULT_FARNEBACK_POLY_SIGMA

        return cls(
            backend=backend,
            raft_model=raft_model,
            visualization=viz,
            opacity=opacity,
            quiver_step=quiver_step,
            quiver_gain=quiver_gain,
            stable_hsv=stable_hsv,
            farneback_pyr_scale=pyr_scale,
            farneback_levels=levels,
            farneback_winsize=winsize,
            farneback_iterations=iterations,
            farneback_poly_n=poly_n,
            farneback_poly_sigma=poly_sigma,
        )

    def to_profile_dict(self) -> Dict[str, Any]:
        """Serialize to the canonical settings-profile dictionary schema."""
        return {
            "backend": self.backend,
            "raft_model": self.raft_model,
            "visualization": self.visualization,
            "opacity": int(self.opacity),
            "quiver_step": int(self.quiver_step),
            "quiver_gain": float(self.quiver_gain),
            "stable_hsv": bool(self.stable_hsv),
            "farneback_pyr_scale": float(self.farneback_pyr_scale),
            "farneback_levels": int(self.farneback_levels),
            "farneback_winsize": int(self.farneback_winsize),
            "farneback_iterations": int(self.farneback_iterations),
            "farneback_poly_n": int(self.farneback_poly_n),
            "farneback_poly_sigma": float(self.farneback_poly_sigma),
        }

    def qsettings_items(self) -> Tuple[Tuple[str, Any], ...]:
        """Return `(qsettings_key, value)` pairs for persistence."""
        return (
            ("optical_flow/backend", self.backend),
            ("optical_flow/raft_model", self.raft_model),
            ("optical_flow/visualization", self.visualization),
            ("optical_flow/opacity", int(self.opacity)),
            ("optical_flow/quiver_step", int(self.quiver_step)),
            ("optical_flow/quiver_gain", float(self.quiver_gain)),
            ("optical_flow/stable_hsv", bool(self.stable_hsv)),
            ("optical_flow/farneback_pyr_scale", float(self.farneback_pyr_scale)),
            ("optical_flow/farneback_levels", int(self.farneback_levels)),
            ("optical_flow/farneback_winsize", int(self.farneback_winsize)),
            ("optical_flow/farneback_iterations", int(self.farneback_iterations)),
            ("optical_flow/farneback_poly_n", int(self.farneback_poly_n)),
            ("optical_flow/farneback_poly_sigma", float(self.farneback_poly_sigma)),
        )

    def window_attribute_items(self) -> Tuple[Tuple[str, Any], ...]:
        """Return `(attr_name, value)` pairs mirrored onto manager/window objects."""
        return (
            ("optical_flow_backend", self.backend),
            ("optical_flow_raft_model", self.raft_model),
            ("flow_visualization", self.visualization),
            ("flow_opacity", int(self.opacity)),
            ("flow_quiver_step", int(self.quiver_step)),
            ("flow_quiver_gain", float(self.quiver_gain)),
            ("flow_stable_hsv", bool(self.stable_hsv)),
            ("flow_farneback_pyr_scale", float(self.farneback_pyr_scale)),
            ("flow_farneback_levels", int(self.farneback_levels)),
            ("flow_farneback_winsize", int(self.farneback_winsize)),
            ("flow_farneback_iterations", int(self.farneback_iterations)),
            ("flow_farneback_poly_n", int(self.farneback_poly_n)),
            ("flow_farneback_poly_sigma", float(self.farneback_poly_sigma)),
        )

    def apply_to_target(self, target: Any) -> None:
        """Mirror preference attributes onto a window or manager instance."""
        for attr, value in self.window_attribute_items():
            setattr(target, attr, value)


def optical_flow_settings_from(source: Any) -> Dict[str, Any]:
    """
    Return a normalized optical-flow settings mapping using the same keys
    the Settings menu exposes (mirrored onto the window/manager).

    This is the preferred way to thread optical flow preferences into
    prediction/tracking code without duplicating per-key plumbing.
    """
    compute_enabled = coerce_bool(
        _read_setting(
            source,
            "compute_optical_flow",
            _read_setting(source, "optical_flow/compute", False),
        ),
        False,
    )
    prefs = OpticalFlowPreferences.from_source(source)

    scale = _coerce_float(_read_setting(source, "optical_flow_scale", 1.0), 1.0)
    if scale <= 0:
        scale = 1.0
    scale = min(1.0, scale)
    max_dim = _coerce_pos_int_or_none(_read_setting(source, "optical_flow_max_dim"))
    torch_device_value = _read_setting(source, "optical_flow_torch_device")
    torch_device = (
        str(torch_device_value).strip() if torch_device_value is not None else None
    )
    if not torch_device or torch_device.lower() == "auto":
        torch_device = None

    return {
        "compute_optical_flow": compute_enabled,
        "optical_flow_backend": prefs.backend,
        "optical_flow_raft_model": prefs.raft_model,
        "flow_farneback_pyr_scale": prefs.farneback_pyr_scale,
        "flow_farneback_levels": prefs.farneback_levels,
        "flow_farneback_winsize": prefs.farneback_winsize,
        "flow_farneback_iterations": prefs.farneback_iterations,
        "flow_farneback_poly_n": prefs.farneback_poly_n,
        "flow_farneback_poly_sigma": prefs.farneback_poly_sigma,
        "optical_flow_scale": scale,
        "optical_flow_max_dim": max_dim,
        "optical_flow_torch_device": torch_device,
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
      - `optical_flow_torch_device`
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
        "torch_device": normalized["optical_flow_torch_device"],
    }


def _to_gray(frame: np.ndarray) -> np.ndarray:
    """Convert BGR, BGRA, or grayscale input to a 2D single-channel grayscale array."""
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3:
        channels = frame.shape[2]
        if channels == 1:
            return frame[..., 0]
        if channels == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if channels == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported frame shape for grayscale conversion: {frame.shape}")


def _to_rgb_for_raft(frame: np.ndarray) -> np.ndarray:
    """Convert OpenCV BGR/BGRA/grayscale image to contiguous 3-channel RGB for RAFT."""
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    if frame.ndim == 3:
        channels = frame.shape[2]
        if channels == 1:
            return cv2.cvtColor(frame[..., 0], cv2.COLOR_GRAY2RGB)
        if channels == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if channels == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
    raise ValueError(f"Unsupported frame shape for RAFT optical flow: {frame.shape}")


def _cuda_flow_available() -> bool:
    """Check whether CUDA Farneback is available."""
    return bool(
        getattr(cv2, "cuda", None)
        and hasattr(cv2.cuda, "getCudaEnabledDeviceCount")
        and cv2.cuda.getCudaEnabledDeviceCount() > 0
        and hasattr(cv2.cuda, "FarnebackOpticalFlow_create")
    )


def _get_cuda_farneback(
    *,
    levels: int,
    pyr_scale: float,
    winsize: int,
    iterations: int,
    poly_n: int,
    poly_sigma: float,
    flags: int = 0,
) -> Any:
    key = (levels, pyr_scale, winsize, iterations, poly_n, poly_sigma, flags)
    with _CUDA_FLOW_LOCK:
        cached = _CUDA_FLOW_CACHE.get(key)
        if cached is not None:
            _CUDA_FLOW_CACHE.move_to_end(key)
            return cached
        cuda_flow = cv2.cuda.FarnebackOpticalFlow_create(
            numLevels=levels,
            pyrScale=pyr_scale,
            fastPyramids=False,
            winSize=winsize,
            numIters=iterations,
            polyN=poly_n,
            polySigma=poly_sigma,
            flags=flags,
        )
        _CUDA_FLOW_CACHE[key] = cuda_flow
        while len(_CUDA_FLOW_CACHE) > _MAX_CUDA_FLOW_CACHE:
            _CUDA_FLOW_CACHE.popitem(last=False)
        return cuda_flow


def torch_farneback_supports_parameters(*, winsize: int, poly_n: int) -> bool:
    """Return whether Torch Farneback is reliable for the requested parameters."""
    return not (
        int(winsize) == LEGACY_FARNEBACK_WINSIZE
        and int(poly_n) == LEGACY_FARNEBACK_POLY_N
    )


def _compute_effective_scale(
    orig_h: int, orig_w: int, scale: float, max_dim: Optional[int]
) -> float:
    eff_scale = float(max(scale, 1e-3))
    if max_dim is not None and max_dim > 0 and max(orig_h, orig_w) > 0:
        dynamic_scale = min(1.0, float(max_dim) / float(max(orig_h, orig_w)))
        eff_scale = min(eff_scale, dynamic_scale)
    return eff_scale


def compute_optical_flow_field(
    prev_frame: np.ndarray,
    current_frame: np.ndarray,
    scale: float = 1.0,
    max_dim: Optional[int] = None,
    use_umat: Optional[bool] = None,
    prefer_cuda: Optional[bool] = None,
    use_raft: bool = False,
    raft_model: str = "small",
    use_torch_farneback: bool = False,
    torch_device: Optional[str] = None,
    farneback_pyr_scale: float = DEFAULT_FARNEBACK_PYR_SCALE,
    farneback_levels: int = DEFAULT_FARNEBACK_LEVELS,
    farneback_winsize: int = DEFAULT_FARNEBACK_WINSIZE,
    farneback_iterations: int = DEFAULT_FARNEBACK_ITERATIONS,
    farneback_poly_n: int = DEFAULT_FARNEBACK_POLY_N,
    farneback_poly_sigma: float = DEFAULT_FARNEBACK_POLY_SIGMA,
    backend: Optional[str] = None,
) -> np.ndarray:
    """
    Compute dense `(H, W, 2)` float32 optical flow without building the HSV summary.

    Prefer this helper in pipelines that only consume `(dx, dy)` flow vectors
    (such as `flow_runner` and CUTIE segment processing) to avoid redundant
    polar/normalization allocations.
    """
    if prev_frame is None or current_frame is None:
        raise ValueError("prev_frame and current_frame must be non-None arrays.")
    if prev_frame.shape != current_frame.shape:
        raise ValueError(
            f"prev_frame and current_frame must share the same shape, "
            f"got {prev_frame.shape} vs {current_frame.shape}"
        )

    use_raft, use_torch_farneback = resolve_optical_flow_backend_flags(
        backend,
        use_raft=use_raft,
        use_torch_farneback=use_torch_farneback,
    )

    orig_h, orig_w = prev_frame.shape[:2]
    eff_scale = _compute_effective_scale(orig_h, orig_w, scale, max_dim)

    scaled_h, scaled_w = orig_h, orig_w
    scale_x = 1.0
    scale_y = 1.0
    if eff_scale != 1.0:
        scaled_w = max(1, int(round(orig_w * eff_scale)))
        scaled_h = max(1, int(round(orig_h * eff_scale)))
        scale_x = float(scaled_w) / float(orig_w)
        scale_y = float(scaled_h) / float(orig_h)

    resize_interp = cv2.INTER_AREA if eff_scale < 1.0 else cv2.INTER_LINEAR

    # Lazily prepare grayscale frames only once and resize in grayscale when RAFT is unused.
    prev_work = prev_frame
    curr_work = current_frame
    prev_gray_cached: Optional[np.ndarray] = None
    curr_gray_cached: Optional[np.ndarray] = None

    if eff_scale != 1.0 and use_raft:
        prev_work = cv2.resize(
            prev_frame, (scaled_w, scaled_h), interpolation=resize_interp
        )
        curr_work = cv2.resize(
            current_frame, (scaled_w, scaled_h), interpolation=resize_interp
        )

    def _get_gray_pair() -> Tuple[np.ndarray, np.ndarray]:
        nonlocal prev_gray_cached, curr_gray_cached
        if prev_gray_cached is not None and curr_gray_cached is not None:
            return prev_gray_cached, curr_gray_cached
        if eff_scale != 1.0 and not use_raft:
            prev_g = _to_gray(prev_frame)
            curr_g = _to_gray(current_frame)
            prev_gray_cached = cv2.resize(
                prev_g, (scaled_w, scaled_h), interpolation=resize_interp
            )
            curr_gray_cached = cv2.resize(
                curr_g, (scaled_w, scaled_h), interpolation=resize_interp
            )
        else:
            prev_gray_cached = _to_gray(prev_work)
            curr_gray_cached = _to_gray(curr_work)
        return prev_gray_cached, curr_gray_cached

    prefer_cuda = True if prefer_cuda is None else bool(prefer_cuda)
    opencl_available = cv2.ocl.haveOpenCL()
    if use_umat is None:
        use_umat = opencl_available and cv2.ocl.useOpenCL()
    elif use_umat and opencl_available:
        cv2.ocl.setUseOpenCL(True)
        use_umat = cv2.ocl.useOpenCL()
    else:
        use_umat = False

    torch_parameters_supported = torch_farneback_supports_parameters(
        winsize=farneback_winsize,
        poly_n=farneback_poly_n,
    )
    if use_torch_farneback and not torch_parameters_supported:
        legacy_key = (int(farneback_winsize), int(farneback_poly_n))
        if legacy_key not in _LOGGED_TORCH_PARAMETER_FALLBACKS:
            _LOGGED_TORCH_PARAMETER_FALLBACKS.add(legacy_key)
            logger.warning(
                "Torch Farneback is not reliable for legacy parameters "
                "winsize=%s, poly_n=%s; using OpenCV Farneback with the saved "
                "parameters. Set winsize=%s and poly_n=%s to enable Torch/GPU flow.",
                farneback_winsize,
                farneback_poly_n,
                DEFAULT_FARNEBACK_WINSIZE,
                DEFAULT_FARNEBACK_POLY_N,
            )
        use_torch_farneback = False

    pyr_scale_f = float(farneback_pyr_scale)
    levels_i = int(farneback_levels)
    winsize_i = int(farneback_winsize)
    iterations_i = int(farneback_iterations)
    poly_n_i = int(farneback_poly_n)
    poly_sigma_f = float(farneback_poly_sigma)

    flow: Optional[np.ndarray] = None
    resolved_torch_device = (
        (torch_device or get_device()) if use_raft or use_torch_farneback else None
    )

    # Try RAFT (torchvision) first if requested and available.
    if use_raft:
        try:
            flow = compute_optical_flow_raft(
                prev_work,
                curr_work,
                model=raft_model,
                device=resolved_torch_device,
            )
        except Exception as exc:
            failure_key = ("raft", str(resolved_torch_device))
            if failure_key not in _LOGGED_TORCH_BACKEND_FAILURES:
                _LOGGED_TORCH_BACKEND_FAILURES.add(failure_key)
                logger.warning(
                    "RAFT optical flow failed on %s; falling back to Farneback: %s",
                    resolved_torch_device,
                    exc,
                )
            clear_device_cache(device=resolved_torch_device)
            flow = None

    # Torch Farneback (optional).
    if flow is None and use_torch_farneback:
        try:
            from annolid.motion.farneback_torch import calc_optical_flow_farneback_torch

            prev_gray, current_gray = _get_gray_pair()
            flow = calc_optical_flow_farneback_torch(
                prev_gray,
                current_gray,
                pyr_scale=pyr_scale_f,
                levels=levels_i,
                winsize=winsize_i,
                iterations=iterations_i,
                poly_n=poly_n_i,
                poly_sigma=poly_sigma_f,
                flags=0,
                device=resolved_torch_device,
            )
        except Exception as exc:
            failure_key = ("farneback_torch", str(resolved_torch_device))
            if failure_key not in _LOGGED_TORCH_BACKEND_FAILURES:
                _LOGGED_TORCH_BACKEND_FAILURES.add(failure_key)
                logger.warning(
                    "Torch Farneback failed on %s; falling back to OpenCV: %s",
                    resolved_torch_device,
                    exc,
                )
            clear_device_cache(device=resolved_torch_device)
            flow = None

    # Fastest OpenCV path: CUDA Farneback.
    if flow is None and prefer_cuda and _cuda_flow_available():
        try:
            cuda_flow = _get_cuda_farneback(
                levels=levels_i,
                pyr_scale=pyr_scale_f,
                winsize=winsize_i,
                iterations=iterations_i,
                poly_n=poly_n_i,
                poly_sigma=poly_sigma_f,
                flags=0,
            )
            prev_gray, current_gray = _get_gray_pair()
            prev_gpu = cv2.cuda_GpuMat()
            curr_gpu = cv2.cuda_GpuMat()
            prev_gpu.upload(prev_gray)
            curr_gpu.upload(current_gray)
            flow_gpu = cuda_flow.calc(prev_gpu, curr_gpu, None)
            flow = flow_gpu.download()
        except Exception:
            flow = None  # Fall back to OpenCL/CPU

    # Next: OpenCL via UMat.
    if flow is None:
        prev_gray, current_gray = _get_gray_pair()
        used_umat = False
        if use_umat and opencl_available:
            try:
                flow = cv2.calcOpticalFlowFarneback(
                    cv2.UMat(prev_gray),
                    cv2.UMat(current_gray),
                    None,
                    pyr_scale=pyr_scale_f,
                    levels=levels_i,
                    winsize=winsize_i,
                    iterations=iterations_i,
                    poly_n=poly_n_i,
                    poly_sigma=poly_sigma_f,
                    flags=0,
                )
                used_umat = True
            except Exception:
                flow = None
        if flow is None and not used_umat:
            # CPU fallback.
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray,
                current_gray,
                None,
                pyr_scale=pyr_scale_f,
                levels=levels_i,
                winsize=winsize_i,
                iterations=iterations_i,
                poly_n=poly_n_i,
                poly_sigma=poly_sigma_f,
                flags=0,
            )

    # Ensure float32 ndarray regardless of UMat path.
    if hasattr(flow, "get"):
        flow = flow.get()
    flow = np.asarray(flow, dtype=np.float32)

    # Upscale flow to original resolution if downscaled.
    if scaled_w != orig_w or scaled_h != orig_h:
        flow = cv2.resize(flow, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        flow[..., 0] /= scale_x
        flow[..., 1] /= scale_y

    if not np.isfinite(flow).all():
        flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)

    return flow


def flow_to_hsv_map(flow: np.ndarray) -> np.ndarray:
    """
    Build the `(H, W, 3)` float32 flow summary map used by Annolid pipelines:
      - channel 0: OpenCV hue angle in `[0, 180)` degrees/2 (uint8 quantized into float32)
      - channel 1: display-friendly `[0, 255]` min-max normalized magnitude
      - channel 2: raw unscaled float32 magnitude (used for motion index calculations)
    """
    flow_f32 = np.asarray(flow, dtype=np.float32)
    orig_h, orig_w = flow_f32.shape[:2]
    magnitude, angle_deg = cv2.cartToPolar(
        flow_f32[..., 0], flow_f32[..., 1], angleInDegrees=True
    )
    if not np.isfinite(magnitude).all():
        magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)
    if not np.isfinite(angle_deg).all():
        angle_deg = np.nan_to_num(angle_deg, nan=0.0, posinf=0.0, neginf=0.0)

    magnitude_normalized = np.clip(
        cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX), 0, 255
    ).astype(np.uint8)
    angle_half = np.clip(angle_deg * 0.5, 0, 179).astype(np.uint8)

    flow_hsv = np.empty((orig_h, orig_w, 3), dtype=np.float32)
    flow_hsv[..., 0] = angle_half
    flow_hsv[..., 1] = magnitude_normalized
    flow_hsv[..., 2] = magnitude
    return flow_hsv


def flow_hsv_to_bgr(flow_hsv: np.ndarray) -> np.ndarray:
    """
    Convert Annolid's float32 `flow_hsv` `(angle_half, norm_mag, raw_mag)` map
    into a `uint8` BGR image suitable for `cv2.addWeighted` overlays.
    """
    arr = np.asarray(flow_hsv)
    if arr.ndim != 3 or arr.shape[2] < 2:
        raise ValueError(f"Expected flow_hsv with shape (H, W, 3), got {arr.shape}")
    h, w = arr.shape[:2]
    hsv_u8 = np.empty((h, w, 3), dtype=np.uint8)
    hsv_u8[..., 0] = np.clip(np.nan_to_num(arr[..., 0]), 0, 179).astype(np.uint8)
    hsv_u8[..., 1] = 255
    hsv_u8[..., 2] = np.clip(np.nan_to_num(arr[..., 1]), 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv_u8, cv2.COLOR_HSV2BGR)


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
    torch_device: Optional[str] = None,
    farneback_pyr_scale: float = DEFAULT_FARNEBACK_PYR_SCALE,
    farneback_levels: int = DEFAULT_FARNEBACK_LEVELS,
    farneback_winsize: int = DEFAULT_FARNEBACK_WINSIZE,
    farneback_iterations: int = DEFAULT_FARNEBACK_ITERATIONS,
    farneback_poly_n: int = DEFAULT_FARNEBACK_POLY_N,
    farneback_poly_sigma: float = DEFAULT_FARNEBACK_POLY_SIGMA,
    backend: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute dense optical flow with automatic device selection.

    When use_raft=True (or backend='raft') and torch/torchvision RAFT are
    available, uses RAFT first; otherwise falls back to Torch Farneback (when
    requested), CUDA Farneback, OpenCL/UMat, then CPU.
    Frames can be optionally downscaled for speed via `scale` and/or `max_dim`.
    The returned flow is always resized back to the original resolution and
    scaled so downstream consumers do not need to adjust.

    The historical `winsize=1, poly_n=3` combination remains supported through
    OpenCV. Torch Farneback is bypassed for that ill-conditioned combination so
    existing projects keep their saved numerical behavior.
    """
    flow = compute_optical_flow_field(
        prev_frame,
        current_frame,
        scale=scale,
        max_dim=max_dim,
        use_umat=use_umat,
        prefer_cuda=prefer_cuda,
        use_raft=use_raft,
        raft_model=raft_model,
        use_torch_farneback=use_torch_farneback,
        torch_device=torch_device,
        farneback_pyr_scale=farneback_pyr_scale,
        farneback_levels=farneback_levels,
        farneback_winsize=farneback_winsize,
        farneback_iterations=farneback_iterations,
        farneback_poly_n=farneback_poly_n,
        farneback_poly_sigma=farneback_poly_sigma,
        backend=backend,
    )
    flow_hsv = flow_to_hsv_map(flow)
    return flow_hsv, flow


def compute_optical_flow_raft(
    prev_frame: np.ndarray,
    current_frame: np.ndarray,
    model: str = "small",
    device: Optional[str] = None,
) -> np.ndarray:
    """
    Compute RAFT optical flow via torchvision and return (H, W, 2) float32 numpy array.
    This is optional and requires torch + torchvision with optical_flow models.

    OpenCV BGR/BGRA/grayscale frames are converted to RGB and passed as uint8
    (or float when already floating-point) so torchvision RAFT receives properly
    normalized [-1, 1] RGB inputs.
    """
    try:
        import torch
        from annolid.motion.raft_wrapper import compute_raft_flow
    except Exception as exc:  # pragma: no cover - optional
        raise ImportError("RAFT flow requires torch + torchvision") from exc

    prev_rgb = np.ascontiguousarray(_to_rgb_for_raft(prev_frame))
    curr_rgb = np.ascontiguousarray(_to_rgb_for_raft(current_frame))
    prev_t = torch.from_numpy(prev_rgb).permute(2, 0, 1).unsqueeze(0)
    curr_t = torch.from_numpy(curr_rgb).permute(2, 0, 1).unsqueeze(0)
    flow_t = compute_raft_flow(prev_t, curr_t, model_type=model, device=device)
    return (
        flow_t.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32, copy=False)
    )
