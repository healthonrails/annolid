from dataclasses import dataclass
import importlib
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

from annolid.infrastructure.capabilities import ensure_capability
from annolid.utils.logger import logger


def _looks_like_ultralytics_asset_weight(name: str) -> bool:
    name_lower = str(name or "").strip().lower()
    if not name_lower.endswith(".pt"):
        return False
    if name_lower in {"best.pt", "last.pt"}:
        return False
    return name_lower.startswith(("yolo", "yolov"))


def _maybe_cache_asset_weight(resolved_path: Path) -> Path:
    """
    For Ultralytics-provided asset weights, prefer (and optionally populate) Annolid's cache.

    Ultralytics will happily load from any absolute path. By copying an already-present weight
    from the working directory into the cache, we ensure subsequent runs reuse the cached copy.
    """
    try:
        if not resolved_path.is_file():
            return resolved_path
    except OSError:
        return resolved_path

    if not _looks_like_ultralytics_asset_weight(resolved_path.name):
        return resolved_path

    cache_target = get_ultralytics_weights_cache_dir() / resolved_path.name
    try:
        if resolved_path.resolve() == cache_target.resolve():
            return resolved_path
    except OSError:
        return resolved_path

    try:
        cache_target.parent.mkdir(parents=True, exist_ok=True)
        if cache_target.is_file():
            try:
                if cache_target.stat().st_size == resolved_path.stat().st_size:
                    return cache_target
            except OSError:
                return resolved_path
        shutil.copy2(resolved_path, cache_target)
        logger.info("Cached YOLO weight %s -> %s", resolved_path, cache_target)
        return cache_target
    except OSError as exc:
        logger.debug("Failed to cache YOLO weight '%s': %s", resolved_path, exc)
        return resolved_path


def get_cache_root() -> Path:
    """Return the base cache directory (XDG_CACHE_HOME or ~/.cache)."""
    env_override = os.getenv("XDG_CACHE_HOME")
    if env_override:
        return Path(env_override).expanduser()
    return Path.home() / ".cache"


def get_ultralytics_weights_cache_dir() -> Path:
    """
    Return the directory where Annolid should cache Ultralytics YOLO weights.

    This prevents pretrained checkpoints (e.g. yolo11x-pose.pt) from being
    downloaded into the current working directory.
    """
    return get_cache_root() / "annolid" / "ultralytics" / "weights"


def configure_ultralytics_cache(weights_dir: Optional[Path] = None) -> Path:
    """
    Configure Ultralytics to download/load weights from a stable cache directory.

    Note: This updates Ultralytics' in-memory SETTINGS for the current process
    (it does not persist changes to the user's Ultralytics settings file).
    """
    target = (weights_dir or get_ultralytics_weights_cache_dir()).expanduser()
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.debug("Unable to create Ultralytics cache dir '%s': %s", target, exc)
        return target

    try:
        from ultralytics.utils import SETTINGS  # type: ignore
    except Exception as exc:
        logger.debug("Ultralytics SETTINGS unavailable: %s", exc)
        return target

    # Ensure an absolute path so downloads never end up in the CWD.
    SETTINGS["weights_dir"] = str(target.resolve())
    return target


def ensure_yolo_runtime() -> tuple[str, ...]:
    """Ensure the optional Ultralytics YOLO runtime is available."""
    return ensure_capability("yolo")


def import_ultralytics_symbol(
    symbol_name: str,
    *,
    module_name: str = "ultralytics",
    purpose: str = "YOLO",
) -> Any:
    """Import an Ultralytics symbol after running Annolid's YOLO capability check."""
    ensure_yolo_runtime()
    configure_ultralytics_cache()
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"{purpose} requires the optional YOLO runtime. "
            'Install it with `python -m pip install "annolid[yolo]"`.'
        ) from exc

    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"Installed Ultralytics runtime does not provide {module_name}.{symbol_name}."
        ) from exc


def ensure_ultralytics_asset_cached(asset_name: str) -> Path:
    """
    Ensure an Ultralytics GitHub asset is present inside Annolid's Ultralytics weights cache.

    Some Ultralytics components (notably YOLOE text encoders) call
    `attempt_download_asset("file.ext")` with a bare filename which otherwise downloads into the
    current working directory. Prefetching into `SETTINGS["weights_dir"]` ensures subsequent
    Ultralytics calls resolve the asset from the configured cache directory.
    """
    asset = str(asset_name or "").strip()
    if not asset:
        raise ValueError("asset_name must be non-empty")

    weights_dir = configure_ultralytics_cache()
    target = (weights_dir / asset).expanduser()
    try:
        if target.is_file():
            return target
    except OSError:
        return target

    try:
        ensure_yolo_runtime()
        from ultralytics.utils.downloads import attempt_download_asset  # type: ignore
    except Exception as exc:
        raise RuntimeError("Ultralytics is required to download YOLO assets.") from exc

    downloaded = Path(attempt_download_asset(str(target))).expanduser()
    return downloaded


def _is_supported_model_path(candidate: Path) -> bool:
    """
    Determine whether the candidate path points to a supported YOLO export.
    Accepts regular files (e.g., .pt, .engine, .onnx) and CoreML bundles (.mlpackage).
    """
    try:
        if candidate.is_file():
            return True
        if candidate.is_dir() and candidate.suffix.lower() == ".mlpackage":
            return True
    except OSError:
        return False
    return False


@dataclass(frozen=True)
class YOLOModelSpec:
    """Resolved configuration for loading a YOLO model."""

    weight_path: Path
    device: Any
    backend: str


def _normalize_weight_path(weight_name: str) -> Path:
    weight_path = Path(weight_name)
    if weight_path.suffix == "":
        return weight_path.with_suffix(".pt")
    return weight_path


def _runtime_roots() -> tuple[Path, Path, Path]:
    module_root = Path(__file__).resolve().parent  # annolid/yolo
    annolid_root = module_root.parent  # annolid/
    project_root = annolid_root.parent  # repo root
    return module_root, annolid_root, project_root


def _default_weight_search_roots() -> tuple[Path, ...]:
    module_root, annolid_root, project_root = _runtime_roots()
    package_roots = tuple(
        annolid_root / subdir
        for subdir in ("realtime", "segmentation", "detector", "gui")
    )
    return (
        module_root,
        annolid_root,
        project_root,
        Path.cwd(),
        Path.home() / "Downloads",
        get_ultralytics_weights_cache_dir(),
        Path.home() / ".cache" / "ultralytics",
        *package_roots,
        project_root / "runs",
        annolid_root / "runs",
    )


def _dedupe_paths(paths: Iterable[Path]) -> Iterable[Path]:
    seen: set[Path] = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        yield path


def _training_run_weight_paths(weight_name: str, weight_path: Path) -> Iterable[Path]:
    _, _, project_root = _runtime_roots()
    weight_lower = weight_name.lower()
    run_dirs: Tuple[Tuple[str, str], ...] = (
        ("pose", "runs/pose/train/weights/best.pt"),
        ("segment", "runs/segment/train/weights/best.pt"),
        ("detect", "runs/detect/train/weights/best.pt"),
    )
    for keyword, rel_path in run_dirs:
        if keyword in weight_lower or weight_path.name == "best.pt":
            yield project_root / rel_path
            yield Path.home() / "Downloads" / rel_path


def _candidate_weight_paths(
    weight_name: str, search_roots: Optional[Iterable[Path]] = None
) -> Iterable[Path]:
    """
    Yield candidate weight paths for a given model name.

    The search order prioritises user-provided paths, the Annolid project root,
    the current working directory, and common Ultralytics export locations.
    """
    weight_path = _normalize_weight_path(weight_name)
    roots: list[Path] = []
    if search_roots:
        roots.extend(Path(root) for root in search_roots)
    roots.extend(_default_weight_search_roots())

    seen = set()

    # Absolute or relative path provided by user.
    yield weight_path

    for root in _dedupe_paths(roots):
        candidate = (root / weight_path).resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate

    for candidate in _training_run_weight_paths(weight_name, weight_path):
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


def _ultralytics_asset_cache_target(weight_name: str) -> Optional[Path]:
    requested = str(weight_name or "").strip()
    if not requested:
        return None

    requested_path = Path(requested)
    requested_name = requested_path.name
    has_path_separators = ("/" in requested) or ("\\" in requested)
    is_remote = requested.startswith(("http://", "https://"))
    candidate_name = requested_name if requested_path.suffix else f"{requested_name}.pt"
    looks_like_ultralytics_asset = _looks_like_ultralytics_asset_weight(candidate_name)
    if (
        is_remote
        or requested_path.is_absolute()
        or has_path_separators
        or not looks_like_ultralytics_asset
    ):
        return None

    return get_ultralytics_weights_cache_dir() / candidate_name


def resolve_weight_path(
    weight_name: str, search_roots: Optional[Iterable[Path]] = None
) -> Path:
    """
    Resolve a YOLO weight reference to an existing file path if possible.

    Args:
        weight_name: Name or path provided by the user or registry.
        search_roots: Optional iterable of additional directories to search.

    Returns:
        Path: Existing model path if found; otherwise the original value.
    """
    for candidate in _candidate_weight_paths(weight_name, search_roots):
        if _is_supported_model_path(candidate):
            resolved = _maybe_cache_asset_weight(candidate)
            logger.info("YOLO weight resolved to %s", resolved)
            return resolved

    # If this looks like an Ultralytics asset weight name (e.g. yolo11n.pt), direct downloads
    # to Annolid's cache directory by returning an absolute target path.
    #
    # Ultralytics' internal download helper saves to the provided filename (defaulting to the
    # current working directory), so passing a cache path is the most reliable way to ensure
    # weights are cached for reuse across runs.
    cache_target = _ultralytics_asset_cache_target(weight_name)
    if cache_target is not None:
        logger.info(
            "YOLO weight '%s' not found; will download to cache path %s",
            weight_name,
            cache_target,
        )
        return cache_target

    logger.info(
        "YOLO weight '%s' not found on disk; relying on Ultralytics resolution.",
        weight_name,
    )
    return Path(weight_name)


def ensure_mps_fallback_env() -> bool:
    """Enable PyTorch MPS CPU fallback on macOS unless explicitly overridden by user."""
    if sys.platform != "darwin":
        return False
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    return os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def _mps_is_available(torch_module: Any) -> bool:
    try:
        mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
        return bool(mps_backend is not None and mps_backend.is_available())
    except Exception:
        return False


def resolve_yolo_device(requested: Optional[Any] = None) -> str:
    """Resolve an Ultralytics-compatible device token ('0', 'mps', 'cpu', etc.)."""
    raw = "" if requested is None else str(requested).strip()
    lowered = raw.lower()
    if lowered in {"cpu"}:
        return "cpu"
    if lowered in {"mps"}:
        ensure_mps_fallback_env()
        return "mps"
    if lowered in {"cuda", "gpu"}:
        return "0"
    if raw and lowered != "auto":
        return raw

    try:
        import torch
    except Exception:
        return "cpu"

    if torch.cuda.is_available():
        return "0"
    if _mps_is_available(torch):
        ensure_mps_fallback_env()
        return "mps"
    return "cpu"


def list_available_yolo_devices() -> list[tuple[str, str]]:
    """Return UI-friendly (label, value) device choices for YOLO training/inference."""
    options: list[tuple[str, str]] = [("Auto (recommended)", "")]
    try:
        import torch

        if torch.cuda.is_available():
            try:
                count = int(torch.cuda.device_count() or 1)
            except Exception:
                count = 1
            for idx in range(count):
                try:
                    name = torch.cuda.get_device_name(idx)
                    options.append((f"CUDA (GPU {idx}) — {name}", str(idx)))
                except Exception:
                    options.append((f"CUDA (GPU {idx})", str(idx)))
        if _mps_is_available(torch):
            options.append(("Apple MPS", "mps"))
    except Exception:
        pass
    options.append(("CPU", "cpu"))
    return options


def select_backend(
    weight_name: str,
    search_roots: Optional[Iterable[Path]] = None,
    *,
    device: Optional[str] = None,
) -> YOLOModelSpec:
    """
    Detect available hardware and return the best backend for the provided model.

    The selection order follows TensorRT (if engine exists + CUDA) -> CUDA -> Apple MPS -> CPU.
    """
    import torch

    resolved = resolve_weight_path(weight_name, search_roots=search_roots)

    # CoreML packages are tied to Apple devices and should bypass CUDA/TensorRT selection.
    if resolved.suffix.lower() == ".mlpackage":
        if not resolved.exists():
            raise FileNotFoundError(f"CoreML package '{resolved}' not found on disk.")
        logger.info("Using CoreML package for %s", resolved)
        return YOLOModelSpec(
            weight_path=resolved, device=torch.device("cpu"), backend="CoreML (CPU)"
        )

    requested = "" if device is None else str(device).strip().lower()
    prefer_cpu = requested == "cpu"
    prefer_mps = requested == "mps"
    prefer_cuda = (
        requested.startswith("cuda") or requested.isdigit() or requested == "gpu"
    )

    # TensorRT takes precedence when CUDA is used and an engine exists alongside the weight.
    if not prefer_cpu and not prefer_mps and torch.cuda.is_available():
        cuda_idx = "0"
        if requested.isdigit():
            cuda_idx = requested
        elif requested.startswith("cuda:") and requested.split(":", 1)[1].isdigit():
            cuda_idx = requested.split(":", 1)[1]
        torch_device = torch.device(f"cuda:{cuda_idx}")
        engine_path = resolved.with_suffix(".engine")
        if engine_path.is_file():
            logger.info("Using TensorRT engine for %s", engine_path)
            return YOLOModelSpec(
                weight_path=engine_path, device=torch_device, backend="TensorRT (GPU)"
            )

        # Fall back to the resolved weight on GPU.
        if resolved.suffix == ".pt" or resolved.exists():
            logger.info("Using PyTorch backend on CUDA for %s", resolved)
            return YOLOModelSpec(
                weight_path=resolved, device=torch_device, backend="PyTorch (GPU)"
            )
        raise FileNotFoundError(
            f"CUDA available but weight '{resolved}' could not be located."
        )

    # Apple Silicon MPS acceleration
    if not prefer_cpu and not prefer_cuda and _mps_is_available(torch):
        ensure_mps_fallback_env()
        torch_device = torch.device("mps")
        if resolved.suffix == ".pt" or resolved.exists():
            logger.info("Using PyTorch backend on Apple MPS for %s", resolved)
            return YOLOModelSpec(
                weight_path=resolved, device=torch_device, backend="PyTorch (MPS)"
            )
        raise FileNotFoundError(
            f"MPS available but weight '{resolved}' could not be located."
        )

    # CPU fallback
    torch_device = torch.device("cpu")
    if resolved.suffix == ".pt" or resolved.exists():
        if not prefer_cpu:
            logger.warning(
                "No compatible GPU detected; running %s on CPU. Expect reduced performance.",
                resolved,
            )
        return YOLOModelSpec(
            weight_path=resolved, device=torch_device, backend="PyTorch (CPU)"
        )

    raise FileNotFoundError(
        f"Unable to locate model weights for '{weight_name}'. Provide a valid path."
    )


def load_yolo_model(
    weight_name: str,
    search_roots: Optional[Iterable[Path]] = None,
    *,
    device: Optional[str] = None,
):
    """
    Load a YOLO model with automatic backend selection.

    Returns:
        tuple(model, YOLOModelSpec)
    """
    configure_ultralytics_cache()
    spec = select_backend(weight_name, search_roots=search_roots, device=device)
    YOLO = import_ultralytics_symbol("YOLO", purpose="YOLO inference")

    model = YOLO(str(spec.weight_path))
    if (
        "TensorRT" not in spec.backend
        and spec.weight_path.suffix.lower() != ".mlpackage"
    ):
        model.to(spec.device)
    return model, spec


def export_yolo_weights(
    weight_name: str,
    *,
    export_format: str = "onnx",
    imgsz: int = 640,
    half: bool = False,
    device: Optional[str] = None,
    **kwargs: Any,
) -> Any:
    """Export YOLO weights to an Ultralytics-supported deployment format."""
    configure_ultralytics_cache()
    resolved = resolve_weight_path(weight_name)
    YOLO = import_ultralytics_symbol("YOLO", purpose="YOLO export")
    model = YOLO(str(resolved))
    export_kwargs: dict[str, Any] = {
        "format": str(export_format),
        "imgsz": int(imgsz),
        "half": bool(half),
    }
    if device is not None and str(device).strip():
        export_kwargs["device"] = str(device).strip()
    export_kwargs.update(kwargs)
    return model.export(**export_kwargs)
