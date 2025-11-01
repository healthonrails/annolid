from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch

from annolid.utils.logger import logger


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
    device: torch.device
    backend: str


def _candidate_weight_paths(weight_name: str,
                            search_roots: Optional[Iterable[Path]] = None) -> Iterable[Path]:
    """
    Yield candidate weight paths for a given model name.

    The search order prioritises user-provided paths, the Annolid project root,
    the current working directory, and common Ultralytics export locations.
    """
    weight_path = Path(weight_name)
    if weight_path.suffix == "":
        weight_path = weight_path.with_suffix(".pt")
    roots = []
    if search_roots:
        roots.extend(Path(root) for root in search_roots)

    module_root = Path(__file__).resolve().parent           # annolid/yolo
    annolid_root = module_root.parent                       # annolid/
    project_root = annolid_root.parent                      # repo root

    default_roots = [
        module_root,
        annolid_root,
        project_root,
        Path.cwd(),
        Path.home() / "Downloads",
    ]

    for subdir in ("realtime", "segmentation", "detector", "gui"):
        default_roots.append(annolid_root / subdir)

    default_roots.append(project_root / "runs")
    default_roots.append(annolid_root / "runs")

    seen_root = set()
    for root in default_roots:
        if root not in seen_root:
            roots.append(root)
            seen_root.add(root)

    seen = set()

    # Absolute or relative path provided by user.
    yield weight_path

    for root in roots:
        candidate = (root / weight_path).resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        yield candidate

    weight_lower = weight_name.lower()
    run_dirs: Tuple[Tuple[str, str], ...] = (
        ("pose", "runs/pose/train/weights/best.pt"),
        ("segment", "runs/segment/train/weights/best.pt"),
        ("detect", "runs/detect/train/weights/best.pt"),
    )
    for keyword, rel_path in run_dirs:
        if keyword in weight_lower or weight_path.name == "best.pt":
            candidate = project_root / rel_path
            if candidate not in seen:
                seen.add(candidate)
                yield candidate
            downloads_candidate = Path.home() / "Downloads" / rel_path
            if downloads_candidate not in seen:
                seen.add(downloads_candidate)
                yield downloads_candidate


def resolve_weight_path(weight_name: str,
                        search_roots: Optional[Iterable[Path]] = None) -> Path:
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
            logger.info("YOLO weight resolved to %s", candidate)
            return candidate

    logger.info(
        "YOLO weight '%s' not found on disk; relying on Ultralytics resolution.",
        weight_name,
    )
    return Path(weight_name)


def select_backend(weight_name: str,
                   search_roots: Optional[Iterable[Path]] = None) -> YOLOModelSpec:
    """
    Detect available hardware and return the best backend for the provided model.

    The selection order follows TensorRT (if installed) -> CUDA -> CPU.
    """
    resolved = resolve_weight_path(weight_name, search_roots=search_roots)

    # CoreML packages are tied to Apple devices and should bypass CUDA/TensorRT selection.
    if resolved.suffix.lower() == ".mlpackage":
        if not resolved.exists():
            raise FileNotFoundError(
                f"CoreML package '{resolved}' not found on disk.")
        logger.info("Using CoreML package for %s", resolved)
        return YOLOModelSpec(weight_path=resolved,
                             device=torch.device("cpu"),
                             backend="CoreML (CPU)")

    # TensorRT takes precedence when available and an engine exists alongside the weight.
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        engine_path = resolved.with_suffix(".engine")
        if engine_path.is_file():
            logger.info("Using TensorRT engine for %s", engine_path)
            return YOLOModelSpec(weight_path=engine_path,
                                 device=device,
                                 backend="TensorRT (GPU)")

        # Fall back to the resolved weight on GPU.
        if resolved.suffix == ".pt" or resolved.exists():
            logger.info("Using PyTorch backend on CUDA for %s", resolved)
            return YOLOModelSpec(weight_path=resolved,
                                 device=device,
                                 backend="PyTorch (GPU)")
        raise FileNotFoundError(
            f"CUDA available but weight '{resolved}' could not be located."
        )

    # CPU fallback
    device = torch.device("cpu")
    if resolved.suffix == ".pt" or resolved.exists():
        logger.warning(
            "No compatible GPU detected; running %s on CPU. Expect reduced performance.",
            resolved,
        )
        return YOLOModelSpec(weight_path=resolved,
                             device=device,
                             backend="PyTorch (CPU)")

    raise FileNotFoundError(
        f"Unable to locate model weights for '{weight_name}'. Provide a valid path."
    )


def load_yolo_model(weight_name: str,
                    search_roots: Optional[Iterable[Path]] = None):
    """
    Load a YOLO model with automatic backend selection.

    Returns:
        tuple(model, YOLOModelSpec)
    """
    spec = select_backend(weight_name, search_roots=search_roots)
    from ultralytics import YOLO  # Imported lazily to avoid heavy import cost.

    model = YOLO(str(spec.weight_path))
    if "TensorRT" not in spec.backend and spec.weight_path.suffix.lower() != ".mlpackage":
        model.to(spec.device)
    return model, spec
