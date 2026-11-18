from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from annolid.utils.model_assets import (
    candidate_model_paths,
    model_downloads_dir,
    resolve_existing_model_path,
)
from annolid.features.dino_models import (
    DEFAULT_DINO_FEATURE_MODEL_ID,
    iter_dino_feature_models,
)


@dataclass
class ModelConfig:
    display_name: str
    identifier: str
    weight_file: str


# List of model configurations
MODEL_REGISTRY = [
    ModelConfig("SAM_HQ", "sam_hq", "sam_hq.pt"),
    ModelConfig("EfficientVit_SAM", "efficientvit_sam", "efficientvit_sam.pt"),
    ModelConfig("EfficientTAM_s", "efficienttam_s", "efficienttam_s.pt"),
    ModelConfig("EfficientTAM_ti", "efficienttam_ti", "efficienttam_ti.pt"),
    ModelConfig("Cutie", "Cutie", "Cutie.pt"),
    ModelConfig(
        "Cutie (SAM3 Body Parts)",
        "cutie_sam3_bodyparts",
        "cutie-sam3-bodyparts.pth",
    ),
    ModelConfig(
        "Cutie (Original Base Mega)",
        "cutie_original",
        "cutie-base-mega.orig.pth",
    ),
    ModelConfig("CoTracker", "CoTracker", "CoTracker.pt"),
    ModelConfig("CoWTracker", "CoWTracker", "facebook/cowtracker"),
    ModelConfig("TAPNext (ONNX)", "TAPNext", "downloads/tapnext.onnx"),
    ModelConfig(
        "VideoMT ViT-S (ONNX)",
        "videomt",
        "downloads/videomt_yt_2019_vit_small_52.8.onnx",
    ),
    ModelConfig("INSID3 Video (DINOv3)", "insid3_video", "INSID3_VIDEO_DINOV3"),
    ModelConfig(
        "Instance Body Part Segmentation",
        "body_part_seg",
        "INSTANCE_BODY_PART_SEG",
    ),
    ModelConfig("sam2_hiera_s", "sam2_hiera_s", "sam2_hiera_s.pt"),
    ModelConfig("sam2_hiera_l", "sam2_hiera_l", "sam2_hiera_l.pt"),
    ModelConfig("SAM3", "sam3", "sam3"),
    ModelConfig("YOLO11n", "yolo11n", "yolo11n-seg.pt"),
    ModelConfig("YOLO11x", "yolo11x", "yolo11x-seg.pt"),
    ModelConfig("YOLO11n-pose", "yolo11n-pose", "yolo11n-pose.pt"),
    ModelConfig("YOLO11x-pose", "yolo11x-pose", "yolo11x-pose.pt"),
    ModelConfig("yoloe-11s-seg.pt", "yoloe-11s-seg.pt", "yoloe-11s-seg.pt"),
    ModelConfig("yoloe-11l-seg.pt", "yoloe-11l-seg.pt", "yoloe-11l-seg.pt"),
    # YOLOE-26 (prompted: text + visual prompts)
    ModelConfig("YOLOE-26s-seg (Prompted)", "yoloe-26s-seg", "yoloe-26s-seg.pt"),
    ModelConfig("YOLOE-26m-seg (Prompted)", "yoloe-26m-seg", "yoloe-26m-seg.pt"),
    ModelConfig("YOLOE-26l-seg (Prompted)", "yoloe-26l-seg", "yoloe-26l-seg.pt"),
    # YOLOE-26 prompt-free (built-in vocabulary; do not provide prompts)
    ModelConfig(
        "YOLOE-26s-seg (Prompt-free)", "yoloe-26s-seg-pf", "yoloe-26s-seg-pf.pt"
    ),
    ModelConfig(
        "YOLOE-26m-seg (Prompt-free)", "yoloe-26m-seg-pf", "yoloe-26m-seg-pf.pt"
    ),
    ModelConfig(
        "YOLOE-26l-seg (Prompt-free)", "yoloe-26l-seg-pf", "yoloe-26l-seg-pf.pt"
    ),
    ModelConfig("YOLO26n-seg", "yolo26n-seg", "yolo26n-seg.pt"),
    ModelConfig("YOLO26x-seg", "yolo26x-seg", "yolo26x-seg.pt"),
    ModelConfig("YOLO26n-pose", "yolo26n-pose", "yolo26n-pose.pt"),
    ModelConfig("YOLO26x-pose", "yolo26x-pose", "yolo26x-pose.pt"),
    ModelConfig(
        "DINOv3 Keypoint Tracker", "dinov3_keypoint_tracker", "DINO_KEYPOINT_TRACKER"
    ),
    ModelConfig(
        "Cutie + DINOv3 Keypoint Segmentation",
        "dino_kpseg_tracker",
        "DINO_KPSEG_TRACKER_DEFAULT",
    ),
    ModelConfig(
        "DINOv3 Keypoint Segmentation",
        "dino_kpseg",
        "DINO_KPSEG_DEFAULT",
    ),
    ModelConfig(
        "MediaPipe Pose",
        "mediapipe_pose",
        "mediapipe_pose",
    ),
    ModelConfig(
        "MediaPipe Hands",
        "mediapipe_hands",
        "mediapipe_hands",
    ),
    ModelConfig(
        "MediaPipe Face",
        "mediapipe_face",
        "mediapipe_face",
    ),
]


# Registry for patch-similarity (DINO) backbones. These identifiers correspond
# to Hugging Face model IDs and may require gated access for DINOv3 checkpoints.
PATCH_SIMILARITY_MODELS = [
    ModelConfig(model.display_name, model.model_id, "")
    for model in iter_dino_feature_models()
]

PATCH_SIMILARITY_DEFAULT_MODEL = DEFAULT_DINO_FEATURE_MODEL_ID


MODEL_PATH_DEFAULTS: Dict[str, str] = {
    "dino_kpseg": "runs/dino_kpseg/train/weights/best.pt",
    "dino_kpseg_tracker": "runs/dino_kpseg/train/weights/best.pt",
    "videomt": "downloads/videomt_yt_2019_vit_small_52.8.onnx",
}


def _config_override_path(
    config: Optional[Dict[str, Any]], identifier: str
) -> Optional[str]:
    root_cfg = config if isinstance(config, dict) else {}
    ai_cfg = root_cfg.get("ai") if isinstance(root_cfg.get("ai"), dict) else {}
    path_cfg = (
        ai_cfg.get("model_path_defaults")
        if isinstance(ai_cfg.get("model_path_defaults"), dict)
        else {}
    )
    value = path_cfg.get(identifier)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _settings_override_path(settings: Any, identifier: str) -> Optional[str]:
    if settings is None:
        return None
    try:
        raw = settings.value(f"ai/model_paths/{identifier}", "")
    except Exception:
        return None
    text = str(raw or "").strip()
    return text or None


def resolve_model_weight_path(
    model: ModelConfig,
    *,
    config: Optional[Dict[str, Any]] = None,
    settings: Any = None,
) -> str:
    """Resolve runtime weight path for models with configurable local defaults."""
    identifier = str(model.identifier)
    settings_override = _settings_override_path(settings, identifier)
    config_override = _config_override_path(config, identifier)
    default_override = MODEL_PATH_DEFAULTS.get(identifier)

    override = settings_override or config_override or default_override
    if override:
        existing = resolve_existing_model_path(str(override))
        if existing is not None:
            return str(existing)

        # Respect explicit user/project overrides even when they are missing.
        # Auto-discovery fallback should only apply to built-in defaults.
        explicit_override = settings_override or config_override
        if not explicit_override and identifier in ("dino_kpseg", "dino_kpseg_tracker"):
            from annolid.utils.runs import find_latest_checkpoint

            latest = find_latest_checkpoint(
                task="dino_kpseg", model="train", filename="best.pt"
            )
            if latest is not None and latest.exists():
                return str(latest)

        return str(override)
    return str(model.weight_file)


def get_runtime_model_registry(
    *,
    config: Optional[Dict[str, Any]] = None,
    settings: Any = None,
) -> List[ModelConfig]:
    """Return model registry with runtime-resolved model paths."""
    resolved: List[ModelConfig] = []
    for model in MODEL_REGISTRY:
        weight_path = resolve_model_weight_path(model, config=config, settings=settings)
        if weight_path != model.weight_file:
            resolved.append(replace(model, weight_file=weight_path))
        else:
            resolved.append(model)
    return resolved


def validate_model_registry_entries(
    registry: Iterable[ModelConfig],
) -> Tuple[bool, List[str], List[str]]:
    """
    Validate schema and runtime integrity for model registry entries.

    Returns:
        (is_valid, errors, warnings)
    """
    errors: List[str] = []
    warnings: List[str] = []
    seen_ids: Dict[str, str] = {}

    for entry in registry:
        display = str(entry.display_name or "").strip()
        identifier = str(entry.identifier or "").strip()
        weight_file = str(entry.weight_file or "").strip()

        if not display:
            errors.append("Model entry has empty display_name.")
        if not identifier:
            errors.append(f"Model '{display or '<unknown>'}' has empty identifier.")
        if not weight_file:
            errors.append(f"Model '{display or identifier}' has empty weight_file.")

        lowered = identifier.lower()
        if lowered:
            existing = seen_ids.get(lowered)
            if existing is not None and existing != identifier:
                errors.append(
                    f"Duplicate model identifier (case-insensitive): '{identifier}' conflicts with '{existing}'."
                )
            else:
                seen_ids[lowered] = identifier

        if identifier in MODEL_PATH_DEFAULTS:
            resolved = resolve_existing_model_path(weight_file)
            if resolved is None:
                preferred = candidate_model_paths(weight_file)[0]
                warnings.append(
                    f"Model '{display}' currently unavailable: expected local weights at '{preferred}'."
                )

    return len(errors) == 0, errors, warnings


def get_model_unavailable_reason(model: ModelConfig) -> Optional[str]:
    """
    Return user-facing availability reason for model entries that require local paths.
    """
    identifier = str(model.identifier)
    if identifier not in MODEL_PATH_DEFAULTS:
        return None
    resolved = resolve_existing_model_path(str(model.weight_file))
    if resolved is not None:
        return None
    preferred = candidate_model_paths(str(model.weight_file))[0]
    return f"Missing local weights: {preferred}"


CUTIE_MODEL_DESCRIPTIONS: Dict[str, str] = {
    "Cutie": (
        "Original Cutie VOS (default) — upstream dense 35.0M-parameter Cutie "
        "Base Mega checkpoint (134 MB) for general video object segmentation."
    ),
    "cutie_sam3_bodyparts": (
        "Improved Compact Cutie VOS — 15.8M parameters (-55%), 30 MB checkpoint, "
        "1.5x–2.7x faster, SAM 3.1 + DINOv3 distilled for multi-instance and "
        "coarse/fine anatomical body-part tracking."
    ),
    "cutie_original": (
        "Original Upstream Cutie Base Mega — explicitly pins the untouched "
        "35.0M-parameter dense checkpoint (134 MB) even when an override is set."
    ),
}


def inspect_pytorch_checkpoint_keys(
    path: Path,
    *,
    max_raw_bytes: int = 262_144,
    max_strings: int = 4096,
) -> frozenset[str]:
    """Safely extract string keys from a PyTorch ``.pth``/``.pt`` file without ``torch`` or unpickling.

    Inspects ``*/data.pkl`` inside PyTorch ZIP archives (or raw pickle streams for legacy
    checkpoints) via ``pickletools.genops``, which tokenizes opcodes as inert data without
    importing modules, instantiating classes, executing pickle gadgets, or reading tensors.
    """
    import io
    import pickletools
    import zipfile

    if not path.is_file():
        return frozenset()

    def _extract_strings(buf: bytes, *, max_streams: int = 6) -> set[str]:
        found: set[str] = set()
        stream = io.BytesIO(buf)
        for _ in range(max_streams):
            pos_before = stream.tell()
            if pos_before >= len(buf) or len(found) >= max_strings:
                break
            saw_stop = False
            try:
                for opcode, arg, _pos in pickletools.genops(stream):
                    if isinstance(arg, str) and arg:
                        found.add(arg)
                        if len(found) >= max_strings:
                            break
                    elif isinstance(arg, (bytes, bytearray)) and 0 < len(arg) <= 128:
                        try:
                            found.add(arg.decode("utf-8"))
                        except UnicodeDecodeError:
                            pass
                    if opcode.name == "STOP":
                        saw_stop = True
                        break
            except Exception:
                break
            if not saw_stop or stream.tell() <= pos_before:
                break
        return found

    try:
        if zipfile.is_zipfile(path):
            collected: set[str] = set()
            with zipfile.ZipFile(path, "r") as zf:
                pkl_names = [
                    name
                    for name in zf.namelist()
                    if name.endswith("data.pkl") or name.endswith(".pkl")
                ]
                for name in pkl_names[:4]:
                    with zf.open(name, "r") as entry:
                        collected.update(_extract_strings(entry.read(max_raw_bytes)))
                    if len(collected) >= max_strings:
                        break
            return frozenset(collected)

        with path.open("rb") as fh:
            prefix = fh.read(max_raw_bytes)
        return frozenset(_extract_strings(prefix))
    except OSError:
        return frozenset()


def is_cutie_checkpoint_file(path: Path) -> bool:
    """Return True when ``path`` is a compact or dense Cutie VOS ``.pth``/``.pt`` checkpoint."""
    if not path.is_file() or path.suffix.lower() not in {".pth", ".pt"}:
        return False
    lowered = path.name.lower()
    if "cutie" in lowered and "yolo" not in lowered and "kpseg" not in lowered:
        return True
    keys = inspect_pytorch_checkpoint_keys(path)
    if not keys:
        return False
    if "__annolid_cutie_format__" in keys:
        return True
    has_pix_enc = any(
        k.startswith("pixel_encoder.") or ".pixel_encoder." in k for k in keys
    )
    has_mask_or_obj = any(
        k.startswith(("mask_decoder.", "object_transformer."))
        or ".mask_decoder." in k
        or ".object_transformer." in k
        for k in keys
    )
    return has_pix_enc and has_mask_or_obj


def describe_cutie_checkpoint_status_fast(
    weights_spec: Optional[str] = None,
    *,
    weights_dir: Optional[Path] = None,
) -> Tuple[bool, str]:
    """Inspect Cutie checkpoint availability using pure pathlib (no torch import)."""
    raw = str(weights_spec or "").strip()
    pkg_weights_dir = (
        Path(weights_dir).expanduser()
        if weights_dir is not None
        else Path(__file__).resolve().parents[1]
        / "segmentation"
        / "cutie_vos"
        / "weights"
    )
    dl_dir = model_downloads_dir()

    def _find_managed(fname: str) -> Optional[Path]:
        for cand in (pkg_weights_dir / fname, dl_dir / fname):
            try:
                if cand.is_file():
                    return cand
            except OSError:
                continue
        return None

    if (
        not raw
        or raw.lower() == "cutie.pt"
        or raw in ("cutie-base-mega.pth", "cutie-base-mega.orig.pth")
    ):
        preferred = (
            "cutie-base-mega.orig.pth"
            if raw == "cutie-base-mega.orig.pth"
            else "cutie-base-mega.pth"
        )
        fallback = (
            "cutie-base-mega.pth"
            if preferred == "cutie-base-mega.orig.pth"
            else "cutie-base-mega.orig.pth"
        )
        found = _find_managed(preferred) or _find_managed(fallback)
        if found is not None:
            size_mb = found.stat().st_size / (1024.0 * 1024.0)
            return (
                True,
                f"Ready locally ({size_mb:.1f} MB — Original Upstream Base Mega, 35.0M params)",
            )
        return (
            True,
            "Auto-downloads on first run (~133.9 MB — Original Upstream Base Mega)",
        )

    if raw == "cutie-sam3-bodyparts.pth":
        found = _find_managed("cutie-sam3-bodyparts.pth")
        if found is not None:
            size_mb = found.stat().st_size / (1024.0 * 1024.0)
            if size_mb < 60.0:
                return (
                    True,
                    f"Ready locally ({size_mb:.1f} MB compact — SAM3 + DINOv3, 15.8M params)",
                )
            return (
                True,
                f"Ready locally ({size_mb:.1f} MB — auto-compresses to 30.3 MB on first load)",
            )
        return (
            True,
            "Auto-downloads on first run (~30.3 MB — Improved Compact SAM3 + DINOv3)",
        )

    cand = Path(raw).expanduser()
    if cand.is_file():
        size_mb = cand.stat().st_size / (1024.0 * 1024.0)
        return (True, f"Ready locally ({size_mb:.1f} MB — Custom: {cand.name})")
    if not cand.is_absolute():
        in_pkg = pkg_weights_dir / cand.name
        if in_pkg.is_file():
            size_mb = in_pkg.stat().st_size / (1024.0 * 1024.0)
            return (True, f"Ready locally ({size_mb:.1f} MB — Custom: {in_pkg.name})")
    resolved = resolve_existing_model_path(raw)
    if resolved is not None and resolved.is_file():
        size_mb = resolved.stat().st_size / (1024.0 * 1024.0)
        return (True, f"Ready locally ({size_mb:.1f} MB — Custom: {resolved.name})")
    return (False, f"Custom checkpoint not found: {raw}")


def get_model_description(model: ModelConfig) -> str:
    """Return a rich user-facing tooltip/description for ``model``."""
    identifier = str(model.identifier or "").strip()
    base_desc = CUTIE_MODEL_DESCRIPTIONS.get(identifier, "")
    if base_desc:
        _ok, status = describe_cutie_checkpoint_status_fast(model.weight_file)
        return f"{base_desc}\nStatus: {status}"
    if identifier.lower().startswith("cutie_custom:") or (
        "cutie" in identifier.lower() and "kpseg" not in identifier.lower()
    ):
        _ok, status = describe_cutie_checkpoint_status_fast(model.weight_file)
        return f"Custom Cutie VOS checkpoint ({model.weight_file}).\nStatus: {status}"
    return ""
