from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


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
    ModelConfig("CoTracker", "CoTracker", "CoTracker.pt"),
    ModelConfig("CoWTracker", "CoWTracker", "facebook/cowtracker"),
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
# to Hugging Face model IDs and may require gated-access for certain DINOv3
# checkpoints.
PATCH_SIMILARITY_MODELS = [
    ModelConfig("DINOv2 Base (open)", "facebook/dinov2-base", ""),
    ModelConfig("DINOv2 Large (open)", "facebook/dinov2-large", ""),
    ModelConfig(
        "DINOv3 ViT-S/16 (gated)", "facebook/dinov3-vits16-pretrain-lvd1689m", ""
    ),
    ModelConfig(
        "DINOv3 ViT-S/16+ (gated)", "facebook/dinov3-vits16plus-pretrain-lvd1689m", ""
    ),
    ModelConfig(
        "DINOv3 ViT-L/16 (gated)", "facebook/dinov3-vitl16-pretrain-lvd1689m", ""
    ),
    ModelConfig(
        "DINOv3 ViT-H/16+ (gated)", "facebook/dinov3-vith16plus-pretrain-lvd1689m", ""
    ),
    ModelConfig(
        "DINOv3 ViT-7B/16 (gated)", "facebook/dinov3-vit7b16-pretrain-lvd1689m", ""
    ),
    ModelConfig("NVIDIA RADIOv4-SO400M", "nvidia/C-RADIOv4-SO400M", ""),
]

PATCH_SIMILARITY_DEFAULT_MODEL = PATCH_SIMILARITY_MODELS[2].identifier


MODEL_PATH_DEFAULTS: Dict[str, str] = {
    "dino_kpseg": "runs/dino_kpseg/train/weights/best.pt",
    "dino_kpseg_tracker": "runs/dino_kpseg/train/weights/best.pt",
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
    override = _settings_override_path(settings, identifier)
    if not override:
        override = _config_override_path(config, identifier)
    if not override:
        override = MODEL_PATH_DEFAULTS.get(identifier)
    if override:
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
            resolved = Path(weight_file).expanduser()
            if not resolved.exists():
                warnings.append(
                    f"Model '{display}' currently unavailable: expected local weights at '{resolved}'."
                )

    return len(errors) == 0, errors, warnings


def get_model_unavailable_reason(model: ModelConfig) -> Optional[str]:
    """
    Return user-facing availability reason for model entries that require local paths.
    """
    identifier = str(model.identifier)
    if identifier not in MODEL_PATH_DEFAULTS:
        return None
    resolved = Path(str(model.weight_file)).expanduser()
    if resolved.exists():
        return None
    return f"Missing local weights: {resolved}"
