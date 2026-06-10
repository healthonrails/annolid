"""Shared catalog and download helpers for DINO-family feature backbones."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_DINOV3_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
DEFAULT_DINO_FEATURE_MODEL_ID = DEFAULT_DINOV3_MODEL_ID
DINO_MODEL_RUNTIME_KEYS = ("patch_model_name", "dinov3_model_name")


@dataclass(frozen=True)
class DinoModelInfo:
    display_name: str
    model_id: str
    family: str
    gated: bool = False
    aliases: tuple[str, ...] = ()
    note: str = ""


DINO_FEATURE_MODEL_CATALOG: tuple[DinoModelInfo, ...] = (
    DinoModelInfo(
        "DINOv2 Base (open)",
        "facebook/dinov2-base",
        "dinov2",
        note="Open fallback with lower memory use than large DINOv3 checkpoints.",
    ),
    DinoModelInfo(
        "DINOv2 Large (open)",
        "facebook/dinov2-large",
        "dinov2",
        note="Open fallback with stronger features than DINOv2 Base.",
    ),
    DinoModelInfo(
        "DINOv3 ViT-S/16 (gated, recommended)",
        DEFAULT_DINOV3_MODEL_ID,
        "dinov3",
        gated=True,
        aliases=("dinov3_vits16", "vits16", "small"),
        note="Default balance for interactive tracking on CPU, MPS, or modest GPUs.",
    ),
    DinoModelInfo(
        "DINOv3 ViT-S/16+ (gated)",
        "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        "dinov3",
        gated=True,
        aliases=("dinov3_vits16plus", "vits16plus", "small-plus"),
        note="Slightly stronger small backbone when memory allows.",
    ),
    DinoModelInfo(
        "DINOv3 ViT-B/16 (gated)",
        "facebook/dinov3-vitb16-pretrain-lvd1426",
        "dinov3",
        gated=True,
        aliases=("dinov3_vitb16", "vitb16", "base"),
        note="Medium backbone for better descriptors at higher memory cost.",
    ),
    DinoModelInfo(
        "DINOv3 ViT-L/16 (gated)",
        "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "dinov3",
        gated=True,
        aliases=("dinov3_vitl16", "vitl16", "large"),
        note="Large backbone for offline or high-memory tracking runs.",
    ),
    DinoModelInfo(
        "DINOv3 ViT-H/16+ (gated)",
        "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        "dinov3",
        gated=True,
        aliases=("dinov3_vith16plus", "vith16plus", "huge-plus"),
        note="High-memory backbone; prefer pre-downloading before long videos.",
    ),
    DinoModelInfo(
        "DINOv3 ViT-7B/16 (gated)",
        "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        "dinov3",
        gated=True,
        aliases=("dinov3_vit7b16", "vit7b16", "7b"),
        note="Very large checkpoint for specialized offline workflows.",
    ),
    DinoModelInfo(
        "NVIDIA RADIOv4-SO400M",
        "nvidia/C-RADIOv4-SO400M",
        "radio",
        aliases=("radio", "radio_v4", "radio_so400m"),
        note="Compatible feature backbone; requires open-clip-torch.",
    ),
)

DINOV3_MODEL_CATALOG: tuple[DinoModelInfo, ...] = tuple(
    model for model in DINO_FEATURE_MODEL_CATALOG if model.family == "dinov3"
)

LEGACY_ALIAS_TO_HF_ID: dict[str, str] = {
    alias: model.model_id
    for model in DINO_FEATURE_MODEL_CATALOG
    for alias in (model.aliases + (model.model_id, model.display_name))
}


def iter_dino_feature_models(*, dinov3_only: bool = False) -> Iterable[DinoModelInfo]:
    """Yield supported feature backbones in UI order."""
    if dinov3_only:
        return iter(DINOV3_MODEL_CATALOG)
    return iter(DINO_FEATURE_MODEL_CATALOG)


def resolve_dino_model_id(
    value: object, *, default: str = DEFAULT_DINO_FEATURE_MODEL_ID
) -> str:
    """Resolve a display label, alias, or Hugging Face id to a model id."""
    text = str(value or "").strip()
    if not text:
        return str(default)
    lowered = text.lower()
    for key, model_id in LEGACY_ALIAS_TO_HF_ID.items():
        if lowered == str(key).strip().lower():
            return model_id
    return text


def resolve_dinov3_model_id(value: object) -> str:
    """Backward-compatible resolver name for DINOv3 extractor callers."""
    return resolve_dino_model_id(value)


def resolve_dino_model_from_runtime(
    runtime: object,
    *,
    fallback: object = DEFAULT_DINO_FEATURE_MODEL_ID,
) -> str:
    """Resolve the selected DINO model from runtime config aliases."""
    for key in DINO_MODEL_RUNTIME_KEYS:
        if isinstance(runtime, dict):
            raw = runtime.get(key)
        else:
            raw = getattr(runtime, key, None)
        if str(raw or "").strip():
            return resolve_dino_model_id(raw)
    return resolve_dino_model_id(fallback)


def set_dino_model_on_runtime(runtime: object, model_name: object) -> str:
    """Set both supported runtime aliases to the same resolved DINO model id."""
    if isinstance(model_name, dict):
        selected_model = resolve_dino_model_from_runtime(model_name)
    else:
        selected_model = resolve_dino_model_id(model_name)
    for key in DINO_MODEL_RUNTIME_KEYS:
        if isinstance(runtime, dict):
            runtime[key] = selected_model
        elif hasattr(runtime, key):
            setattr(runtime, key, selected_model)
    return selected_model


def get_dino_cache_dir(cache_dir: Optional[str | Path] = None) -> Optional[Path]:
    """Return the explicit DINO cache dir, honoring DINOV3_LOCATION."""
    raw = cache_dir if cache_dir is not None else os.getenv("DINOV3_LOCATION")
    if raw in (None, ""):
        return None
    return Path(raw).expanduser()


def download_dino_model(
    model_name: object = DEFAULT_DINO_FEATURE_MODEL_ID,
    *,
    cache_dir: Optional[str | Path] = None,
    local_files_only: bool = False,
    token: Optional[str] = None,
) -> Path:
    """Download or verify a DINO-family model snapshot in the Hugging Face cache."""
    model_id = resolve_dino_model_id(model_name)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "huggingface-hub is required to download DINO models. "
            "Install it with: pip install 'huggingface-hub>=0.22.0'"
        ) from exc

    try:
        resolved_cache_dir = get_dino_cache_dir(cache_dir)
        path = snapshot_download(
            repo_id=model_id,
            cache_dir=str(resolved_cache_dir) if resolved_cache_dir else None,
            local_files_only=bool(local_files_only),
            token=token,
        )
    except Exception as exc:
        hint = (
            "For gated DINOv3 checkpoints, accept the model license on Hugging Face "
            "and authenticate with `hf auth login` or set HF_TOKEN."
        )
        raise RuntimeError(
            f"Failed to download DINO model '{model_id}': {exc}. {hint}"
        ) from exc
    return Path(path)
