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
    DinoModelInfo(
        "Google EmbeddingGemma 2 (open, Matryoshka)",
        "google/embeddinggemma-2",
        "embeddinggemma",
        aliases=("embeddinggemma-2", "embeddinggemma2", "embedding_gemma_2"),
        note=(
            "Lightweight 170M vision tower (768d with 512/256/128 Matryoshka dims); "
            "selective tower loading in bfloat16/float32."
        ),
    ),
    DinoModelInfo(
        "Google SigLIP 2 Base/16 (open)",
        "google/siglip2-base-patch16-224",
        "siglip2",
        aliases=("siglip2", "siglip2_base", "siglip2-b16"),
        note="Open ViT-B/16 dense vision encoder with strong local landmark features.",
    ),
    DinoModelInfo(
        "Google SigLIP 2 So400m/16 (open)",
        "google/siglip2-so400m-patch16-384",
        "siglip2",
        aliases=("siglip2_so400m", "siglip2-so400m"),
        note="Higher-capacity SigLIP 2 shape-optimized vision encoder.",
    ),
    DinoModelInfo(
        "Google MobileNetV4 Small (on-device, fast)",
        "timm/mobilenetv4_conv_small.e2400_r224_in1k",
        "mobilenet",
        aliases=("mobilenetv4", "mobilenetv4_small", "mnv4_small"),
        note="Ultra-fast on-device stride-8 CNN pyramid for CPU/MPS keypoint tracking.",
    ),
    DinoModelInfo(
        "Google MobileNetV4 Hybrid Medium (on-device)",
        "timm/mobilenetv4_hybrid_medium.e500_r224_in1k",
        "mobilenet",
        aliases=("mobilenetv4_hybrid", "mnv4_hybrid"),
        note="On-device MobileNetV4 hybrid pyramid with mobile attention blocks.",
    ),
    DinoModelInfo(
        "MobileNetV3 Small (torchvision, ultra-light)",
        "torchvision/mobilenet_v3_small",
        "mobilenet",
        aliases=("mobilenetv3_small", "mnv3_small"),
        note="Lightweight stride-8 CNN pyramid via torchvision.",
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
    resolved_values: list[str] = []
    for key in DINO_MODEL_RUNTIME_KEYS:
        if isinstance(runtime, dict):
            raw = runtime.get(key)
        else:
            raw = getattr(runtime, key, None)
        if str(raw or "").strip():
            resolved_values.append(resolve_dino_model_id(raw))
    for model_id in resolved_values:
        if model_id != DEFAULT_DINO_FEATURE_MODEL_ID:
            return model_id
    if resolved_values:
        return resolved_values[0]
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


def is_google_vision_model(value: object) -> bool:
    """Return True when the model id/alias targets Google or on-device vision backbones."""
    resolved = resolve_dino_model_id(value, default="").strip().lower()
    if not resolved:
        return False
    return (
        "embeddinggemma" in resolved
        or "embedding_gemma" in resolved
        or "siglip" in resolved
        or "mobilenet" in resolved
        or resolved.startswith("timm/")
        or resolved.startswith("torchvision/")
    )


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
    resolved_cache_dir = get_dino_cache_dir(cache_dir)
    if model_id.lower().startswith("torchvision/"):
        import torch

        hub_dir = (
            resolved_cache_dir
            if resolved_cache_dir is not None
            else Path(torch.hub.get_dir())
        )
        checkpoints_dir = hub_dir / "checkpoints"
        if local_files_only:
            if checkpoints_dir.exists() and any(checkpoints_dir.glob("mobilenet_v3*")):
                return checkpoints_dir
            raise RuntimeError(
                f"Torchvision model '{model_id}' is not cached in {checkpoints_dir}."
            )
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        prev_hub_dir = torch.hub.get_dir()
        try:
            torch.hub.set_dir(str(hub_dir))
            from torchvision import models as tv_models

            arch = model_id.split("/", 1)[1].strip().lower()
            if arch == "mobilenet_v3_large":
                tv_models.mobilenet_v3_large(
                    weights=tv_models.MobileNet_V3_Large_Weights.DEFAULT
                )
            else:
                tv_models.mobilenet_v3_small(
                    weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT
                )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download torchvision model '{model_id}': {exc}"
            ) from exc
        finally:
            torch.hub.set_dir(prev_hub_dir)
        return checkpoints_dir

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "huggingface-hub is required to download DINO models. "
            "Install it with: pip install 'huggingface-hub>=0.22.0'"
        ) from exc

    try:
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
