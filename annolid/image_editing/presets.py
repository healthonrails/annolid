from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class StableDiffusionCppPreset:
    """Bundle of weights used by stable-diffusion.cpp for a model family."""

    name: str
    description: str

    diffusion_repo_id: str
    diffusion_filename: str

    vae_repo_id: str
    vae_filename: str

    llm_repo_id: str
    llm_filename: str

    llm_vision_repo_id: Optional[str] = None
    llm_vision_filename: Optional[str] = None

    extra_args: Tuple[str, ...] = ()


def _normalize_quant(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    # Users often type "Q2-K" while repos use "Q2_K".
    return (value or "").strip().replace("-", "_")


def _qwen_image_2512_gguf(
    *,
    quant: str = "Q2_K",
    llm_quant: str = "Q2_K",
) -> StableDiffusionCppPreset:
    quant = _normalize_quant(quant) or "Q2_K"
    llm_quant = _normalize_quant(llm_quant) or "Q2_K"
    diffusion_filename = f"qwen-image-2512-{quant}.gguf"
    # unsloth repo uses dash-separated quant suffixes ("-Q8_0.gguf").
    llm_filename = f"Qwen2.5-VL-7B-Instruct-{llm_quant}.gguf"
    return StableDiffusionCppPreset(
        name="qwen-image-2512-gguf",
        description="Qwen-Image-2512 diffusion GGUF + Qwen2.5-VL 7B text encoder (stable-diffusion.cpp).",
        diffusion_repo_id="unsloth/Qwen-Image-2512-GGUF",
        diffusion_filename=diffusion_filename,
        vae_repo_id="Comfy-Org/Qwen-Image_ComfyUI",
        vae_filename="split_files/vae/qwen_image_vae.safetensors",
        llm_repo_id="unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
        llm_filename=llm_filename,
        llm_vision_repo_id="unsloth/Qwen2.5-VL-7B-Instruct-GGUF",
        llm_vision_filename="mmproj-F16.gguf",
        extra_args=(
            "--diffusion-fa",
            "--flow-shift",
            "3",
        ),
    )


_PRESET_BUILDERS: Dict[str, object] = {
    "qwen-image-2512-gguf": _qwen_image_2512_gguf,
}


def list_stable_diffusion_cpp_presets() -> List[str]:
    return sorted(_PRESET_BUILDERS.keys())


def get_stable_diffusion_cpp_preset(
    name: str,
    *,
    quant: Optional[str] = None,
    llm_quant: Optional[str] = None,
) -> StableDiffusionCppPreset:
    key = (name or "").strip().lower()
    if key not in _PRESET_BUILDers_lower():
        raise KeyError(
            f"Unknown preset {name!r}. Available: {', '.join(list_stable_diffusion_cpp_presets())}"
        )
    builder = _PRESET_BUILDers_lower()[key]
    if key == "qwen-image-2512-gguf":
        return builder(
            quant=_normalize_quant(quant) or "Q2_K",
            llm_quant=_normalize_quant(llm_quant) or "Q2_K",
        )
    return builder()  # type: ignore[misc]


def _PRESET_BUILDers_lower() -> Dict[str, object]:
    return {k.lower(): v for k, v in _PRESET_BUILDERS.items()}


def _as_sequence(value: Sequence[str] | None) -> Tuple[str, ...]:
    return tuple(str(v) for v in (value or ()))
