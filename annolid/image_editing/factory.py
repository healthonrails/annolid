from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from annolid.image_editing.backends.base import ImageEditingBackend
from annolid.image_editing.backends import (
    DiffusersBackend,
    PillowBackend,
    StableDiffusionCppBackend,
)
from annolid.image_editing.downloads import download_stable_diffusion_cpp_preset
from annolid.image_editing.presets import get_stable_diffusion_cpp_preset


@dataclass(frozen=True)
class StableDiffusionCppPresetConfig:
    name: str
    quant: Optional[str] = None
    llm_quant: Optional[str] = None


def create_backend(
    backend: str,
    *,
    diffusers_model_id: Optional[str] = None,
    diffusers_device: str = "auto",
    diffusers_dtype: str = "auto",
    diffusers_local_files_only: bool = False,
    sd_cli_path: Optional[str] = None,
    sdcpp_model_path: Optional[str] = None,
    sdcpp_diffusion_model_path: Optional[str] = None,
    sdcpp_vae_path: Optional[str] = None,
    sdcpp_llm_path: Optional[str] = None,
    sdcpp_llm_vision_path: Optional[str] = None,
    sdcpp_preset: Optional[StableDiffusionCppPresetConfig] = None,
    sdcpp_extra_args: Tuple[str, ...] = (),
) -> ImageEditingBackend:
    key = (backend or "").strip().lower()
    if key == "pillow":
        return PillowBackend()

    if key == "diffusers":
        model_id = (diffusers_model_id or "").strip()
        if not model_id:
            raise ValueError("diffusers_model_id is required for diffusers backend")
        return DiffusersBackend(
            model_id=model_id,
            device=diffusers_device,
            dtype=diffusers_dtype,
            local_files_only=bool(diffusers_local_files_only),
        )

    if key in {"sdcpp", "stable-diffusion.cpp"}:
        if not sd_cli_path:
            raise ValueError("sd_cli_path is required for stable-diffusion.cpp backend")

        if sdcpp_preset is not None:
            preset = get_stable_diffusion_cpp_preset(
                sdcpp_preset.name,
                quant=sdcpp_preset.quant,
                llm_quant=sdcpp_preset.llm_quant,
            )
            weights = download_stable_diffusion_cpp_preset(preset)
            extra_args = tuple(preset.extra_args) + tuple(sdcpp_extra_args)
            return StableDiffusionCppBackend(
                sd_cli_path=sd_cli_path,
                diffusion_model_path=str(weights.diffusion_model),
                vae_path=str(weights.vae),
                llm_path=str(weights.llm),
                llm_vision_path=str(weights.llm_vision)
                if weights.llm_vision is not None
                else None,
                extra_args=extra_args,
            )

        return StableDiffusionCppBackend(
            sd_cli_path=sd_cli_path,
            model_path=sdcpp_model_path,
            diffusion_model_path=sdcpp_diffusion_model_path,
            vae_path=sdcpp_vae_path,
            llm_path=sdcpp_llm_path,
            llm_vision_path=sdcpp_llm_vision_path,
            extra_args=tuple(sdcpp_extra_args),
        )

    raise ValueError(
        f"Unknown backend {backend!r}. Expected one of: pillow, diffusers, sdcpp"
    )
