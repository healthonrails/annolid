from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from annolid.image_editing.presets import StableDiffusionCppPreset


def _split_hf_path(path: str) -> Tuple[Optional[str], str]:
    parts = [p for p in (path or "").split("/") if p]
    if len(parts) <= 1:
        return None, (parts[0] if parts else "")
    return "/".join(parts[:-1]), parts[-1]


def hf_download(repo_id: str, *, filename: str, revision: Optional[str] = None) -> Path:
    """Download a file from Hugging Face Hub (returns absolute on-disk path)."""
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:
        raise ImportError(
            "huggingface_hub is required to download model assets. "
            "Install with: pip install huggingface-hub"
        ) from exc

    subfolder, base = _split_hf_path(filename)
    downloaded = hf_hub_download(
        repo_id,
        filename=base,
        subfolder=subfolder,
        revision=revision,
    )
    return Path(downloaded).expanduser().resolve()


@dataclass(frozen=True)
class StableDiffusionCppWeights:
    diffusion_model: Path
    vae: Path
    llm: Path
    llm_vision: Optional[Path] = None


def download_stable_diffusion_cpp_preset(
    preset: StableDiffusionCppPreset,
    *,
    revision: Optional[str] = None,
) -> StableDiffusionCppWeights:
    diffusion = hf_download(
        preset.diffusion_repo_id, filename=preset.diffusion_filename, revision=revision
    )
    vae = hf_download(
        preset.vae_repo_id, filename=preset.vae_filename, revision=revision
    )
    llm = hf_download(
        preset.llm_repo_id, filename=preset.llm_filename, revision=revision
    )
    llm_vision = None
    if preset.llm_vision_repo_id and preset.llm_vision_filename:
        llm_vision = hf_download(
            preset.llm_vision_repo_id,
            filename=preset.llm_vision_filename,
            revision=revision,
        )
    return StableDiffusionCppWeights(
        diffusion_model=diffusion,
        vae=vae,
        llm=llm,
        llm_vision=llm_vision,
    )
