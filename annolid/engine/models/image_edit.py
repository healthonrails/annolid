from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from annolid.engine.registry import ModelPluginBase, register_model


@register_model
class ImageEditPlugin(ModelPluginBase):
    name = "image-edit"
    description = "Generate/edit images via Diffusers or stable-diffusion.cpp (supports GGUF presets like Qwen-Image-2512)."

    def add_predict_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--backend",
            choices=("diffusers", "sdcpp", "pillow"),
            default="diffusers",
            help="Execution backend (default: diffusers).",
        )
        parser.add_argument(
            "--mode",
            choices=("text_to_image", "image_to_image", "inpaint", "ref_image_edit"),
            default=None,
            help="Edit mode (default: inferred from provided inputs).",
        )
        parser.add_argument("--prompt", required=True, help="Text prompt/instruction.")
        parser.add_argument("--negative-prompt", default="", help="Negative prompt.")
        parser.add_argument("--width", type=int, default=1024)
        parser.add_argument("--height", type=int, default=1024)
        parser.add_argument("--steps", type=int, default=20)
        parser.add_argument("--cfg-scale", type=float, default=2.5)
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Random seed; omit for random.",
        )
        parser.add_argument(
            "--strength",
            type=float,
            default=0.75,
            help="Strength for img2img/inpaint (default: 0.75).",
        )
        parser.add_argument(
            "--num-images",
            type=int,
            default=1,
            help="Number of images to generate (default: 1).",
        )
        parser.add_argument(
            "--output",
            default="output.png",
            help="Output image path (default: output.png). For multiple images, a suffix is added automatically.",
        )

        parser.add_argument(
            "--init-img",
            default=None,
            help="Init image path for img2img/inpaint.",
        )
        parser.add_argument(
            "--mask",
            default=None,
            help="Mask image path for inpaint (white = edit).",
        )
        parser.add_argument(
            "--ref-image",
            action="append",
            default=[],
            help="Reference image path (repeatable) for ref_image_edit.",
        )

        # Diffusers
        parser.add_argument(
            "--model-id",
            default="Qwen/Qwen-Image-2512",
            help="Hugging Face model id or local path for diffusers (default: Qwen/Qwen-Image-2512).",
        )
        parser.add_argument(
            "--device",
            default="auto",
            help="Diffusers device: auto|cpu|cuda|mps (default: auto).",
        )
        parser.add_argument(
            "--dtype",
            default="auto",
            help="Diffusers dtype: auto|float32|float16|bfloat16 (default: auto).",
        )
        parser.add_argument(
            "--local-files-only",
            action="store_true",
            help="Do not download weights from the Hugging Face Hub.",
        )

        # stable-diffusion.cpp
        parser.add_argument(
            "--sd-cli",
            default=None,
            help="Path to stable-diffusion.cpp `sd-cli` binary (required for sdcpp backend).",
        )
        parser.add_argument(
            "--preset",
            default=None,
            help="stable-diffusion.cpp preset name (downloads required weights).",
        )
        parser.add_argument(
            "--quant",
            default=None,
            help="Preset quantization (e.g. Q4_K_M) where supported.",
        )
        parser.add_argument(
            "--llm-quant",
            default=None,
            help="Preset LLM quantization (e.g. Q8_0) where supported.",
        )
        parser.add_argument(
            "--diffusion-model",
            default=None,
            help="Path to diffusion model (GGUF) for sdcpp backend (when not using --preset).",
        )
        parser.add_argument(
            "--vae",
            default=None,
            help="Path to VAE for sdcpp backend (when not using --preset).",
        )
        parser.add_argument(
            "--llm",
            default=None,
            help="Path to LLM text encoder for sdcpp backend (when not using --preset).",
        )
        parser.add_argument(
            "--llm-vision",
            default=None,
            help="Path to LLM vision encoder for sdcpp backend (optional).",
        )
        parser.add_argument(
            "--extra-arg",
            action="append",
            default=[],
            help="Extra sd-cli argument (repeatable). Example: --extra-arg=--qwen-image-zero-cond-t",
        )

    def predict(self, args: argparse.Namespace) -> int:
        from PIL import Image

        from annolid.image_editing.errors import ImageEditingError
        from annolid.image_editing.factory import (
            StableDiffusionCppPresetConfig,
            create_backend,
        )
        from annolid.image_editing.presets import list_stable_diffusion_cpp_presets
        from annolid.image_editing.types import ImageEditRequest

        mode = args.mode
        if not mode:
            if args.ref_image:
                mode = "ref_image_edit"
            elif args.mask:
                mode = "inpaint"
            elif args.init_img:
                mode = "image_to_image"
            else:
                mode = "text_to_image"

        init_image = Image.open(args.init_img).convert("RGB") if args.init_img else None
        mask_image = Image.open(args.mask).convert("L") if args.mask else None
        ref_images: Optional[List[Image.Image]] = None
        if args.ref_image:
            ref_images = [Image.open(p).convert("RGB") for p in args.ref_image]

        request = ImageEditRequest(
            mode=mode,
            prompt=str(args.prompt),
            negative_prompt=str(args.negative_prompt or ""),
            width=int(args.width),
            height=int(args.height),
            steps=int(args.steps),
            cfg_scale=float(args.cfg_scale),
            seed=(int(args.seed) if args.seed is not None else None),
            strength=float(args.strength),
            init_image=init_image,
            mask_image=mask_image,
            ref_images=ref_images,
            num_images=int(args.num_images),
        )

        preset_cfg = None
        if args.preset:
            available = list_stable_diffusion_cpp_presets()
            if str(args.preset) not in available:
                raise ValueError(
                    f"Unknown --preset {args.preset!r}. Available: {', '.join(available)}"
                )
            preset_cfg = StableDiffusionCppPresetConfig(
                name=str(args.preset),
                quant=(str(args.quant) if args.quant else None),
                llm_quant=(str(args.llm_quant) if args.llm_quant else None),
            )

        backend = create_backend(
            str(args.backend),
            diffusers_model_id=str(args.model_id),
            diffusers_device=str(args.device),
            diffusers_dtype=str(args.dtype),
            diffusers_local_files_only=bool(args.local_files_only),
            sd_cli_path=(str(args.sd_cli) if args.sd_cli else None),
            sdcpp_diffusion_model_path=(
                str(args.diffusion_model) if args.diffusion_model else None
            ),
            sdcpp_vae_path=(str(args.vae) if args.vae else None),
            sdcpp_llm_path=(str(args.llm) if args.llm else None),
            sdcpp_llm_vision_path=(str(args.llm_vision) if args.llm_vision else None),
            sdcpp_preset=preset_cfg,
            sdcpp_extra_args=tuple(args.extra_arg or ()),
        )

        try:
            result = backend.run(request)
        except ImageEditingError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        out_paths = _save_images(
            result.images,
            output=Path(str(args.output)).expanduser().resolve(),
        )
        summary = {
            "backend": getattr(backend, "name", str(args.backend)),
            "mode": mode,
            "outputs": [str(p) for p in out_paths],
            "meta": dict(getattr(result, "meta", {}) or {}),
        }
        print(json.dumps(summary, indent=2))
        return 0


def _save_images(images, *, output: Path) -> List[Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if len(images) == 1:
        images[0].save(output)
        return [output]

    stem = output.stem
    suffix = output.suffix or ".png"
    out_paths: List[Path] = []
    for idx, img in enumerate(images):
        path = output.with_name(f"{stem}_{idx:03d}{suffix}")
        img.save(path)
        out_paths.append(path)
    return out_paths
