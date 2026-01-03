from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from annolid.image_editing.backends.base import ImageEditingBackend
from annolid.image_editing.errors import ExternalCommandError, InvalidRequestError
from annolid.image_editing.types import ImageEditRequest, ImageEditResult


def _ensure_file(path: str, label: str) -> str:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"{label} not found: {p}")
    return str(p)


@dataclass
class StableDiffusionCppBackend(ImageEditingBackend):
    """Backend that delegates to stable-diffusion.cpp's `sd-cli` binary."""

    sd_cli_path: str

    # Either provide a monolithic model via --model, or split weights.
    model_path: Optional[str] = None
    diffusion_model_path: Optional[str] = None
    vae_path: Optional[str] = None
    llm_path: Optional[str] = None
    llm_vision_path: Optional[str] = None

    offload_to_cpu: bool = True
    mmap: bool = True
    extra_args: Tuple[str, ...] = ()

    name: str = "stable-diffusion.cpp"

    def run(self, request: ImageEditRequest) -> ImageEditResult:
        sd_cli = _ensure_file(self.sd_cli_path, "sd-cli")
        ctx_args = self._build_context_args()

        with tempfile.TemporaryDirectory(prefix="annolid_sdcpp_") as tmp_dir:
            tmp = Path(tmp_dir)
            init_img_path = None
            mask_img_path = None
            ref_img_paths: List[str] = []

            if request.init_image is not None:
                init_img_path = str(tmp / "init.png")
                request.init_image.save(init_img_path)
            if request.mask_image is not None:
                mask_img_path = str(tmp / "mask.png")
                request.mask_image.convert("L").save(mask_img_path)
            if request.ref_images:
                for idx, img in enumerate(request.ref_images):
                    path = str(tmp / f"ref_{idx}.png")
                    img.save(path)
                    ref_img_paths.append(path)

            output_pattern = str(tmp / "output_%03d.png")
            cmd = self.build_command(
                request,
                sd_cli=sd_cli,
                ctx_args=ctx_args,
                output=output_pattern,
                init_img=init_img_path,
                mask_img=mask_img_path,
                ref_imgs=tuple(ref_img_paths),
            )

            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                env=os.environ.copy(),
            )
            if proc.returncode != 0:
                stderr = proc.stderr or ""
                hint = ""
                if "unsupported op 'DIAG_MASK_INF'" in stderr and "ggml_metal" in stderr:
                    hint = (
                        "Metal backend crash: ggml-metal does not support DIAG_MASK_INF for this build.\n"
                        "Fix: update/rebuild stable-diffusion.cpp from a newer commit, or build a CPU-only sd-cli by disabling Metal "
                        "(e.g. `cmake -B build-cpu -DGGML_METAL=OFF`), then point Annolid to that sd-cli.\n"
                    )
                raise ExternalCommandError(
                    hint + "stable-diffusion.cpp failed",
                    command=cmd,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    returncode=proc.returncode,
                )

            images: List[Image.Image] = []
            for i in range(int(request.num_images)):
                out_path = Path(output_pattern % i)
                if not out_path.exists():
                    raise ExternalCommandError(
                        f"Expected output image not found: {out_path}",
                        command=cmd,
                        stdout=proc.stdout,
                        stderr=proc.stderr,
                        returncode=proc.returncode,
                    )
                images.append(Image.open(out_path).copy())

            return ImageEditResult(
                images=images,
                meta={
                    "backend": self.name,
                    "command": cmd,
                },
            )

    def _build_context_args(self) -> List[str]:
        args: List[str] = []

        if self.model_path:
            args.extend(["--model", _ensure_file(self.model_path, "model")])
        else:
            if not self.diffusion_model_path:
                raise InvalidRequestError(
                    "StableDiffusionCppBackend requires either model_path or diffusion_model_path"
                )
            args.extend(
                ["--diffusion-model", _ensure_file(self.diffusion_model_path, "diffusion model")]
            )
            if not self.vae_path:
                raise InvalidRequestError(
                    "vae_path is required when using diffusion_model_path"
                )
            args.extend(["--vae", _ensure_file(self.vae_path, "vae")])
            if not self.llm_path:
                raise InvalidRequestError(
                    "llm_path is required for Qwen/Flux-like models"
                )
            args.extend(["--llm", _ensure_file(self.llm_path, "llm")])
            if self.llm_vision_path:
                args.extend(
                    ["--llm_vision", _ensure_file(self.llm_vision_path, "llm_vision")]
                )

        if self.offload_to_cpu:
            args.append("--offload-to-cpu")
        if self.mmap:
            args.append("--mmap")
        args.extend(list(self.extra_args or ()))
        return args

    @staticmethod
    def build_command(
        request: ImageEditRequest,
        *,
        sd_cli: str,
        ctx_args: Sequence[str],
        output: str,
        init_img: Optional[str],
        mask_img: Optional[str],
        ref_imgs: Sequence[str],
    ) -> List[str]:
        cmd: List[str] = [sd_cli, *list(ctx_args)]

        cmd.extend(["--output", str(output)])
        cmd.extend(["--output-begin-idx", "0"])
        cmd.extend(["--batch-count", str(int(request.num_images))])

        cmd.extend(["--prompt", request.prompt])
        if request.negative_prompt:
            cmd.extend(["--negative-prompt", request.negative_prompt])

        cmd.extend(["--height", str(int(request.height))])
        cmd.extend(["--width", str(int(request.width))])
        cmd.extend(["--steps", str(int(request.steps))])
        cmd.extend(["--cfg-scale", str(float(request.cfg_scale))])

        if request.seed is None:
            cmd.extend(["--seed", "-1"])
        else:
            cmd.extend(["--seed", str(int(request.seed))])

        if request.mode in ("image_to_image", "inpaint"):
            if not init_img:
                raise InvalidRequestError(
                    "init_img is required for image_to_image/inpaint")
            cmd.extend(["--init-img", init_img])
            cmd.extend(["--strength", str(float(request.strength))])

        if request.mode == "inpaint":
            if not mask_img:
                raise InvalidRequestError("mask_img is required for inpaint")
            cmd.extend(["--mask", mask_img])

        if request.mode == "ref_image_edit":
            if not ref_imgs:
                raise InvalidRequestError(
                    "ref_imgs is required for ref_image_edit")
            for path in ref_imgs:
                cmd.extend(["--ref-image", path])

        return cmd
