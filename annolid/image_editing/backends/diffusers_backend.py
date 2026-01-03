from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from PIL import Image

from annolid.image_editing.backends.base import ImageEditingBackend, filter_kwargs
from annolid.image_editing.errors import BackendNotAvailableError, ImageEditingError
from annolid.image_editing.types import ImageEditRequest, ImageEditResult


def _resolve_device(device: str) -> str:
    device = (device or "auto").strip().lower()
    if device != "auto":
        return device
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_dtype(dtype: str, *, device: str) -> str:
    dtype = (dtype or "auto").strip().lower()
    if dtype != "auto":
        return dtype
    if device == "cuda":
        # Qwen-Image README recommends bfloat16 on CUDA.
        return "bfloat16"
    if device == "mps":
        return "float16"
    return "float32"


def _torch_dtype(dtype: str):
    import torch

    dtype = (dtype or "float32").strip().lower()
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype not in mapping:
        raise ValueError(
            f"Unsupported dtype {dtype!r}. Choose from: {', '.join(sorted(mapping))}"
        )
    return mapping[dtype]


@dataclass
class DiffusersBackend(ImageEditingBackend):
    model_id: str
    device: str = "auto"
    dtype: str = "auto"
    local_files_only: bool = False
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)

    name: str = "diffusers"

    _pipe: Any = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def run(self, request: ImageEditRequest) -> ImageEditResult:
        pipe = self._get_pipe()
        call = getattr(pipe, "__call__", pipe)

        generator = None
        try:
            import torch

            device = _resolve_device(self.device)
            if request.seed is not None:
                if device.startswith("cuda"):
                    generator = [
                        torch.Generator(device=device).manual_seed(int(request.seed) + i)
                        for i in range(int(request.num_images))
                    ]
                else:
                    # Most pipelines accept CPU generators even on MPS.
                    generator = [
                        torch.Generator(device="cpu").manual_seed(int(request.seed) + i)
                        for i in range(int(request.num_images))
                    ]
        except Exception:
            generator = None

        kwargs: Dict[str, Any] = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "height": int(request.height),
            "width": int(request.width),
            "num_inference_steps": int(request.steps),
            "guidance_scale": float(request.cfg_scale),
            "generator": generator[0] if isinstance(generator, list) and len(generator) == 1 else generator,
            "num_images_per_prompt": int(request.num_images),
        }

        if request.init_image is not None:
            kwargs.setdefault("image", request.init_image)
        if request.mask_image is not None:
            kwargs.setdefault("mask_image", request.mask_image)

        kwargs.update(self.extra_kwargs or {})
        call_kwargs = filter_kwargs(call, kwargs)

        try:
            out = call(**call_kwargs)
        except TypeError as exc:
            raise ImageEditingError(
                f"Diffusers pipeline did not accept provided arguments: {exc}"
            ) from exc
        except Exception as exc:
            raise ImageEditingError(f"Diffusers inference failed: {exc}") from exc

        images: Optional[Sequence[Image.Image]] = None
        if hasattr(out, "images"):
            images = out.images
        elif isinstance(out, dict) and "images" in out:
            images = out["images"]
        if not images:
            raise ImageEditingError(
                "Diffusers pipeline returned no images (unexpected output format)."
            )
        return ImageEditResult(
            images=list(images),
            meta={
                "backend": self.name,
                "model_id": self.model_id,
                "device": _resolve_device(self.device),
                "dtype": _resolve_dtype(self.dtype, device=_resolve_device(self.device)),
            },
        )

    def _get_pipe(self):
        with self._lock:
            if self._pipe is not None:
                return self._pipe
            try:
                from diffusers import DiffusionPipeline  # type: ignore
            except Exception as exc:
                raise BackendNotAvailableError(
                    "Diffusers backend requires the optional dependency 'diffusers'. "
                    "Install with: pip install diffusers"
                ) from exc

            try:
                import torch
            except Exception as exc:
                raise BackendNotAvailableError(
                    "Diffusers backend requires 'torch' to run inference."
                ) from exc

            device = _resolve_device(self.device)
            dtype = _resolve_dtype(self.dtype, device=device)

            pipe = DiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=_torch_dtype(dtype),
                local_files_only=bool(self.local_files_only),
                **(self.extra_kwargs or {}),
            )
            pipe = pipe.to(device)
            try:
                pipe.set_progress_bar_config(disable=True)
            except Exception:
                pass
            try:
                pipe.enable_attention_slicing()
            except Exception:
                pass
            try:
                pipe.enable_vae_slicing()
            except Exception:
                pass

            self._pipe = pipe
            return pipe

