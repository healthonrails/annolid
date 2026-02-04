from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont, ImageOps

from annolid.image_editing.backends.base import ImageEditingBackend
from annolid.image_editing.types import ImageEditRequest, ImageEditResult


@dataclass
class PillowBackend(ImageEditingBackend):
    """A lightweight fallback backend for demos/tests (no ML involved)."""

    name: str = "pillow"

    def run(self, request: ImageEditRequest) -> ImageEditResult:
        images: List[Image.Image] = []
        for idx in range(int(request.num_images)):
            images.append(self._render_one(request, offset=idx))
        meta: Dict[str, Any] = {
            "backend": self.name,
            "mode": request.mode,
        }
        return ImageEditResult(images=images, meta=meta)

    def _render_one(self, request: ImageEditRequest, *, offset: int) -> Image.Image:
        if request.mode == "text_to_image" or request.init_image is None:
            img = Image.new(
                "RGB", (int(request.width), int(request.height)), (255, 255, 255)
            )
        else:
            img = request.init_image.convert("RGB").copy()

        prompt = request.prompt.strip()
        # Simple prompt-driven effects for basic "editing" semantics.
        prompt_lower = prompt.lower()
        if "invert" in prompt_lower:
            img = ImageOps.invert(img)
        if "grayscale" in prompt_lower or "grey" in prompt_lower:
            img = ImageOps.grayscale(img).convert("RGB")

        self._draw_prompt(img, prompt=prompt, line_offset=offset)
        return img

    @staticmethod
    def _draw_prompt(img: Image.Image, *, prompt: str, line_offset: int = 0) -> None:
        draw = ImageDraw.Draw(img)
        font: Optional[ImageFont.ImageFont]
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        text = prompt if len(prompt) <= 120 else prompt[:117] + "..."
        draw.rectangle([(0, 0), (img.width, 32)], fill=(0, 0, 0))
        draw.text((8, 8 + 10 * line_offset), text, fill=(255, 255, 255), font=font)
