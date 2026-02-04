from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from PIL import Image

from .errors import InvalidRequestError

ImageEditMode = Literal[
    "text_to_image",
    "image_to_image",
    "inpaint",
    "ref_image_edit",
]


@dataclass(frozen=True)
class ImageEditRequest:
    mode: ImageEditMode
    prompt: str
    negative_prompt: str = ""

    width: int = 512
    height: int = 512
    steps: int = 20
    cfg_scale: float = 7.0
    seed: Optional[int] = None
    strength: float = 0.75

    init_image: Optional[Image.Image] = None
    mask_image: Optional[Image.Image] = None
    ref_images: Optional[List[Image.Image]] = None

    num_images: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        prompt = (self.prompt or "").strip()
        if not prompt:
            raise InvalidRequestError("prompt must be non-empty")
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise InvalidRequestError("width and height must be positive")
        if int(self.steps) <= 0:
            raise InvalidRequestError("steps must be positive")
        if float(self.cfg_scale) <= 0:
            raise InvalidRequestError("cfg_scale must be positive")
        if int(self.num_images) <= 0:
            raise InvalidRequestError("num_images must be positive")

        if self.mode == "text_to_image":
            return

        if self.mode == "image_to_image":
            if self.init_image is None:
                raise InvalidRequestError(
                    "init_image is required for image_to_image mode"
                )
            return

        if self.mode == "inpaint":
            if self.init_image is None or self.mask_image is None:
                raise InvalidRequestError(
                    "init_image and mask_image are required for inpaint mode"
                )
            return

        if self.mode == "ref_image_edit":
            if not self.ref_images:
                raise InvalidRequestError(
                    "ref_images is required for ref_image_edit mode"
                )
            return

        raise InvalidRequestError(f"Unknown mode: {self.mode!r}")


@dataclass(frozen=True)
class ImageEditResult:
    images: List[Image.Image]
    meta: Dict[str, Any] = field(default_factory=dict)
