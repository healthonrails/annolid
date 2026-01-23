from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from annolid.core.models.base import ModelRequest, RuntimeModel

from .base import FrameBatch, Tool, ToolContext


@dataclass(frozen=True)
class CaptionResult:
    frames: dict[int, str]


class CaptionTool(Tool[FrameBatch, CaptionResult]):
    """Generate captions for frames using a RuntimeModel."""

    name = "caption"

    def __init__(
        self,
        *,
        model: RuntimeModel,
        prompt: Optional[str] = None,
        config: Optional[dict[str, object]] = None,
    ) -> None:
        super().__init__(config=config)
        self._model = model
        self._prompt = prompt

    def run(self, ctx: ToolContext, payload: FrameBatch) -> CaptionResult:
        results: dict[int, str] = {}
        with self._model:
            for frame in payload:
                response = self._model.predict(
                    ModelRequest(
                        task="caption",
                        text=self._prompt,
                        image=frame.image_rgb,
                        image_path=str(frame.image_path) if frame.image_path else None,
                    )
                )
                text = response.text or (response.output or {}).get("text") or ""
                results[int(frame.ref.frame_index)] = str(text)
        return CaptionResult(frames=results)
