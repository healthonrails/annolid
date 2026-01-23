from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from annolid.core.models.base import ModelRequest, RuntimeModel
from annolid.core.types import FrameRef

from .base import FrameBatch, Tool, ToolContext


@dataclass(frozen=True)
class EmbeddingResult:
    frames: Sequence[FrameRef]
    embeddings: Sequence[Sequence[float]]


class EmbeddingTool(Tool[FrameBatch, EmbeddingResult]):
    """Generate embeddings for frames or text payloads."""

    name = "embedding"

    def __init__(
        self,
        *,
        model: RuntimeModel,
        config: Optional[dict[str, object]] = None,
    ) -> None:
        super().__init__(config=config)
        self._model = model

    def run(self, ctx: ToolContext, payload: FrameBatch) -> EmbeddingResult:
        frames: List[FrameRef] = []
        vectors: List[Sequence[float]] = []
        with self._model:
            for frame in payload:
                meta = frame.meta if isinstance(frame.meta, dict) else {}
                text = meta.get("text")
                video = meta.get("video") or meta.get("video_path")
                params = dict(self.config or {})
                if video is not None:
                    params["video"] = video
                request = ModelRequest(
                    task="embed",
                    text=text if text is not None else None,
                    image=frame.image_rgb,
                    image_path=str(frame.image_path) if frame.image_path else None,
                    params=params,
                )
                response = self._model.predict(request)
                embedding = (response.output or {}).get("embedding")
                if embedding is None:
                    raise ValueError("Embedding model returned no embedding.")
                vectors.append(list(embedding))
                frames.append(frame.ref)
        return EmbeddingResult(frames=frames, embeddings=vectors)
