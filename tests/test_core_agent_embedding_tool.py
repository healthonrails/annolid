from __future__ import annotations

from dataclasses import dataclass

from annolid.core.agent.tools.base import FrameBatch, FrameData, ToolContext
from annolid.core.agent.tools.embedding import EmbeddingTool
from annolid.core.models.base import (
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    RuntimeModel,
)
from annolid.core.types import FrameRef


@dataclass
class _FakeEmbedder(RuntimeModel):
    @property
    def model_id(self) -> str:
        return "fake-embedder"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(tasks=("embed",))

    def load(self) -> None:
        return None

    def predict(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(task="embed", output={"embedding": [1.0, 0.0]})

    def close(self) -> None:
        return None


def test_embedding_tool_collects_vectors(tmp_path) -> None:
    tool = EmbeddingTool(model=_FakeEmbedder())
    ctx = ToolContext(video_path=tmp_path / "v.mp4", results_dir=tmp_path, run_id="r1")
    batch = FrameBatch(frames=[FrameData(ref=FrameRef(frame_index=0))])
    result = tool.run(ctx, batch)
    assert list(result.embeddings[0]) == [1.0, 0.0]
