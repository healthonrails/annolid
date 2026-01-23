from __future__ import annotations

import pytest

from annolid.core.agent.tools.vector_index import NumpyEmbeddingIndex
from annolid.core.types import FrameRef


pytest.importorskip("numpy")


def test_numpy_embedding_index_search() -> None:
    frames = [FrameRef(frame_index=0), FrameRef(frame_index=1)]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]
    index = NumpyEmbeddingIndex(embeddings, frames)
    results = index.search([0.9, 0.1], top_k=1)
    assert results[0].frame.frame_index == 0
