"""Service-layer entry points for search workflows."""

from __future__ import annotations

from typing import Any

from annolid.services.embedding_search import EmbeddingSearchMatch, run_embedding_search


def search_indexed_frames(*args: Any, **kwargs: Any):
    from annolid.agents.frame_search import search_frames

    return search_frames(*args, **kwargs)


def search_video_frames(*args: Any, **kwargs: Any):
    from annolid.agents.frame_search import search_video

    return search_video(*args, **kwargs)


__all__ = [
    "EmbeddingSearchMatch",
    "run_embedding_search",
    "search_indexed_frames",
    "search_video_frames",
]
