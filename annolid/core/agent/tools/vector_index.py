from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from annolid.core.types import FrameRef


@dataclass(frozen=True)
class SearchResult:
    frame: FrameRef
    score: float


class NumpyEmbeddingIndex:
    """Simple cosine-similarity index backed by NumPy."""

    def __init__(
        self, embeddings: Sequence[Sequence[float]], frames: Sequence[FrameRef]
    ) -> None:
        if len(embeddings) != len(frames):
            raise ValueError("embeddings and frames length mismatch.")
        try:
            import numpy as np  # type: ignore
        except ImportError as exc:
            raise ImportError("NumpyEmbeddingIndex requires numpy.") from exc
        self._np = np
        self._frames = list(frames)
        matrix = np.array(embeddings, dtype="float32")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = matrix / norms

    def search(self, query: Sequence[float], *, top_k: int = 5) -> List[SearchResult]:
        np = self._np
        q = np.array(query, dtype="float32")
        denom = np.linalg.norm(q)
        if denom == 0:
            denom = 1.0
        q = q / denom
        scores = self._matrix @ q
        top_k = max(1, int(top_k))
        idx = np.argsort(scores)[::-1][:top_k]
        return [
            SearchResult(frame=self._frames[int(i)], score=float(scores[int(i)]))
            for i in idx
        ]
