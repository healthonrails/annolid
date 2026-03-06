from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np


@dataclass(frozen=True)
class EmbeddingSearchMatch:
    frame_index: int
    similarity: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "frame_index": int(self.frame_index),
            "similarity": float(self.similarity),
        }


def run_embedding_search(
    *,
    query_vector: np.ndarray,
    query_frame_index: int,
    total_frames: int,
    stride: int,
    max_frames: Optional[int],
    threshold: float,
    top_k: int,
    embed_frame: Callable[[int], np.ndarray],
    is_stopped: Optional[Callable[[], bool]] = None,
    on_match: Optional[Callable[[EmbeddingSearchMatch], None]] = None,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[EmbeddingSearchMatch]:
    """Run vector similarity search over frames using caller-provided embedding IO."""
    if total_frames <= 0:
        raise ValueError("No frames selected for search.")

    stride = max(1, int(stride))
    top_k = max(1, int(top_k))
    threshold = float(threshold)
    max_steps = (int(total_frames) + stride - 1) // stride
    if max_frames is not None:
        max_steps = min(max_steps, max(1, int(max_frames)))
    if max_steps <= 0:
        raise ValueError("No frames selected for search.")

    qvec = np.asarray(query_vector, dtype=np.float32)
    matches: List[EmbeddingSearchMatch] = []
    step = 0

    for frame_idx in range(0, int(total_frames), stride):
        if step >= max_steps:
            break
        step += 1
        if is_stopped is not None and is_stopped():
            break
        if int(frame_idx) == int(query_frame_index):
            continue

        vec = np.asarray(embed_frame(int(frame_idx)), dtype=np.float32)
        sim = float(vec @ qvec)
        if sim >= threshold:
            match = EmbeddingSearchMatch(frame_index=int(frame_idx), similarity=sim)
            matches.append(match)
            if on_match is not None:
                on_match(match)

        if on_progress is not None and (step % 10 == 0 or step == max_steps):
            on_progress(step, max_steps)

    matches.sort(key=lambda m: m.similarity, reverse=True)
    return matches[:top_k]


__all__ = [
    "EmbeddingSearchMatch",
    "run_embedding_search",
]
