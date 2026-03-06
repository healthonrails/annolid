from __future__ import annotations

import numpy as np

from annolid.services.embedding_search import run_embedding_search


def test_run_embedding_search_returns_sorted_top_k() -> None:
    vectors = {
        0: np.asarray([1.0, 0.0], dtype=np.float32),  # query
        1: np.asarray([0.8, 0.2], dtype=np.float32),
        2: np.asarray([0.2, 0.8], dtype=np.float32),
        3: np.asarray([0.9, 0.1], dtype=np.float32),
    }

    matches = run_embedding_search(
        query_vector=vectors[0],
        query_frame_index=0,
        total_frames=4,
        stride=1,
        max_frames=None,
        threshold=0.5,
        top_k=2,
        embed_frame=lambda idx: vectors[idx],
    )

    assert [m.frame_index for m in matches] == [3, 1]
    assert matches[0].similarity >= matches[1].similarity


def test_run_embedding_search_honors_stop() -> None:
    calls = {"count": 0}
    stop = {"flag": False}

    def _embed(_idx: int) -> np.ndarray:
        calls["count"] += 1
        if calls["count"] >= 1:
            stop["flag"] = True
        return np.asarray([1.0, 0.0], dtype=np.float32)

    def _stopped() -> bool:
        return bool(stop["flag"])

    matches = run_embedding_search(
        query_vector=np.asarray([1.0, 0.0], dtype=np.float32),
        query_frame_index=0,
        total_frames=10,
        stride=1,
        max_frames=None,
        threshold=0.0,
        top_k=10,
        embed_frame=_embed,
        is_stopped=_stopped,
    )

    assert len(matches) <= 1
    assert calls["count"] <= 1
