from __future__ import annotations

import pytest
import torch

pytest.importorskip("shapely")
from annolid.tracker.cotracker.track import CoTrackerProcessor


def test_build_chunk_queries_clamps_to_local_chunk_range() -> None:
    processor = CoTrackerProcessor.__new__(CoTrackerProcessor)
    processor.queries = torch.tensor(
        [
            [1676.0, 100.0, 120.0],  # before chunk start
            [1677.0, 110.0, 130.0],  # at chunk start
            [1684.0, 120.0, 140.0],  # inside chunk
            [2000.0, 130.0, 150.0],  # after chunk end
        ],
        dtype=torch.float32,
    )

    q = processor._build_chunk_queries(chunk_start_frame=1677, chunk_num_frames=8)

    assert q is not None
    assert q.shape == (1, 4, 3)
    # Local time indices clamped to [0, 7].
    assert q[0, :, 0].tolist() == [0.0, 0.0, 7.0, 7.0]
    # Coordinates remain unchanged.
    assert q[0, :, 1:].tolist() == [
        [100.0, 120.0],
        [110.0, 130.0],
        [120.0, 140.0],
        [130.0, 150.0],
    ]
