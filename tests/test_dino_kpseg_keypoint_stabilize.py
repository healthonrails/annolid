from __future__ import annotations

from pathlib import Path

from annolid.segmentation.dino_kpseg.keypoints import (
    LRStabilizeConfig,
    stabilize_symmetric_keypoints_xy,
    symmetric_pairs_from_flip_idx,
)
from annolid.segmentation.dino_kpseg.predictor import DinoKPSEGPredictor


def test_symmetric_pairs_from_flip_idx() -> None:
    assert symmetric_pairs_from_flip_idx([1, 0, 2]) == [(0, 1)]
    assert symmetric_pairs_from_flip_idx([0, 1, 2]) == []


def test_stabilize_swaps_left_right_pair() -> None:
    prev = [(0.0, 0.0), (10.0, 0.0), (5.0, 5.0)]
    curr = [(10.0, 0.0), (0.0, 0.0), (5.0, 5.0)]
    pairs = [(0, 1)]
    out = stabilize_symmetric_keypoints_xy(
        prev, list(curr), pairs=pairs, cfg=LRStabilizeConfig(enabled=True)
    )
    assert out[0] == prev[0]
    assert out[1] == prev[1]
    assert out[2] == prev[2]


def test_stabilize_respects_min_improvement() -> None:
    prev = [(0.0, 0.0), (10.0, 0.0)]
    curr = [(6.0, 0.0), (4.0, 0.0)]
    # Swapping makes both points move the same total distance (no improvement).
    out = stabilize_symmetric_keypoints_xy(
        prev,
        list(curr),
        pairs=[(0, 1)],
        cfg=LRStabilizeConfig(enabled=True, min_improvement_px=100.0),
    )
    assert out == curr


def test_predictor_resolves_checkpoint_dir(tmp_path: Path) -> None:
    (tmp_path / "weights").mkdir(parents=True, exist_ok=True)
    ckpt = tmp_path / "weights" / "best.pt"
    ckpt.write_bytes(b"not a real checkpoint")
    assert DinoKPSEGPredictor._resolve_checkpoint_path(tmp_path) == ckpt.resolve()
