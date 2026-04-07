from __future__ import annotations

import torch

from annolid.segmentation.SAM.sam3.windowed_runner import (
    build_window_seed_segments,
    compute_window_reuse_shift,
    first_manual_seed_frame,
    normalize_window_schedule,
    resolve_window_schedule,
    shift_annotations_to_window,
)


def test_normalize_window_schedule_clamps_stride() -> None:
    assert normalize_window_schedule(window_size=5, stride=5) == (5, 4)
    assert normalize_window_schedule(window_size=1, stride=0) == (1, 1)


def test_resolve_window_schedule_uses_user_values() -> None:
    schedule = resolve_window_schedule(
        resolved_device=torch.device("cpu"),
        total_frames=120,
        user_window_size=9,
        user_stride=3,
    )
    assert schedule == (9, 3)


def test_compute_window_reuse_shift_uses_true_overlap() -> None:
    assert (
        compute_window_reuse_shift(
            previous_window_end_idx=4,
            window_start_idx=4,
            frame_count=4,
            previous_window_frame_count=4,
        )
        == 0
    )
    assert (
        compute_window_reuse_shift(
            previous_window_end_idx=4,
            window_start_idx=2,
            frame_count=4,
            previous_window_frame_count=4,
        )
        == 2
    )
    assert (
        compute_window_reuse_shift(
            previous_window_end_idx=8,
            window_start_idx=6,
            frame_count=4,
            previous_window_frame_count=4,
        )
        == 2
    )


def test_shift_annotations_to_window_rewrites_local_indices() -> None:
    grouped = shift_annotations_to_window(
        [
            {"ann_frame_idx": 9, "obj_id": 1},
            {"ann_frame_idx": 11, "obj_id": 2},
            {"ann_frame_idx": 13, "obj_id": 3},
        ],
        start_idx=10,
        end_idx=13,
    )
    assert sorted(grouped.keys()) == [1]
    assert grouped[1][0]["ann_frame_idx"] == 1
    assert grouped[1][0]["obj_id"] == 2


def test_first_manual_seed_frame_picks_earliest_valid_frame() -> None:
    frame_idx = first_manual_seed_frame(
        [{"ann_frame_idx": -1}, {"ann_frame_idx": 8}, {"ann_frame_idx": 3}]
    )
    assert frame_idx == 3


def test_build_window_seed_segments_text_fallback() -> None:
    assert build_window_seed_segments([], 6, has_text_prompt=False) == []
    assert build_window_seed_segments([], 6, has_text_prompt=True) == [(0, 6)]
    assert build_window_seed_segments([4, 1, 4], 6, has_text_prompt=False) == [
        (1, 4),
        (4, 6),
    ]
