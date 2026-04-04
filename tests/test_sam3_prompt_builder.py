from __future__ import annotations

import numpy as np

from annolid.segmentation.SAM.sam3.prompt_builder import (
    build_prompts_from_annotations,
    label_hints_from_ids,
)


def test_prompt_builder_text_only_fallback_uses_first_frame() -> None:
    result = build_prompts_from_annotations(
        annotations=[],
        text_prompt="mouse",
        frame_shape=(32, 48, 3),
        video_dir="/tmp/video",
        first_frame_index=lambda: 7,
        shape_points_to_mask=lambda _pts, shape, _shape_type: np.zeros(
            shape[:2], dtype=np.uint8
        ),
    )
    assert result.frame_idx == 7
    assert result.boxes == []
    assert result.mask_inputs == []
    assert result.points == []


def test_prompt_builder_builds_box_mask_and_point_prompts() -> None:
    mask = np.pad(np.ones((4, 4), dtype=np.uint8), ((2, 2), (3, 3)), mode="constant")
    annotations = [
        {
            "type": "box",
            "ann_frame_idx": 5,
            "box": [2.0, 4.0, 10.0, 12.0],
            "labels": [3],
            "obj_id": 3,
        },
        {"type": "mask", "ann_frame_idx": 5, "mask": mask, "labels": [4], "obj_id": 4},
        {
            "type": "points",
            "ann_frame_idx": 5,
            "points": [[8.0, 9.0]],
            "labels": [1],
            "obj_id": 5,
        },
    ]
    result = build_prompts_from_annotations(
        annotations=annotations,
        text_prompt=None,
        frame_shape=(16, 16, 3),
        video_dir="/tmp/video",
        first_frame_index=lambda: 0,
        shape_points_to_mask=lambda _pts, shape, _shape_type: np.zeros(
            shape[:2], dtype=np.uint8
        ),
    )
    assert result.frame_idx == 5
    assert len(result.boxes) == 1
    assert result.box_labels == [3]
    assert len(result.mask_inputs) == 1
    assert result.mask_labels == [1]
    assert len(result.points) == 1
    assert result.point_labels == [1]
    assert 3 in result.obj_ids and 4 in result.obj_ids and 5 in result.obj_ids


def test_prompt_builder_label_hints_from_ids() -> None:
    hints = label_hints_from_ids([1, 2, 3], {1: "mouse", 3: "vole"})
    assert hints == ["mouse", "2", "vole"]
