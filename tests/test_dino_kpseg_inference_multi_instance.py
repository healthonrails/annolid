from __future__ import annotations

import numpy as np
import pytest
import torch

from annolid.segmentation.dino_kpseg.predictor import (
    DinoKPSEGPrediction,
    DinoKPSEGPredictor,
)
from annolid.segmentation.yolos import InferenceProcessor


def test_predictor_resolve_instance_ids_deduplicates_collisions() -> None:
    resolved = DinoKPSEGPredictor._resolve_instance_ids(
        count=4,
        instance_ids=[7, 7, "7", None],
    )
    assert resolved == [7, 8, 9, 3]


def test_predict_instances_uses_collision_safe_instance_ids() -> None:
    predictor = DinoKPSEGPredictor.__new__(DinoKPSEGPredictor)
    seen_ids = []

    def _extract_features(_crop):
        return torch.zeros((1, 2, 2), dtype=torch.float32)

    def _predict_from_features(
        _feats,
        *,
        frame_shape,
        threshold=None,
        return_patch_masks=False,
        stabilize_lr=False,
        stabilize_cfg=None,
        instance_id=None,
        mask=None,
    ):
        _ = (
            frame_shape,
            threshold,
            return_patch_masks,
            stabilize_lr,
            stabilize_cfg,
            mask,
        )
        seen_ids.append(int(instance_id))
        return DinoKPSEGPrediction(
            keypoints_xy=[(1.0, 2.0)],
            keypoint_scores=[0.9],
            masks_patch=None,
            resized_hw=(2, 2),
            patch_size=16,
        )

    predictor.extract_features = _extract_features  # type: ignore[assignment]
    predictor.predict_from_features = _predict_from_features  # type: ignore[assignment]

    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    out = predictor.predict_instances(
        frame,
        bboxes_xyxy=[[1, 1, 10, 12], [12, 1, 22, 12]],
        instance_ids=[5, 5],
    )

    assert seen_ids == [5, 6]
    assert [int(instance_id) for instance_id, _ in out] == [5, 6]


def _new_inference_processor_for_shapes() -> InferenceProcessor:
    proc = InferenceProcessor.__new__(InferenceProcessor)
    proc.pose_schema = None
    proc._instance_label_to_gid = {}
    return proc


def test_instance_masks_from_shapes_assigns_distinct_ids_for_duplicate_labels() -> None:
    pytest.importorskip("cv2")
    processor = _new_inference_processor_for_shapes()
    shapes = [
        {
            "label": "mouse",
            "shape_type": "polygon",
            "points": [[2, 2], [10, 2], [10, 10], [2, 10]],
        },
        {
            "label": "mouse",
            "shape_type": "polygon",
            "points": [[14, 2], [22, 2], [22, 10], [14, 10]],
        },
    ]
    masks = processor._instance_masks_from_shapes(shapes, frame_hw=(32, 32))
    assert len(masks) == 2
    assert {int(gid) for gid, _mask in masks} == {0, 1}


def test_instance_masks_from_shapes_remaps_duplicate_group_ids() -> None:
    pytest.importorskip("cv2")
    processor = _new_inference_processor_for_shapes()
    shapes = [
        {
            "label": "mouse_a",
            "group_id": 4,
            "shape_type": "polygon",
            "points": [[2, 2], [10, 2], [10, 10], [2, 10]],
        },
        {
            "label": "mouse_b",
            "group_id": 4,
            "shape_type": "polygon",
            "points": [[14, 2], [22, 2], [22, 10], [14, 10]],
        },
    ]
    masks = processor._instance_masks_from_shapes(shapes, frame_hw=(32, 32))
    assert len(masks) == 2
    gids = [int(gid) for gid, _mask in masks]
    assert gids[0] == 4
    assert gids[1] != 4


def test_extract_dino_kpseg_results_forwards_instance_ids_to_predictor() -> None:
    class _DummyModel:
        def __init__(self) -> None:
            self.instance_id_calls = []

        def predict_instances(self, _frame_bgr, *, bboxes_xyxy, instance_ids=None, **_):
            _ = bboxes_xyxy
            ids = [] if instance_ids is None else [int(v) for v in list(instance_ids)]
            self.instance_id_calls.append(ids)
            return [
                (
                    int(instance_id),
                    DinoKPSEGPrediction(
                        keypoints_xy=[(4.0, 5.0)],
                        keypoint_scores=[0.8],
                        masks_patch=None,
                        resized_hw=(8, 8),
                        patch_size=16,
                    ),
                )
                for instance_id in ids
            ]

    processor = InferenceProcessor.__new__(InferenceProcessor)
    processor.model = _DummyModel()
    processor.model_type = "dinokpseg"
    processor.keypoint_names = ["nose"]
    processor.pose_schema = None
    processor._instance_label_to_gid = {}

    frame = np.zeros((24, 24, 3), dtype=np.uint8)
    bboxes = np.array([[1, 1, 8, 8], [10, 1, 18, 8]], dtype=np.float32)
    out = processor.extract_dino_kpseg_results(
        frame,
        bboxes=bboxes,
        instance_ids=np.array([12, 21], dtype=object),
    )

    assert processor.model.instance_id_calls == [[12, 21]]
    assert len(out) == 2
    assert {int(shape.group_id) for shape in out} == {12, 21}
