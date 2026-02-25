from __future__ import annotations

from annolid.segmentation.dino_kpseg.inference_utils import filter_keypoints_by_score
from annolid.segmentation.dino_kpseg.predictor import DinoKPSEGPrediction


def test_filter_keypoints_by_score_drops_low_confidence_points() -> None:
    pred = DinoKPSEGPrediction(
        keypoints_xy=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
        keypoint_scores=[0.1, 0.5, 0.9],
        masks_patch=None,
        resized_hw=(256, 256),
        patch_size=16,
    )
    filtered = filter_keypoints_by_score(pred, min_score=0.5)
    assert filtered.keypoints_xy == [(3.0, 4.0), (5.0, 6.0)]
    assert filtered.keypoint_scores == [0.5, 0.9]


def test_filter_keypoints_by_score_is_noop_for_non_positive_threshold() -> None:
    pred = DinoKPSEGPrediction(
        keypoints_xy=[(1.0, 2.0)],
        keypoint_scores=[0.2],
        masks_patch=None,
        resized_hw=(128, 128),
        patch_size=16,
    )
    assert filter_keypoints_by_score(pred, min_score=0.0) == pred


def test_filter_keypoints_by_score_returns_kept_indices() -> None:
    pred = DinoKPSEGPrediction(
        keypoints_xy=[(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)],
        keypoint_scores=[0.2, 0.8, 0.7],
        masks_patch=None,
        resized_hw=(256, 256),
        patch_size=16,
    )
    filtered, kept = filter_keypoints_by_score(
        pred, min_score=0.75, return_indices=True
    )
    assert kept == [1]
    assert filtered.keypoints_xy == [(3.0, 4.0)]
    assert filtered.keypoint_scores == [0.8]
