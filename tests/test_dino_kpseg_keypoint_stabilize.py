from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from annolid.segmentation.dino_kpseg.inference_utils import (
    DinoKPSEGInstanceCrop,
    predict_on_instance_crops,
)
from annolid.segmentation.dino_kpseg.keypoints import (
    LRStabilizeConfig,
    stabilize_symmetric_keypoints_xy,
    symmetric_pairs_from_flip_idx,
)
from annolid.segmentation.dino_kpseg.predictor import (
    DinoKPSEGPredictor,
    TemporalFusionConfig,
    _KeypointTrackStateStore,
)


def _make_stub_crop_predictor() -> DinoKPSEGPredictor:
    class _IdentityHead(torch.nn.Module):
        in_dim = 2

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class _StubExtractor:
        patch_size = 1

        def quantize_mask(self, mask_u8: np.ndarray) -> torch.Tensor:
            return torch.from_numpy((mask_u8 > 0).astype(np.float32))

    predictor = DinoKPSEGPredictor.__new__(DinoKPSEGPredictor)
    predictor.meta = type("Meta", (), {"threshold": 0.5})()
    predictor.keypoint_names = ["left_ear", "right_ear"]
    predictor.flip_idx = [1, 0]
    predictor._symmetric_pairs = [(0, 1)]
    predictor._track_state = _KeypointTrackStateStore()
    predictor._temporal_fusion = TemporalFusionConfig(
        enable=True,
        alpha=0.6,
        low_conf_threshold=0.5,
        max_instances=16,
    )
    predictor._max_track_instances = 16
    predictor.device = torch.device("cpu")
    predictor.extractor = _StubExtractor()
    predictor.head = _IdentityHead()
    return predictor


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


def test_predict_on_instance_crops_keeps_full_frame_state_with_manual_seed() -> None:
    """Full-frame seed_instance_state must not pull cropped temporal fusion to crop borders."""
    predictor = _make_stub_crop_predictor()

    # Seed full-frame coordinates inside an instance crop at offset (800, 580).
    predictor.seed_instance_state(
        7,
        keypoints_xy=[(802.5, 583.5), (807.5, 583.5)],
        keypoint_scores=[1.0, 1.0],
    )

    # Frame 1: crop is 10x10 at offset (800, 580). Model emits low-confidence peaks
    # with swapped left/right channels at crop (7.5, 3.5) and (2.5, 3.5).
    feats = torch.full((2, 10, 10), -10.0, dtype=torch.float32)
    feats[0, 3, 7] = -1.1  # sigmoid ~0.25 (< 0.5 low_conf_threshold) at crop x=7.5
    feats[1, 3, 2] = -1.1  # sigmoid ~0.25 at crop x=2.5
    predictor.extract_features = lambda _crop: feats  # type: ignore[assignment]

    crop = DinoKPSEGInstanceCrop(
        instance_id=7,
        bbox_xyxy=(800, 580, 810, 590),
        crop_bgr=np.zeros((10, 10, 3), dtype=np.uint8),
        crop_mask=np.ones((10, 10), dtype=np.uint8),
        offset_xy=(800, 580),
    )
    preds = predict_on_instance_crops(predictor, [crop], stabilize_lr=True)
    assert len(preds) == 1
    inst_id, pred = preds[0]
    assert inst_id == 7
    # L/R swap is corrected and temporal fusion stays at full-frame (802.5, 583.5) / (807.5, 583.5),
    # rather than clipping toward the bottom-right crop border (809.0, 589.0).
    assert pred.keypoints_xy[0] == pytest.approx((802.5, 583.5))
    assert pred.keypoints_xy[1] == pytest.approx((807.5, 583.5))

    stored_xy, _ = predictor._track_state.get(instance_id=7)
    assert stored_xy is not None
    assert stored_xy[0] == pytest.approx((802.5, 583.5))
    assert stored_xy[1] == pytest.approx((807.5, 583.5))


def test_predict_instances_preserves_global_coords_across_shifting_crop_boxes() -> None:
    """Moving crop offsets across frames must not inject synthetic drift into _track_state."""
    predictor = _make_stub_crop_predictor()

    call_idx = 0

    def _extract_features(_crop: np.ndarray) -> torch.Tensor:
        nonlocal call_idx
        feats = torch.full((2, 20, 20), -10.0, dtype=torch.float32)
        if call_idx == 0:
            # Frame 0 bbox=[100, 100, 120, 120]: high-conf peaks at crop (5.5, 6.5) & (13.5, 6.5)
            # => global (105.5, 106.5) & (113.5, 106.5).
            feats[0, 6, 5] = 6.0
            feats[1, 6, 13] = 6.0
        else:
            # Frame 1 bbox=[96, 98, 116, 118] (crop shifted by -4, -2 while animal stayed still):
            # stationary global points correspond to crop (9.5, 8.5) & (17.5, 8.5), emitted low-conf.
            feats[0, 8, 9] = -1.1
            feats[1, 8, 17] = -1.1
        call_idx += 1
        return feats

    predictor.extract_features = _extract_features  # type: ignore[assignment]
    frame = np.zeros((200, 200, 3), dtype=np.uint8)

    out0 = predictor.predict_instances(
        frame,
        bboxes_xyxy=[[100, 100, 120, 120]],
        instance_ids=[3],
        stabilize_lr=True,
    )
    assert out0[0][1].keypoints_xy[0] == pytest.approx((105.5, 106.5))
    assert out0[0][1].keypoints_xy[1] == pytest.approx((113.5, 106.5))

    out1 = predictor.predict_instances(
        frame,
        bboxes_xyxy=[[96, 98, 116, 118]],
        instance_ids=[3],
        stabilize_lr=True,
    )
    # Stationary animal with shifted bbox stays anchored at global (105.5, 106.5) & (113.5, 106.5).
    assert out1[0][1].keypoints_xy[0] == pytest.approx((105.5, 106.5))
    assert out1[0][1].keypoints_xy[1] == pytest.approx((113.5, 106.5))
