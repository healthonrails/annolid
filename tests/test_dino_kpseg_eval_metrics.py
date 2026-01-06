from __future__ import annotations

import numpy as np
import pytest

from annolid.segmentation.dino_kpseg.eval import DinoKPSEGEvalAccumulator


def test_eval_accumulator_basic_metrics() -> None:
    acc = DinoKPSEGEvalAccumulator(kpt_count=2, thresholds_px=[4.0, 8.0])
    gt = np.array(
        [
            [0.0, 0.0, 2.0],
            [0.9, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    acc.update(
        pred_xy=[(0.0, 0.0), (10.0, 0.0)],
        gt_instances=[gt],
        image_hw=(10, 10),
        lr_pairs=[],
    )
    summary = acc.summary()
    assert summary["keypoints_visible_total"] == 2
    assert summary["mean_error_px"] == pytest.approx(0.5)
    assert summary["pck"]["4.0"] == pytest.approx(1.0)
    assert summary["pck"]["8.0"] == pytest.approx(1.0)


def test_eval_accumulator_swap_rate() -> None:
    acc = DinoKPSEGEvalAccumulator(kpt_count=2, thresholds_px=[4.0])
    gt = np.array(
        [
            [0.0, 0.0, 2.0],
            [0.9, 0.0, 2.0],
        ],
        dtype=np.float32,
    )
    acc.update(
        pred_xy=[(9.0, 0.0), (0.0, 0.0)],
        gt_instances=[gt],
        image_hw=(10, 10),
        lr_pairs=[(0, 1)],
    )
    summary = acc.summary()
    assert summary["swap_pairs_total"] == 1
    assert summary["swap_pairs_swapped"] == 1
    assert summary["swap_rate"] == pytest.approx(1.0)
