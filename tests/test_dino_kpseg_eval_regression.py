from __future__ import annotations

import numpy as np
import pytest

from annolid.segmentation.dino_kpseg.eval import DinoKPSEGEvalAccumulator


def test_eval_accumulator_pck_and_swap_rate_regression() -> None:
    acc = DinoKPSEGEvalAccumulator(
        kpt_count=3, thresholds_px=[2.0, 4.0, 8.0, 16.0]
    )

    gt1 = np.array(
        [
            [0.10, 0.10, 2.0],
            [0.90, 0.10, 2.0],
            [0.50, 0.50, 2.0],
        ],
        dtype=np.float32,
    )
    pred1 = [(90.0, 10.0), (10.0, 10.0), (50.0, 55.0)]
    acc.update(
        pred_xy=pred1,
        gt_instances=[gt1],
        image_hw=(100, 100),
        lr_pairs=[(0, 1)],
    )

    gt2 = np.array(
        [
            [0.20, 0.20, 2.0],
            [0.80, 0.20, 2.0],
            [0.50, 0.60, 2.0],
        ],
        dtype=np.float32,
    )
    pred2 = [(22.0, 21.0), (80.0, 23.0), (48.0, 62.0)]
    acc.update(
        pred_xy=pred2,
        gt_instances=[gt2],
        image_hw=(100, 100),
        lr_pairs=[(0, 1)],
    )

    summary = acc.summary()
    assert summary["pck"]["2.0"] == pytest.approx(0.0)
    assert summary["pck"]["4.0"] == pytest.approx(0.5)
    assert summary["pck"]["8.0"] == pytest.approx(4.0 / 6.0)
    assert summary["pck"]["16.0"] == pytest.approx(4.0 / 6.0)
    assert summary["swap_rate"] == pytest.approx(0.5)
