from __future__ import annotations

import torch
import pytest

from annolid.segmentation.dino_kpseg.predictor import DinoKPSEGPredictor


def test_local_peaks_2d_returns_topk_sorted() -> None:
    heatmap = torch.zeros((5, 5), dtype=torch.float32)
    heatmap[1, 1] = 0.9  # (x=1,y=1)
    heatmap[2, 1] = 0.85  # neighbor, should not be selected as a local max
    heatmap[3, 3] = 0.8

    coords, scores = DinoKPSEGPredictor._local_peaks_2d(
        heatmap, topk=2, threshold=0.0, nms_radius=1
    )
    assert coords == [(1, 1), (3, 3)]
    assert scores == pytest.approx([0.9, 0.8])


def test_local_peaks_2d_falls_back_to_global_max() -> None:
    heatmap = torch.zeros((4, 4), dtype=torch.float32)
    heatmap[0, 2] = 0.7
    heatmap[3, 1] = 0.6

    coords, scores = DinoKPSEGPredictor._local_peaks_2d(
        heatmap, topk=3, threshold=0.95, nms_radius=2
    )
    assert coords == [(2, 0)]
    assert scores == pytest.approx([0.7])
