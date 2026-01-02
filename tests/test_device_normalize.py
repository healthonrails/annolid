from __future__ import annotations

import torch

from annolid.features.dinov3_extractor import Dinov3FeatureExtractor
from annolid.segmentation.dino_kpseg.cli_utils import normalize_device


def _patch_mps_unavailable(monkeypatch) -> None:
    if hasattr(torch.backends, "mps") and hasattr(torch.backends.mps, "is_available"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


def test_normalize_device_numeric_to_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _patch_mps_unavailable(monkeypatch)
    assert normalize_device("0") == "cuda:0"


def test_normalize_device_cuda_suffix(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _patch_mps_unavailable(monkeypatch)
    assert normalize_device("cuda0") == "cuda:0"


def test_dinov3_select_device_numeric(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    _patch_mps_unavailable(monkeypatch)
    assert Dinov3FeatureExtractor._select_device("0") == torch.device("cuda:0")
