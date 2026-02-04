from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from annolid.segmentation.dino_kpseg.data import DinoKPSEGPoseDataset


@dataclass
class _StubModelConfig:
    hidden_size: int = 8


@dataclass
class _StubModel:
    config: _StubModelConfig = field(default_factory=_StubModelConfig)


class _StubExtractor:
    def __init__(self, *, patch_size: int = 16) -> None:
        self.patch_size = int(patch_size)
        self.model_id = "stub"
        self.cfg = SimpleNamespace(short_side=64, layers=(-1,))
        self.model = _StubModel()
        self.calls = 0

    def extract(self, pil: Image.Image, *, return_type: str = "torch") -> torch.Tensor:
        assert return_type == "torch"
        self.calls += 1
        width, height = pil.size
        h_p = max(1, int(height) // int(self.patch_size))
        w_p = max(1, int(width) // int(self.patch_size))
        return torch.zeros((8, h_p, w_p), dtype=torch.float32)


def _write_minimal_yolo_pose_example(root: Path) -> Path:
    img_dir = root / "images" / "train"
    lbl_dir = root / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    img_path = img_dir / "frame.png"
    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(img_path)

    # Two pose instances (two animals) in the same frame.
    # Format: cls cx cy w h (x y v)*K, with v in {0,1,2}.
    lines = [
        "0 0.25 0.5 0.4 0.6  0.20 0.40 2  0.30 0.60 2",
        "0 0.75 0.5 0.4 0.6  0.70 0.40 2  0.80 0.60 2",
    ]
    (lbl_dir / "frame.txt").write_text("\n".join(lines), encoding="utf-8")
    return img_path


def test_dino_kpseg_dataset_instance_mode_auto_selects_per_instance(
    tmp_path: Path,
) -> None:
    img_path = _write_minimal_yolo_pose_example(tmp_path)
    extractor = _StubExtractor(patch_size=16)

    ds = DinoKPSEGPoseDataset(
        [img_path],
        kpt_count=2,
        kpt_dims=3,
        radius_px=6.0,
        extractor=extractor,  # type: ignore[arg-type]
        instance_mode="auto",
        cache_dir=None,
    )
    assert ds.instance_mode == "per_instance"
    assert len(ds) == 2


def test_dino_kpseg_dataset_per_instance_cache_avoids_collisions(
    tmp_path: Path,
) -> None:
    img_path = _write_minimal_yolo_pose_example(tmp_path)
    cache_dir = tmp_path / "cache"
    extractor = _StubExtractor(patch_size=16)

    ds = DinoKPSEGPoseDataset(
        [img_path],
        kpt_count=2,
        kpt_dims=3,
        radius_px=6.0,
        extractor=extractor,  # type: ignore[arg-type]
        instance_mode="auto",
        cache_dir=cache_dir,
        cache_dtype=torch.float32,
    )
    assert ds.instance_mode == "per_instance"

    _ = ds[0]
    _ = ds[1]
    assert extractor.calls == 2

    # Second pass should load both crops from cache.
    _ = ds[0]
    _ = ds[1]
    assert extractor.calls == 2

    cache_files = sorted(cache_dir.glob("*.pt"))
    assert len(cache_files) == 2
