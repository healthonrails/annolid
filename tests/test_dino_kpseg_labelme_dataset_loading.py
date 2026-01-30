from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from annolid.segmentation.dino_kpseg.data import (
    DinoKPSEGPoseDataset,
    load_labelme_pose_spec,
    summarize_labelme_pose_labels,
)


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


def _write_minimal_labelme_pose_example(root: Path) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    img_path = root / "frame.png"
    Image.new("RGB", (64, 64), color=(0, 0, 0)).save(img_path)

    # Two instances, grouped via group_id. Keypoints are points labeled by name.
    # Polygons are optional but used for bbox crops in per-instance mode.
    payload = {
        "version": "5.0.0",
        "imagePath": img_path.name,
        "imageHeight": 64,
        "imageWidth": 64,
        "shapes": [
            # instance 0
            {
                "label": "animal",
                "shape_type": "polygon",
                "group_id": 0,
                "points": [[8.0, 10.0], [28.0, 10.0], [28.0, 40.0], [8.0, 40.0]],
            },
            {
                "label": "k0",
                "shape_type": "point",
                "group_id": 0,
                "points": [[12.0, 18.0]],
            },
            {
                "label": "k1",
                "shape_type": "point",
                "group_id": 0,
                "points": [[20.0, 30.0]],
            },
            # instance 1
            {
                "label": "animal",
                "shape_type": "polygon",
                "group_id": 1,
                "points": [[36.0, 12.0], [56.0, 12.0], [56.0, 44.0], [36.0, 44.0]],
            },
            {
                "label": "k0",
                "shape_type": "point",
                "group_id": 1,
                "points": [[40.0, 22.0]],
            },
            {
                "label": "k1",
                "shape_type": "point",
                "group_id": 1,
                "points": [[52.0, 34.0]],
            },
        ],
    }
    json_path = root / "frame.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    return img_path, json_path


def test_labelme_pose_spec_discovers_json_dir_and_keypoints(tmp_path: Path) -> None:
    img_path, json_path = _write_minimal_labelme_pose_example(tmp_path / "labelme")
    spec_yaml = tmp_path / "labelme_spec.yaml"
    spec_yaml.write_text(
        "\n".join(
            [
                "format: labelme",
                f"path: {str(tmp_path)}",
                f"train: {str((tmp_path / 'labelme').relative_to(tmp_path))}",
                f"val: {str((tmp_path / 'labelme').relative_to(tmp_path))}",
                "kpt_shape: [2, 3]",
                "keypoint_names: [k0, k1]",
            ]
        ),
        encoding="utf-8",
    )

    spec = load_labelme_pose_spec(spec_yaml)
    assert spec.kpt_count == 2
    assert spec.kpt_dims == 3
    assert spec.keypoint_names == ["k0", "k1"]
    assert img_path in spec.train_images
    assert json_path in [p for p in spec.train_json if p is not None]

    summary = summarize_labelme_pose_labels(
        spec.train_images,
        label_paths=spec.train_json,
        keypoint_names=spec.keypoint_names,
        kpt_dims=spec.kpt_dims,
    )
    assert summary.images_with_pose_instances == 1
    assert summary.pose_instances_total == 2


def test_dino_kpseg_dataset_labelme_auto_instance_mode(tmp_path: Path) -> None:
    img_path, json_path = _write_minimal_labelme_pose_example(tmp_path / "labelme")
    extractor = _StubExtractor(patch_size=16)

    ds = DinoKPSEGPoseDataset(
        [img_path],
        kpt_count=2,
        kpt_dims=3,
        radius_px=6.0,
        extractor=extractor,  # type: ignore[arg-type]
        label_format="labelme",
        label_paths=[json_path],
        keypoint_names=["k0", "k1"],
        instance_mode="auto",
        cache_dir=None,
    )
    assert ds.instance_mode == "per_instance"
    assert len(ds) == 2

    sample0 = ds[0]
    assert sample0["masks"].shape[0] == 2
    assert sample0["coords"].shape[-1] == 2
