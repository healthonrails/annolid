from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import yaml

from annolid.segmentation.dino_kpseg.data import (
    load_coco_pose_spec,
    load_yolo_pose_spec,
    materialize_coco_pose_as_yolo,
    summarize_yolo_pose_labels,
)


def _write_minimal_coco_dataset(root: Path) -> tuple[Path, Path]:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (100, 80), color=(0, 0, 0)).save(root / "images" / "a.png")
    Image.new("RGB", (120, 90), color=(0, 0, 0)).save(root / "images" / "b.png")

    categories = [
        {
            "id": 1,
            "name": "mouse",
            "supercategory": "animal",
            "keypoints": ["nose", "left_ear", "right_ear"],
        }
    ]

    train_payload = {
        "images": [
            {"id": 1, "file_name": "images/a.png", "width": 100, "height": 80},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 20.0, 40.0, 20.0],
                # include -1 visibility to verify remapping to 0
                "keypoints": [
                    20.0,
                    30.0,
                    2.0,
                    12.0,
                    22.0,
                    -1.0,
                    36.0,
                    24.0,
                    1.0,
                ],
            }
        ],
        "categories": categories,
    }
    val_payload = {
        "images": [
            {"id": 2, "file_name": "images/b.png", "width": 120, "height": 90},
        ],
        "annotations": [
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [12.0, 18.0, 48.0, 24.0],
                "keypoints": [
                    22.0,
                    30.0,
                    2.0,
                    18.0,
                    26.0,
                    1.0,
                    40.0,
                    28.0,
                    2.0,
                ],
            }
        ],
        "categories": categories,
    }

    train_json = root / "annotations" / "train.json"
    val_json = root / "annotations" / "val.json"
    train_json.write_text(json.dumps(train_payload), encoding="utf-8")
    val_json.write_text(json.dumps(val_payload), encoding="utf-8")
    return train_json, val_json


def _write_train_only_coco_dataset(root: Path) -> Path:
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (100, 80), color=(0, 0, 0)).save(root / "images" / "a.png")
    Image.new("RGB", (120, 90), color=(0, 0, 0)).save(root / "images" / "b.png")

    categories = [
        {
            "id": 1,
            "name": "mouse",
            "supercategory": "animal",
            "keypoints": ["nose", "left_ear", "right_ear"],
        }
    ]
    train_payload = {
        "images": [
            {"id": 1, "file_name": "images/a.png", "width": 100, "height": 80},
            {"id": 2, "file_name": "images/b.png", "width": 120, "height": 90},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10.0, 20.0, 40.0, 20.0],
                "keypoints": [20.0, 30.0, 2.0, 12.0, 22.0, 1.0, 36.0, 24.0, 1.0],
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [12.0, 18.0, 48.0, 24.0],
                "keypoints": [22.0, 30.0, 2.0, 18.0, 26.0, 1.0, 40.0, 28.0, 2.0],
            },
        ],
        "categories": categories,
    }
    train_json = root / "annotations" / "train_only.json"
    train_json.write_text(json.dumps(train_payload), encoding="utf-8")
    return train_json


def test_load_and_materialize_coco_pose_spec(tmp_path: Path) -> None:
    _write_minimal_coco_dataset(tmp_path)

    spec_yaml = tmp_path / "coco_spec.yaml"
    spec_yaml.write_text(
        "\n".join(
            [
                "format: coco",
                f"path: {tmp_path.as_posix()}",
                "train: annotations/train.json",
                "val: annotations/val.json",
                "",
            ]
        ),
        encoding="utf-8",
    )

    spec = load_coco_pose_spec(spec_yaml)
    assert spec.kpt_count == 3
    assert spec.kpt_dims == 3
    assert spec.keypoint_names == ["nose", "left_ear", "right_ear"]

    out = tmp_path / "yolo_from_coco"
    yolo_yaml = materialize_coco_pose_as_yolo(spec=spec, output_dir=out)
    yolo_spec = load_yolo_pose_spec(yolo_yaml)

    assert len(yolo_spec.train_images) == 1
    assert len(yolo_spec.val_images) == 1
    assert yolo_spec.kpt_count == 3
    assert yolo_spec.kpt_dims == 3
    assert yolo_spec.keypoint_names == ["nose", "left_ear", "right_ear"]

    train_label = out / "labels" / "train" / f"{yolo_spec.train_images[0].stem}.txt"
    tokens = train_label.read_text(encoding="utf-8").strip().split()
    assert len(tokens) == 5 + (3 * 3)
    # visibility for the second keypoint was -1 in COCO and should be clamped to 0
    assert tokens[5 + 5] == "0"

    summary = summarize_yolo_pose_labels(
        yolo_spec.train_images,
        kpt_count=yolo_spec.kpt_count,
        kpt_dims=yolo_spec.kpt_dims,
    )
    assert summary.images_with_pose_instances == 1


def test_materialize_coco_pose_auto_builds_val_split_when_missing(
    tmp_path: Path,
) -> None:
    train_json = _write_train_only_coco_dataset(tmp_path)

    spec_yaml = tmp_path / "coco_spec_auto_val.yaml"
    spec_yaml.write_text(
        "\n".join(
            [
                "format: coco",
                f"path: {tmp_path.as_posix()}",
                f"train: {train_json.relative_to(tmp_path).as_posix()}",
                "val_split: 0.5",
                "val_seed: 123",
                "",
            ]
        ),
        encoding="utf-8",
    )

    spec = load_coco_pose_spec(spec_yaml)
    out = tmp_path / "yolo_from_coco_auto_val"
    yolo_yaml = materialize_coco_pose_as_yolo(spec=spec, output_dir=out)
    yolo_spec = load_yolo_pose_spec(yolo_yaml)

    assert len(yolo_spec.train_images) == 1
    assert len(yolo_spec.val_images) == 1

    payload = yaml.safe_load(yolo_yaml.read_text(encoding="utf-8")) or {}
    assert payload.get("auto_val_split") is True
    assert float(payload.get("val_split")) == 0.5
    assert int(payload.get("val_seed")) == 123
    assert int(payload.get("images_train_count")) == 1
    assert int(payload.get("images_val_count")) == 1


def test_materialize_coco_pose_auto_val_split_can_be_disabled(tmp_path: Path) -> None:
    train_json = _write_train_only_coco_dataset(tmp_path)

    spec_yaml = tmp_path / "coco_spec_no_auto_val.yaml"
    spec_yaml.write_text(
        "\n".join(
            [
                "format: coco",
                f"path: {tmp_path.as_posix()}",
                f"train: {train_json.relative_to(tmp_path).as_posix()}",
                "auto_val_split: false",
                "val_split: 0.5",
                "",
            ]
        ),
        encoding="utf-8",
    )

    spec = load_coco_pose_spec(spec_yaml)
    out = tmp_path / "yolo_from_coco_no_auto_val"
    yolo_yaml = materialize_coco_pose_as_yolo(spec=spec, output_dir=out)
    yolo_spec = load_yolo_pose_spec(yolo_yaml)

    assert len(yolo_spec.train_images) == 2
    assert len(yolo_spec.val_images) == 0
