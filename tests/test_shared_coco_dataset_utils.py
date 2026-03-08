from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from annolid.datasets.coco import (
    build_coco_spec_from_annotations_dir,
    infer_coco_task,
    load_coco_category_id_map,
    load_coco_class_names,
    load_coco_keypoint_meta,
    materialize_coco_spec_as_yolo,
    read_yaml_dict,
)


def _write_detection_coco_json(path: Path, *, file_name: str, image_id: int) -> None:
    payload = {
        "images": [
            {
                "id": image_id,
                "file_name": file_name,
                "width": 32,
                "height": 24,
            }
        ],
        "annotations": [
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 3,
                "bbox": [4, 6, 10, 8],
                "area": 80,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 3, "name": "mouse"}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_coco_spec_from_annotations_dir_accepts_instances_files(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    images_dir = root / "images"
    annotations_dir = root / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    Image.new("RGB", (32, 24)).save(images_dir / "train.png")
    _write_detection_coco_json(
        annotations_dir / "instances_train.json",
        file_name="train.png",
        image_id=1,
    )

    payload = build_coco_spec_from_annotations_dir(annotations_dir)
    assert payload is not None
    assert payload["format"] == "coco"
    assert payload["image_root"] == "images"
    assert payload["train"] == "annotations/instances_train.json"


def test_materialize_coco_spec_as_yolo_rejects_detection_when_pose_expected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    images_dir = root / "images"
    annotations_dir = root / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    Image.new("RGB", (32, 24)).save(images_dir / "train.png")
    _write_detection_coco_json(
        annotations_dir / "instances_train.json",
        file_name="train.png",
        image_id=1,
    )

    spec_path = root / "coco_detect.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "format: coco",
                f"path: {root}",
                "image_root: images",
                "train: annotations/instances_train.json",
            ]
        ),
        encoding="utf-8",
    )

    payload = read_yaml_dict(spec_path)
    assert infer_coco_task(config_path=spec_path, payload=payload) == "detect"

    try:
        materialize_coco_spec_as_yolo(
            config_path=spec_path,
            output_dir=tmp_path / "out",
            expected_task="pose",
        )
    except ValueError as exc:
        assert "Expected a COCO pose dataset" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-pose COCO spec.")


def test_shared_coco_metadata_helpers_parse_categories(tmp_path: Path) -> None:
    ann_path = tmp_path / "annotations.json"
    ann_path.write_text(
        json.dumps(
            {
                "images": [],
                "annotations": [],
                "categories": [
                    {
                        "id": 5,
                        "name": "mouse",
                        "keypoints": ["nose", "tail"],
                        "skeleton": [[1, 2]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert load_coco_class_names(ann_path) == ["mouse"]
    assert load_coco_category_id_map(ann_path) == {5: 0}
    assert load_coco_keypoint_meta(ann_path) == {
        "num_keypoints": 2,
        "keypoint_names": ["nose", "tail"],
        "skeleton": [[1, 2]],
    }
