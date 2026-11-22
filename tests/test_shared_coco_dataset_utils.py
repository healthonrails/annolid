from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from annolid.datasets.coco import (
    build_coco_spec_from_annotations_dir,
    build_coco_spec_from_dataset_path,
    discover_coco_annotations_dir,
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


def test_discover_coco_annotations_dir_from_dataset_root(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    annotations_dir = root / "annotations"
    annotations_dir.mkdir(parents=True)
    (annotations_dir / "train.json").write_text("{}", encoding="utf-8")
    assert discover_coco_annotations_dir(root) == annotations_dir.resolve()


def test_build_coco_spec_from_dataset_path_accepts_split_local_annotations_json(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    train_dir = root / "train"
    val_dir = root / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    Image.new("RGB", (32, 24)).save(train_dir / "train.png")
    Image.new("RGB", (32, 24)).save(val_dir / "val.png")
    _write_detection_coco_json(
        train_dir / "annotations.json",
        file_name="train.png",
        image_id=1,
    )
    _write_detection_coco_json(
        val_dir / "annotations.json",
        file_name="val.png",
        image_id=2,
    )

    payload = build_coco_spec_from_dataset_path(root)
    assert payload is not None
    assert payload["format"] == "coco"
    assert payload["image_root"] == "."
    assert payload["train"] == "train/annotations.json"
    assert payload["val"] == "val/annotations.json"


def test_materialize_coco_spec_as_yolo_accepts_split_local_annotations_json(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    train_dir = root / "train"
    val_dir = root / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    Image.new("RGB", (32, 24)).save(train_dir / "train.png")
    Image.new("RGB", (32, 24)).save(val_dir / "val.png")
    _write_detection_coco_json(
        train_dir / "annotations.json",
        file_name="train.png",
        image_id=1,
    )
    _write_detection_coco_json(
        val_dir / "annotations.json",
        file_name="val.png",
        image_id=2,
    )

    spec = build_coco_spec_from_dataset_path(root)
    assert spec is not None
    spec_path = root / "coco_detect.yaml"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    yolo_yaml = materialize_coco_spec_as_yolo(
        config_path=spec_path,
        output_dir=tmp_path / "out",
    )
    payload = read_yaml_dict(yolo_yaml)
    assert payload["names"] == ["mouse"]
    assert (yolo_yaml.parent / "images" / "train").exists()
    assert (yolo_yaml.parent / "images" / "val").exists()


def test_build_coco_spec_prefers_person_keypoints_over_instances(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dataset"
    images_dir = root / "images"
    annotations_dir = root / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    # Presence is enough for selection priority in spec creation.
    (annotations_dir / "instances_train.json").write_text("{}", encoding="utf-8")
    (annotations_dir / "instances_val.json").write_text("{}", encoding="utf-8")
    (annotations_dir / "person_keypoints_train.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (annotations_dir / "person_keypoints_val.json").write_text(
        "{}",
        encoding="utf-8",
    )

    payload = build_coco_spec_from_annotations_dir(annotations_dir)
    assert payload is not None
    assert payload.get("train") == "annotations/person_keypoints_train.json"
    assert payload.get("val") == "annotations/person_keypoints_val.json"


def test_materialize_coco_segmentation_supports_polygons_multi_parts_and_rle(
    tmp_path: Path,
) -> None:
    import numpy as np

    from annolid.annotation.masks import binary_mask_to_coco_rle

    root = tmp_path / "coco_seg"
    images_dir = root / "images"
    ann_dir = root / "annotations"
    images_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)

    Image.new("RGB", (40, 30), color=(80, 90, 100)).save(images_dir / "poly.png")
    Image.new("RGB", (40, 30), color=(60, 70, 80)).save(images_dir / "rle.png")
    Image.new("RGB", (40, 30), color=(40, 50, 60)).save(images_dir / "val.png")

    # Binary mask for compressed RLE test
    mask = np.zeros((30, 40), dtype=np.uint8)
    mask[5:15, 8:20] = 1
    compressed_rle = binary_mask_to_coco_rle(mask)

    # Uncompressed RLE mask (10x10 square inside 30x40 in column-major order)
    uncompressed_mask = np.zeros((30, 40), dtype=np.uint8)
    uncompressed_mask[10:20, 22:32] = 1
    flat_f = uncompressed_mask.flatten(order="F")
    counts: list[int] = []
    curr = 0
    run_len = 0
    for px in flat_f:
        if int(px) == curr:
            run_len += 1
        else:
            counts.append(run_len)
            curr = int(px)
            run_len = 1
    counts.append(run_len)
    uncompressed_rle = {"size": [30, 40], "counts": counts}

    train_coco = {
        "images": [
            {"id": 1, "file_name": "poly.png", "width": 40, "height": 30},
            {"id": 2, "file_name": "rle.png", "width": 40, "height": 30},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [4, 4, 24, 16],
                "area": 100,
                "iscrowd": 0,
                # Multi-part polygon (2 disconnected squares)
                "segmentation": [
                    [4, 4, 12, 4, 12, 12, 4, 12],
                    [20, 12, 28, 12, 28, 20, 20, 20],
                ],
            },
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [8, 5, 12, 10],
                "area": 120,
                "iscrowd": 0,
                "segmentation": compressed_rle,
            },
            {
                "id": 3,
                "image_id": 2,
                "category_id": 1,
                "bbox": [22, 10, 10, 10],
                "area": 100,
                "iscrowd": 1,
                "segmentation": uncompressed_rle,
            },
        ],
        "categories": [{"id": 1, "name": "mouse"}],
    }
    val_coco = {
        "images": [{"id": 3, "file_name": "val.png", "width": 40, "height": 30}],
        "annotations": [
            {
                "id": 4,
                "image_id": 3,
                "category_id": 1,
                "bbox": [6, 6, 10, 10],
                "area": 100,
                "iscrowd": 0,
                "segmentation": [[6, 6, 16, 6, 16, 16, 6, 16]],
            }
        ],
        "categories": [{"id": 1, "name": "mouse"}],
    }
    (ann_dir / "instances_train.json").write_text(
        json.dumps(train_coco), encoding="utf-8"
    )
    (ann_dir / "instances_val.json").write_text(json.dumps(val_coco), encoding="utf-8")

    spec = build_coco_spec_from_dataset_path(root)
    assert spec is not None
    spec_path = root / "coco_seg.yaml"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")

    assert infer_coco_task(config_path=spec_path, payload=spec) == "seg"

    # Materialize as YOLO segmentation
    seg_yaml = materialize_coco_spec_as_yolo(
        config_path=spec_path,
        output_dir=tmp_path / "out_seg",
        expected_task="seg",
    )
    seg_train_labels = sorted((seg_yaml.parent / "labels" / "train").glob("*.txt"))
    assert len(seg_train_labels) == 2

    # Multi-part polygon merged into a single YOLO-seg line (> 8 vertices -> > 17 tokens)
    poly_lines = (
        (seg_yaml.parent / "labels" / "train" / "poly__1.txt")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    assert len(poly_lines) == 1
    poly_tokens = poly_lines[0].split()
    assert poly_tokens[0] == "0"
    assert len(poly_tokens) >= 1 + 16
    assert (len(poly_tokens) - 1) % 2 == 0
    assert all(0.0 <= float(v) <= 1.0 for v in poly_tokens[1:])

    # Both compressed and uncompressed RLE annotations materialize into valid polygons
    rle_lines = (
        (seg_yaml.parent / "labels" / "train" / "rle__2.txt")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    assert len(rle_lines) == 2
    for line in rle_lines:
        toks = line.split()
        assert toks[0] == "0"
        assert len(toks) >= 7
        assert (len(toks) - 1) % 2 == 0

    # Same COCO seg dataset can also materialize detection boxes when expected_task="detect"
    det_yaml = materialize_coco_spec_as_yolo(
        config_path=spec_path,
        output_dir=tmp_path / "out_det",
        expected_task="detect",
    )
    det_lines = (
        (det_yaml.parent / "labels" / "train" / "poly__1.txt")
        .read_text(encoding="utf-8")
        .strip()
        .splitlines()
    )
    assert len(det_lines) == 1
    assert len(det_lines[0].split()) == 5
