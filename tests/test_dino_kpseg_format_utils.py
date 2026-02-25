from __future__ import annotations

from annolid.segmentation.dino_kpseg.format_utils import (
    normalize_dino_kpseg_data_format,
)


def test_normalize_explicit_coco_format() -> None:
    payload = {"train": "annotations/train.json"}
    fmt = normalize_dino_kpseg_data_format(payload, data_format="coco")
    assert fmt == "coco"


def test_auto_detect_coco_from_format_token() -> None:
    payload = {"format": "coco_pose", "train": "annotations/train.json"}
    fmt = normalize_dino_kpseg_data_format(payload, data_format="auto")
    assert fmt == "coco"


def test_auto_detect_coco_from_annotation_path_heuristic() -> None:
    payload = {
        "path": "/dataset",
        "train": "annotations/train.json",
        "val": "annotations/val.json",
    }
    fmt = normalize_dino_kpseg_data_format(payload, data_format="auto")
    assert fmt == "coco"


def test_auto_detect_labelme_from_jsonl_heuristic() -> None:
    payload = {"train": "annolid_logs/train.jsonl"}
    fmt = normalize_dino_kpseg_data_format(payload, data_format="auto")
    assert fmt == "labelme"


def test_auto_falls_back_to_yolo() -> None:
    payload = {"train": "images/train", "val": "images/val"}
    fmt = normalize_dino_kpseg_data_format(payload, data_format="auto")
    assert fmt == "yolo"
