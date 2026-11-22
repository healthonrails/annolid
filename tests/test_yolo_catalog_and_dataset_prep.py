from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import yaml
from PIL import Image

from annolid.yolo.catalog import (
    compute_yolo_weight_name,
    discover_yolo_checkpoints,
    get_yolo_train_preset,
    infer_yolo_task,
    is_prompt_free_yoloe,
    is_yoloe_model,
    iter_yolo_train_presets,
    parse_yolo_weight_spec,
    recommend_preset_for_dataset,
)
from annolid.yolo.dataset_prep import (
    YOLODatasetError,
    check_training_preflight,
    inspect_yolo_dataset,
    prepare_yolo_dataset,
)


def test_weight_naming_parsing_and_task_inference(tmp_path: Path) -> None:
    assert compute_yolo_weight_name("seg", "s", family="yolo11") == "yolo11s-seg.pt"
    assert (
        compute_yolo_weight_name("Pose Estimation", "m", family="yolo26")
        == "yolo26m-pose.pt"
    )
    assert compute_yolo_weight_name("Detection", "n", family="yolo11") == "yolo11n.pt"

    assert parse_yolo_weight_spec("yolo26s-seg.pt") == (
        "yolo26",
        "Instance Segmentation",
        "S",
    )
    assert parse_yolo_weight_spec("yolo11l-pose.pt") == (
        "yolo11",
        "Pose Estimation",
        "L",
    )
    assert parse_yolo_weight_spec("yolo11n.pt") == ("yolo11", "Detection", "N")
    assert parse_yolo_weight_spec("custom_best.pt") is None

    assert infer_yolo_task("yolo26s-pose.pt") == "pose"
    assert infer_yolo_task("yolo11n-seg.pt") == "seg"
    assert infer_yolo_task("yolo11s.pt") == "detect"
    assert infer_yolo_task("custom_best.pt") is None

    # Sibling args.yaml task inference for custom trained checkpoint
    run_dir = tmp_path / "run_1"
    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True)
    ckpt = weights_dir / "best.pt"
    ckpt.write_bytes(b"pt")
    (run_dir / "args.yaml").write_text("task: segment\n", encoding="utf-8")
    assert infer_yolo_task(str(ckpt)) == "seg"

    assert is_yoloe_model("yoloe-26s-seg.pt") is True
    assert is_yoloe_model("yolo11n-seg.pt") is False
    assert is_prompt_free_yoloe("yoloe-26s-seg-pf.pt") is True
    assert is_prompt_free_yoloe("yoloe-26s-seg.pt") is False


def test_presets_and_dataset_size_recommendation() -> None:
    keys = [p.key for p in iter_yolo_train_presets()]
    assert {"balanced", "small_dataset", "fast", "high_accuracy"}.issubset(set(keys))

    small = get_yolo_train_preset("small_dataset")
    assert small.freeze == 10
    assert small.optimizer == "AdamW"
    assert small.cos_lr is True

    assert recommend_preset_for_dataset(40).key == "small_dataset"
    assert recommend_preset_for_dataset(150).key == "small_dataset"
    assert recommend_preset_for_dataset(500).key == "balanced"


def test_discover_yolo_checkpoints_scans_managed_and_legacy_layouts(
    tmp_path: Path,
) -> None:
    managed = (
        tmp_path / "yolo" / "yolo11n-seg" / "20260101_120000" / "weights" / "best.pt"
    )
    legacy = tmp_path / "train" / "exp1" / "weights" / "best.pt"
    ultralytics_seg = tmp_path / "segment" / "run2" / "weights" / "best.pt"
    for p in (managed, legacy, ultralytics_seg):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"pt")

    now = time.time()
    os.utime(legacy, (now - 30, now - 30))
    os.utime(ultralytics_seg, (now - 10, now - 10))
    os.utime(managed, (now, now))

    discovered = discover_yolo_checkpoints(runs_root=tmp_path)
    paths = [c.path for c in discovered]
    assert paths == [managed.resolve(), ultralytics_seg.resolve(), legacy.resolve()]
    assert "yolo11n-seg" in discovered[0].display_name


def test_inspect_and_prepare_yolo_dataset_detects_seg_vs_detect_and_cleans_up(
    tmp_path: Path,
) -> None:
    root = tmp_path / "seg_ds"
    train_imgs = root / "images" / "train"
    val_imgs = root / "images" / "val"
    train_lbls = root / "labels" / "train"
    val_lbls = root / "labels" / "val"
    for d in (train_imgs, val_imgs, train_lbls, val_lbls):
        d.mkdir(parents=True)

    Image.new("RGB", (32, 32)).save(train_imgs / "img1.jpg")
    Image.new("RGB", (32, 32)).save(val_imgs / "img2.jpg")
    # Polygon label (>= 7 tokens -> segmentation)
    (train_lbls / "img1.txt").write_text(
        "0 0.1 0.1 0.5 0.1 0.5 0.5 0.1 0.5\n", encoding="utf-8"
    )
    (val_lbls / "img2.txt").write_text(
        "0 0.2 0.2 0.6 0.2 0.6 0.6 0.2 0.6\n", encoding="utf-8"
    )

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": ".",
                "train": "images/train",
                "val": "images/val",
                "names": {0: "mouse"},
            }
        ),
        encoding="utf-8",
    )

    summary = inspect_yolo_dataset(root)
    assert summary.is_valid is True
    assert summary.format_kind == "yolo"
    assert summary.task == "seg"
    assert summary.train_images == 1
    assert summary.val_images == 1
    assert summary.class_names == ("mouse",)

    with prepare_yolo_dataset(root, expected_task="seg") as prepared:
        temp_yaml = prepared.data_yaml
        assert temp_yaml.is_file()
        assert prepared.data_cfg["nc"] == 1
        assert prepared.data_cfg["names"] == ["mouse"]
        warnings = check_training_preflight(
            temp_yaml,
            batch_size=8,
            device="cpu",
            plots=False,
            expected_task="seg",
        )
        assert any("Batch size" in w for w in warnings)
        assert any("CPU" in w for w in warnings)

    # Temp YAML is cleaned up on context exit, while user's original YAML remains intact
    assert not temp_yaml.exists()
    assert data_yaml.is_file()


def test_inspect_coco_directory_and_prepare_errors(tmp_path: Path) -> None:
    coco_root = tmp_path / "coco_ds"
    (coco_root / "images").mkdir(parents=True)
    (coco_root / "annotations").mkdir(parents=True)
    Image.new("RGB", (32, 24)).save(coco_root / "images" / "a.png")
    Image.new("RGB", (32, 24)).save(coco_root / "images" / "b.png")
    payload_train = {
        "images": [{"id": 1, "file_name": "a.png", "width": 32, "height": 24}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [2, 2, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "rat"}],
    }
    payload_val = {
        "images": [{"id": 2, "file_name": "b.png", "width": 32, "height": 24}],
        "annotations": [
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [2, 2, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "rat"}],
    }
    (coco_root / "annotations" / "instances_train.json").write_text(
        json.dumps(payload_train), encoding="utf-8"
    )
    (coco_root / "annotations" / "instances_val.json").write_text(
        json.dumps(payload_val), encoding="utf-8"
    )

    summary = inspect_yolo_dataset(coco_root)
    assert summary.is_valid is True
    assert summary.format_kind == "coco_dir"
    assert summary.task == "detect"
    assert summary.class_names == ("rat",)
    assert summary.train_images == 1
    assert summary.val_images == 1

    with pytest.raises(YOLODatasetError):
        prepare_yolo_dataset(tmp_path / "nonexistent.yaml")


def test_prepare_yolo_dataset_materializes_coco_polygon_segmentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    coco_root = tmp_path / "coco_seg"
    (coco_root / "images").mkdir(parents=True)
    (coco_root / "annotations").mkdir(parents=True)
    Image.new("RGB", (32, 24)).save(coco_root / "images" / "a.png")
    Image.new("RGB", (32, 24)).save(coco_root / "images" / "b.png")
    train_payload = {
        "images": [{"id": 1, "file_name": "a.png", "width": 32, "height": 24}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [4, 4, 12, 10],
                "area": 120,
                "iscrowd": 0,
                "segmentation": [[4, 4, 16, 4, 16, 14, 4, 14]],
            }
        ],
        "categories": [{"id": 1, "name": "mouse"}],
    }
    val_payload = {
        "images": [{"id": 2, "file_name": "b.png", "width": 32, "height": 24}],
        "annotations": [
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [4, 4, 12, 10],
                "area": 120,
                "iscrowd": 0,
                "segmentation": [[4, 4, 16, 4, 16, 14, 4, 14]],
            }
        ],
        "categories": [{"id": 1, "name": "mouse"}],
    }
    (coco_root / "annotations" / "instances_train.json").write_text(
        json.dumps(train_payload), encoding="utf-8"
    )
    (coco_root / "annotations" / "instances_val.json").write_text(
        json.dumps(val_payload), encoding="utf-8"
    )

    summary = inspect_yolo_dataset(coco_root)
    assert summary.is_valid is True
    assert summary.task == "seg"

    monkeypatch.setattr(
        "annolid.yolo.runtime.resolve_yolo_device", lambda _d=None: "mps"
    )
    with prepare_yolo_dataset(coco_root, expected_task="seg") as prepared:
        assert prepared.data_yaml.is_file()
        staged_labels = list(
            (Path(prepared.data_cfg["path"]) / "labels" / "train").glob("*.txt")
        )
        assert len(staged_labels) == 1
        tokens = staged_labels[0].read_text(encoding="utf-8").strip().split()
        assert len(tokens) == 9  # class_id + 4 (x, y) pairs
        warnings = check_training_preflight(
            prepared.data_yaml,
            batch_size=1,
            device="",
            plots=False,
            expected_task="seg",
        )
        assert not any("Auto" in w or "CPU" in w for w in warnings)
