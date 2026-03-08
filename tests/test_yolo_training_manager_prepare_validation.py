from __future__ import annotations

import json
import os
from pathlib import Path

import yaml
from PIL import Image
from qtpy import QtWidgets

from annolid.gui.yolo_training_manager import YOLOTrainingManager


os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


def test_prepare_data_config_infers_standard_split_dirs(
    tmp_path: Path, monkeypatch
) -> None:
    _ensure_qapp()

    def _fail(*args, **kwargs):
        raise AssertionError("Unexpected QMessageBox during test.")

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _fail)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", _fail)

    root = tmp_path / "dataset"
    train_dir = root / "images" / "train"
    val_dir = root / "images" / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(train_dir / "frame_0001.jpg")
    Image.new("RGB", (16, 16)).save(val_dir / "frame_0001.jpg")

    config_path = root / "data.yaml"
    config_path.write_text(
        "\n".join(
            [
                "path: .",
                "nc: 1",
                "names: [mouse]",
                "kpt_shape: [1, 3]",
            ]
        ),
        encoding="utf-8",
    )

    window = QtWidgets.QMainWindow()
    manager = YOLOTrainingManager(window)
    try:
        prepared = manager.prepare_data_config(str(config_path))
        assert prepared is not None
        payload = yaml.safe_load(Path(prepared).read_text(encoding="utf-8"))
        assert payload["train"] == str(train_dir.resolve())
        assert payload["val"] == str(val_dir.resolve())
    finally:
        manager.cleanup()
        window.close()


def test_start_training_skips_repreparing_manager_temp_config(
    monkeypatch, tmp_path: Path
) -> None:
    _ensure_qapp()

    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        "path: .\ntrain: images/train\nval: images/val\n", encoding="utf-8"
    )

    window = QtWidgets.QMainWindow()
    manager = YOLOTrainingManager(window)
    try:
        prepared = Path(
            manager._write_temp_config({"train": "a", "val": "b"}, config_path)
        )  # type: ignore[attr-defined]
        calls = []

        def _unexpected_prepare(path: str):
            calls.append(path)
            raise AssertionError(
                "start_training should not re-prepare manager temp config"
            )

        monkeypatch.setattr(manager, "prepare_data_config", _unexpected_prepare)
        monkeypatch.setattr(manager, "_confirm_preflight", lambda **kwargs: False)

        started = manager.start_training(
            yolo_model_file="yolo11n-pose.pt",
            model_path=None,
            data_config_path=str(prepared),
            epochs=1,
            image_size=640,
            batch_size=1,
            device=None,
            plots=False,
            train_overrides=None,
            out_dir=None,
        )

        assert started is False
        assert calls == []
    finally:
        manager.cleanup()
        window.close()


def test_prepare_data_config_infers_names_and_nc_for_pose_yaml(
    tmp_path: Path, monkeypatch
) -> None:
    _ensure_qapp()

    def _fail(*args, **kwargs):
        raise AssertionError("Unexpected QMessageBox during test.")

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _fail)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", _fail)

    root = tmp_path / "dataset"
    train_dir = root / "images" / "train"
    val_dir = root / "images" / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(train_dir / "frame_0001.jpg")
    Image.new("RGB", (16, 16)).save(val_dir / "frame_0001.jpg")

    config_path = root / "data.yaml"
    config_path.write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/val",
                "kpt_shape: [3, 3]",
                "kpt_names:",
                "  0: [nose, left_ear, right_ear]",
            ]
        ),
        encoding="utf-8",
    )

    window = QtWidgets.QMainWindow()
    manager = YOLOTrainingManager(window)
    try:
        prepared = manager.prepare_data_config(str(config_path))
        assert prepared is not None
        payload = yaml.safe_load(Path(prepared).read_text(encoding="utf-8"))
        assert payload["names"] == ["class_0"]
        assert payload["nc"] == 1
    finally:
        manager.cleanup()
        window.close()


def test_prepare_data_config_infers_names_nc_from_nearby_coco_annotations(
    tmp_path: Path, monkeypatch
) -> None:
    _ensure_qapp()

    def _fail(*args, **kwargs):
        raise AssertionError("Unexpected QMessageBox during test.")

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _fail)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", _fail)

    root = tmp_path / "dataset"
    train_dir = root / "images" / "train"
    val_dir = root / "images" / "val"
    ann_dir = root / "annotations"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(train_dir / "frame_0001.jpg")
    Image.new("RGB", (16, 16)).save(val_dir / "frame_0001.jpg")

    coco_payload = {
        "images": [{"id": 1, "file_name": "frame_0001.jpg", "width": 16, "height": 16}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 7,
                "bbox": [2, 2, 8, 8],
                "area": 64,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 7, "name": "mouse"}],
    }
    (ann_dir / "train.json").write_text(json.dumps(coco_payload), encoding="utf-8")
    (ann_dir / "val.json").write_text(json.dumps(coco_payload), encoding="utf-8")

    config_path = root / "data.yaml"
    config_path.write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/val",
            ]
        ),
        encoding="utf-8",
    )

    window = QtWidgets.QMainWindow()
    manager = YOLOTrainingManager(window)
    try:
        prepared = manager.prepare_data_config(str(config_path))
        assert prepared is not None
        payload = yaml.safe_load(Path(prepared).read_text(encoding="utf-8"))
        assert payload["names"] == ["mouse"]
        assert payload["nc"] == 1
    finally:
        manager.cleanup()
        window.close()


def test_prepare_data_config_upgrades_pose_yaml_from_nearby_coco_keypoints(
    tmp_path: Path, monkeypatch
) -> None:
    _ensure_qapp()

    def _fail(*args, **kwargs):
        raise AssertionError("Unexpected QMessageBox during test.")

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _fail)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", _fail)

    root = tmp_path / "dataset"
    train_dir = root / "images" / "train"
    val_dir = root / "images" / "val"
    ann_dir = root / "annotations"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    ann_dir.mkdir(parents=True)
    Image.new("RGB", (16, 16)).save(train_dir / "frame_0001.jpg")
    Image.new("RGB", (16, 16)).save(val_dir / "frame_0001.jpg")

    coco_payload = {
        "images": [
            {"id": 1, "file_name": "train/frame_0001.jpg", "width": 16, "height": 16}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [2, 2, 8, 8],
                "area": 64,
                "iscrowd": 0,
                "num_keypoints": 3,
                "keypoints": [4, 4, 2, 6, 6, 2, 8, 8, 2],
            }
        ],
        "categories": [
            {
                "id": 1,
                "name": "mouse",
                "keypoints": ["nose", "left_ear", "right_ear"],
                "skeleton": [[1, 2], [1, 3]],
            }
        ],
    }
    (ann_dir / "person_keypoints_train.json").write_text(
        json.dumps(coco_payload),
        encoding="utf-8",
    )
    val_payload = dict(coco_payload)
    val_payload["images"] = [
        {"id": 2, "file_name": "val/frame_0001.jpg", "width": 16, "height": 16}
    ]
    val_payload["annotations"] = [
        {
            "id": 2,
            "image_id": 2,
            "category_id": 1,
            "bbox": [3, 3, 7, 7],
            "area": 49,
            "iscrowd": 0,
            "num_keypoints": 3,
            "keypoints": [5, 5, 2, 7, 7, 2, 9, 9, 2],
        }
    ]
    (ann_dir / "person_keypoints_val.json").write_text(
        json.dumps(val_payload),
        encoding="utf-8",
    )

    config_path = root / "data.yaml"
    config_path.write_text(
        "\n".join(
            [
                "path: .",
                "train: images/train",
                "val: images/val",
            ]
        ),
        encoding="utf-8",
    )

    window = QtWidgets.QMainWindow()
    manager = YOLOTrainingManager(window)
    try:
        prepared = manager.prepare_data_config(str(config_path), expected_task="pose")
        assert prepared is not None
        payload = yaml.safe_load(Path(prepared).read_text(encoding="utf-8"))
        assert payload["kpt_shape"] == [3, 3]
        assert payload["names"] == ["mouse"]
        assert payload["nc"] == 1
        assert Path(payload["train"]).exists()
        assert Path(payload["val"]).exists()
    finally:
        manager.cleanup()
        window.close()
