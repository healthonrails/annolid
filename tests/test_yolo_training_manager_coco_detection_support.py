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


def _patch_no_dialogs(monkeypatch) -> None:
    def _fail(*args, **kwargs):
        raise AssertionError("Unexpected QMessageBox during test.")

    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _fail)
    monkeypatch.setattr(QtWidgets.QMessageBox, "critical", _fail)


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


def test_prepare_data_config_accepts_coco_detection_spec(
    monkeypatch, tmp_path: Path
) -> None:
    _ensure_qapp()
    _patch_no_dialogs(monkeypatch)

    root = tmp_path / "dataset"
    images_dir = root / "images"
    annotations_dir = root / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    Image.new("RGB", (32, 24), color=(120, 120, 120)).save(images_dir / "train.png")
    Image.new("RGB", (32, 24), color=(100, 100, 100)).save(images_dir / "val.png")
    _write_detection_coco_json(
        annotations_dir / "instances_train.json", file_name="train.png", image_id=1
    )
    _write_detection_coco_json(
        annotations_dir / "instances_val.json", file_name="val.png", image_id=2
    )

    spec_path = root / "coco_detect.yaml"
    spec_path.write_text(
        "\n".join(
            [
                "format: coco",
                f"path: {root}",
                "image_root: images",
                "train: annotations/instances_train.json",
                "val: annotations/instances_val.json",
            ]
        ),
        encoding="utf-8",
    )

    window = QtWidgets.QMainWindow()
    manager = YOLOTrainingManager(window)
    try:
        prepared = manager.prepare_data_config(str(spec_path))
        assert prepared is not None
        payload = yaml.safe_load(Path(prepared).read_text(encoding="utf-8"))
        assert payload["names"] == ["mouse"]
        assert payload["nc"] == 1
        assert "kpt_shape" not in payload
        assert Path(str(payload["train"])).exists()
        assert Path(str(payload["val"])).exists()

        train_labels = Path(str(payload["path"])) / "labels" / "train"
        assert any(train_labels.glob("*.txt"))
    finally:
        manager.cleanup()
        window.close()


def test_prepare_data_config_accepts_instances_annotations_dir(
    monkeypatch, tmp_path: Path
) -> None:
    _ensure_qapp()
    _patch_no_dialogs(monkeypatch)

    root = tmp_path / "dataset"
    images_dir = root / "images"
    annotations_dir = root / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    Image.new("RGB", (32, 24), color=(120, 120, 120)).save(images_dir / "train.png")
    Image.new("RGB", (32, 24), color=(100, 100, 100)).save(images_dir / "val.png")
    _write_detection_coco_json(
        annotations_dir / "instances_train.json", file_name="train.png", image_id=1
    )
    _write_detection_coco_json(
        annotations_dir / "instances_val.json", file_name="val.png", image_id=2
    )

    window = QtWidgets.QMainWindow()
    manager = YOLOTrainingManager(window)
    try:
        prepared = manager.prepare_data_config(str(annotations_dir))
        assert prepared is not None
        payload = yaml.safe_load(Path(prepared).read_text(encoding="utf-8"))
        assert payload["names"] == ["mouse"]
        assert payload["nc"] == 1
        assert Path(str(payload["train"])).exists()
        assert Path(str(payload["val"])).exists()
    finally:
        manager.cleanup()
        window.close()
