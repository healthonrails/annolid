from __future__ import annotations

import os
from pathlib import Path

import yaml
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


def test_prepare_data_config_accepts_coco_pose_spec(monkeypatch) -> None:
    _ensure_qapp()
    _patch_no_dialogs(monkeypatch)

    spec_path = Path("tests/fixtures/dino_kpseg_coco_tiny/coco_spec.yaml").resolve()
    window = QtWidgets.QMainWindow()
    manager = YOLOTrainingManager(window)
    try:
        prepared = manager.prepare_data_config(str(spec_path))
        assert prepared is not None
        prepared_path = Path(prepared)
        assert prepared_path.exists()

        payload = yaml.safe_load(prepared_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        assert payload.get("nc") == 1
        assert payload.get("names") == ["mouse"]
        assert Path(str(payload["train"])).exists()
        assert Path(str(payload["val"])).exists()
        assert payload.get("kpt_shape") == [3, 3]

        staged_dirs = list(manager._temp_dataset_dirs)
        assert staged_dirs
    finally:
        manager.cleanup()
        window.close()

    for d in staged_dirs:
        assert not d.exists()
    assert not prepared_path.exists()


def test_prepare_data_config_accepts_coco_annotations_folder(monkeypatch) -> None:
    _ensure_qapp()
    _patch_no_dialogs(monkeypatch)

    annotations_dir = Path("tests/fixtures/dino_kpseg_coco_tiny/annotations").resolve()
    window = QtWidgets.QMainWindow()
    manager = YOLOTrainingManager(window)
    try:
        prepared = manager.prepare_data_config(str(annotations_dir))
        assert prepared is not None
        prepared_path = Path(prepared)
        assert prepared_path.exists()

        payload = yaml.safe_load(prepared_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        assert payload.get("nc") == 1
        assert payload.get("names") == ["mouse"]
        assert Path(str(payload["train"])).exists()
        assert Path(str(payload["val"])).exists()
        assert payload.get("kpt_shape") == [3, 3]
    finally:
        manager.cleanup()
        window.close()


def test_prepare_data_config_accepts_coco_dataset_root(
    monkeypatch, tmp_path: Path
) -> None:
    _ensure_qapp()
    _patch_no_dialogs(monkeypatch)

    root = tmp_path / "dataset"
    images_dir = root / "images"
    annotations_dir = root / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    fixture_root = Path("tests/fixtures/dino_kpseg_coco_tiny").resolve()
    for src_name, dst_name in (
        ("mouse_train_001.png", "mouse_train_001.png"),
        ("mouse_train_002.png", "mouse_train_002.png"),
        ("mouse_val_001.png", "mouse_val_001.png"),
    ):
        src = fixture_root / "images" / src_name
        dst = images_dir / dst_name
        dst.write_bytes(src.read_bytes())

    for split in ("train", "val"):
        src = fixture_root / "annotations" / f"{split}.json"
        dst = annotations_dir / f"{split}.json"
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    window = QtWidgets.QMainWindow()
    manager = YOLOTrainingManager(window)
    try:
        prepared = manager.prepare_data_config(str(root))
        assert prepared is not None
        prepared_path = Path(prepared)
        payload = yaml.safe_load(prepared_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        assert payload.get("nc") == 1
        assert payload.get("names") == ["mouse"]
        assert Path(str(payload["train"])).exists()
        assert Path(str(payload["val"])).exists()
    finally:
        manager.cleanup()
        window.close()
