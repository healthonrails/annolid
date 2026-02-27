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
