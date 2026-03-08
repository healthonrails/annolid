from __future__ import annotations

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
