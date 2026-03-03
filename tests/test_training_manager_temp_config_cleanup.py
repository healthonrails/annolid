from __future__ import annotations

import os
from pathlib import Path

from qtpy import QtWidgets

from annolid.gui.dino_kpseg_training_manager import DinoKPSEGTrainingManager
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


def _assert_release_only_deletes_manager_temp_configs(
    manager: DinoKPSEGTrainingManager | YOLOTrainingManager,
    config_path: Path,
) -> None:
    assert config_path.exists()

    # Simulate a release call with the user-selected config path.
    manager._release_temp_config(str(config_path))
    assert config_path.exists()

    resolved_path = Path(
        manager._write_temp_config(  # type: ignore[attr-defined]
            {"path": str(config_path.parent)},
            config_path,
        )
    )
    assert resolved_path.exists()

    manager._release_temp_config(str(resolved_path))
    assert not resolved_path.exists()


def test_dino_manager_does_not_delete_user_data_yaml(tmp_path: Path) -> None:
    _ensure_qapp()

    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        "path: .\ntrain: images/train\nval: images/val\n", encoding="utf-8"
    )

    window = QtWidgets.QMainWindow()
    manager = DinoKPSEGTrainingManager(window)
    try:
        _assert_release_only_deletes_manager_temp_configs(manager, config_path)
    finally:
        manager.cleanup()
        window.close()


def test_yolo_manager_does_not_delete_user_data_yaml(tmp_path: Path) -> None:
    _ensure_qapp()

    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        "path: .\ntrain: images/train\nval: images/val\n", encoding="utf-8"
    )

    window = QtWidgets.QMainWindow()
    manager = YOLOTrainingManager(window)
    try:
        _assert_release_only_deletes_manager_temp_configs(manager, config_path)
    finally:
        manager.cleanup()
        window.close()
