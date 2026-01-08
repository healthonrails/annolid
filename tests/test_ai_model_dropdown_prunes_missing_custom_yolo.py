from annolid.gui.model_manager import AIModelManager
from qtpy import QtCore, QtWidgets
import json
import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")


_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


def test_custom_yolo_weights_removed_from_dropdown_when_deleted(tmp_path: Path):
    _ensure_qapp()

    weights_ok = tmp_path / "weights_ok.pt"
    weights_ok.write_bytes(b"")
    weights_missing = tmp_path / "weights_missing.pt"

    settings_path = tmp_path / "settings.ini"
    settings = QtCore.QSettings(str(settings_path), QtCore.QSettings.IniFormat)

    payload = [
        {
            "display": "Custom YOLO OK",
            "identifier": str(weights_ok),
            "weight": str(weights_ok),
        },
        {
            "display": "Custom YOLO Missing",
            "identifier": str(weights_missing),
            "weight": str(weights_missing),
        },
    ]
    settings.setValue("ai/custom_yolo_models", json.dumps(payload))

    parent = QtWidgets.QWidget()
    combo = QtWidgets.QComboBox(parent)

    manager = AIModelManager(
        parent=parent,
        combo=combo,
        settings=settings,
        base_config={"ai": {"default": "Custom YOLO OK"}},
        canvas_getter=lambda: None,
    )
    manager.initialize(default_selection="Custom YOLO OK")

    assert combo.findText("Custom YOLO OK") != -1
    assert combo.findText("Custom YOLO Missing") == -1
    assert "Custom YOLO Missing" not in manager.custom_model_names

    weights_ok.unlink()
    manager.refresh()

    assert combo.findText("Custom YOLO OK") == -1
    assert "Custom YOLO OK" not in manager.custom_model_names

    weights_ok.write_bytes(b"")
    manager.refresh()

    # Deleted weights are pruned from settings; restoring the file does not
    # automatically re-add the entry.
    assert combo.findText("Custom YOLO OK") == -1
