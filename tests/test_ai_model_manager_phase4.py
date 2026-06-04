from __future__ import annotations

from qtpy import QtCore, QtWidgets

from annolid.gui.model_manager import AIModelManager, RECOMMENDED_AI_MODEL_NAMES


def test_ai_model_manager_disables_unavailable_runtime_models(tmp_path) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app

    settings_path = tmp_path / "settings.ini"
    settings = QtCore.QSettings(str(settings_path), QtCore.QSettings.IniFormat)

    parent = QtWidgets.QWidget()
    combo = QtWidgets.QComboBox(parent)

    manager = AIModelManager(
        parent=parent,
        combo=combo,
        settings=settings,
        base_config={
            "ai": {
                "model_path_defaults": {
                    "dino_kpseg": str(tmp_path / "missing_kpseg.pt"),
                    "dino_kpseg_tracker": str(tmp_path / "missing_kpseg_tracker.pt"),
                },
                "default": "DINOv3 Keypoint Segmentation",
            }
        },
        canvas_getter=lambda: None,
    )
    manager.initialize()

    idx = combo.findText("DINOv3 Keypoint Segmentation")
    assert idx >= 0
    model_obj = combo.model()
    item = model_obj.item(idx) if hasattr(model_obj, "item") else None
    assert item is not None
    assert item.isEnabled() is False
    tooltip = str(combo.itemData(idx, QtCore.Qt.ToolTipRole) or "")
    assert "Missing local weights:" in tooltip
    assert combo.currentText() != "DINOv3 Keypoint Segmentation"


def test_ai_model_manager_keeps_toolbar_model_list_compact(tmp_path) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app

    settings_path = tmp_path / "settings.ini"
    settings = QtCore.QSettings(str(settings_path), QtCore.QSettings.IniFormat)

    parent = QtWidgets.QWidget()
    combo = QtWidgets.QComboBox(parent)
    manager = AIModelManager(
        parent=parent,
        combo=combo,
        settings=settings,
        base_config={"ai": {"default": "Cutie"}},
        canvas_getter=lambda: None,
    )
    manager.initialize()

    visible = [combo.itemText(index) for index in range(combo.count())]

    assert "More models..." in visible
    assert "Browse Custom Model…" in visible
    assert "YOLO11x" not in visible
    assert len(visible) <= len(RECOMMENDED_AI_MODEL_NAMES) + 2

    catalog_names = {name for name, _group, _cfg in manager._catalog_entries()}
    assert "YOLO11x" in catalog_names


def test_ai_model_manager_preserves_non_recommended_selection(tmp_path) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app

    settings_path = tmp_path / "settings.ini"
    settings = QtCore.QSettings(str(settings_path), QtCore.QSettings.IniFormat)

    parent = QtWidgets.QWidget()
    combo = QtWidgets.QComboBox(parent)
    manager = AIModelManager(
        parent=parent,
        combo=combo,
        settings=settings,
        base_config={"ai": {"default": "Cutie"}},
        canvas_getter=lambda: None,
    )
    manager.initialize(default_selection="YOLO11x")

    assert combo.findText("YOLO11x") != -1
    assert combo.currentText() == "YOLO11x"
