from __future__ import annotations

import os
from types import SimpleNamespace

from qtpy import QtCore, QtWidgets

from annolid.gui.controllers.menu import MenuController


os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


class _DummyWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = QtCore.QSettings(
            str(QtCore.QDir.tempPath() + "/annolid_menu_test.ini"),
            QtCore.QSettings.IniFormat,
        )
        self.actions = SimpleNamespace()
        self.menus = SimpleNamespace()


def test_settings_profile_action_specs_cover_all_workflows() -> None:
    _ensure_qapp()
    window = _DummyWindow()
    controller = MenuController(window)

    specs = controller._settings_profile_action_specs()

    assert set(specs.keys()) == {
        "advanced_parameters",
        "optical_flow",
        "video_depth_anything",
        "sam3d",
        "patch_similarity",
        "pca_map",
    }
    for workflow, pair in specs.items():
        assert pair["apply"]["name"].startswith("apply_")
        assert pair["save"]["name"].startswith("save_")
        assert callable(pair["apply"]["slot"])
        assert callable(pair["save"]["slot"])
        assert workflow in {
            "advanced_parameters",
            "optical_flow",
            "video_depth_anything",
            "sam3d",
            "patch_similarity",
            "pca_map",
        }


def test_settings_profile_actions_trigger_controller_methods(monkeypatch) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    controller = MenuController(window)
    specs = controller._settings_profile_action_specs()
    calls: list[tuple[str, str, str]] = []

    monkeypatch.setattr(
        controller,
        "_apply_settings_profile",
        lambda workflow, panel: calls.append(("apply", workflow, panel)),
    )
    monkeypatch.setattr(
        controller,
        "_save_settings_profile",
        lambda workflow, panel: calls.append(("save", workflow, panel)),
    )

    for workflow, pair in specs.items():
        apply_action = controller._create_action_from_spec(pair["apply"])
        save_action = controller._create_action_from_spec(pair["save"])
        apply_action.trigger()
        save_action.trigger()

    assert calls == [
        ("apply", "advanced_parameters", "Advanced Parameters"),
        ("save", "advanced_parameters", "Advanced Parameters"),
        ("apply", "optical_flow", "Optical Flow"),
        ("save", "optical_flow", "Optical Flow"),
        ("apply", "video_depth_anything", "Depth"),
        ("save", "video_depth_anything", "Depth"),
        ("apply", "sam3d", "SAM 3D"),
        ("save", "sam3d", "SAM 3D"),
        ("apply", "patch_similarity", "Patch Similarity"),
        ("save", "patch_similarity", "Patch Similarity"),
        ("apply", "pca_map", "PCA Feature Map"),
        ("save", "pca_map", "PCA Feature Map"),
    ]
