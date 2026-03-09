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


def test_reorder_top_menus_keeps_qmenu_instances_alive() -> None:
    _ensure_qapp()
    window = _DummyWindow()
    controller = MenuController(window)

    window.menus.file = window.menuBar().addMenu(window.tr("&File"))
    window.menus.edit = window.menuBar().addMenu(window.tr("&Edit"))
    window.menus.view = window.menuBar().addMenu(window.tr("&View"))
    window.menus.help = window.menuBar().addMenu(window.tr("&Help"))
    window.menus.analysis = QtWidgets.QMenu(window.tr("&Analysis"), window)
    window.menus.ai_models = QtWidgets.QMenu(window.tr("&AI && Models"), window)
    window.menus.video_tools = QtWidgets.QMenu(window.tr("&Video Tools"), window)
    window.menus.settings = QtWidgets.QMenu(window.tr("&Settings"), window)
    window.menus.convert = QtWidgets.QMenu(window.tr("&Convert"), window)

    bar = window.menuBar()
    bar.addMenu(window.menus.help)
    bar.addMenu(window.menus.analysis)
    bar.addMenu(window.menus.file)
    bar.addMenu(window.menus.settings)
    bar.addMenu(window.menus.edit)
    bar.addMenu(window.menus.view)
    bar.addMenu(window.menus.ai_models)
    bar.addMenu(window.menus.convert)
    bar.addMenu(window.menus.video_tools)

    controller._reorder_top_menus()

    assert [action.text() for action in bar.actions()] == [
        "&File",
        "&Edit",
        "&View",
        "&Video Tools",
        "&AI && Models",
        "&Analysis",
        "&Convert",
        "&Settings",
        "&Help",
    ]
    assert window.menus.help.title() == "&Help"
    assert [action.menu() for action in bar.actions()] == [
        window.menus.file,
        window.menus.edit,
        window.menus.view,
        window.menus.video_tools,
        window.menus.ai_models,
        window.menus.analysis,
        window.menus.convert,
        window.menus.settings,
        window.menus.help,
    ]


def test_reorder_top_menus_recreates_stale_menu_reference(monkeypatch) -> None:
    _ensure_qapp()
    window = _DummyWindow()
    controller = MenuController(window)

    window.menus.file = window.menuBar().addMenu(window.tr("&File"))
    window.menus.edit = window.menuBar().addMenu(window.tr("&Edit"))
    window.menus.view = window.menuBar().addMenu(window.tr("&View"))
    window.menus.help = window.menuBar().addMenu(window.tr("&Help"))
    window.menus.analysis = QtWidgets.QMenu(window.tr("&Analysis"), window)
    window.menus.ai_models = QtWidgets.QMenu(window.tr("&AI && Models"), window)
    window.menus.video_tools = QtWidgets.QMenu(window.tr("&Video Tools"), window)
    window.menus.settings = QtWidgets.QMenu(window.tr("&Settings"), window)
    window.menus.convert = QtWidgets.QMenu(window.tr("&Convert"), window)

    stale_analysis = window.menus.analysis
    original_is_menu_alive = controller._is_menu_alive

    def _is_menu_alive(menu) -> bool:
        if menu is stale_analysis:
            return False
        return original_is_menu_alive(menu)

    monkeypatch.setattr(controller, "_is_menu_alive", _is_menu_alive)

    controller._reorder_top_menus()

    assert window.menus.analysis is not stale_analysis
    assert [action.text() for action in window.menuBar().actions()] == [
        "&File",
        "&Edit",
        "&View",
        "&Video Tools",
        "&AI && Models",
        "&Analysis",
        "&Convert",
        "&Settings",
        "&Help",
    ]
