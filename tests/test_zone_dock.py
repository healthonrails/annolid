from __future__ import annotations

from types import SimpleNamespace

from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.mixins.tooling_dialogs_mixin import ToolingDialogsMixin
from annolid.gui.widgets.label_dialog import AnnolidLabelDialog
import annolid.gui.widgets.zone_dock as zone_dock_module
from annolid.gui.widgets.zone_dock import ZoneDockWidget


def _ensure_qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class _DummyCanvas(QtCore.QObject):
    newShape = QtCore.Signal()
    selectionChanged = QtCore.Signal(list)
    shapeMoved = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self.pixmap = QtGui.QPixmap(64, 64)
        self.pixmap.fill(QtGui.QColor("white"))
        self.shapes = []
        self.editing = None
        self.loaded_shapes = None

    def setEditing(self, value: bool) -> None:
        self.editing = bool(value)

    def loadShapes(self, shapes, replace=True) -> None:  # noqa: ARG002
        self.loaded_shapes = list(shapes)
        self.shapes = list(shapes)

    def selectShapes(self, shapes) -> None:  # noqa: ARG002
        return None

    def storeShapes(self) -> None:
        return None

    def setFocus(self) -> None:
        return None


class _DummyWindow(QtWidgets.QWidget, ToolingDialogsMixin):
    def __init__(self) -> None:
        super().__init__()
        self.canvas = _DummyCanvas()
        self.video_file = None
        self.filename = ""
        self._config = {
            "epsilon": 2.0,
            "canvas": {"double_click": "close", "num_backups": 10, "crosshair": False},
            "sam": {},
            "display_label_popup": False,
            "validate_label": "none",
        }
        self.labelDialog = SimpleNamespace(
            popUp=lambda *args, **kwargs: (None, {}, None, "")
        )
        self.uniqLabelList = QtWidgets.QListWidget(self)
        self.actions = SimpleNamespace(
            editMode=SimpleNamespace(setEnabled=lambda *_: None)
        )


class _DockStub:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def refresh_from_current_canvas(self) -> None:
        self.calls.append("refresh")

    def show(self) -> None:
        self.calls.append("show")

    def raise_(self) -> None:
        self.calls.append("raise")

    def activateWindow(self) -> None:
        self.calls.append("activate")


def test_zone_dock_wraps_zone_panel(monkeypatch) -> None:
    _ensure_qapp()
    window = _DummyWindow()

    class _FakeZonePanel(QtWidgets.QWidget):
        def __init__(self, parent=None, zone_path=None):  # noqa: ARG002
            super().__init__(parent)
            self.canvas = window.canvas
            self.close_button = QtWidgets.QPushButton("Close", self)
            self.close_button.clicked.connect(self.close)
            self.refreshed = False
            self.cleared = False
            self._dirty = False

        def refresh_from_current_canvas(self) -> None:
            self.refreshed = True

        def save_zone_file(self) -> bool:
            return True

        def _clear_zone_defaults(self) -> None:
            self.cleared = True

    monkeypatch.setattr(zone_dock_module, "ZonePanelWidget", _FakeZonePanel)
    dock = ZoneDockWidget(window)

    assert dock.widget() is dock.zone_panel
    assert dock.objectName() == "zoneDock"
    assert dock.zone_panel.canvas is window.canvas
    dock.refresh_from_current_canvas()
    assert dock.zone_panel.refreshed is True


def test_open_zone_manager_raises_existing_zone_dock() -> None:
    _ensure_qapp()
    window = _DummyWindow()
    window.zone_dock = _DockStub()

    window.open_zone_manager()

    assert window.zone_dock.calls == ["refresh", "show", "raise", "activate"]


def test_zone_popup_hides_flags_for_zone_authoring(monkeypatch) -> None:
    _ensure_qapp()
    parent = QtWidgets.QWidget()
    dialog = AnnolidLabelDialog(parent=parent, config={})
    monkeypatch.setattr(dialog, "exec_", lambda: QtWidgets.QDialog.Accepted)

    text, flags, group_id, description = dialog.popUp(
        "north_chamber",
        flags={"zone_kind": True, "semantic_type": True},
        group_id=None,
        description="north chamber",
        show_flags=False,
    )

    assert text == "north_chamber"
    assert description == "north chamber"
    assert group_id is None
    assert dialog._flags_box.isVisible() is False
    assert flags["zone_kind"] is True
