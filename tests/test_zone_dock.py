from __future__ import annotations

from types import SimpleNamespace

from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.mixins.tooling_dialogs_mixin import ToolingDialogsMixin
from annolid.gui.shape import Shape
from annolid.gui.widgets.label_dialog import AnnolidLabelDialog
import annolid.gui.widgets.zone_dock as zone_dock_module
from annolid.gui.widgets.zone_dock import ZoneDockWidget
from annolid.gui.widgets.zone_panel import ZonePanelWidget


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

    def deleteSelected(self):
        return []

    def storeShapes(self) -> None:
        return None

    def setFocus(self) -> None:
        return None

    def update(self) -> None:
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


def _build_zone_shape(label: str = "left_zone") -> Shape:
    shape = Shape(
        label=label,
        shape_type="polygon",
        flags={
            "semantic_type": "zone",
            "zone_kind": "chamber",
            "phase": "phase_1",
            "occupant_role": "rover",
            "access_state": "open",
            "tags": ["social"],
        },
        description="left chamber",
    )
    shape.addPoint(QtCore.QPointF(0, 0))
    shape.addPoint(QtCore.QPointF(10, 0))
    shape.addPoint(QtCore.QPointF(10, 8))
    shape.addPoint(QtCore.QPointF(0, 8))
    shape.close()
    return shape


def _build_test_panel(window: _DummyWindow) -> ZonePanelWidget:
    original = ZonePanelWidget._create_metadata_combo

    def _safe_combo(self, options):
        combo = QtWidgets.QComboBox()
        combo.addItems(options)
        return combo

    ZonePanelWidget._create_metadata_combo = _safe_combo
    try:
        return ZonePanelWidget(window)
    finally:
        ZonePanelWidget._create_metadata_combo = original


def test_zone_panel_applies_selected_zone_metadata() -> None:
    _ensure_qapp()
    window = _DummyWindow()
    zone = _build_zone_shape()
    window.canvas.shapes = [zone]

    panel = _build_test_panel(window)
    panel.refresh_from_current_canvas()
    panel._selected_shape = zone
    panel._sync_selected_fields()

    panel.zone_label_edit.setText("social_left")
    panel.zone_description_edit.setText("left social zone")
    panel.zone_kind_combo.setCurrentText("interaction_zone")
    panel.phase_combo.setCurrentText("phase_2")
    panel.occupant_combo.setCurrentText("stim")
    panel.access_combo.setCurrentText("blocked")
    panel.tags_edit.setText("social, mesh")
    panel.barrier_adjacent_checkbox.setChecked(True)
    panel._apply_selected_zone_details()

    assert zone.label == "social_left"
    assert zone.description == "left social zone"
    assert zone.flags["zone_kind"] == "interaction_zone"
    assert zone.flags["phase"] == "phase_2"
    assert zone.flags["occupant_role"] == "stim"
    assert zone.flags["access_state"] == "blocked"
    assert zone.flags["barrier_adjacent"] is True
    assert zone.flags["tags"] == ["social", "mesh"]
    assert panel.selected_zone_value.text() == "social_left"
    assert (
        "Barrier-adjacent summary is enabled." in panel.selected_metric_summary.text()
    )


def test_zone_panel_updates_area_and_default_preview() -> None:
    _ensure_qapp()
    window = _DummyWindow()
    zone = _build_zone_shape("arena_zone")
    window.canvas.shapes = [zone]

    panel = _build_test_panel(window)
    panel.refresh_from_current_canvas()

    assert panel.zone_count_value.text() == "1"
    assert panel.metrics_ready_value.text() == "1"

    panel._selected_shape = zone
    panel._sync_selected_fields()

    assert panel.selected_area_value.text() == "80.0 px²"
    assert "arena_zone" in panel.selected_metric_summary.text()

    panel._selected_shape = None
    panel.default_zone_kind_combo.setCurrentText("doorway")
    panel.default_phase_combo.setCurrentText("phase_2")
    panel.default_occupant_combo.setCurrentText("stim")
    panel.default_access_combo.setCurrentText("tethered")
    panel._publish_zone_defaults()

    assert "kind 'doorway'" in panel.selected_metric_summary.text()
    assert window._zone_authoring_defaults["flags"]["zone_kind"] == "doorway"


def test_zone_panel_uses_scroll_areas_for_compact_layout() -> None:
    _ensure_qapp()
    window = _DummyWindow()
    panel = _build_test_panel(window)

    scroll_areas = panel.findChildren(QtWidgets.QScrollArea)
    assert scroll_areas
    assert all(area.widgetResizable() for area in scroll_areas)
