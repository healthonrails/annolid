from __future__ import annotations

from qtpy import QtCore, QtWidgets

from annolid.gui.mixins.settings_timeline_mixin import SettingsTimelineMixin
from annolid.gui.widgets.timeline_panel import TimelinePanel


def _ensure_qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class _Settings:
    def __init__(self, values: dict[str, object]) -> None:
        self._values = dict(values)

    def value(self, key: str, default=None, type=None):  # noqa: A002
        return self._values.get(key, default)

    def setValue(self, key: str, value) -> None:
        self._values[key] = value


class _Host(QtWidgets.QMainWindow, SettingsTimelineMixin):
    def __init__(self) -> None:
        super().__init__()
        self.settings = _Settings({"timeline/show_dock": True})
        self.timeline_panel = QtWidgets.QWidget()
        self.timeline_dock = QtWidgets.QDockWidget("Timeline", self)
        self.timeline_dock.setWidget(self.timeline_panel)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.timeline_dock)
        self._toggle_timeline_action = QtWidgets.QAction("Timeline", self)
        self._toggle_timeline_action.setCheckable(True)
        self._toggle_timeline_action.setChecked(False)
        self._toggle_timeline_action.setEnabled(False)

    def _apply_fixed_dock_sizes(self) -> None:
        pass


def test_timeline_dock_visibility_uses_single_policy() -> None:
    _ensure_qapp()
    host = _Host()
    try:
        host.show()
        _ensure_qapp().processEvents()

        host._apply_timeline_dock_visibility(video_open=True)
        _ensure_qapp().processEvents()
        assert host._toggle_timeline_action.isEnabled() is True
        assert host._toggle_timeline_action.isChecked() is True
        assert host.timeline_panel.isEnabled() is True
        assert host.timeline_dock.isVisible() is True

        host.settings.setValue("timeline/show_dock", False)
        host._apply_timeline_dock_visibility(video_open=True)
        _ensure_qapp().processEvents()
        assert host._toggle_timeline_action.isEnabled() is True
        assert host._toggle_timeline_action.isChecked() is False
        assert host.timeline_dock.isVisible() is False

        host._apply_timeline_dock_visibility(video_open=False)
        _ensure_qapp().processEvents()
        assert host._toggle_timeline_action.isEnabled() is False
        assert host.timeline_panel.isEnabled() is True
    finally:
        host.close()


def test_timeline_edit_toggle_for_event_rows_forces_edit_mode_off() -> None:
    _ensure_qapp()

    class _ViewStub:
        def __init__(self) -> None:
            self.calls: list[bool] = []

        def set_edit_mode(self, enabled: bool) -> None:
            self.calls.append(bool(enabled))

    host = type("TimelineHost", (), {})()
    host._row_mode = "Event"
    host._edit_toggle = QtWidgets.QToolButton()
    host._edit_toggle.setCheckable(True)
    host._edit_toggle.setChecked(True)
    host._view = _ViewStub()

    TimelinePanel._on_edit_toggled(host, True)

    assert host._edit_toggle.isChecked() is False
    assert host._view.calls == [False]
