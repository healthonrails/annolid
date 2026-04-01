from __future__ import annotations

from qtpy import QtCore, QtWidgets

from annolid.gui.controllers.flags import FlagsController
from annolid.gui.widgets.flags import FlagTableWidget


def _ensure_qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class _WindowStub:
    def __init__(self) -> None:
        self.here = __import__("pathlib").Path("/tmp")
        self.canvas = type(
            "CanvasStub", (), {"setBehaviorText": lambda self, text: None}
        )()
        self.seekbar = None
        self.event_type = None

    def statusBar(self):
        return type(
            "StatusStub",
            (),
            {"showMessage": lambda *args, **kwargs: None},
        )()


def test_flag_table_widget_emits_rows_changed_on_row_updates() -> None:
    _ensure_qapp()
    widget = FlagTableWidget()
    rows_changed = []
    widget.rowsChanged.connect(lambda: rows_changed.append(True))

    widget.add_row("grooming", True)
    assert rows_changed

    rows_changed.clear()
    widget.loadFlags({"resting": False})
    assert rows_changed


def test_flags_controller_emits_changed_when_state_updates() -> None:
    _ensure_qapp()

    class _WidgetStub(QtWidgets.QWidget):
        startButtonClicked = QtCore.Signal(str)
        endButtonClicked = QtCore.Signal(str)
        flagsSaved = QtCore.Signal(dict)
        rowSelected = QtCore.Signal(str)
        flagToggled = QtCore.Signal(str, bool)

        def __init__(self) -> None:
            super().__init__()
            self.loaded = []
            self.cleared = 0

        def loadFlags(self, flags):
            self.loaded.append(dict(flags))

        def clear(self):
            self.cleared += 1

        def _get_existing_flag_names(self):
            return {}

        def _update_row_value(self, *_args, **_kwargs):
            pass

    window = _WindowStub()
    widget = _WidgetStub()
    controller = FlagsController(window=window, widget=widget)
    emitted = []
    controller.flagsChanged.connect(lambda flags: emitted.append(dict(flags)))

    controller.set_flags({"grooming": False}, persist=False)
    controller.clear_flags()

    assert emitted[0] == {"grooming": False}
    assert emitted[-1] == {}
    assert widget.loaded[0] == {"grooming": False}
    assert widget.cleared == 1
