from __future__ import annotations

import gc
import os
from types import SimpleNamespace
import warnings
import weakref

from qtpy import QtCore, QtWidgets

from annolid.gui.app import AnnolidWindow
from annolid.gui.mixins.label_panel_mixin import LabelPanelMixin
from annolid.gui.mixins.playback_draw_mixin import PlaybackDrawMixin

_LEAKED_WIDGET_OBJECT_NAME = "annolid_test_intentional_leaked_widget"


class _WarnOnDisconnectSignal:
    def __init__(self, warning_text: str) -> None:
        self._warning_text = warning_text
        self.connected = []

    def disconnect(self) -> None:
        warnings.warn(self._warning_text, RuntimeWarning, stacklevel=2)
        raise RuntimeError(self._warning_text)

    def connect(self, slot) -> None:
        self.connected.append(slot)


def test_label_panel_mixin_suppresses_pyside6_611_disconnect_warnings() -> None:
    msg = 'libpyside: Failed to disconnect (None) from signal "itemSelectionChanged()"'
    label_list = SimpleNamespace(
        itemSelectionChanged=_WarnOnDisconnectSignal(msg),
        itemDoubleClicked=_WarnOnDisconnectSignal(msg),
        shapeVisibilityChanged=_WarnOnDisconnectSignal(msg),
        shapeDeleteRequested=_WarnOnDisconnectSignal(msg),
        shapesDeleteRequested=_WarnOnDisconnectSignal(msg),
        shapesSwitchRequested=_WarnOnDisconnectSignal(msg),
        labelColorChangeRequested=_WarnOnDisconnectSignal(msg),
        labelColorResetRequested=_WarnOnDisconnectSignal(msg),
        selectedItems=lambda: [],
    )
    uniq_list = SimpleNamespace(
        labelColorChangeRequested=_WarnOnDisconnectSignal(msg),
        labelColorResetRequested=_WarnOnDisconnectSignal(msg),
    )
    dummy = SimpleNamespace(
        labelList=label_list,
        uniqLabelList=uniq_list,
        canvas=SimpleNamespace(selectShapes=lambda _shapes: None),
        editLabel=lambda *_args: None,
        _choose_label_color=lambda *_args: None,
        _reset_label_color=lambda *_args: None,
        _on_shape_list_order_changed=lambda *_args: None,
        _on_unique_label_order_changed=lambda *_args: None,
        _label_list_shortcuts_setup=True,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        LabelPanelMixin._setup_label_list_connections(dummy)

    disconnect_warnings = [
        w for w in caught if "Failed to disconnect" in str(w.message)
    ]
    assert disconnect_warnings == []
    assert len(label_list.itemSelectionChanged.connected) == 1


def test_playback_draw_mixin_suppresses_pyside6_611_disconnect_warnings() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    parent = QtWidgets.QWidget()
    msg = 'libpyside: Failed to disconnect (None) from signal "triggered()"'

    class _FakeAction:
        def __init__(self) -> None:
            self.triggered = _WarnOnDisconnectSignal(msg)
            self.toggled = _WarnOnDisconnectSignal(msg)
            self.checkable = False

        def setCheckable(self, value: bool) -> None:  # noqa: N802
            self.checkable = value

        def setChecked(self, _value: bool) -> None:  # noqa: N802
            return None

    class _FakeActionGroup:
        def __init__(self, _owner) -> None:
            self.exclusive = False
            self.actions = []

        def setExclusive(self, value: bool) -> None:  # noqa: N802
            self.exclusive = value

        def addAction(self, action) -> None:  # noqa: N802
            self.actions.append(action)

    action = _FakeAction()
    parent._draw_mode_actions_initialized = False
    parent.actions = SimpleNamespace(
        editMode=action,
        createMode=None,
        createPolygonSAMMode=None,
        createRectangleMode=None,
        createCircleMode=None,
        createLineMode=None,
        createPointMode=None,
        createLineStripMode=None,
        createAiPolygonMode=None,
        createAiMaskMode=None,
        createGroundingSAMMode=None,
    )
    parent.toggleDrawMode = lambda *_args, **_kwargs: None
    parent.segmentAnything = lambda *_args, **_kwargs: None

    orig_group = QtWidgets.QActionGroup
    QtWidgets.QActionGroup = _FakeActionGroup  # type: ignore[misc]
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            PlaybackDrawMixin._setup_drawing_mode_actions(parent)
    finally:
        QtWidgets.QActionGroup = orig_group  # type: ignore[misc]
        parent.close()
        parent.deleteLater()

    disconnect_warnings = [
        w for w in caught if "Failed to disconnect" in str(w.message)
    ]
    assert disconnect_warnings == []
    assert len(action.toggled.connected) == 1


def test_annolid_window_can_be_garbage_collected_after_close_and_delete_later() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = AnnolidWindow(config={})
    win_ref = weakref.ref(window)

    window.setClean()
    window.close()
    window.deleteLater()
    del window

    deferred_delete = getattr(
        getattr(QtCore.QEvent, "Type", QtCore.QEvent),
        "DeferredDelete",
        QtCore.QEvent.DeferredDelete,
    )
    for _ in range(3):
        app.sendPostedEvents(None, deferred_delete)
        app.processEvents()
    gc.collect()
    app.sendPostedEvents(None, deferred_delete)
    app.processEvents()
    gc.collect()

    assert win_ref() is None


def test_conftest_teardown_reclaims_leaked_top_level_widget_step1() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    leaked = QtWidgets.QWidget()
    leaked.setObjectName(_LEAKED_WIDGET_OBJECT_NAME)
    assert any(
        w.objectName() == _LEAKED_WIDGET_OBJECT_NAME for w in app.topLevelWidgets()
    )


def test_conftest_teardown_reclaims_leaked_top_level_widget_step2() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    remaining = [
        w for w in app.topLevelWidgets() if w.objectName() == _LEAKED_WIDGET_OBJECT_NAME
    ]
    assert remaining == []


def test_pytest_sets_non_interactive_matplotlib_backend() -> None:
    assert os.environ.get("MPLBACKEND", "").lower() == "agg"


def test_conftest_teardown_skips_cpp_owned_qcompleter_popup() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    editor = QtWidgets.QLineEdit()
    completer = QtWidgets.QCompleter(["alpha", "beta"], editor)
    editor.setCompleter(completer)
    popup = completer.popup()
    assert popup is not None
    editor.close()
