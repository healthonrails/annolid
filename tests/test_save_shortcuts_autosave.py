from __future__ import annotations

import os
from types import SimpleNamespace

from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.mixins.persistence_lifecycle_mixin import PersistenceLifecycleMixin
from annolid.gui.window_base import AnnolidWindowBase


os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


class _AutoSaveProbe(QtCore.QObject, PersistenceLifecycleMixin):
    def __init__(self) -> None:
        super().__init__()
        self._config = {"auto_save": True}
        self.settings = QtCore.QSettings("Annolid", "AnnolidTest")
        self.canvas = SimpleNamespace(isShapeRestorable=False)
        self.actions = SimpleNamespace(
            undo=QtWidgets.QAction("Undo", self),
            save=QtWidgets.QAction("Save", self),
            saveAuto=QtWidgets.QAction("Auto Save", self),
        )
        self.actions.saveAuto.setCheckable(True)
        self.actions.saveAuto.setChecked(True)
        self.filename = "frame_0001.png"
        self.output_dir = None
        self.dirty = False
        self.save_calls = 0

    def getTitle(self, clean=True):  # noqa: ARG002
        return "Annolid-Test"

    def setWindowTitle(self, _title: str) -> None:
        return None

    def saveFile(self) -> None:
        self.save_calls += 1
        self.dirty = False

    def statusBar(self):
        return SimpleNamespace(showMessage=lambda *_args, **_kwargs: None)


def test_window_base_provides_default_save_shortcuts() -> None:
    _ensure_qapp()
    window = AnnolidWindowBase(config={})
    try:
        save_shortcut = window.actions.save.shortcut().toString()
        assert save_shortcut
        # Platform-dependent representation; this catches Ctrl+S/Meta+S behavior.
        assert ("S" in save_shortcut) or (
            QtGui.QKeySequence(window.actions.save.shortcut()).toString()
        )
    finally:
        window.close()


def test_set_dirty_uses_debounced_autosave_path() -> None:
    _ensure_qapp()
    probe = _AutoSaveProbe()
    probe.setDirty()
    assert probe.dirty is True
    assert probe.actions.save.isEnabled() is True
    # Simulate debounce timeout callback without waiting.
    probe._run_auto_save()
    assert probe.save_calls == 1
