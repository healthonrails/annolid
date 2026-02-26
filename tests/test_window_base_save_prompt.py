import os

from qtpy import QtWidgets

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


def test_save_file_suppresses_unsaved_prompt_during_save(monkeypatch) -> None:
    _ensure_qapp()
    window = AnnolidWindowBase(config={})
    try:
        window.filename = "/tmp/frame_0001.png"
        window.dirty = True
        warned = {"count": 0}

        def _warn(*args, **kwargs):
            warned["count"] += 1
            return QtWidgets.QMessageBox.Cancel

        monkeypatch.setattr(QtWidgets.QMessageBox, "warning", _warn)
        window._getLabelFile = (  # type: ignore[method-assign]
            lambda _filename: "/tmp/frame_0001.json"
        )

        called = {"ok": False}

        def _fake_save(_target):
            called["ok"] = True
            # Internal callbacks may call mayContinue while dirty; this should
            # be suppressed during an explicit save action.
            assert window.mayContinue() is True
            window.dirty = False

        window._saveFile = _fake_save  # type: ignore[method-assign]
        window.saveFile()

        assert called["ok"] is True
        assert warned["count"] == 0
    finally:
        window.close()
