from __future__ import annotations

from qtpy import QtWidgets

from annolid.gui.widgets.ai_chat_manager import AIChatManager


def _ensure_qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


class _WindowStub(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.canvas = None

    def set_unrelated_docks_visible(self, *_args, **_kwargs) -> None:
        return


def test_initialize_annolid_bot_hidden_does_not_construct_dock(monkeypatch) -> None:
    _ensure_qapp()
    window = _WindowStub()
    manager = AIChatManager(window)

    calls = {"ensure_dock": 0, "start_bg": 0}

    def _count_ensure_dock():
        calls["ensure_dock"] += 1
        raise AssertionError("_ensure_dock should not run for hidden startup")

    monkeypatch.setattr(manager, "_ensure_dock", _count_ensure_dock)
    monkeypatch.setattr(
        manager,
        "_start_background_services",
        lambda: calls.__setitem__("start_bg", calls["start_bg"] + 1),
    )

    manager.initialize_annolid_bot(start_visible=False)

    assert calls["ensure_dock"] == 0
    assert calls["start_bg"] == 1


def test_start_background_services_is_idempotent(monkeypatch) -> None:
    _ensure_qapp()
    window = _WindowStub()
    manager = AIChatManager(window)

    starts = {"count": 0}

    class _FakeThread:
        def __init__(self, *args, **kwargs) -> None:  # noqa: ARG002
            self._alive = False

        def start(self) -> None:
            starts["count"] += 1
            self._alive = True

        def is_alive(self) -> bool:
            return bool(self._alive)

        def join(self, timeout=None) -> None:  # noqa: ARG002
            self._alive = False

    monkeypatch.setattr(
        "annolid.gui.widgets.ai_chat_manager.threading.Thread", _FakeThread
    )

    manager._start_background_services()
    manager._start_background_services()

    assert starts["count"] == 1
