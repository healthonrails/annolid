from __future__ import annotations

from annolid.gui.qt_env import configure_qt_api


def test_configure_qt_api_prefers_pyside6_when_both_available(monkeypatch) -> None:
    from annolid.gui import qt_env

    monkeypatch.delenv("QT_API", raising=False)

    def _fake_find_spec(name: str):
        if name in {"PyQt5", "PySide6"}:
            return object()
        return None

    monkeypatch.setattr(qt_env.importlib_util, "find_spec", _fake_find_spec)
    env = {}
    configure_qt_api(env)
    assert env["QT_API"] == "pyside6"


def test_configure_qt_api_leaves_env_unset_when_pyside6_missing(monkeypatch) -> None:
    from annolid.gui import qt_env

    monkeypatch.delenv("QT_API", raising=False)

    def _fake_find_spec(name: str):
        return None

    monkeypatch.setattr(qt_env.importlib_util, "find_spec", _fake_find_spec)
    env = {}
    configure_qt_api(env)
    assert "QT_API" not in env


def test_configure_qt_api_respects_existing_setting(monkeypatch) -> None:
    from annolid.gui import qt_env

    env = {"QT_API": "pyside6"}
    monkeypatch.setattr(qt_env.importlib_util, "find_spec", lambda _name: object())
    configure_qt_api(env)
    assert env["QT_API"] == "pyside6"
