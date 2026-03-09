from __future__ import annotations

import os

from annolid.gui.qt_env import configure_qt_api
from annolid.infrastructure import runtime


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


def test_sanitize_qt_plugin_env_removes_cv2_qt_entries_on_linux() -> None:
    env = {
        "QT_QPA_PLATFORM_PLUGIN_PATH": "/tmp/site-packages/cv2/qt/plugins",
        "QT_QPA_FONTDIR": "/tmp/site-packages/cv2/qt/plugins/fonts",
        "QT_PLUGIN_PATH": os.pathsep.join(
            [
                "/tmp/site-packages/cv2/qt/plugins",
                "/opt/qt/plugins",
            ]
        ),
    }

    from annolid.gui.qt_env import sanitize_qt_plugin_env

    sanitize_qt_plugin_env(env, is_linux=True)

    assert "QT_QPA_PLATFORM_PLUGIN_PATH" not in env
    assert "QT_QPA_FONTDIR" not in env
    assert env["QT_PLUGIN_PATH"] == "/opt/qt/plugins"


def test_configure_qt_runtime_applies_binding_and_webengine_env(monkeypatch) -> None:
    env: dict[str, str] = {
        "QT_PLUGIN_PATH": os.pathsep.join(
            [
                "/tmp/site-packages/cv2/qt/plugins",
                "/opt/qt/plugins",
            ]
        ),
        "QTWEBENGINE_CHROMIUM_FLAGS": "--use-gl=desktop",
    }
    monkeypatch.setattr(runtime.os, "environ", env, raising=False)
    monkeypatch.setattr(
        runtime,
        "configure_qt_api",
        lambda target_env: target_env.__setitem__("QT_API", "pyside6"),
    )
    monkeypatch.setattr(
        runtime,
        "sanitize_qt_plugin_env",
        lambda target_env: target_env.__setitem__("QT_PLUGIN_PATH", "/opt/qt/plugins"),
    )
    monkeypatch.setattr(
        runtime,
        "configure_qtwebengine_resource_paths",
        lambda: env.__setitem__("QTWEBENGINE_RESOURCES_PATH", "/opt/qt/resources"),
    )

    runtime.configure_qt_runtime()

    assert env["QT_API"] == "pyside6"
    assert env["QT_PLUGIN_PATH"] == "/opt/qt/plugins"
    assert env["QTWEBENGINE_CHROMIUM_FLAGS"] == "--disable-skia-graphite"
    assert env["QTWEBENGINE_RESOURCES_PATH"] == "/opt/qt/resources"
