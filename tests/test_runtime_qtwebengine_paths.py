from __future__ import annotations

import os
from annolid.infrastructure import runtime


def test_configure_qtwebengine_resource_paths_prefers_pyside6_framework_layout(
    monkeypatch, tmp_path
) -> None:
    qt_root = tmp_path / "PySide6" / "Qt"
    resources_dir = qt_root / "lib" / "QtWebEngineCore.framework" / "Resources"
    locales_dir = resources_dir / "qtwebengine_locales"
    locales_dir.mkdir(parents=True)

    monkeypatch.setenv("QT_API", "pyside6")
    monkeypatch.setenv("PYSIDE6_QT_LIB", str(qt_root))
    monkeypatch.delenv("QTWEBENGINE_RESOURCES_PATH", raising=False)
    monkeypatch.delenv("QTWEBENGINE_LOCALES_PATH", raising=False)

    runtime.configure_qtwebengine_resource_paths()

    assert os.environ["QTWEBENGINE_RESOURCES_PATH"] == str(resources_dir)
    assert os.environ["QTWEBENGINE_LOCALES_PATH"] == str(locales_dir)


def test_configure_qtwebengine_resource_paths_respects_selected_qt_api(
    monkeypatch, tmp_path
) -> None:
    pyside_root = tmp_path / "PySide6" / "Qt"
    pyside_resources = pyside_root / "lib" / "QtWebEngineCore.framework" / "Resources"
    (pyside_resources / "qtwebengine_locales").mkdir(parents=True)

    monkeypatch.setenv("QT_API", "pyside6")
    monkeypatch.setenv("PYSIDE6_QT_LIB", str(pyside_root))
    monkeypatch.delenv("QTWEBENGINE_RESOURCES_PATH", raising=False)
    monkeypatch.delenv("QTWEBENGINE_LOCALES_PATH", raising=False)

    runtime.configure_qtwebengine_resource_paths()

    assert os.environ["QTWEBENGINE_RESOURCES_PATH"] == str(pyside_resources)


def test_configure_qtwebengine_chromium_flags_disables_graphite(monkeypatch) -> None:
    monkeypatch.delenv("QTWEBENGINE_CHROMIUM_FLAGS", raising=False)

    runtime.configure_qtwebengine_chromium_flags()

    assert "--disable-skia-graphite" in os.environ["QTWEBENGINE_CHROMIUM_FLAGS"]


def test_configure_qtwebengine_chromium_flags_strips_unsupported_gl(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "QTWEBENGINE_CHROMIUM_FLAGS",
        "--use-gl=desktop --foo",
    )

    runtime.configure_qtwebengine_chromium_flags()

    flags = os.environ["QTWEBENGINE_CHROMIUM_FLAGS"].split()
    assert "--use-gl=desktop" not in flags
    assert "--disable-skia-graphite" in flags
    assert "--foo" in flags
