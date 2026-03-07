"""Runtime patching/environment adapters behind the infrastructure layer."""

from __future__ import annotations

import os
from pathlib import Path

from annolid.gui.application import create_qapp
from annolid.gui.qt_env import sanitize_qt_plugin_env
from annolid.utils.macos_fixes import apply_macos_webengine_sandbox_patch


def _candidate_qt_resource_roots() -> list[Path]:
    roots: list[Path] = []
    for env_key in ("PYQT6_QT_LIB", "PYSIDE6_QT_LIB"):
        raw = str(os.environ.get(env_key, "")).strip()
        if raw:
            roots.append(Path(raw))

    try:
        import PyQt6  # type: ignore

        roots.append(Path(PyQt6.__file__).resolve().parent / "Qt6")
    except Exception:
        pass
    try:
        import PySide6  # type: ignore

        roots.append(Path(PySide6.__file__).resolve().parent / "Qt")
    except Exception:
        pass
    return roots


def configure_qtwebengine_resource_paths() -> None:
    """
    Set Qt WebEngine resource paths when running from source/venv layouts.

    This avoids startup warnings like "Qt WebEngine resources not found at ...".
    """
    roots = _candidate_qt_resource_roots()
    if not roots:
        return

    if not str(os.environ.get("QTWEBENGINE_RESOURCES_PATH", "")).strip():
        for root in roots:
            candidate = root / "resources"
            if candidate.is_dir():
                os.environ["QTWEBENGINE_RESOURCES_PATH"] = str(candidate)
                break

    if not str(os.environ.get("QTWEBENGINE_LOCALES_PATH", "")).strip():
        for root in roots:
            candidate = root / "translations" / "qtwebengine_locales"
            if candidate.is_dir():
                os.environ["QTWEBENGINE_LOCALES_PATH"] = str(candidate)
                break


__all__ = [
    "apply_macos_webengine_sandbox_patch",
    "configure_qtwebengine_resource_paths",
    "create_qapp",
    "sanitize_qt_plugin_env",
]
