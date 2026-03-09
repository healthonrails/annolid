"""Runtime patching/environment adapters behind the infrastructure layer."""

from __future__ import annotations

import os
from importlib import util as importlib_util
from pathlib import Path

from annolid.gui.qt_env import configure_qt_api, sanitize_qt_plugin_env


_QT_PACKAGE_LAYOUTS = {
    "pyside6": ("PySide6", "PYSIDE6_QT_LIB", Path("Qt")),
}

_QTWEBENGINE_DEFAULT_CHROMIUM_FLAGS = ("--disable-skia-graphite",)

_QTWEBENGINE_UNSUPPORTED_CHROMIUM_FLAGS = ("--use-gl=desktop",)


def _append_env_flags(var_name: str, flags: tuple[str, ...]) -> None:
    current = os.environ.get(var_name, "")
    active = current.split()
    to_add = [flag for flag in flags if flag and flag not in active]
    if to_add:
        os.environ[var_name] = f"{current} {' '.join(to_add)}".strip()


def _strip_env_flags(var_name: str, flags: tuple[str, ...]) -> None:
    current = os.environ.get(var_name, "")
    if not current:
        return
    blocked = set(flags)
    active = [flag for flag in current.split() if flag and flag not in blocked]
    if active:
        os.environ[var_name] = " ".join(active)
    else:
        os.environ.pop(var_name, None)


def _candidate_qt_package_configs() -> list[tuple[str, Path]]:
    configs: list[tuple[str, Path]] = []
    configure_qt_api(os.environ)
    selected_api = str(os.environ.get("QT_API", "")).strip().lower()
    ordered_layouts: list[tuple[str, str, str, Path]] = []
    if selected_api in _QT_PACKAGE_LAYOUTS:
        package_name, env_key, qt_folder = _QT_PACKAGE_LAYOUTS[selected_api]
        ordered_layouts.append((selected_api, package_name, env_key, qt_folder))
    for api_name, (package_name, env_key, qt_folder) in _QT_PACKAGE_LAYOUTS.items():
        if api_name != selected_api:
            ordered_layouts.append((api_name, package_name, env_key, qt_folder))

    for api_name, package_name, env_key, qt_folder in ordered_layouts:
        raw = str(os.environ.get(env_key, "")).strip()
        if raw:
            configs.append((api_name, Path(raw)))
            continue
        try:
            spec = importlib_util.find_spec(package_name)
        except Exception:
            spec = None
        origin = str(getattr(spec, "origin", "") or "").strip()
        if not origin:
            continue
        configs.append((api_name, Path(origin).resolve().parent / qt_folder))
    return configs


def _candidate_qt_webengine_paths() -> list[tuple[Path, Path | None]]:
    candidates: list[tuple[Path, Path | None]] = []
    seen: set[tuple[str, str]] = set()
    for _, qt_root in _candidate_qt_package_configs():
        resource_candidates = [
            (qt_root / "resources", qt_root / "translations" / "qtwebengine_locales"),
            (
                qt_root / "lib" / "QtWebEngineCore.framework" / "Resources",
                qt_root
                / "lib"
                / "QtWebEngineCore.framework"
                / "Resources"
                / "qtwebengine_locales",
            ),
            (
                qt_root / "QtWebEngineCore.framework" / "Resources",
                qt_root
                / "QtWebEngineCore.framework"
                / "Resources"
                / "qtwebengine_locales",
            ),
        ]
        for resources_dir, locales_dir in resource_candidates:
            key = (str(resources_dir), str(locales_dir))
            if key in seen:
                continue
            seen.add(key)
            candidates.append((resources_dir, locales_dir))
    return candidates


def configure_qtwebengine_resource_paths() -> None:
    """
    Set Qt WebEngine resource paths when running from source/venv layouts.

    This avoids startup warnings like "Qt WebEngine resources not found at ...".
    """
    candidates = _candidate_qt_webengine_paths()
    if not candidates:
        return

    if not str(os.environ.get("QTWEBENGINE_RESOURCES_PATH", "")).strip():
        for resources_dir, _ in candidates:
            if resources_dir.is_dir():
                os.environ["QTWEBENGINE_RESOURCES_PATH"] = str(resources_dir)
                break

    if not str(os.environ.get("QTWEBENGINE_LOCALES_PATH", "")).strip():
        for _, locales_dir in candidates:
            if locales_dir is not None and locales_dir.is_dir():
                os.environ["QTWEBENGINE_LOCALES_PATH"] = str(locales_dir)
                break


def configure_qtwebengine_chromium_flags() -> None:
    """Apply stable WebEngine flags before QtWebEngine is imported."""
    _strip_env_flags(
        "QTWEBENGINE_CHROMIUM_FLAGS",
        _QTWEBENGINE_UNSUPPORTED_CHROMIUM_FLAGS,
    )
    _append_env_flags(
        "QTWEBENGINE_CHROMIUM_FLAGS",
        _QTWEBENGINE_DEFAULT_CHROMIUM_FLAGS,
    )


def create_qapp(argv=None):
    """Create QApplication after QT_API has been selected deterministically."""
    configure_qt_api(os.environ)
    from annolid.gui.application import create_qapp as _create_qapp

    return _create_qapp(argv)


__all__ = [
    "configure_qtwebengine_chromium_flags",
    "configure_qtwebengine_resource_paths",
    "create_qapp",
    "sanitize_qt_plugin_env",
]
