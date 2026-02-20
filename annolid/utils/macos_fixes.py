from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

_REQUIRED_FRAMEWORKS = (
    "QtCore.framework",
    "QtGui.framework",
    "QtNetwork.framework",
    "QtQml.framework",
    "QtQmlModels.framework",
    "QtQuick.framework",
    "QtWebChannel.framework",
    "QtWebEngineCore.framework",
    "QtPositioning.framework",
)

_BASE_CHROMIUM_FLAGS = (
    "--no-sandbox",
    "--ignore-gpu-blocklist",
    "--use-gl=desktop",
)


# ---------------------------------------------------------------------------
# Framework symlink repair
# ---------------------------------------------------------------------------


def _fix_framework_symlinks(framework_dir: Path) -> None:
    """Restore standard macOS framework symlinks that pip wheels strip.

    PyPI-distributed Qt frameworks have a flat layout missing:
      - Versions/Current -> 5
      - Versions/5/Resources -> ../../Resources
      - <Framework> -> Versions/Current/<Framework>

    Chromium's icu_util.cc uses dladdr() to locate the QtWebEngineCore
    binary at Versions/5/QtWebEngineCore, then looks for Resources/icudtl.dat
    as a sibling.  Without Versions/5/Resources the lookup fails.
    """
    versions_dir = framework_dir / "Versions"
    top_resources = framework_dir / "Resources"
    if not versions_dir.is_dir() or not top_resources.is_dir():
        return

    version_name = "5"  # Qt 5
    version_dir = versions_dir / version_name
    if not version_dir.is_dir():
        return

    # Versions/Current -> 5
    current_link = versions_dir / "Current"
    if not current_link.exists():
        try:
            current_link.symlink_to(version_name)
        except Exception:
            pass

    # Versions/5/Resources -> ../../Resources
    versioned_resources = version_dir / "Resources"
    if not versioned_resources.exists():
        try:
            versioned_resources.symlink_to(Path("..", "..", "Resources"))
        except Exception:
            pass

    # Top-level binary symlink: <Framework> -> Versions/Current/<Binary>
    framework_name = framework_dir.name.replace(".framework", "")
    binary_in_version = version_dir / framework_name
    top_binary = framework_dir / framework_name
    if binary_in_version.exists() and not top_binary.exists():
        try:
            top_binary.symlink_to(Path("Versions", "Current", framework_name))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _prepend_env_path(var_name: str, entries: Iterable[str]) -> None:
    existing = [p for p in os.environ.get(var_name, "").split(os.pathsep) if p]
    for entry in reversed([str(e) for e in entries if e]):
        if entry not in existing:
            existing.insert(0, entry)
    if existing:
        os.environ[var_name] = os.pathsep.join(existing)


def _append_chromium_flags(flags: Iterable[str]) -> None:
    current = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
    active = current.split()
    to_add = [flag for flag in flags if flag and flag not in active]
    if to_add:
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (
            f"{current} {' '.join(to_add)}".strip()
        )


def _safe_remove(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    try:
        if path.is_symlink() or path.is_file():
            path.unlink(missing_ok=True)
        else:
            shutil.rmtree(path)
    except Exception:
        subprocess.run(
            ["/bin/rm", "-rf", str(path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


# ---------------------------------------------------------------------------
# Resource env vars
# ---------------------------------------------------------------------------


def _set_framework_resource_env(qt_lib_dir: Path) -> None:
    """Point QtWebEngine to the framework Resources directory explicitly."""
    source_resources = qt_lib_dir / "QtWebEngineCore.framework" / "Resources"
    if not source_resources.exists():
        return
    os.environ["QTWEBENGINE_RESOURCES_PATH"] = str(source_resources)
    locales_dir = source_resources / "qtwebengine_locales"
    if locales_dir.exists():
        os.environ["QTWEBENGINE_LOCALES_PATH"] = str(locales_dir)


# ---------------------------------------------------------------------------
# Bare-binary helper setup (the actual fix for the ICU mmap error)
# ---------------------------------------------------------------------------


def _setup_bare_binary_helper(qt_lib_dir: Path) -> None:
    """Copy QtWebEngineProcess as a bare binary to bypass macOS sandbox.

    The default ``QtWebEngineProcess.app`` inside ``.venv`` is subject to
    strict macOS sandboxing that blocks it from loading Qt frameworks and
    ICU data (dyld error: ``file system sandbox blocked open``).

    This function:
    1. Copies the binary (not the .app bundle) to a user cache directory
    2. Copies resources (icudtl.dat, .pak files) flat next to it
    3. Copies required Qt frameworks alongside it
    4. Patches rpath to ``@loader_path/Frameworks``
    5. Ad-hoc codesigns the binary
    6. Sets ``QTWEBENGINEPROCESS_PATH`` to this unrestricted copy

    Stripping the .app bundle structure avoids macOS automatically applying
    the ``com.apple.WebEngine`` sandbox profile which blocks .venv access.
    """
    webengine_core = qt_lib_dir / "QtWebEngineCore.framework"
    src_process_app = webengine_core / "Helpers" / "QtWebEngineProcess.app"
    src_resources = webengine_core / "Resources"

    if not src_process_app.exists():
        return

    cache_base = Path.home() / "Library" / "Caches" / "Annolid" / "QtWebEngine"
    target_executable = cache_base / "QtWebEngineProcess"

    # Re-create cache if binary is missing
    if not target_executable.exists():
        if cache_base.exists():
            shutil.rmtree(cache_base)
        cache_base.mkdir(parents=True, exist_ok=True)

        src_executable = src_process_app / "Contents" / "MacOS" / "QtWebEngineProcess"
        shutil.copy2(src_executable, target_executable)

    # Clean up old "Resources" subdirectory from previous layout versions
    old_resources_dir = cache_base / "Resources"
    if old_resources_dir.exists():
        if old_resources_dir.is_dir():
            shutil.rmtree(old_resources_dir)
        else:
            old_resources_dir.unlink()

    # Copy resources flat next to binary (bare binary expects them as siblings).
    # Use a stamp file to avoid re-copying on every launch.
    stamp_file = cache_base / ".resources_copied_v2"
    if not stamp_file.exists():
        for item in src_resources.iterdir():
            dest = cache_base / item.name
            if dest.exists() or dest.is_symlink():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

        # Copy required frameworks so the isolated process can load them
        frameworks_dir = cache_base / "Frameworks"
        if frameworks_dir.exists():
            shutil.rmtree(frameworks_dir)
        frameworks_dir.mkdir()

        for fw_name in _REQUIRED_FRAMEWORKS:
            src_fw = qt_lib_dir / fw_name
            if src_fw.exists():
                shutil.copytree(src_fw, frameworks_dir / fw_name, symlinks=True)

        # Create empty qt.conf to prevent Qt from looking elsewhere
        with open(cache_base / "qt.conf", "w") as f:
            f.write("[Paths]\nPrefix = .\n")

        stamp_file.touch()

    # Patch rpath to prefer local Frameworks
    install_name_tool = "/usr/bin/install_name_tool"
    if not os.path.exists(install_name_tool):
        install_name_tool = "install_name_tool"

    subprocess.run(
        [
            install_name_tool,
            "-add_rpath",
            "@loader_path/Frameworks",
            str(target_executable),
        ],
        capture_output=True,
    )
    # Remove old absolute rpath if present
    qt_lib_str = str(qt_lib_dir.resolve())
    subprocess.run(
        [install_name_tool, "-delete_rpath", qt_lib_str, str(target_executable)],
        capture_output=True,
    )

    # Ad-hoc sign
    subprocess.run(
        ["codesign", "--force", "--sign", "-", str(target_executable)],
        capture_output=True,
    )

    os.environ["QTWEBENGINEPROCESS_PATH"] = str(target_executable)
    logger.info(
        "Redirected QtWebEngineProcess to cached bare binary: %s",
        target_executable,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def apply_macos_webengine_sandbox_patch() -> None:
    """Configure macOS QtWebEngine so icudtl.dat is found and sandbox is off.

    The essential fixes are:
    1. Restore standard framework symlinks so Chromium's icu_util.cc can
       resolve icudtl.dat via dladdr() -> Versions/5/Resources/icudtl.dat.
    2. Copy QtWebEngineProcess as a bare binary (without .app bundle) to
       a cache directory, with resources and frameworks alongside it.
       This avoids the macOS sandbox profile that blocks .venv access.
    3. Disable the sandbox explicitly via env vars and Chromium flags.
    """
    if platform.system() != "Darwin":
        return

    os.environ["QT_MAC_WANTS_LAYER"] = "1"
    os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"
    _append_chromium_flags(_BASE_CHROMIUM_FLAGS)

    try:
        import PyQt5  # type: ignore
    except Exception:
        return

    try:
        qt_lib_dir = Path(PyQt5.__file__).parent / "Qt5" / "lib"

        # Fix the framework's stripped symlinks so that Chromium's
        # icu_util.cc can locate icudtl.dat via dladdr().
        _fix_framework_symlinks(qt_lib_dir / "QtWebEngineCore.framework")

        # Set QTWEBENGINE_RESOURCES_PATH (the env var Qt actually honours).
        _set_framework_resource_env(qt_lib_dir)

        # Set up the bare-binary helper to bypass macOS sandbox.
        # This is the primary fix for the ICU mmap error.
        _setup_bare_binary_helper(qt_lib_dir)

        # Ensure the helper process can resolve Qt frameworks at runtime.
        _prepend_env_path("DYLD_FRAMEWORK_PATH", [str(qt_lib_dir)])
    except Exception as exc:
        print(
            f"Warning: Failed to apply macOS QtWebEngine fix: "
            f"{exc.__class__.__name__}: {exc}"
        )
