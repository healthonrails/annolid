import os
import sys
import types
from pathlib import Path

from annolid.utils import macos_fixes


def test_apply_macos_fix_sets_framework_resources_path(monkeypatch, tmp_path) -> None:
    qt_lib_dir = tmp_path / "PyQt5" / "Qt5" / "lib"
    resources_dir = qt_lib_dir / "QtWebEngineCore.framework" / "Resources"
    locales_dir = resources_dir / "qtwebengine_locales"
    process_bin = (
        qt_lib_dir
        / "QtWebEngineCore.framework"
        / "Helpers"
        / "QtWebEngineProcess.app"
        / "Contents"
        / "MacOS"
        / "QtWebEngineProcess"
    )
    locales_dir.mkdir(parents=True)
    process_bin.parent.mkdir(parents=True)
    (resources_dir / "icudtl.dat").write_bytes(b"icu")
    (locales_dir / "en-US.pak").write_bytes(b"pak")
    process_bin.write_text("")

    fake_pyqt = types.SimpleNamespace(__file__=str(tmp_path / "PyQt5" / "__init__.py"))
    monkeypatch.setitem(sys.modules, "PyQt5", fake_pyqt)
    monkeypatch.setattr(macos_fixes.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(macos_fixes, "_fix_framework_symlinks", lambda _: None)
    monkeypatch.setattr(macos_fixes, "_setup_bare_binary_helper", lambda _: None)
    monkeypatch.setattr(macos_fixes, "_prepend_env_path", lambda *_: None)

    monkeypatch.delenv("QTWEBENGINE_RESOURCES_PATH", raising=False)
    monkeypatch.delenv("QTWEBENGINE_LOCALES_PATH", raising=False)
    monkeypatch.delenv("QTWEBENGINEPROCESS_PATH", raising=False)
    monkeypatch.delenv("QTWEBENGINE_CHROMIUM_FLAGS", raising=False)

    macos_fixes.apply_macos_webengine_sandbox_patch()

    assert os.environ["QTWEBENGINE_RESOURCES_PATH"] == str(resources_dir)
    assert os.environ["QTWEBENGINE_LOCALES_PATH"] == str(locales_dir)
    assert "--no-sandbox" in os.environ["QTWEBENGINE_CHROMIUM_FLAGS"]


def test_bare_binary_helper_copies_executable(monkeypatch, tmp_path) -> None:
    """Verify _setup_bare_binary_helper copies the binary and resources."""
    qt_lib_dir = tmp_path / "qt_lib"
    webengine_core = qt_lib_dir / "QtWebEngineCore.framework"
    src_resources = webengine_core / "Resources"
    src_locales = src_resources / "qtwebengine_locales"
    src_executable = (
        webengine_core
        / "Helpers"
        / "QtWebEngineProcess.app"
        / "Contents"
        / "MacOS"
        / "QtWebEngineProcess"
    )

    # Set up source tree
    src_locales.mkdir(parents=True)
    src_executable.parent.mkdir(parents=True)
    (src_resources / "icudtl.dat").write_bytes(b"icu-data")
    (src_resources / "qtwebengine_resources.pak").write_bytes(b"pak-data")
    (src_locales / "en-US.pak").write_bytes(b"en")
    src_executable.write_bytes(b"binary-content")

    # Create a minimal framework for copying
    fw_dir = qt_lib_dir / "QtCore.framework"
    fw_dir.mkdir(parents=True)
    (fw_dir / "QtCore").write_bytes(b"fw")

    # Redirect cache to tmp_path and stub out subprocess calls
    monkeypatch.setattr(
        macos_fixes,
        "_setup_bare_binary_helper",
        macos_fixes._setup_bare_binary_helper.__wrapped__
        if hasattr(macos_fixes._setup_bare_binary_helper, "__wrapped__")
        else macos_fixes._setup_bare_binary_helper,
    )

    # Monkey-patch Path.home to redirect cache
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "fakehome")
    (tmp_path / "fakehome" / "Library" / "Caches" / "Annolid").mkdir(
        parents=True, exist_ok=True
    )

    # Stub subprocess.run so we don't actually call install_name_tool/codesign
    calls = []
    monkeypatch.setattr(
        macos_fixes.subprocess,
        "run",
        lambda cmd, **kw: calls.append(cmd) or type("R", (), {"returncode": 0})(),
    )

    monkeypatch.delenv("QTWEBENGINEPROCESS_PATH", raising=False)

    macos_fixes._setup_bare_binary_helper(qt_lib_dir)

    expected_cache = (
        tmp_path / "fakehome" / "Library" / "Caches" / "Annolid" / "QtWebEngine"
    )
    target_bin = expected_cache / "QtWebEngineProcess"
    assert target_bin.exists()
    assert target_bin.read_bytes() == b"binary-content"
    assert (expected_cache / "icudtl.dat").exists()
    assert os.environ["QTWEBENGINEPROCESS_PATH"] == str(target_bin)
    # Verify subprocess was called (rpath patch + codesign)
    assert len(calls) >= 2
