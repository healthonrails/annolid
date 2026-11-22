from __future__ import annotations

import ast
import sys
import types
from pathlib import Path

import pytest

from annolid.gui import cli as gui_cli
from annolid.infrastructure import runtime as infra_runtime


def _load_spec_namespace(spec_path: Path) -> dict[str, object]:
    """Evaluate annolid.spec with stubbed PyInstaller callables to inspect config."""
    captured: dict[str, object] = {}

    class _StubObj:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self.binaries = []
            self.datas = []
            self.pure = []
            self.zipped_data = []
            self.scripts = []
            self.zipfiles = []

    def _analysis(*args, **kwargs):
        obj = _StubObj(*args, **kwargs)
        captured["analysis_kwargs"] = kwargs
        return obj

    def _bundle(*args, **kwargs):
        obj = _StubObj(*args, **kwargs)
        captured["bundle_kwargs"] = kwargs
        return obj

    fake_hooks = types.ModuleType("PyInstaller.utils.hooks")
    fake_hooks.copy_metadata = lambda name: [(f"{name}.dist-info", ".")]  # type: ignore[attr-defined]
    fake_utils = types.ModuleType("PyInstaller.utils")
    fake_utils.hooks = fake_hooks  # type: ignore[attr-defined]
    fake_pyi = types.ModuleType("PyInstaller")
    fake_pyi.utils = fake_utils  # type: ignore[attr-defined]

    saved_modules = {
        name: sys.modules.get(name)
        for name in ("PyInstaller", "PyInstaller.utils", "PyInstaller.utils.hooks")
    }
    sys.modules["PyInstaller"] = fake_pyi
    sys.modules["PyInstaller.utils"] = fake_utils
    sys.modules["PyInstaller.utils.hooks"] = fake_hooks

    namespace: dict[str, object] = {
        "SPECPATH": str(spec_path.resolve().parent),
        "Analysis": _analysis,
        "PYZ": _StubObj,
        "EXE": _StubObj,
        "COLLECT": _StubObj,
        "BUNDLE": _bundle,
    }
    try:
        exec(
            compile(spec_path.read_text(encoding="utf-8"), str(spec_path), "exec"),
            namespace,
        )
    finally:
        for name, original in saved_modules.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original

    namespace["_captured"] = captured
    return namespace


def _analysis_hiddenimports(spec_path: Path) -> set[str]:
    ns = _load_spec_namespace(spec_path)
    captured = ns["_captured"]
    assert isinstance(captured, dict)
    analysis_kwargs = captured["analysis_kwargs"]
    assert isinstance(analysis_kwargs, dict)
    hiddenimports = analysis_kwargs["hiddenimports"]
    assert isinstance(hiddenimports, list)
    return {str(item) for item in hiddenimports}


def _lazy_exported_modules(init_path: Path) -> set[str]:
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("annolid.")
        ):
            modules.add(node.value)
    return modules


def test_pyinstaller_spec_includes_launcher_dynamic_gui_import() -> None:
    hiddenimports = _analysis_hiddenimports(Path("annolid.spec"))

    assert "annolid.gui.app" in hiddenimports


def test_pyinstaller_spec_includes_gui_startup_lazy_namespace_imports() -> None:
    hiddenimports = _analysis_hiddenimports(Path("annolid.spec"))

    assert "annolid.domain.project_schema" in hiddenimports
    assert "annolid.domain.behavior_events" in hiddenimports
    assert "annolid.infrastructure.persistence" in hiddenimports


def test_pyinstaller_spec_covers_all_first_party_lazy_export_modules() -> None:
    hiddenimports = _analysis_hiddenimports(Path("annolid.spec"))

    lazy_entrypoints = [
        Path("annolid/domain/__init__.py"),
        Path("annolid/domain/datasets.py"),
        Path("annolid/infrastructure/__init__.py"),
        Path("annolid/gui/widgets/__init__.py"),
        Path("annolid/gui/controllers/__init__.py"),
        Path("annolid/services/__init__.py"),
        Path("annolid/services/behavior_agent/__init__.py"),
        Path("annolid/postprocessing/__init__.py"),
        Path("annolid/features/__init__.py"),
        Path("annolid/interfaces/background/__init__.py"),
        Path("annolid/interfaces/bot/__init__.py"),
        Path("annolid/interfaces/cli/__init__.py"),
        Path("annolid/interfaces/gui/__init__.py"),
    ]
    missing: dict[str, list[str]] = {}
    for entry in lazy_entrypoints:
        expected = _lazy_exported_modules(entry)
        absent = sorted(expected - hiddenimports)
        if absent:
            missing[str(entry)] = absent

    assert not missing, f"Lazy-exported modules missing from annolid.spec: {missing}"


def test_pyinstaller_spec_macos_bundle_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    ns = _load_spec_namespace(Path("annolid.spec"))
    captured = ns["_captured"]
    assert isinstance(captured, dict)
    bundle_kwargs = captured["bundle_kwargs"]
    assert isinstance(bundle_kwargs, dict)
    assert bundle_kwargs["name"] == "Annolid.app"
    assert bundle_kwargs["bundle_identifier"] == "com.healthonrails.annolid"
    info_plist = bundle_kwargs["info_plist"]
    assert isinstance(info_plist, dict)
    assert info_plist["NSHighResolutionCapable"] is True
    assert info_plist["CFBundleShortVersionString"] != "0.0.0"
    assert info_plist["CFBundleVersion"] == info_plist["CFBundleShortVersionString"]


def test_parse_cli_strips_macos_launch_services_psn_arg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_file = tmp_path / "config.yaml"
    config_file.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(sys, "platform", "darwin")

    cfg, _namespace, version_requested = gui_cli.parse_cli(
        ["-psn_0_123456", "--config", str(config_file)]
    )

    assert version_requested is False
    assert isinstance(cfg, dict)


def test_parse_cli_version_does_not_load_user_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_args, **_kwargs):
        raise AssertionError("get_config should not be called for --version")

    monkeypatch.setattr(gui_cli, "get_config", _boom)

    cfg, _namespace, version_requested = gui_cli.parse_cli(["--version"])

    assert version_requested is True
    assert cfg == {}


def test_macos_stderr_filter_handles_none_stderr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(sys, "stderr", None)

    infra_runtime._configure_macos_stderr_filter()

    assert sys.stderr is None
