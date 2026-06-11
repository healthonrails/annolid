from __future__ import annotations

import subprocess
import sys

import pytest

from annolid.segmentation.cutie_vos.dependencies import (
    AUTO_INSTALL_ENV,
    RuntimeDependency,
    ensure_cutie_runtime_dependencies,
    missing_cutie_runtime_dependencies,
)


def test_missing_cutie_runtime_dependencies_reports_missing_imports() -> None:
    dependencies = (
        RuntimeDependency("present_pkg", "present-package"),
        RuntimeDependency("missing_pkg", "missing-package"),
    )

    missing = missing_cutie_runtime_dependencies(
        dependencies=dependencies,
        finder=lambda name: object() if name == "present_pkg" else None,
    )

    assert missing == [RuntimeDependency("missing_pkg", "missing-package")]


def test_ensure_cutie_runtime_dependencies_noops_when_present() -> None:
    installed = ensure_cutie_runtime_dependencies(
        finder=lambda _name: object(),
        runner=lambda *args, **kwargs: pytest.fail("pip should not run"),
    )

    assert installed == []


def test_ensure_cutie_runtime_dependencies_installs_missing_packages() -> None:
    calls = []
    available = {"present_pkg"}

    def _finder(name: str):
        return object() if name in available else None

    def _runner(cmd, **kwargs):
        calls.append((cmd, kwargs))
        available.add("missing_pkg")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    installed = ensure_cutie_runtime_dependencies(
        finder=_finder,
        runner=_runner,
        dependencies=(
            RuntimeDependency("present_pkg", "present-package"),
            RuntimeDependency("missing_pkg", "missing-package>=1"),
        ),
    )

    assert installed == ["missing-package>=1"]
    assert calls
    assert calls[0][0][-1] == "missing-package>=1"
    assert calls[0][1]["capture_output"] is True


def test_cutie_runtime_install_command_prefers_uv_for_active_python(
    monkeypatch,
) -> None:
    calls = []
    available = set()

    monkeypatch.setattr(
        "annolid.segmentation.cutie_vos.dependencies.shutil.which",
        lambda name: "/usr/bin/uv" if name == "uv" else None,
    )

    def _finder(name: str):
        return object() if name in available else None

    def _runner(cmd, **kwargs):
        calls.append(cmd)
        available.add("missing_pkg")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    ensure_cutie_runtime_dependencies(
        finder=_finder,
        runner=_runner,
        dependencies=(RuntimeDependency("missing_pkg", "missing-package>=1"),),
    )

    assert calls
    assert calls[0][:5] == ["/usr/bin/uv", "pip", "install", "--python", sys.executable]
    assert calls[0][-1] == "missing-package>=1"


def test_ensure_cutie_runtime_dependencies_can_disable_auto_install(
    monkeypatch,
) -> None:
    monkeypatch.setenv(AUTO_INSTALL_ENV, "0")

    with pytest.raises(RuntimeError) as exc_info:
        ensure_cutie_runtime_dependencies(
            finder=lambda _name: None,
            dependencies=(RuntimeDependency("missing_pkg", "missing-package"),),
        )

    assert "missing_pkg" in str(exc_info.value)
    assert "python" in str(exc_info.value)
    assert "missing-package" in str(exc_info.value)
