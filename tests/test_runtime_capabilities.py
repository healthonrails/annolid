from __future__ import annotations

import subprocess
import sys

import pytest

from annolid.infrastructure.capabilities import (
    AUTO_INSTALL_OPTIONAL_DEPS_ENV,
    capability_message,
    check_capability,
    ensure_capability,
    format_capability_report,
    list_capabilities,
)


def test_capability_available_when_all_imports_resolve() -> None:
    status = check_capability("cutie", find_spec=lambda name: object())

    assert status.available is True
    assert status.state == "available"
    assert status.missing_imports == ()
    assert "annolid[cutie]" in status.install_command


def test_capability_reports_missing_imports_and_install_hint() -> None:
    missing = {"torch", "hydra"}
    status = check_capability(
        "cutie", find_spec=lambda name: None if name in missing else object()
    )

    assert status.available is False
    assert status.state == "missing"
    assert status.missing_imports == ("torch", "hydra")
    message = capability_message(status)
    assert "torch, hydra" in message
    assert "python -m pip install" in message
    assert "annolid[cutie]" in message


def test_format_capability_report_is_deterministic_for_selected_names() -> None:
    report = format_capability_report(("sam3", "yolo"), find_spec=lambda name: object())

    assert report.splitlines() == ["sam3: available", "yolo: available"]


def test_unknown_capability_lists_known_names() -> None:
    with pytest.raises(KeyError, match="Known capabilities"):
        check_capability("missing-feature", find_spec=lambda name: object())


def test_list_capabilities_exposes_professional_install_tiers() -> None:
    capabilities = set(list_capabilities())

    assert {"ml", "tracking", "cutie", "sam3", "yolo", "bot"}.issubset(capabilities)


def test_ensure_capability_installs_missing_runtime_packages(monkeypatch) -> None:
    calls = []
    available = {"lap"}

    monkeypatch.setattr(
        "annolid.infrastructure.capabilities.shutil.which",
        lambda name: "/usr/bin/uv" if name == "uv" else None,
    )

    def _finder(name: str):
        return object() if name in available else None

    def _runner(cmd, **kwargs):
        calls.append((cmd, kwargs))
        available.add("ultralytics")
        return subprocess.CompletedProcess(cmd, returncode=0, stdout="", stderr="")

    installed = ensure_capability("yolo", find_spec=_finder, runner=_runner)

    assert installed == ("ultralytics>=8.4.0",)
    assert calls
    assert calls[0][0][:5] == [
        "/usr/bin/uv",
        "pip",
        "install",
        "--python",
        sys.executable,
    ]
    assert calls[0][1]["capture_output"] is True


def test_ensure_capability_can_disable_auto_install(monkeypatch) -> None:
    monkeypatch.setenv(AUTO_INSTALL_OPTIONAL_DEPS_ENV, "0")

    with pytest.raises(RuntimeError) as exc_info:
        ensure_capability("sam3", find_spec=lambda _name: None)

    message = str(exc_info.value)
    assert "sam3: missing optional runtime packages" in message
    assert "annolid[sam3]" in message


def test_ensure_capability_does_not_mutate_frozen_bundle(monkeypatch) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)

    with pytest.raises(RuntimeError) as exc_info:
        ensure_capability("yolo", find_spec=lambda _name: None)

    assert "Frozen desktop bundles are read-only" in str(exc_info.value)
