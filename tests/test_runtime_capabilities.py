from __future__ import annotations

import pytest

from annolid.infrastructure.capabilities import (
    capability_message,
    check_capability,
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
