from __future__ import annotations

import importlib.util
import os

import pytest


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _simulation_skip_reason() -> str | None:
    if os.environ.get("ANNOLID_RUN_FLYBODY_RUNTIME") != "1":
        return (
            "set ANNOLID_RUN_FLYBODY_RUNTIME=1 to run optional simulation runtime tests"
        )

    required_modules = ("flybody", "dm_control", "mujoco")
    missing = [name for name in required_modules if not _has_module(name)]
    if missing:
        missing_text = ", ".join(missing)
        return (
            f"optional FlyBody runtime dependencies are not installed: {missing_text}"
        )

    return None


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    del config
    reason = _simulation_skip_reason()
    if reason is None:
        return

    skip_marker = pytest.mark.skip(reason=reason)
    for item in items:
        if item.get_closest_marker("simulation") is not None:
            item.add_marker(skip_marker)
