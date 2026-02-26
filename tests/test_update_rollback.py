from __future__ import annotations

from annolid.core.agent.update_manager.rollback import (
    build_rollback_plan,
    execute_rollback,
)


def test_update_rollback_package_dry_run_plan() -> None:
    plan = build_rollback_plan(
        install_mode="package",
        project="annolid",
        previous_version="1.2.3",
    )
    assert plan.manual_required is False
    payload = execute_rollback(plan, execute=False)
    assert payload["ok"] is True
    assert payload["executed"] is False
    assert payload["steps"][0]["ok"] is True


def test_update_rollback_manual_required_for_source_execute() -> None:
    plan = build_rollback_plan(
        install_mode="source",
        project="annolid",
        previous_version="1.2.3",
    )
    payload = execute_rollback(plan, execute=True)
    assert payload["ok"] is False
    assert payload["reason"] == "manual_required"


def test_update_rollback_execute_failure(monkeypatch) -> None:
    import annolid.core.agent.update_manager.rollback as rollback_mod

    class _Proc:
        returncode = 2
        stdout = ""
        stderr = "failure"

    monkeypatch.setattr(rollback_mod.subprocess, "run", lambda *a, **k: _Proc())
    plan = build_rollback_plan(
        install_mode="package",
        project="annolid",
        previous_version="1.2.3",
    )
    payload = execute_rollback(plan, execute=True)
    assert payload["ok"] is False
    assert payload["reason"] == "rollback_command_failed"
