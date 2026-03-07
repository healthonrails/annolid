from __future__ import annotations

import json
from pathlib import Path

from annolid.services.agent_update import (
    check_for_agent_update,
    rollback_agent_update,
    run_legacy_agent_update,
    run_agent_update,
)


def test_check_for_agent_update_stages_plan(monkeypatch) -> None:
    import annolid.core.agent.update_manager.manager as manager_mod

    class _Plan:
        def to_dict(self):
            return {"status": "staged", "channel": "stable"}

    class _Manager:
        def __init__(self, *, project: str) -> None:
            self.project = project

        def stage(self, *, channel: str, timeout_s: float, require_signature: bool):
            assert channel == "stable"
            assert timeout_s == 4.0
            assert require_signature is True
            return _Plan()

    monkeypatch.setattr(manager_mod, "SignedUpdateManager", _Manager)

    payload = check_for_agent_update(require_signature=True)

    assert payload["status"] == "staged"
    assert payload["channel"] == "stable"


def test_run_agent_update_reads_metrics_and_maps_exit_code(
    monkeypatch, tmp_path: Path
) -> None:
    import annolid.core.agent.update_manager.canary as canary_mod
    import annolid.core.agent.update_manager.manager as manager_mod

    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"sample_count": 20}), encoding="utf-8")
    captured: dict[str, object] = {}

    class _Plan:
        pass

    class _Manager:
        def __init__(self, *, project: str) -> None:
            captured["project"] = project

        def stage(self, *, channel: str, timeout_s: float, require_signature: bool):
            captured["stage"] = {
                "channel": channel,
                "timeout_s": timeout_s,
                "require_signature": require_signature,
            }
            return _Plan()

        def run(
            self,
            plan,
            *,
            execute: bool,
            run_post_check: bool,
            canary_metrics,
            canary_policy,
        ):
            captured["run"] = {
                "execute": execute,
                "run_post_check": run_post_check,
                "canary_metrics": canary_metrics,
                "canary_policy": canary_policy,
            }
            return {"status": "updated"}

    monkeypatch.setattr(manager_mod, "SignedUpdateManager", _Manager)
    monkeypatch.setattr(canary_mod, "CanaryPolicy", lambda **kwargs: {"policy": kwargs})

    payload, exit_code = run_agent_update(
        canary_metrics=metrics_path,
        execute=True,
        skip_post_check=True,
        canary_min_samples=5,
        canary_max_failure_rate=0.1,
        canary_max_regressions=2,
    )

    assert exit_code == 0
    assert payload["status"] == "updated"
    assert captured["run"]["execute"] is True
    assert captured["run"]["run_post_check"] is False
    assert captured["run"]["canary_metrics"]["sample_count"] == 20
    assert captured["run"]["canary_policy"]["policy"]["max_regressions"] == 2


def test_rollback_agent_update_maps_failure(monkeypatch) -> None:
    import annolid.core.agent.update_manager.rollback as rollback_mod

    monkeypatch.setattr(
        rollback_mod,
        "build_rollback_plan",
        lambda **kwargs: {"plan": kwargs},
    )
    monkeypatch.setattr(
        rollback_mod,
        "execute_rollback",
        lambda plan, *, execute: {"ok": False, "executed": execute, "plan": plan},
    )

    payload, exit_code = rollback_agent_update(
        install_mode="package",
        project="annolid",
        previous_version="1.2.3",
        execute=True,
    )

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["executed"] is True


def test_run_legacy_agent_update_maps_failed_steps(monkeypatch) -> None:
    import annolid.core.agent.updater as updater_mod

    class _Report:
        def to_dict(self):
            return {"status": "update_available"}

    monkeypatch.setattr(updater_mod, "check_for_updates", lambda **kwargs: _Report())
    monkeypatch.setattr(
        updater_mod,
        "apply_update",
        lambda report, *, execute, run_doctor: {
            "status": "update_available",
            "steps": [{"ok": False}],
        },
    )

    payload, exit_code = run_legacy_agent_update(
        apply=True,
        execute=True,
        skip_doctor=True,
        require_signature=True,
    )

    assert exit_code == 1
    assert payload["steps"][0]["ok"] is False
