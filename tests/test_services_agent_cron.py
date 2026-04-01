from __future__ import annotations

from pathlib import Path

from annolid.services.agent_cron import (
    add_agent_cron_job,
    get_agent_status,
    list_agent_cron_jobs,
    onboard_agent_workspace,
    remove_agent_cron_job,
    restore_agent_workspace_backup,
    run_agent_cron_job,
    set_agent_cron_job_enabled,
)


def test_onboard_agent_workspace_and_status(monkeypatch, tmp_path: Path) -> None:
    import annolid.core.agent as agent_mod
    import annolid.core.agent.utils as utils_mod
    import annolid.services.agent_cron as cron_mod

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    store_path = tmp_path / "cron.json"

    monkeypatch.setattr(
        utils_mod, "get_agent_workspace_path", lambda value=None: workspace
    )
    monkeypatch.setattr(utils_mod, "get_agent_data_path", lambda: data_dir)
    monkeypatch.setattr(
        agent_mod,
        "bootstrap_workspace",
        lambda root, overwrite=False, dry_run=False, backup_root=None: {
            "AGENTS.md": "created"
        },
    )
    monkeypatch.setattr(
        agent_mod,
        "prune_bootstrap_workspace",
        lambda root, dry_run=False, backup_root=None: {"legacy/OLD.md": "removed"},
    )
    monkeypatch.setattr(cron_mod, "_default_agent_cron_store_path", lambda: store_path)

    class _Service:
        def status(self):
            return {"jobs": 1}

    monkeypatch.setattr(cron_mod, "_agent_cron_service", lambda: _Service())

    onboard = onboard_agent_workspace(
        workspace=str(workspace), overwrite=True, prune_bootstrap=True
    )
    status = get_agent_status()

    assert onboard["files"]["AGENTS.md"] == "created"
    assert onboard["dry_run"] is False
    assert onboard["backup_enabled"] is True
    assert onboard["prune_bootstrap"] is True
    assert onboard["pruned_files"]["legacy/OLD.md"] == "removed"
    assert isinstance(onboard["summary"], dict)
    assert status["data_dir"] == str(data_dir)
    assert status["cron_store_path"] == str(store_path)
    assert status["cron"] == {"jobs": 1}


def test_list_add_remove_enable_and_run_cron(monkeypatch) -> None:
    import annolid.core.agent.cron as cron_pkg
    import annolid.services.agent_cron as cron_mod

    class _Schedule:
        def __init__(self, **kwargs):
            self.kind = kwargs.get("kind")
            self.at_ms = kwargs.get("at_ms")
            self.every_ms = kwargs.get("every_ms")
            self.expr = kwargs.get("expr")
            self.tz = kwargs.get("tz")

    class _Payload:
        def __init__(self, **kwargs):
            self.kind = kwargs.get("kind")
            self.message = kwargs.get("message")
            self.deliver = kwargs.get("deliver")
            self.channel = kwargs.get("channel")
            self.to = kwargs.get("to")

    monkeypatch.setattr(cron_pkg, "CronSchedule", _Schedule)
    monkeypatch.setattr(cron_pkg, "CronPayload", _Payload)

    class _Job:
        def __init__(self):
            self.id = "job-1"
            self.name = "nightly"
            self.enabled = True
            self.schedule = _Schedule(
                kind="every", every_ms=5000, at_ms=None, expr=None, tz=None
            )
            self.payload = _Payload(
                kind="agent_turn", message="hello", deliver=False, channel=None, to=None
            )
            self.state = type(
                "State",
                (),
                {
                    "next_run_at_ms": 10,
                    "last_run_at_ms": 5,
                    "last_status": "ok",
                    "last_error": "",
                },
            )()

    class _Service:
        def list_jobs(self, include_disabled=False):
            return [_Job()]

        def add_job(self, **kwargs):
            return _Job()

        def remove_job(self, job_id):
            return job_id == "job-1"

        def enable_job(self, job_id, enabled=True):
            if job_id != "job-1":
                return None
            job = _Job()
            job.enabled = enabled
            return job

        async def run_job(self, job_id, force=False):
            return job_id == "job-1" and force is True

    monkeypatch.setattr(cron_mod, "_agent_cron_service", lambda: _Service())

    rows = list_agent_cron_jobs(include_all=True)
    created = add_agent_cron_job(name="nightly", message="hello", every=5)
    removed, remove_code = remove_agent_cron_job(job_id="job-1")
    enabled, enable_code = set_agent_cron_job_enabled(job_id="job-1", enabled=False)
    ran, run_code = run_agent_cron_job(job_id="job-1", force=True)

    assert rows[0]["id"] == "job-1"
    assert created["name"] == "nightly"
    assert remove_code == 0 and removed["removed"] is True
    assert enable_code == 0 and enabled["enabled"] is False
    assert run_code == 0 and ran["ran"] is True


def test_restore_agent_workspace_backup_latest(tmp_path: Path, monkeypatch) -> None:
    import annolid.core.agent.utils as utils_mod

    workspace = tmp_path / "workspace"
    backup = workspace / ".annolid" / "bootstrap-backups" / "20260101-000000"
    (backup / "AGENTS.md").parent.mkdir(parents=True, exist_ok=True)
    (backup / "AGENTS.md").write_text("backup agents", encoding="utf-8")
    (workspace / "AGENTS.md").parent.mkdir(parents=True, exist_ok=True)
    (workspace / "AGENTS.md").write_text("current agents", encoding="utf-8")
    monkeypatch.setattr(
        utils_mod, "get_agent_workspace_path", lambda value=None: workspace
    )

    payload = restore_agent_workspace_backup(workspace=str(workspace), latest=True)
    assert payload["restored"] is True
    assert payload["files"]["AGENTS.md"] == "overwritten"
    assert (workspace / "AGENTS.md").read_text(encoding="utf-8") == "backup agents"
    pre_backup = Path(str(payload["pre_restore_backup_dir"]))
    assert (pre_backup / "AGENTS.md").read_text(encoding="utf-8") == "current agents"


def test_restore_agent_workspace_backup_when_missing(
    tmp_path: Path, monkeypatch
) -> None:
    import annolid.core.agent.utils as utils_mod

    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        utils_mod, "get_agent_workspace_path", lambda value=None: workspace
    )

    payload = restore_agent_workspace_backup(workspace=str(workspace), latest=True)
    assert payload["restored"] is False
    assert payload["backup_dir"] is None
