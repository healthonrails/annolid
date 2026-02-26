from __future__ import annotations

import asyncio
import json
from pathlib import Path

from annolid.core.agent.tools.function_admin import (
    AdminEvalRunTool,
    AdminMemoryFlushTool,
    AdminSkillsRefreshTool,
    AdminUpdateRunTool,
)


def test_admin_skills_refresh_tool(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills" / "local_test_skill"
    skills_dir.mkdir(parents=True, exist_ok=True)
    (skills_dir / "SKILL.md").write_text(
        "---\ndescription: local test\n---\n\ncontent\n",
        encoding="utf-8",
    )

    tool = AdminSkillsRefreshTool()
    payload = json.loads(asyncio.run(tool.execute(workspace=str(tmp_path))))
    assert payload["ok"] is True
    assert "local_test_skill" in payload["names"]


def test_admin_memory_flush_tool(tmp_path: Path) -> None:
    tool = AdminMemoryFlushTool()
    payload = json.loads(
        asyncio.run(
            tool.execute(
                workspace=str(tmp_path),
                session_id="s1",
                note="manual flush",
            )
        )
    )
    assert payload["ok"] is True
    assert "manual flush" in payload["entry"]
    assert Path(payload["today_file"]).exists()
    assert Path(payload["history_file"]).exists()


def test_admin_eval_run_tool(tmp_path: Path) -> None:
    traces = tmp_path / "traces.jsonl"
    traces.write_text(
        json.dumps(
            {
                "trace_id": "t1",
                "user_message": "hello",
                "expected_substring": "world",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    candidate = tmp_path / "candidate.jsonl"
    candidate.write_text(
        json.dumps({"trace_id": "t1", "content": "hello world"}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "report.json"
    tool = AdminEvalRunTool()
    payload = json.loads(
        asyncio.run(
            tool.execute(
                traces_path=str(traces),
                candidate_responses_path=str(candidate),
                output_path=str(out),
                max_regressions=0,
                min_pass_rate=0.5,
            )
        )
    )
    assert payload["ok"] is True
    assert Path(payload["output_path"]).exists()
    assert float(payload["gate"]["pass_rate"]) >= 1.0


def test_admin_update_run_tool_dry_run(monkeypatch) -> None:
    import annolid.core.agent.tools.function_admin as admin_mod

    class _FakeService:
        def __init__(self, project: str = "annolid") -> None:
            self.project = project

        def run_transaction(
            self,
            *,
            channel: str = "stable",
            timeout_s: float = 4.0,
            require_signature: bool = False,
            execute: bool = False,
            run_post_check: bool = True,
        ):
            return {
                "status": "staged",
                "channel": channel,
                "executed": execute,
                "steps": [
                    {
                        "phase": "preflight",
                        "ok": True,
                        "require_signature": require_signature,
                        "timeout_s": timeout_s,
                        "run_post_check": run_post_check,
                    }
                ],
            }

        @property
        def manager(self):
            class _Manager:
                @staticmethod
                def preflight():
                    return {"install_mode": "package"}

            return _Manager()

    monkeypatch.setattr(admin_mod, "UpdateManagerService", _FakeService)
    tool = AdminUpdateRunTool(project="annolid")
    payload = json.loads(
        asyncio.run(
            tool.execute(
                channel="stable",
                execute=False,
                run_post_check=False,
            )
        )
    )
    assert payload["status"] == "staged"
    assert payload["ok"] is True


def test_admin_update_run_tool_blocks_execute_without_consent(monkeypatch) -> None:
    import annolid.core.agent.tools.function_admin as admin_mod

    class _FakeService:
        def __init__(self, project: str = "annolid") -> None:
            self.project = project

        def run_transaction(self, **_kwargs):
            raise AssertionError("run_transaction should not be called without consent")

    monkeypatch.setattr(admin_mod, "UpdateManagerService", _FakeService)
    monkeypatch.setenv("ANNOLID_BOT_UPDATE_REQUIRE_CONSENT", "1")
    tool = AdminUpdateRunTool(project="annolid")
    payload = json.loads(asyncio.run(tool.execute(channel="stable", execute=True)))
    assert payload["ok"] is False
    assert payload["reason"] == "operator_consent_required"


def test_admin_update_run_tool_execute_with_consent(monkeypatch) -> None:
    import annolid.core.agent.tools.function_admin as admin_mod

    class _FakeService:
        def __init__(self, project: str = "annolid") -> None:
            self.project = project

        def run_transaction(self, **_kwargs):
            return {"status": "updated", "steps": []}

    monkeypatch.setattr(admin_mod, "UpdateManagerService", _FakeService)
    monkeypatch.setenv("ANNOLID_BOT_UPDATE_REQUIRE_CONSENT", "1")
    monkeypatch.setenv("ANNOLID_OPERATOR_UPDATE_CONSENT_PHRASE", "YES_UPDATE")
    tool = AdminUpdateRunTool(project="annolid")
    payload = json.loads(
        asyncio.run(
            tool.execute(
                channel="stable",
                execute=True,
                operator_consent="YES_UPDATE",
            )
        )
    )
    assert payload["ok"] is True
    assert payload["status"] == "updated"
