from __future__ import annotations

import asyncio
from pathlib import Path

from annolid.core.agent.heartbeat import HeartbeatService, is_heartbeat_empty


def test_is_heartbeat_empty_rules() -> None:
    assert is_heartbeat_empty(None) is True
    assert is_heartbeat_empty("") is True
    assert is_heartbeat_empty("# Title\n\n- [ ]\n<!-- note -->\n") is True
    assert is_heartbeat_empty("Run maintenance check") is False


def test_heartbeat_trigger_idle_without_file(tmp_path: Path) -> None:
    svc = HeartbeatService(workspace=tmp_path, enabled=True)
    result = asyncio.run(svc.trigger_now())
    assert result.status == "idle"


def test_heartbeat_trigger_skipped_without_handler(tmp_path: Path) -> None:
    (tmp_path / "HEARTBEAT.md").write_text("Do task A\n", encoding="utf-8")
    svc = HeartbeatService(workspace=tmp_path, enabled=True)
    result = asyncio.run(svc.trigger_now())
    assert result.status == "skipped"


def test_heartbeat_trigger_ok_and_action(tmp_path: Path) -> None:
    (tmp_path / "HEARTBEAT.md").write_text("Do task A\n", encoding="utf-8")

    async def _ok_handler(prompt: str) -> str:
        assert "HEARTBEAT_OK" in prompt
        return "heartbeat ok"

    svc_ok = HeartbeatService(
        workspace=tmp_path,
        on_heartbeat=_ok_handler,
        enabled=True,
    )
    ok_result = asyncio.run(svc_ok.trigger_now())
    assert ok_result.status == "ok"

    async def _action_handler(prompt: str) -> str:
        del prompt
        return "Completed follow-up task."

    svc_action = HeartbeatService(
        workspace=tmp_path,
        on_heartbeat=_action_handler,
        enabled=True,
    )
    action_result = asyncio.run(svc_action.trigger_now())
    assert action_result.status == "action"


def test_heartbeat_timeout(tmp_path: Path) -> None:
    (tmp_path / "HEARTBEAT.md").write_text("Do task A\n", encoding="utf-8")

    async def _slow_handler(prompt: str) -> str:
        del prompt
        await asyncio.sleep(0.2)
        return "done"

    svc = HeartbeatService(
        workspace=tmp_path,
        on_heartbeat=_slow_handler,
        enabled=True,
        timeout_s=0.05,
    )
    result = asyncio.run(svc.trigger_now())
    assert result.status == "timeout"
