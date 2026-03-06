from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path

from annolid.core.agent.acp_stdio_bridge import ACPStdioBridge
from annolid.core.agent.coding_harness import CodingHarnessManager
from annolid.core.agent.session_manager import AgentSessionManager
from annolid.engine import cli


def test_acp_stdio_bridge_initialize_and_shutdown_over_stdio(tmp_path: Path) -> None:
    input_stream = io.StringIO(
        "\n".join(
            [
                json.dumps(
                    {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
                ),
                json.dumps(
                    {"jsonrpc": "2.0", "id": 2, "method": "shutdown", "params": {}}
                ),
            ]
        )
        + "\n"
    )
    output_stream = io.StringIO()
    bridge = ACPStdioBridge(
        workspace=tmp_path,
        input_stream=input_stream,
        output_stream=output_stream,
    )

    rc = asyncio.run(bridge.serve())

    assert rc == 0
    rows = [
        json.loads(line)
        for line in output_stream.getvalue().splitlines()
        if line.strip()
    ]
    assert rows[0]["id"] == 1
    assert rows[0]["result"]["protocol"] == "annolid.acp.stdio"
    assert rows[0]["result"]["runtime"] == "acp"
    assert rows[1]["id"] == 2
    assert rows[1]["result"]["shutdown"] is True


def test_acp_stdio_bridge_session_lifecycle_emits_notification(
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "sessions"
    session_manager = AgentSessionManager(sessions_dir=sessions_dir)
    output_stream = io.StringIO()

    def _invoke_turn(**kwargs):
        prompt = str(kwargs.get("prompt") or "")
        runtime = str(kwargs.get("runtime") or "")
        assert runtime == "acp"
        return prompt, f"completed:{prompt}"

    manager = CodingHarnessManager(
        session_manager=session_manager,
        invoke_turn=_invoke_turn,
    )
    bridge = ACPStdioBridge(
        manager=manager,
        workspace=tmp_path,
        input_stream=io.StringIO(),
        output_stream=output_stream,
    )

    async def _run() -> None:
        previous = manager.set_announce_callback(bridge._handle_session_announcement)
        try:
            spawn = await bridge.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 10,
                    "method": "sessions.spawn",
                    "params": {"task": "review repo", "label": "Repo Review"},
                }
            )
            assert spawn is not None
            session_id = spawn["result"]["session_id"]
            meta = manager.get_session(session_id)
            assert meta is not None
            assert meta.workspace == str(tmp_path.resolve())

            for _ in range(20):
                if meta.turn_count >= 1 and meta.status == "idle":
                    break
                await asyncio.sleep(0.05)
            assert meta.turn_count == 1
            assert meta.status == "idle"

            poll = await bridge.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 11,
                    "method": "sessions.poll",
                    "params": {"session_id": session_id},
                }
            )
            assert poll is not None
            assert poll["result"]["ok"] is True
            assert poll["result"]["last_response"] == "completed:review repo"

            listed = await bridge.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 12,
                    "method": "sessions.list",
                    "params": {},
                }
            )
            assert listed is not None
            assert listed["result"]["sessions"][0]["session_id"] == session_id

            closed = await bridge.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 13,
                    "method": "sessions.close",
                    "params": {"session_id": session_id},
                }
            )
            assert closed is not None
            assert closed["result"]["closed"] is True
            assert meta.worker_task is not None
            await asyncio.wait_for(meta.worker_task, timeout=1.0)
        finally:
            manager.set_announce_callback(previous)

    asyncio.run(_run())

    lines = [
        json.loads(line)
        for line in output_stream.getvalue().splitlines()
        if line.strip()
    ]
    notifications = [row for row in lines if row.get("method") == "sessions.updated"]
    assert notifications
    assert notifications[-1]["params"]["status"] == "idle"
    assert notifications[-1]["params"]["text"] == "completed:review repo"


def test_agent_acp_bridge_operator_command(monkeypatch, tmp_path: Path) -> None:
    called = {}

    def _fake_run_stdio_acp_bridge(*, workspace=None) -> int:
        called["workspace"] = workspace
        return 0

    monkeypatch.setattr(
        "annolid.core.agent.acp_stdio_bridge.run_stdio_acp_bridge",
        _fake_run_stdio_acp_bridge,
    )

    rc = cli.main(["agent", "acp", "bridge", "--workspace", str(tmp_path)])

    assert rc == 0
    assert called["workspace"] == str(tmp_path)


def test_acp_stdio_bridge_supports_openclaw_acp_lifecycle(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    session_manager = AgentSessionManager(sessions_dir=sessions_dir)
    output_stream = io.StringIO()

    def _invoke_turn(**kwargs):
        prompt = str(kwargs.get("prompt") or "")
        return prompt, f"prompt:{prompt}"

    manager = CodingHarnessManager(
        session_manager=session_manager,
        invoke_turn=_invoke_turn,
    )
    bridge = ACPStdioBridge(
        manager=manager,
        workspace=tmp_path,
        input_stream=io.StringIO(),
        output_stream=output_stream,
    )

    async def _run() -> None:
        previous = manager.set_announce_callback(bridge._handle_session_announcement)
        try:
            new_session = await bridge.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 30,
                    "method": "newSession",
                    "params": {
                        "cwd": str(tmp_path),
                        "_meta": {
                            "sessionKey": "acp:test:one",
                            "sessionLabel": "Zed ACP",
                        },
                    },
                }
            )
            assert new_session is not None
            session_id = new_session["result"]["sessionId"]

            listed = await bridge.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 31,
                    "method": "listSessions",
                    "params": {"cwd": str(tmp_path)},
                }
            )
            assert listed is not None
            assert listed["result"]["sessions"][0]["sessionId"] == session_id
            assert (
                listed["result"]["sessions"][0]["_meta"]["sessionKey"] == "acp:test:one"
            )

            prompt_task = asyncio.create_task(
                bridge.handle_request(
                    {
                        "jsonrpc": "2.0",
                        "id": 32,
                        "method": "prompt",
                        "params": {
                            "sessionId": session_id,
                            "prompt": [{"type": "text", "text": "check files"}],
                        },
                    }
                )
            )
            prompt_result = await asyncio.wait_for(prompt_task, timeout=1.0)
            assert prompt_result is not None
            assert prompt_result["result"]["stopReason"] == "end_turn"

            loaded = await bridge.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 33,
                    "method": "loadSession",
                    "params": {
                        "sessionId": session_id,
                        "cwd": str(tmp_path),
                        "_meta": {"sessionKey": "acp:test:loaded"},
                    },
                }
            )
            assert loaded is not None
            assert loaded["result"] == {}
        finally:
            manager.set_announce_callback(previous)

    asyncio.run(_run())

    rows = [
        json.loads(line)
        for line in output_stream.getvalue().splitlines()
        if line.strip()
    ]
    session_updates = [row for row in rows if row.get("method") == "session.update"]
    assert session_updates
    assert (
        session_updates[-1]["params"]["update"]["sessionUpdate"]
        == "agent_message_chunk"
    )
    assert (
        session_updates[-1]["params"]["update"]["content"]["text"]
        == "prompt:check files"
    )


def test_acp_stdio_bridge_accepts_openclaw_style_method_and_field_aliases(
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "sessions"
    session_manager = AgentSessionManager(sessions_dir=sessions_dir)

    def _invoke_turn(**kwargs):
        return str(kwargs.get("prompt") or ""), "alias-ok"

    manager = CodingHarnessManager(
        session_manager=session_manager,
        invoke_turn=_invoke_turn,
    )
    bridge = ACPStdioBridge(
        manager=manager,
        workspace=tmp_path,
        input_stream=io.StringIO(),
        output_stream=io.StringIO(),
    )

    async def _run() -> None:
        spawn = await bridge.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 21,
                "method": "sessions_spawn",
                "params": {
                    "prompt": "implement this",
                    "threadId": "external-thread-1",
                    "channel": "ide",
                },
            }
        )
        assert spawn is not None
        session_id = spawn["result"]["session_id"]
        assert spawn["result"]["sessionId"] == session_id
        assert spawn["result"]["runtime"] == "acp"
        meta = manager.get_session(session_id)
        assert meta is not None
        assert meta.origin_channel == "ide"
        assert meta.origin_chat_id == "external-thread-1"

        for _ in range(20):
            if meta.turn_count >= 1 and meta.status == "idle":
                break
            await asyncio.sleep(0.05)

        poll = await bridge.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 22,
                "method": "sessions_poll",
                "params": {"sessionId": session_id, "tailMessages": 2},
            }
        )
        assert poll is not None
        assert poll["result"]["sessionId"] == session_id
        assert poll["result"]["tail_messages"][-1]["content"] == "alias-ok"

        send = await bridge.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 23,
                "method": "session_send",
                "params": {"id": session_id, "instruction": "follow-up"},
            }
        )
        assert send is not None
        assert send["result"]["queued"] is True

        closed = await bridge.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 24,
                "method": "sessions_close",
                "params": {"thread_id": session_id},
            }
        )
        assert closed is not None
        assert closed["result"]["closed"] is True
        assert meta.worker_task is not None
        await asyncio.wait_for(meta.worker_task, timeout=1.0)

    asyncio.run(_run())


def test_acp_stdio_bridge_cancel_keeps_session_reusable(tmp_path: Path) -> None:
    from threading import Event as ThreadEvent
    import time

    sessions_dir = tmp_path / "sessions"
    session_manager = AgentSessionManager(sessions_dir=sessions_dir)
    started = ThreadEvent()

    def _invoke_turn(**kwargs):
        prompt = str(kwargs.get("prompt") or "")
        cancel_event = kwargs.get("cancel_event")
        if prompt == "slow turn":
            assert cancel_event is not None
            started.set()
            while not cancel_event.is_set():
                time.sleep(0.01)
            raise RuntimeError("Codex CLI request cancelled.")
        return prompt, f"ok:{prompt}"

    manager = CodingHarnessManager(
        session_manager=session_manager,
        invoke_turn=_invoke_turn,
    )
    bridge = ACPStdioBridge(
        manager=manager,
        workspace=tmp_path,
        input_stream=io.StringIO(),
        output_stream=io.StringIO(),
    )

    async def _run() -> None:
        previous = manager.set_announce_callback(bridge._handle_session_announcement)
        try:
            new_session = await bridge.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 40,
                    "method": "newSession",
                    "params": {"cwd": str(tmp_path)},
                }
            )
            assert new_session is not None
            session_id = new_session["result"]["sessionId"]

            prompt_task = asyncio.create_task(
                bridge.handle_request(
                    {
                        "jsonrpc": "2.0",
                        "id": 41,
                        "method": "prompt",
                        "params": {
                            "sessionId": session_id,
                            "prompt": [{"type": "text", "text": "slow turn"}],
                        },
                    }
                )
            )
            for _ in range(20):
                if started.is_set():
                    break
                await asyncio.sleep(0.05)
            assert started.is_set()
            cancelled = await bridge.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 42,
                    "method": "cancel",
                    "params": {"sessionId": session_id},
                }
            )
            assert cancelled is not None
            prompt_result = await asyncio.wait_for(prompt_task, timeout=1.0)
            assert prompt_result is not None
            assert prompt_result["result"]["stopReason"] == "cancelled"

            next_prompt = await bridge.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 43,
                    "method": "prompt",
                    "params": {
                        "sessionId": session_id,
                        "prompt": [{"type": "text", "text": "fast turn"}],
                    },
                }
            )
            assert next_prompt is not None
            assert next_prompt["result"]["stopReason"] == "end_turn"
        finally:
            manager.set_announce_callback(previous)

    asyncio.run(_run())


def test_acp_stdio_bridge_evicts_oldest_idle_client_session(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    session_manager = AgentSessionManager(sessions_dir=sessions_dir)

    def _invoke_turn(**kwargs):
        return str(kwargs.get("prompt") or ""), "ok"

    manager = CodingHarnessManager(
        session_manager=session_manager,
        invoke_turn=_invoke_turn,
    )
    bridge = ACPStdioBridge(
        manager=manager,
        workspace=tmp_path,
        max_client_sessions=1,
        idle_ttl_seconds=3600.0,
        input_stream=io.StringIO(),
        output_stream=io.StringIO(),
    )

    async def _run() -> None:
        first = await bridge.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 50,
                "method": "newSession",
                "params": {"cwd": str(tmp_path), "_meta": {"sessionKey": "acp:first"}},
            }
        )
        assert first is not None
        first_id = first["result"]["sessionId"]

        second = await bridge.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 51,
                "method": "newSession",
                "params": {"cwd": str(tmp_path), "_meta": {"sessionKey": "acp:second"}},
            }
        )
        assert second is not None
        second_id = second["result"]["sessionId"]
        assert second_id != first_id

        listed = await bridge.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 52,
                "method": "listSessions",
                "params": {},
            }
        )
        assert listed is not None
        assert [row["sessionId"] for row in listed["result"]["sessions"]] == [second_id]

        missing = await bridge.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 53,
                "method": "prompt",
                "params": {
                    "sessionId": first_id,
                    "prompt": [{"type": "text", "text": "should fail"}],
                },
            }
        )
        assert missing is not None
        assert missing["error"]["code"] == -32004

    asyncio.run(_run())
