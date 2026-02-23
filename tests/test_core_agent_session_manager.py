from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from annolid.core.agent.loop import AgentLoop
from annolid.core.agent.session_manager import (
    AgentSessionManager,
    PersistentSessionStore,
)
from annolid.core.agent.tools.function_registry import FunctionToolRegistry


def test_session_manager_save_load_list_delete(tmp_path: Path) -> None:
    manager = AgentSessionManager(sessions_dir=tmp_path / "sessions")
    session = manager.get_or_create("gui:chat/1")
    session.add_message({"role": "user", "content": "hello"})
    session.facts["locale"] = "en-US"
    manager.save(session)

    manager2 = AgentSessionManager(sessions_dir=tmp_path / "sessions")
    loaded = manager2.get_or_create("gui:chat/1")
    assert loaded.messages[-1]["content"] == "hello"
    assert loaded.facts["locale"] == "en-US"

    rows = manager2.list_sessions()
    assert rows
    assert rows[0]["key"] == "gui:chat/1"
    assert rows[0]["message_count"] == 1

    assert manager2.delete("gui:chat/1") is True
    assert manager2.delete("gui:chat/1") is False


def test_persistent_session_store_works_across_loop_instances(tmp_path: Path) -> None:
    manager = AgentSessionManager(sessions_dir=tmp_path / "sessions")
    store = PersistentSessionStore(manager)
    registry = FunctionToolRegistry()
    call_state = {"n": 0}

    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del tools, model, on_token
        call_state["n"] += 1
        if call_state["n"] == 2:
            seen = any(
                m.get("role") == "assistant" and m.get("content") == "reply-1"
                for m in messages
            )
            assert seen is True
        return {"content": "reply-1" if call_state["n"] == 1 else "reply-2"}

    loop1 = AgentLoop(
        tools=registry,
        llm_callable=fake_llm,
        model="fake",
        memory_store=store,
    )
    loop1.remember("s1", "animal", "mouse")
    _ = asyncio.run(loop1.run("hello", session_id="s1"))

    # New loop instance with fresh manager should recover stored session state.
    loop2 = AgentLoop(
        tools=registry,
        llm_callable=fake_llm,
        model="fake",
        memory_store=PersistentSessionStore(
            AgentSessionManager(sessions_dir=tmp_path / "sessions")
        ),
    )
    assert loop2.recall("s1", "animal") == "mouse"
    result = asyncio.run(loop2.run("again", session_id="s1"))
    assert result.content == "reply-2"


def test_persistent_session_store_compacts_empty_messages(tmp_path: Path) -> None:
    manager = AgentSessionManager(sessions_dir=tmp_path / "sessions")
    store = PersistentSessionStore(manager)
    session_id = "gui:compact"

    # Seed with an empty assistant row to simulate older polluted history.
    session = manager.get_or_create(session_id)
    session.add_message({"role": "user", "content": "hello"})
    session.add_message({"role": "assistant", "content": ""})
    manager.save(session)

    # Append valid history; store should compact old empty rows and ignore new empty rows.
    store.append_history(
        session_id,
        [
            {"role": "assistant", "content": ""},
            {"role": "assistant", "content": "world"},
        ],
        max_messages=20,
    )

    history = store.get_history(session_id)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "hello"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "world"


def test_session_manager_overview_includes_counts_and_paths(tmp_path: Path) -> None:
    manager = AgentSessionManager(sessions_dir=tmp_path / "sessions")
    session = manager.get_or_create("gui:chat/overview")
    session.add_message({"role": "user", "content": "hello"})
    session.facts["timezone"] = "UTC"
    manager.save(session)

    overview = manager.get_session_overview("gui:chat/overview")
    assert overview["key"] == "gui:chat/overview"
    assert int(overview["message_count"]) == 1
    assert int(overview["fact_count"]) == 1
    assert overview["facts"]["timezone"] == "UTC"
    assert str(overview["path"]).endswith(".jsonl")


def test_persistent_session_store_metadata_roundtrip(tmp_path: Path) -> None:
    manager = AgentSessionManager(sessions_dir=tmp_path / "sessions")
    store = PersistentSessionStore(manager)
    session_id = "gui:meta"

    assert store.get_session_metadata(session_id) == {}
    store.update_session_metadata(session_id, {"last_consolidated": 12})
    meta = store.get_session_metadata(session_id)
    assert int(meta.get("last_consolidated") or 0) == 12


def test_persistent_session_store_records_automation_task_runs(tmp_path: Path) -> None:
    manager = AgentSessionManager(sessions_dir=tmp_path / "sessions")
    store = PersistentSessionStore(manager)
    session_id = "gui:auto"

    store.record_automation_task_run(
        session_id,
        task_name="camera-check",
        status="ok",
        detail="snapshot sent",
    )
    meta = store.get_session_metadata(session_id)
    runs = list(meta.get("automation_runs") or [])
    assert runs
    latest = runs[-1]
    assert latest["task_name"] == "camera-check"
    assert latest["status"] == "ok"
    assert "snapshot" in latest["detail"]


def test_persistent_session_store_memory_layers_apply_quotas(tmp_path: Path) -> None:
    manager = AgentSessionManager(sessions_dir=tmp_path / "sessions")
    store = PersistentSessionStore(
        manager,
        working_memory_max_chars=64,
        long_term_memory_max_chars=64,
    )
    session_id = "gui:memory-layers"
    long_text = "x" * 200
    store.set_working_memory(session_id, long_text, reason="test")
    store.set_long_term_memory(session_id, long_text, reason="test")
    working = store.get_working_memory(session_id)
    long_term = store.get_long_term_memory(session_id)
    assert len(working) <= 80
    assert len(long_term) <= 80
    assert "truncated" in working
    assert "truncated" in long_term


def test_persistent_session_store_memory_audit_trail_records_mutations(
    tmp_path: Path,
) -> None:
    manager = AgentSessionManager(sessions_dir=tmp_path / "sessions")
    store = PersistentSessionStore(manager)
    session_id = "gui:audit"
    store.set_fact(session_id, "animal", "mouse")
    store.set_working_memory(session_id, "user: hello", reason="sync")
    trail = store.get_memory_audit_trail(session_id, limit=20)
    assert trail
    scopes = {str(item.get("scope") or "") for item in trail}
    assert "facts" in scopes
    assert "working_memory" in scopes


def test_persistent_session_store_replay_events_roundtrip(tmp_path: Path) -> None:
    manager = AgentSessionManager(sessions_dir=tmp_path / "sessions")
    store = PersistentSessionStore(manager)
    session_id = "gui:replay"
    store.record_event(
        session_id,
        direction="inbound",
        kind="user",
        payload={"text": "hello"},
    )
    store.record_event(
        session_id,
        direction="outbound",
        kind="assistant",
        payload={"text": "world"},
    )
    inbound = store.replay_events(session_id, direction="inbound", limit=10)
    all_rows = store.replay_events(session_id, limit=10)
    assert len(inbound) == 1
    assert inbound[0]["payload"]["text"] == "hello"
    assert len(all_rows) == 2
