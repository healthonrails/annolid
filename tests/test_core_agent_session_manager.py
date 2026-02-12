from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    ) -> Mapping[str, Any]:
        del tools, model
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
