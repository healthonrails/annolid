from __future__ import annotations

from pathlib import Path

from annolid.core.agent.session_manager import (
    AgentSessionManager,
    PersistentSessionStore,
)
import annolid.gui.widgets.ai_chat_backend as ai_chat_backend
from annolid.gui.widgets.ai_chat_backend import StreamingChatTask


def test_streaming_chat_task_persists_and_loads_session_history(tmp_path: Path) -> None:
    store = PersistentSessionStore(
        AgentSessionManager(sessions_dir=tmp_path / "sessions")
    )
    task = StreamingChatTask(
        prompt="hello",
        widget=None,
        provider="openai",
        model="gpt-4o-mini",
        session_id="gui:test-session",
        session_store=store,
    )

    task._persist_turn("hi", "hello there")
    history = task._load_history_messages()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "hi"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "hello there"


def test_clear_chat_session_removes_persisted_messages(tmp_path: Path) -> None:
    store = PersistentSessionStore(
        AgentSessionManager(sessions_dir=tmp_path / "sessions")
    )
    task = StreamingChatTask(
        prompt="hello",
        widget=None,
        provider="openai",
        model="gpt-4o-mini",
        session_id="gui:test-session-clear",
        session_store=store,
    )
    task._persist_turn("hi", "hello there")
    assert len(task._load_history_messages()) == 2

    # Clear against the explicit store to avoid global singletons in tests.
    task.session_store.clear_session(task.session_id)
    assert task._load_history_messages() == []


def test_plain_mode_fallback_persists_turn(tmp_path: Path, monkeypatch) -> None:
    store = PersistentSessionStore(
        AgentSessionManager(sessions_dir=tmp_path / "sessions")
    )
    task = StreamingChatTask(
        prompt="weather?",
        widget=None,
        provider="ollama",
        model="m",
        session_id="gui:test-plain-mode",
        session_store=store,
    )
    monkeypatch.setattr(task, "_emit_final", lambda message, is_error: None)
    monkeypatch.setattr(task, "_recover_with_plain_ollama_reply", lambda: "sunny")
    ai_chat_backend._OLLAMA_FORCE_PLAIN_CACHE["m"] = True
    try:
        task._run_agent_loop()
    finally:
        ai_chat_backend._OLLAMA_FORCE_PLAIN_CACHE.pop("m", None)

    history = task._load_history_messages()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "weather?"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "sunny"


def test_run_reports_dependency_missing_without_fallback(monkeypatch) -> None:
    task = StreamingChatTask(
        prompt="hello",
        widget=None,
        provider="openai",
        model="gpt-5-mini",
    )
    emitted = {"message": "", "is_error": False}
    called = {"agent_loop": False}

    monkeypatch.setattr(
        task,
        "_provider_dependency_error",
        lambda: "OpenAI-compatible provider requires the `openai` package.",
    )
    monkeypatch.setattr(
        task, "_run_agent_loop", lambda: called.__setitem__("agent_loop", True)
    )
    monkeypatch.setattr(
        task,
        "_emit_final",
        lambda message, is_error: emitted.update(
            {"message": str(message), "is_error": bool(is_error)}
        ),
    )

    task.run()
    assert called["agent_loop"] is False
    assert emitted["is_error"] is True
    assert "openai" in emitted["message"].lower()
