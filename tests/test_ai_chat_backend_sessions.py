from __future__ import annotations

from pathlib import Path
from datetime import date

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
        prompt="hello",
        widget=None,
        provider="ollama",
        model="m",
        session_id="gui:test-plain-mode",
        session_store=store,
    )
    monkeypatch.setattr(task, "_emit_final", lambda message, is_error: None)
    monkeypatch.setattr(task, "_recover_with_plain_ollama_reply", lambda: "sunny")
    # Keep this test independent from evolving prompt/tool-intent heuristics.
    monkeypatch.setattr(task, "_prompt_may_need_tools", lambda _prompt: False)
    ai_chat_backend._OLLAMA_FORCE_PLAIN_CACHE["m"] = True
    try:
        task._run_agent_loop()
    finally:
        ai_chat_backend._OLLAMA_FORCE_PLAIN_CACHE.pop("m", None)

    history = task._load_history_messages()
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "hello"
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
        lambda message, is_error, **_kwargs: emitted.update(
            {"message": str(message), "is_error": bool(is_error)}
        ),
    )

    task.run()
    assert called["agent_loop"] is False
    assert emitted["is_error"] is True
    assert "openai" in emitted["message"].lower()


def test_persist_turn_writes_workspace_daily_memory(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(ai_chat_backend, "get_chat_workspace", lambda: workspace)
    store = PersistentSessionStore(
        AgentSessionManager(sessions_dir=tmp_path / "sessions")
    )
    task = StreamingChatTask(
        prompt="hello",
        widget=None,
        provider="openai",
        model="gpt-4o-mini",
        session_id="gui:annolid_bot:memory-daily",
        session_store=store,
    )

    task._persist_turn("how are you?", "all good")

    today_file = workspace / "memory" / f"{date.today().strftime('%Y-%m-%d')}.md"
    assert today_file.exists()
    text = today_file.read_text(encoding="utf-8")
    assert "gui:annolid_bot:memory-daily" in text
    assert "User: how are you?" in text
    assert "Assistant: all good" in text


def test_persist_turn_can_skip_session_history_write(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(ai_chat_backend, "get_chat_workspace", lambda: workspace)
    store = PersistentSessionStore(
        AgentSessionManager(sessions_dir=tmp_path / "sessions")
    )
    task = StreamingChatTask(
        prompt="hello",
        widget=None,
        provider="openai",
        model="gpt-4o-mini",
        session_id="gui:test-no-dup",
        session_store=store,
    )

    task._persist_turn("q1", "a1", persist_session_history=False)

    assert task._load_history_messages() == []
    today_file = workspace / "memory" / f"{date.today().strftime('%Y-%m-%d')}.md"
    assert not today_file.exists()


def test_persist_turn_does_not_write_chat_transcript_dump_to_history(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(ai_chat_backend, "get_chat_workspace", lambda: workspace)
    store = PersistentSessionStore(
        AgentSessionManager(sessions_dir=tmp_path / "sessions")
    )
    task = StreamingChatTask(
        prompt="hello",
        widget=None,
        provider="openai",
        model="gpt-4o-mini",
        session_id="gui:test-plain-mode",
        session_store=store,
    )

    task._persist_turn("hello", "sunny")

    history_file = workspace / "memory" / "HISTORY.md"
    if history_file.exists():
        text = history_file.read_text(encoding="utf-8")
        assert "## " not in text
        assert "User: hello" not in text
        assert "Assistant: sunny" not in text


def test_persist_turn_skips_daily_md_for_gui_test_session(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(ai_chat_backend, "get_chat_workspace", lambda: workspace)
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

    today_file = workspace / "memory" / f"{date.today().strftime('%Y-%m-%d')}.md"
    if today_file.exists():
        text = today_file.read_text(encoding="utf-8")
        assert "gui:test-session" not in text
        assert "User: hi" not in text
        assert "Assistant: hello there" not in text


def test_persist_turn_strips_raw_tool_call_markup_from_saved_text(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "workspace"
    monkeypatch.setattr(ai_chat_backend, "get_chat_workspace", lambda: workspace)
    store = PersistentSessionStore(
        AgentSessionManager(sessions_dir=tmp_path / "sessions")
    )
    task = StreamingChatTask(
        prompt="check today's weather",
        widget=None,
        provider="nvidia",
        model="moonshotai/kimi-k2.5",
        session_id="gui:annolid_bot:weather",
        session_store=store,
    )

    task._persist_turn(
        "check today's weather",
        (
            "<|tool_calls_section_begin|> <|tool_call_begin|> functions.read_file:0 "
            '<|tool_call_argument_begin|> {"file_path": '
            '"/Users/chenyang/.annolid/workspace/skills/weather/skill.yaml"} '
            "<|tool_call_end|> <|tool_calls_section_end|>"
        ),
    )

    history = task._load_history_messages()
    assert len(history) == 1
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "check today's weather"

    today_file = workspace / "memory" / f"{date.today().strftime('%Y-%m-%d')}.md"
    text = today_file.read_text(encoding="utf-8")
    assert "functions.read_file" not in text
    assert "<|tool_calls_section_begin|>" not in text
    assert "Assistant:" not in text


def test_fallback_timeout_retry_uses_at_least_loop_timeout() -> None:
    task = StreamingChatTask(
        prompt="hello there",
        widget=None,
        provider="nvidia",
        model="moonshotai/kimi-k2.5",
        settings={
            "agent": {
                "fallback_retry_timeout_seconds": 20,
                "loop_llm_timeout_seconds": 60,
                "loop_llm_timeout_seconds_no_tools": 40,
            }
        },
    )
    assert task._fallback_timeout_retry_seconds() == 40


def test_nvidia_timeout_floors_for_agent_loop_and_retry() -> None:
    task = StreamingChatTask(
        prompt="use tools please",
        widget=None,
        provider="nvidia",
        model="moonshotai/kimi-k2.5",
        settings={
            "agent": {
                "loop_llm_timeout_seconds": 300,
                "loop_llm_timeout_seconds_no_tools": 40,
                "fallback_retry_timeout_seconds": 20,
            }
        },
    )
    assert task._agent_loop_llm_timeout_seconds(prompt_needs_tools=True) == 420
    assert task._agent_loop_llm_timeout_seconds(prompt_needs_tools=False) == 180
    assert task._fallback_retry_timeout_seconds() == 60


def test_run_provider_fallback_timeout_is_graceful(monkeypatch) -> None:
    task = StreamingChatTask(
        prompt="hello there",
        widget=None,
        provider="nvidia",
        model="moonshotai/kimi-k2.5",
    )
    emitted = {"message": "", "is_error": False}
    called = {"exception_logged": False}

    monkeypatch.setattr(
        task,
        "_run_openai",
        lambda provider_name, timeout_s, max_tokens: (_ for _ in ()).throw(
            TimeoutError(
                "Provider request timed out after 20s for nvidia:moonshotai/kimi-k2.5."
            )
        ),
    )
    monkeypatch.setattr(
        task,
        "_emit_final",
        lambda message, is_error: emitted.update(
            {"message": str(message), "is_error": bool(is_error)}
        ),
    )
    monkeypatch.setattr(
        ai_chat_backend.logger,
        "exception",
        lambda *args, **kwargs: called.update({"exception_logged": True}),
    )

    task._run_provider_fallback(
        TimeoutError(
            "LLM timed out after 40.0s (iteration=1, model=moonshotai/kimi-k2.5)"
        )
    )
    assert emitted["is_error"] is True
    assert "timed out" in emitted["message"].lower()
    assert called["exception_logged"] is False


def test_codex_cli_bypasses_agent_loop_for_plain_runtime(monkeypatch) -> None:
    task = StreamingChatTask(
        prompt="summarize this",
        widget=None,
        provider="codex_cli",
        model="codex-cli/gpt-5.1-codex",
    )
    calls = {"provider_runtime": 0, "agent_loop": 0}

    async def _no_direct_command() -> bool:
        return False

    monkeypatch.setattr(task, "_try_execute_direct_gui_command", _no_direct_command)
    monkeypatch.setattr(
        task,
        "_run_openai",
        lambda provider_name, timeout_s=None, max_tokens=4096: calls.__setitem__(
            "provider_runtime", calls["provider_runtime"] + 1
        ),
    )
    monkeypatch.setattr(
        task,
        "_build_agent_execution_context",
        lambda include_tools=True: calls.__setitem__(
            "agent_loop", calls["agent_loop"] + 1
        ),
    )

    ai_chat_backend.run_chat_awaitable_sync(task._run_agent_loop_async())
    assert calls["provider_runtime"] == 1
    assert calls["agent_loop"] == 0


def test_streaming_chat_task_cancel_emits_stopped_message(monkeypatch) -> None:
    task = StreamingChatTask(
        prompt="hello",
        widget=None,
        provider="openai",
        model="gpt-5-mini",
    )
    emitted = {"message": "", "is_error": True}

    monkeypatch.setattr(task, "_provider_dependency_error", lambda: None)
    monkeypatch.setattr(
        ai_chat_backend,
        "emit_chat_final",
        lambda **kwargs: emitted.update(
            {
                "message": str(kwargs.get("message") or ""),
                "is_error": bool(kwargs.get("is_error")),
            }
        ),
    )

    task.request_cancel()
    task.run()
    assert emitted["message"] == "Stopped by user."
    assert emitted["is_error"] is False


def test_streaming_chat_task_cancel_skips_persist_turn(tmp_path: Path) -> None:
    store = PersistentSessionStore(
        AgentSessionManager(sessions_dir=tmp_path / "sessions")
    )
    task = StreamingChatTask(
        prompt="hello",
        widget=None,
        provider="openai",
        model="gpt-4o-mini",
        session_id="gui:test-cancel-skip-persist",
        session_store=store,
    )

    task.request_cancel()
    task._persist_turn("hi", "hello there")
    assert task._load_history_messages() == []
