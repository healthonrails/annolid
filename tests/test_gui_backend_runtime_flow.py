from __future__ import annotations

from annolid.core.agent.gui_backend.runtime_flow import (
    emit_agent_loop_result,
    maybe_handle_ollama_plain_mode,
)


class _DummyLogger:
    def __init__(self) -> None:
        self.infos: list[str] = []
        self.warnings: list[str] = []

    def info(self, msg: str, *args) -> None:
        self.infos.append(msg % args if args else msg)

    def warning(self, msg: str, *args) -> None:
        self.warnings.append(msg % args if args else msg)


def test_maybe_handle_ollama_plain_mode_handles_non_tool_prompt() -> None:
    persisted: list[tuple[str, str]] = []
    emitted: list[tuple[str, bool]] = []
    logger = _DummyLogger()

    handled = maybe_handle_ollama_plain_mode(
        provider="ollama",
        model="m",
        prompt="hello",
        show_tool_trace=True,
        prompt_may_need_tools=lambda prompt: False,
        plain_mode_remaining=lambda model: 2,
        plain_mode_decrement=lambda model: 1,
        recover_with_plain_reply=lambda: "plain reply",
        persist_turn=lambda user, assistant: persisted.append((user, assistant)),
        emit_final=lambda message, is_error: emitted.append((message, is_error)),
        logger=logger,
    )

    assert handled is True
    assert persisted and persisted[0][0] == "hello"
    assert "plain reply" in persisted[0][1]
    assert "[Tool Trace]" in persisted[0][1]
    assert emitted and emitted[0][1] is False
    assert logger.warnings


def test_maybe_handle_ollama_plain_mode_bypasses_when_tools_needed() -> None:
    logger = _DummyLogger()
    handled = maybe_handle_ollama_plain_mode(
        provider="ollama",
        model="m",
        prompt="check weather today",
        show_tool_trace=False,
        prompt_may_need_tools=lambda prompt: True,
        plain_mode_remaining=lambda model: 2,
        plain_mode_decrement=lambda model: 1,
        recover_with_plain_reply=lambda: "unused",
        persist_turn=lambda user, assistant: None,
        emit_final=lambda message, is_error: None,
        logger=logger,
    )
    assert handled is False
    assert logger.infos


def test_emit_agent_loop_result_persists_non_empty_only() -> None:
    persisted: list[tuple[str, str]] = []
    emitted: list[tuple[str, bool]] = []

    emit_agent_loop_result(
        prompt="q",
        text="answer",
        persist_turn=lambda user, assistant: persisted.append((user, assistant)),
        emit_final=lambda message, is_error: emitted.append((message, is_error)),
    )
    emit_agent_loop_result(
        prompt="q2",
        text="",
        persist_turn=lambda user, assistant: persisted.append((user, assistant)),
        emit_final=lambda message, is_error: emitted.append((message, is_error)),
    )

    assert ("q", "answer") in persisted
    assert all(user != "q2" for user, _ in persisted)
    assert emitted == [("answer", False), ("", False)]
