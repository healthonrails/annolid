from __future__ import annotations

from annolid.services.chat_provider_runtime import (
    build_chat_agent_loop,
    chat_agent_loop_llm_timeout_seconds,
    chat_agent_loop_tool_timeout_seconds,
    chat_browser_first_for_web,
    chat_fallback_retry_timeout_seconds,
    chat_fallback_timeout_retry_seconds,
    chat_fast_mode_timeout_seconds,
    chat_ollama_agent_plain_timeout_seconds,
    chat_ollama_agent_tool_timeout_seconds,
    chat_ollama_plain_recovery_nudge_timeout_seconds,
    chat_ollama_plain_recovery_timeout_seconds,
    chat_provider_dependency_error,
    format_chat_dependency_error,
    format_chat_provider_config_error,
    has_chat_image_context,
    is_chat_provider_config_error,
    is_chat_provider_timeout_error,
    resolve_chat_provider_kind,
    run_chat_fast_mode,
    run_chat_fast_provider_chat,
    run_chat_gemini,
    run_chat_ollama,
    run_chat_openai,
    run_chat_provider_fallback,
)


def test_chat_provider_runtime_wrappers(monkeypatch) -> None:
    import annolid.services.chat_provider_runtime as runtime_mod

    monkeypatch.setattr(runtime_mod, "gui_is_provider_config_error", lambda exc: True)
    monkeypatch.setattr(
        runtime_mod,
        "gui_format_provider_config_error",
        lambda raw_error, provider: f"{provider}:{raw_error}",
    )
    monkeypatch.setattr(runtime_mod, "gui_is_provider_timeout_error", lambda exc: False)
    monkeypatch.setattr(
        runtime_mod,
        "gui_provider_dependency_error",
        lambda settings, provider: f"missing:{provider}",
    )
    monkeypatch.setattr(
        runtime_mod,
        "gui_format_dependency_error",
        lambda raw_error, settings, provider: f"{provider}:{raw_error}",
    )
    monkeypatch.setattr(
        runtime_mod, "gui_fast_mode_timeout_seconds", lambda settings: 12.0
    )
    monkeypatch.setattr(
        runtime_mod,
        "gui_agent_loop_llm_timeout_seconds",
        lambda settings, prompt_needs_tools=False: 30.0,
    )
    monkeypatch.setattr(
        runtime_mod,
        "gui_agent_loop_tool_timeout_seconds",
        lambda settings, provider=None: 15.0,
    )
    monkeypatch.setattr(runtime_mod, "gui_browser_first_for_web", lambda settings: True)
    monkeypatch.setattr(
        runtime_mod, "gui_ollama_agent_plain_timeout_seconds", lambda settings: 20.0
    )
    monkeypatch.setattr(
        runtime_mod, "gui_ollama_agent_tool_timeout_seconds", lambda settings: 21.0
    )
    monkeypatch.setattr(
        runtime_mod, "gui_ollama_plain_recovery_timeout_seconds", lambda settings: 22.0
    )
    monkeypatch.setattr(
        runtime_mod,
        "gui_ollama_plain_recovery_nudge_timeout_seconds",
        lambda settings: 23.0,
    )
    monkeypatch.setattr(
        runtime_mod, "gui_fallback_retry_timeout_seconds", lambda settings: 24.0
    )
    monkeypatch.setattr(
        runtime_mod,
        "gui_fallback_timeout_retry_seconds",
        lambda settings, prompt_needs_tools=False: 25.0,
    )

    assert is_chat_provider_config_error("x") is True
    assert format_chat_provider_config_error("boom", provider="openai") == "openai:boom"
    assert is_chat_provider_timeout_error("x") is False
    assert (
        chat_provider_dependency_error(settings={}, provider="openai")
        == "missing:openai"
    )
    assert (
        format_chat_dependency_error(raw_error="oops", settings={}, provider="openai")
        == "openai:oops"
    )
    assert chat_fast_mode_timeout_seconds({}) == 12.0
    assert (
        chat_agent_loop_llm_timeout_seconds(
            settings={}, provider="nvidia", prompt_needs_tools=True
        )
        == 420.0
    )
    assert (
        chat_agent_loop_tool_timeout_seconds(
            settings={}, provider="openai", prompt="swarm test"
        )
        == 900.0
    )
    assert chat_browser_first_for_web({}) is True
    assert chat_ollama_agent_plain_timeout_seconds({}) == 20.0
    assert chat_ollama_agent_tool_timeout_seconds({}) == 21.0
    assert chat_ollama_plain_recovery_timeout_seconds({}) == 22.0
    assert chat_ollama_plain_recovery_nudge_timeout_seconds({}) == 23.0
    assert chat_fallback_retry_timeout_seconds(settings={}, provider="nvidia") == 60.0
    assert (
        chat_fallback_timeout_retry_seconds(settings={}, prompt_needs_tools=True)
        == 25.0
    )


def test_chat_provider_runtime_execution_wrappers(monkeypatch) -> None:
    import annolid.services.chat_provider_runtime as runtime_mod

    captured = {}

    monkeypatch.setattr(
        runtime_mod, "gui_provider_kind", lambda settings, provider: "codex_cli"
    )
    monkeypatch.setattr(
        runtime_mod, "gui_has_image_context", lambda image_path: bool(image_path)
    )
    monkeypatch.setattr(
        runtime_mod,
        "gui_run_fast_mode",
        lambda **kwargs: captured.setdefault("fast_mode", kwargs),
    )
    monkeypatch.setattr(
        runtime_mod,
        "gui_run_fast_provider_chat",
        lambda **kwargs: captured.setdefault("fast_provider", kwargs),
    )
    monkeypatch.setattr(
        runtime_mod,
        "gui_run_provider_fallback",
        lambda **kwargs: captured.setdefault("fallback", kwargs),
    )
    monkeypatch.setattr(
        runtime_mod,
        "gui_run_ollama_chat",
        lambda **kwargs: captured.setdefault("ollama", kwargs),
    )
    monkeypatch.setattr(
        runtime_mod,
        "gui_run_openai_chat",
        lambda **kwargs: ("user", f"openai:{kwargs['provider_name']}"),
    )
    monkeypatch.setattr(
        runtime_mod,
        "gui_run_gemini_provider_chat",
        lambda **kwargs: ("user", f"gemini:{kwargs['provider_name']}"),
    )

    assert resolve_chat_provider_kind(settings={}, provider="openai") == "codex_cli"
    assert has_chat_image_context("/tmp/image.png") is True

    run_chat_fast_mode(
        chat_mode="vision_describe", run_fast_provider_chat=lambda *_: None
    )
    assert captured["fast_mode"]["chat_mode"] == "vision_describe"

    run_chat_fast_provider_chat(
        prompt="p",
        image_path="",
        model="m",
        provider="openai",
        settings={},
        session_id="s",
        include_image=False,
        include_history=False,
        load_history_messages=lambda: [],
        fast_mode_timeout_seconds=lambda: 1.0,
        emit_progress=lambda _m: None,
        emit_chunk=lambda _m: None,
        emit_final=lambda _m, _e: None,
        persist_turn=lambda _u, _a: None,
    )
    assert captured["fast_provider"]["provider"] == "openai"

    run_chat_provider_fallback(original_error=RuntimeError("x"))
    assert isinstance(captured["fallback"]["original_error"], RuntimeError)

    run_chat_ollama(prompt="p")
    assert captured["ollama"]["prompt"] == "p"

    assert run_chat_openai(prompt="p", provider_name="openai") == (
        "user",
        "openai:openai",
    )
    assert run_chat_gemini(prompt="p", provider_name="gemini") == (
        "user",
        "gemini:gemini",
    )


def test_build_chat_agent_loop_delegates(monkeypatch) -> None:
    import annolid.services.chat_provider_runtime as runtime_mod

    captured = {}

    class _Loop:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(runtime_mod, "AgentLoop", _Loop)

    loop = build_chat_agent_loop(
        tools="tools",
        llm_callable="llm",
        provider="openai",
        model="gpt",
        memory_store="memory",
        workspace="/tmp/workspace",
        allowed_read_roots=["/tmp/workspace"],
        mcp_servers={"mcp": 1},
        llm_timeout_seconds=30.0,
        tool_timeout_seconds=15.0,
        browser_first_for_web=True,
        strict_runtime_tool_guard=True,
    )

    assert isinstance(loop, _Loop)
    assert captured["provider"] == "openai"
    assert captured["workspace"] == "/tmp/workspace"
