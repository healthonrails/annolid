from __future__ import annotations

from annolid.services.chat_context import (
    load_chat_execution_prerequisites,
    prepare_chat_context_tools,
    register_chat_gui_toolset,
)


def test_load_chat_execution_prerequisites_delegates(monkeypatch) -> None:
    import annolid.services.chat_context as chat_context_mod

    expected = object()
    monkeypatch.setattr(
        chat_context_mod,
        "load_gui_execution_prerequisites",
        lambda: expected,
    )

    assert load_chat_execution_prerequisites() is expected


def test_prepare_chat_context_tools_delegates(monkeypatch) -> None:
    import annolid.services.chat_context as chat_context_mod

    expected = object()

    async def _prepare_gui_context_tools(**kwargs):
        return expected

    monkeypatch.setattr(
        chat_context_mod,
        "prepare_gui_context_tools",
        _prepare_gui_context_tools,
    )

    import asyncio

    result = asyncio.run(
        prepare_chat_context_tools(
            include_tools=True,
            workspace="workspace",
            allowed_read_roots=["/tmp"],
            agent_cfg="cfg",
            register_gui_tools=lambda tools: None,
            provider="openai",
            model="gpt",
            enable_web_tools=True,
            always_disabled_tools={"spawn"},
            web_tools={"web_search"},
            resolve_policy=lambda **kwargs: None,
        )
    )

    assert result is expected


def test_register_chat_gui_toolset_delegates(monkeypatch) -> None:
    import annolid.services.chat_context as chat_context_mod

    captured = {}

    def _register_chat_gui_tools(tools, **kwargs):
        captured["tools"] = tools
        captured.update(kwargs)

    monkeypatch.setattr(
        chat_context_mod,
        "register_chat_gui_tools",
        _register_chat_gui_tools,
    )

    register_chat_gui_toolset(
        "tools",
        context_callback=lambda: None,
        image_path_callback=lambda: None,
        wrap_tool_callback=lambda name, fn: fn,
        handlers={"open_video": lambda: None},
    )

    assert captured["tools"] == "tools"
    assert "handlers" in captured
