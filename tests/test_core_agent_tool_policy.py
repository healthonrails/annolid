from __future__ import annotations

from annolid.core.agent.config import ToolPolicyConfig, ToolsConfig
from annolid.core.agent.tools.policy import resolve_allowed_tools


def test_policy_profile_and_group_allow_deny() -> None:
    cfg = ToolsConfig(
        profile="minimal",
        allow=["group:ui", "web_search"],
        deny=["gui_set_chat_model"],
    )
    all_tools = [
        "gui_context",
        "gui_shared_image_path",
        "gui_open_video",
        "gui_set_chat_model",
        "web_search",
        "exec",
    ]
    resolved = resolve_allowed_tools(
        all_tool_names=all_tools,
        tools_cfg=cfg,
        provider="ollama",
        model="qwen3",
    )
    assert "gui_context" in resolved.allowed_tools
    assert "gui_open_video" in resolved.allowed_tools
    assert "web_search" in resolved.allowed_tools
    assert "gui_set_chat_model" not in resolved.allowed_tools
    assert "exec" not in resolved.allowed_tools


def test_policy_provider_model_override() -> None:
    cfg = ToolsConfig(
        profile="full",
        by_provider={
            "ollama:glm-5:cloud": ToolPolicyConfig(
                profile="minimal",
                allow=["gui_open_video"],
                deny=["gui_shared_image_path"],
            )
        },
    )
    all_tools = [
        "gui_context",
        "gui_shared_image_path",
        "gui_open_video",
        "exec",
        "read_file",
    ]
    resolved = resolve_allowed_tools(
        all_tool_names=all_tools,
        tools_cfg=cfg,
        provider="ollama",
        model="glm-5:cloud",
    )
    assert resolved.source == "ollama:glm-5:cloud"
    assert resolved.allowed_tools == {"gui_context", "gui_open_video"}


def test_policy_wildcard_entries() -> None:
    cfg = ToolsConfig(
        profile="minimal",
        allow=["gui_*"],
        deny=["*chat*"],
    )
    all_tools = [
        "gui_context",
        "gui_open_video",
        "gui_set_chat_prompt",
        "gui_send_chat_prompt",
    ]
    resolved = resolve_allowed_tools(
        all_tool_names=all_tools,
        tools_cfg=cfg,
        provider="openrouter",
        model="gpt-5-mini",
    )
    assert "gui_context" in resolved.allowed_tools
    assert "gui_open_video" in resolved.allowed_tools
    assert "gui_set_chat_prompt" not in resolved.allowed_tools
    assert "gui_send_chat_prompt" not in resolved.allowed_tools
