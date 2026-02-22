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
        "gui_open_pdf",
        "gui_open_threejs",
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
    assert "gui_open_pdf" in resolved.allowed_tools
    assert "gui_open_threejs" in resolved.allowed_tools
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
        "gui_open_pdf",
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
        "gui_open_pdf",
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
    assert "gui_open_pdf" in resolved.allowed_tools
    assert "gui_set_chat_prompt" not in resolved.allowed_tools
    assert "gui_send_chat_prompt" not in resolved.allowed_tools


def test_policy_group_vcs_allows_git_and_github_tools() -> None:
    cfg = ToolsConfig(
        profile="minimal",
        allow=["group:vcs"],
    )
    all_tools = [
        "git_status",
        "git_diff",
        "git_log",
        "github_pr_status",
        "github_pr_checks",
        "exec",
    ]
    resolved = resolve_allowed_tools(
        all_tool_names=all_tools,
        tools_cfg=cfg,
        provider="ollama",
        model="qwen3",
    )
    assert resolved.allowed_tools == {
        "git_status",
        "git_diff",
        "git_log",
        "github_pr_status",
        "github_pr_checks",
    }


def test_policy_group_pdf_includes_open_and_extract_tools() -> None:
    cfg = ToolsConfig(
        profile="minimal",
        allow=["group:pdf"],
    )
    all_tools = [
        "open_pdf",
        "extract_pdf_text",
        "extract_pdf_images",
        "download_pdf",
        "read_file",
    ]
    resolved = resolve_allowed_tools(
        all_tool_names=all_tools,
        tools_cfg=cfg,
        provider="ollama",
        model="qwen3",
    )
    assert resolved.allowed_tools == {
        "open_pdf",
        "extract_pdf_text",
        "extract_pdf_images",
    }


def test_policy_group_fs_includes_rename_file() -> None:
    cfg = ToolsConfig(
        profile="minimal",
        allow=["group:fs"],
    )
    all_tools = [
        "read_file",
        "write_file",
        "edit_file",
        "rename_file",
        "list_dir",
        "exec",
    ]
    resolved = resolve_allowed_tools(
        all_tool_names=all_tools,
        tools_cfg=cfg,
        provider="ollama",
        model="qwen3",
    )
    assert resolved.allowed_tools == {
        "read_file",
        "write_file",
        "edit_file",
        "rename_file",
        "list_dir",
    }


def test_policy_coding_profile_includes_google_calendar() -> None:
    cfg = ToolsConfig(profile="coding")
    all_tools = [
        "read_file",
        "exec",
        "google_calendar",
        "email",
        "list_emails",
        "read_email",
        "gui_context",
    ]
    resolved = resolve_allowed_tools(
        all_tool_names=all_tools,
        tools_cfg=cfg,
        provider="openrouter",
        model="gpt-5-mini",
    )
    assert "google_calendar" in resolved.allowed_tools
    assert "email" in resolved.allowed_tools
    assert "list_emails" in resolved.allowed_tools
    assert "read_email" in resolved.allowed_tools


def test_policy_group_automation_includes_google_calendar() -> None:
    cfg = ToolsConfig(
        profile="minimal",
        allow=["group:automation"],
    )
    all_tools = [
        "cron",
        "spawn",
        "google_calendar",
        "email",
        "list_emails",
        "read_email",
        "exec",
    ]
    resolved = resolve_allowed_tools(
        all_tool_names=all_tools,
        tools_cfg=cfg,
        provider="ollama",
        model="qwen3",
    )
    assert resolved.allowed_tools == {
        "cron",
        "spawn",
        "google_calendar",
        "email",
        "list_emails",
        "read_email",
    }
