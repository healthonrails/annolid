from __future__ import annotations

from annolid.core.agent.gui_backend.command_registry import (
    build_direct_command_alias_line,
    build_root_slash_completion_entries,
    matches_slash_completion_search,
    parse_direct_slash_command,
)


def test_direct_slash_parser_handles_registry_aliases() -> None:
    assert parse_direct_slash_command("/gh checks") == {
        "name": "github_pr_checks",
        "args": {},
    }
    assert parse_direct_slash_command("/caps") == {
        "name": "open_agent_capabilities",
        "args": {},
    }


def test_root_slash_completion_entries_include_registry_commands() -> None:
    entries = build_root_slash_completion_entries()
    searches = {str(row.get("search") or "") for row in entries}
    actions = {str(row.get("action") or "") for row in entries}

    assert "/cron" in searches
    assert "/automation" in searches
    assert "/session" in searches
    assert "open_capabilities" in actions


def test_slash_completion_search_matches_without_literal_slash() -> None:
    assert matches_slash_completion_search("/skill", "skill") is True
    assert matches_slash_completion_search("/capabilities", "caps") is False
    assert matches_slash_completion_search("/session", "/session") is True


def test_direct_command_alias_line_uses_registry_examples() -> None:
    line = build_direct_command_alias_line(["exec_process"])

    assert "'/session list'" in line
    assert "'/session logs <session_id>'" in line
    assert "'/cron status'" not in line
