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
    assert parse_direct_slash_command("/dream") == {
        "name": "dream_memory",
        "args": {"action": "run"},
    }
    assert parse_direct_slash_command("/dreaming status") == {
        "name": "dream_memory",
        "args": {"action": "status"},
    }
    assert parse_direct_slash_command("/dream-log abc123def456") == {
        "name": "dream_memory",
        "args": {"action": "log", "run_id": "abc123def456"},
    }
    assert parse_direct_slash_command("/dream-restore abc123def456") == {
        "name": "dream_memory",
        "args": {"action": "restore", "run_id": "abc123def456"},
    }
    assert parse_direct_slash_command("/track") == {
        "name": "open_track_dialog",
        "args": {},
    }
    assert parse_direct_slash_command(
        '/track video=/tmp/mouse.mp4 prompt="mouse" model=Cutie to_frame=400'
    ) == {
        "name": "segment_track_video",
        "args": {
            "path": "/tmp/mouse.mp4",
            "text_prompt": "mouse",
            "mode": "track",
            "use_countgd": False,
            "model_name": "Cutie",
            "to_frame": 400,
        },
    }
    assert parse_direct_slash_command(
        '/track video=/tmp/mouse.mp4 prompt="mouse" model=SAM3'
    ) == {
        "name": "sam3_agent_video_track",
        "args": {
            "video_path": "/tmp/mouse.mp4",
            "agent_prompt": "mouse",
        },
    }


def test_root_slash_completion_entries_include_registry_commands() -> None:
    entries = build_root_slash_completion_entries()
    searches = {str(row.get("search") or "") for row in entries}
    actions = {str(row.get("action") or "") for row in entries}

    assert "/cron" in searches
    assert "/dream" in searches
    assert "/dreaming" in searches
    assert "/dream-log" in searches
    assert "/dream-restore" in searches
    assert "/automation" in searches
    assert "/session" in searches
    assert "/track" in searches
    assert "open_capabilities" in actions
    assert "open_track_dialog" in actions


def test_slash_completion_search_matches_without_literal_slash() -> None:
    assert matches_slash_completion_search("/skill", "skill") is True
    assert matches_slash_completion_search("/capabilities", "caps") is False
    assert matches_slash_completion_search("/session", "/session") is True


def test_direct_command_alias_line_uses_registry_examples() -> None:
    line = build_direct_command_alias_line(["exec_process", "segment_track_video"])

    assert "'/track video=/path/to/video.mp4 prompt=\"mouse\" model=Cutie'" in line
    assert "'/session list'" in line
    assert "'/session logs <session_id>'" in line
    assert "'/cron status'" not in line
