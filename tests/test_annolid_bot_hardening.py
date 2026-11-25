from __future__ import annotations

import asyncio
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from qtpy import QtWidgets

from annolid.core.agent.bus import MessageBus
from annolid.core.agent.channels.base import BaseChannel
from annolid.core.agent.channels.whatsapp_python_bridge import WhatsAppPythonBridge
from annolid.core.agent.config import (
    AgentConfig,
    ProviderConfig,
    ToolPolicyConfig,
    ToolsConfig,
    load_config,
    save_config,
)
from annolid.core.agent.config.schema import MCPServerConfig
from annolid.core.agent.cron import CronService
from annolid.core.agent.gui_backend.command_registry import (
    build_root_slash_completion_entries,
    parse_direct_slash_command,
)
from annolid.core.agent.gui_backend.commands import parse_direct_gui_command
from annolid.core.agent.gui_backend.provider_dependencies import format_dependency_error
from annolid.core.agent.gui_backend.runtime_config import browser_first_for_web
from annolid.core.agent.gui_backend.session_io import (
    OutboundChatEvent,
    emit_chunk,
    encode_outbound_chat_event,
    load_history_messages,
)
from annolid.core.agent.gui_backend.tool_handlers_citations import (
    resolve_bib_output_path,
)
from annolid.core.agent.gui_backend.tool_handlers_realtime import (
    _redact_camera_source_for_bot_display,
)
from annolid.core.agent.gui_backend.tool_handlers_shape_files import (
    delete_shapes_in_annotation_tool,
)
from annolid.core.agent.gui_backend.tool_handlers_tracking_correction import (
    _repair_temporal_records,
)
from annolid.core.agent.gui_backend.tool_handlers_tracking_stats import (
    analyze_tracking_stats_tool,
)
from annolid.core.agent.loop import AgentLoop
from annolid.core.agent.session_manager import (
    AgentSessionManager,
    PersistentSessionStore,
)
from annolid.core.agent.tools.common import _is_probably_text_file
from annolid.core.agent.tools.filesystem import RenameFileTool
from annolid.core.agent.tools.function_base import FunctionTool
from annolid.core.agent.tools.function_registry import FunctionToolRegistry
from annolid.core.agent.tools.git import GitCliTool, GitDiffTool
from annolid.core.agent.tools.policy import (
    build_tool_permission_context,
    resolve_allowed_tools,
)
from annolid.gui.features.viewers import _LazyAIChatManager
from annolid.gui.widgets.agent_run_dialog import AgentRunDialog
from annolid.gui.widgets.agent_workspace_onboarding_dialog import (
    AgentWorkspaceOnboardingDialog,
)
from annolid.gui.widgets.ai_chat_backend import StreamingChatTask
from annolid.gui.widgets.ai_chat_bubble import _ChatBubble
from annolid.gui.widgets.ai_chat_manager import _sync_dream_cron_job
from annolid.gui.widgets.ai_chat_widget import AIChatWidget
from annolid.services.chat_session import get_chat_session_store
from annolid.services.chat_tracking_stats import run_chat_stats_slash_tool


@pytest.fixture(autouse=True)
def _isolate_bot_hardening_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep bot hardening tests isolated from the developer's real ~/.annolid."""
    import annolid.services.chat_session as chat_session_mod
    import annolid.utils.llm_settings as llm_settings_mod

    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    settings_dir = fake_home / ".annolid"
    monkeypatch.setattr(llm_settings_mod, "_SETTINGS_DIR", settings_dir)
    monkeypatch.setattr(
        llm_settings_mod, "_SETTINGS_FILE", settings_dir / "llm_settings.json"
    )
    monkeypatch.setattr(llm_settings_mod, "_GLOBAL_DOTENV_FILE", settings_dir / ".env")
    monkeypatch.setattr(chat_session_mod, "_SESSION_STORE", None)


def _ensure_qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def test_skills_module_import_has_no_home_mkdir_side_effect(tmp_path: Path) -> None:
    fake_home = tmp_path / "clean_home"
    fake_home.mkdir()
    env = dict(os.environ)
    env["HOME"] = str(fake_home)
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import annolid.core.agent.skills as s; print(s.get_managed_skills_dir())",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert str(fake_home / ".annolid" / "skills") in proc.stdout
    assert not (fake_home / ".annolid").exists()


def test_config_roundtrip_preserves_mcp_env_and_provider_headers_with_0600(
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "config.json"
    cfg = AgentConfig()
    cfg.providers["openrouter"] = ProviderConfig(
        api_key="or-secret",
        extra_headers={
            "HTTP-Referer": "https://annolid.example",
            "X-Title": "Annolid Bot",
        },
    )
    cfg.tools.mcp_servers["myServer"] = MCPServerConfig(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_TOKEN": "ghp_test123", "NODE_ENV": "production"},
    )
    save_config(cfg, cfg_path)

    mode = stat.S_IMODE(cfg_path.stat().st_mode)
    assert mode == 0o600

    reloaded = load_config(cfg_path)
    assert reloaded.providers["openrouter"].extra_headers == {
        "HTTP-Referer": "https://annolid.example",
        "X-Title": "Annolid Bot",
    }
    assert "myServer" in reloaded.tools.mcp_servers
    assert reloaded.tools.mcp_servers["myServer"].env == {
        "GITHUB_TOKEN": "ghp_test123",
        "NODE_ENV": "production",
    }


def test_streaming_chunks_preserve_repeated_tokens_and_filter_stale_turns(
    monkeypatch,
) -> None:
    _ensure_qapp()
    captured: list[str] = []

    class _DummyWidget:
        def enqueue_outbound_bus_message(self, payload: str) -> None:
            captured.append(payload)

    def _fake_invoke(widget, slot_name, _conn, qarg):
        # Simulate PyQt5/PyQt6 returning None on successful void invokeMethod.
        getattr(widget, slot_name)(qarg)
        return None

    import annolid.core.agent.gui_backend.session_io as session_io_mod

    monkeypatch.setattr(session_io_mod.QMetaObject, "invokeMethod", _fake_invoke)
    monkeypatch.setattr(session_io_mod.QtCore, "Q_ARG", lambda _t, val: val)

    dummy = _DummyWidget()
    emit_chunk(widget=dummy, chunk="hello", turn_id="turn-1")
    emit_chunk(widget=dummy, chunk=" ", turn_id="turn-1")
    emit_chunk(widget=dummy, chunk="the", turn_id="turn-1")
    emit_chunk(widget=dummy, chunk=" ", turn_id="turn-1")
    emit_chunk(widget=dummy, chunk="the", turn_id="turn-1")
    # Verify no triple delivery when invokeMethod returns None.
    assert len(captured) == 5

    widget = AIChatWidget()
    try:
        widget._prepare_streaming_turn_ui("hello")
        widget._chat_task_active = True
        widget._active_turn_id = "turn-1"
        for payload in captured:
            widget.consume_outbound_chat_event(payload)
        assert widget._current_response_bubble is not None
        assert widget._current_response_bubble.raw_text() == "hello the the"

        # Stale final event from a cancelled/other turn must be ignored.
        stale_final = encode_outbound_chat_event(
            OutboundChatEvent(
                direction="outbound",
                kind="final",
                text="Stopped by user.",
                turn_id="turn-old",
                turn_status="cancelled",
            )
        )
        widget.consume_outbound_chat_event(stale_final)
        assert widget._chat_task_active is True
        assert widget._active_turn_id == "turn-1"
        assert widget._current_response_bubble.raw_text() == "hello the the"
    finally:
        widget.close()


def test_ai_chat_widget_shares_session_store_and_reads_last_assistant_text(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    widget = AIChatWidget()
    try:
        assert widget._session_store is get_chat_session_store()
        widget._add_bubble("You", "Hi", is_user=True)
        widget._add_bubble("Assistant", "First reply", is_user=False)
        widget._add_bubble("Assistant", "Latest reply", is_user=False)
        assert widget._last_assistant_text() == "Latest reply"

        # Explicit attached image must not be overwritten by hidden canvas checkbox.
        img = tmp_path / "web_img.png"
        img.write_bytes(b"png")
        widget.attach_canvas_checkbox.setChecked(True)
        widget.set_image_path(str(img), explicit=True)
        assert widget._prepare_chat_image(chat_mode="default") == str(img)
    finally:
        widget.close()


def test_chat_bubble_light_mode_uses_dark_readable_text() -> None:
    _ensure_qapp()
    user_bubble = _ChatBubble("You", "Hello", is_user=True)
    bot_bubble = _ChatBubble("Assistant", "World", is_user=False)
    try:
        user_bubble.apply_theme(is_dark=False)
        bot_bubble.apply_theme(is_dark=False)
        assert "#172b4d" in user_bubble.message_doc.defaultStyleSheet()
        assert "#FFFFFF" not in user_bubble.message_doc.defaultStyleSheet()
        assert "#202124" in bot_bubble.message_doc.defaultStyleSheet()
        assert "#E0E0E0" not in bot_bubble.message_doc.defaultStyleSheet()
    finally:
        user_bubble.close()
        bot_bubble.close()


def test_slash_command_word_boundaries_quotes_dedup_and_exec_guard() -> None:
    # Word boundaries prevent typos/unknown commands from triggering side effects.
    assert parse_direct_slash_command("/dreaming unknown-subcommand") == {}
    assert parse_direct_slash_command("/tracker video.mp4") == {}
    assert parse_direct_slash_command("/capsule") == {}
    assert parse_direct_slash_command("/cronjob status") == {}
    assert parse_direct_slash_command("/github status") == {}

    # Unbalanced apostrophe in /track must not raise ValueError.
    parsed_track = parse_direct_slash_command("/track mouse's head")
    assert parsed_track["name"] == "segment_track_video"
    assert "mouse's" in parsed_track["args"]["text_prompt"]

    # Root completion entries must not duplicate /capabilities.
    entries = build_root_slash_completion_entries()
    cap_entries = [row for row in entries if row.get("search") == "/capabilities"]
    assert len(cap_entries) == 1

    # Ordinary English sentences starting with "Execute..." must not run shell commands.
    assert parse_direct_gui_command("Execute this analysis plan") == {}
    assert parse_direct_gui_command("Execution summary for mouse 1") == {}
    assert parse_direct_gui_command("exec pwd") == {
        "name": "exec_command",
        "args": {"command": "pwd"},
    }
    assert parse_direct_gui_command("!echo ok") == {
        "name": "exec_command",
        "args": {"command": "echo ok"},
    }


def test_runtime_config_and_provider_dependency_helpers() -> None:
    assert browser_first_for_web({"agent": {"browser_first_for_web": "false"}}) is False
    assert browser_first_for_web({"agent": {"browser_first_for_web": "0"}}) is False
    assert browser_first_for_web({"agent": {"browser_first_for_web": "true"}}) is True
    msg = format_dependency_error(
        raw_error="No module named 'openai'",
        settings={"providers": {"openai": {"kind": "openai_compat"}}},
        provider="openai",
    )
    assert "pip install openai" in msg


def test_onboarding_dialog_preserves_preview_and_agent_run_dialog_roundtrips(
    monkeypatch, tmp_path: Path
) -> None:
    _ensure_qapp()
    import annolid.gui.widgets.agent_workspace_onboarding_dialog as mod

    monkeypatch.setattr(
        mod,
        "get_agent_status",
        lambda workspace=None: {
            "workspace": str(tmp_path),
            "workspace_templates": {"AGENTS.md": False},
            "workspace_health": {"guidance": []},
            "workspace_backup_count": 0,
        },
    )
    monkeypatch.setattr(
        mod,
        "onboard_agent_workspace",
        lambda **_kw: {
            "workspace": str(tmp_path),
            "dry_run": True,
            "overwrite": False,
            "prune_bootstrap": False,
            "files": {"AGENTS.md": "would_created"},
            "pruned_files": {},
            "summary": {"would_created": 1},
            "workspace_health_after": {"template_missing_count": 1},
        },
    )
    dlg = AgentWorkspaceOnboardingDialog()
    try:
        dlg.workspace_edit.setText(str(tmp_path))
        dlg._run_preview()
        assert "Preview completed." in dlg.summary_label.text()
        assert dlg.results_table.rowCount() == 1
        assert dlg.results_table.item(0, 2).text() == "would_created"
    finally:
        dlg.close()

    run_dlg = AgentRunDialog()
    try:
        values = run_dlg.values()
        assert values["max_frames"] is None
        run_dlg2 = AgentRunDialog(config=values)
        try:
            assert run_dlg2.max_frames_spin.value() == -1
            assert run_dlg2.llm_adapter_combo.currentText() == "none"
            run_dlg2.include_summary_chk.setChecked(True)
            assert run_dlg2.llm_model_edit.isEnabled() is False
        finally:
            run_dlg2.close()
    finally:
        run_dlg.close()


def test_lazy_ai_chat_manager_exposes_background_services() -> None:
    lazy = _LazyAIChatManager(window=None)
    assert lazy._task_scheduler is None
    assert lazy._cron_service is None

    class _FakeManager:
        _task_scheduler = object()
        _cron_service = object()

    fake = _FakeManager()
    lazy._manager = fake
    assert lazy._task_scheduler is fake._task_scheduler
    assert lazy._cron_service is fake._cron_service


def test_stats_slash_tracking_stats_regex_temporal_gap_and_delete_cap(
    tmp_path: Path,
) -> None:
    stats_dir = tmp_path / "stats_root"
    stats_file = stats_dir / "mouse[1](a)" / "mouse[1](a)_tracking_stats.json"
    stats_file.parent.mkdir(parents=True)
    stats_file.write_text(
        json.dumps(
            {
                "version": 4,
                "video_name": "mouse[1](a).mp4",
                "summary": {
                    "manual_frames": 3,
                    "abnormal_segment_events": 4,
                },
                "bad_shape_events": [
                    {"frame": 1, "label": "mouse", "resolved": False},
                    {"frame": 2, "label": "mouse", "resolved": False},
                ],
            }
        ),
        encoding="utf-8",
    )

    filtered = analyze_tracking_stats_tool(
        root_dir=str(stats_dir),
        video_id="mouse[1](a)",
        include_plots=False,
    )
    assert filtered["ok"] is True
    assert filtered["video_count"] == 1

    slash_res = run_chat_stats_slash_tool(path=str(stats_dir))
    assert slash_res["ok"] is True
    assert slash_res["mode"] == "tracking_stats"
    assert "Discovered stats files: 1" in slash_res["summary_text"]
    assert "Abnormal segments: 4" in slash_res["summary_text"]
    assert "Bad-shape events: 2" in slash_res["summary_text"]

    # Temporal gap repair must not hallucinate beyond max_gap_frames.
    records = [
        (
            0,
            {
                "frame": 0,
                "shapes": [
                    {
                        "label": "mouse",
                        "group_id": 1,
                        "points": [[0.0, 0.0], [10.0, 10.0]],
                        "shape_type": "rectangle",
                    }
                ],
            },
        ),
        (1, {"frame": 1, "shapes": []}),
        (2, {"frame": 2, "shapes": []}),
        (3, {"frame": 3, "shapes": []}),
        (4, {"frame": 4, "shapes": []}),
    ]
    repaired, stats = _repair_temporal_records(
        records,
        start_frame=0,
        expected_instance_count=1,
        max_gap_frames=2,
        max_match_distance=50.0,
    )
    assert stats["missing_shapes_filled"] == 2
    assert set(repaired.keys()) == {1, 2}

    # Multi-frame shape deletion must respect max_delete cumulatively.
    store_path = tmp_path / "video_annotations.ndjson"
    store_lines = [
        json.dumps(
            {
                "frame": f,
                "shapes": [
                    {
                        "label": "noise",
                        "points": [[0, 0], [1, 1]],
                        "shape_type": "rectangle",
                    },
                    {
                        "label": "noise",
                        "points": [[2, 2], [3, 3]],
                        "shape_type": "rectangle",
                    },
                ],
            }
        )
        for f in range(3)
    ]
    store_path.write_text("\n".join(store_lines) + "\n", encoding="utf-8")
    del_res = delete_shapes_in_annotation_tool(
        path=str(store_path),
        exact_label="noise",
        apply_all_frames=True,
        max_delete=3,
        workspace=tmp_path,
    )
    assert del_res["ok"] is True
    assert del_res["deleted_shapes"] == 3


def test_citation_path_traversal_and_camera_url_redaction(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    escaped_rel = resolve_bib_output_path("../../evil.bib", workspace=ws)
    assert escaped_rel.parent == ws.resolve()
    assert escaped_rel.name == "evil.bib"

    escaped_abs = resolve_bib_output_path(str(ws / ".." / "outside.bib"), workspace=ws)
    assert escaped_abs.parent == ws.resolve()
    assert escaped_abs.name == "outside.bib"

    redacted = _redact_camera_source_for_bot_display(
        "rtsp://admin:p@ssword@cam.example.org:554/stream?token=secret123&chan=1"
    )
    assert "p@ssword" not in redacted
    assert "ssword" not in redacted
    assert "secret123" not in redacted
    assert "cam.example.org:554" in redacted
    assert "chan=1" in redacted


def test_session_history_preserves_tool_call_assistant_and_paired_tool_messages(
    tmp_path: Path,
) -> None:
    store = PersistentSessionStore(AgentSessionManager(sessions_dir=tmp_path))
    store.append_history(
        "s1",
        [
            {"role": "user", "content": "check file"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "a.txt"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "name": "read_file",
                "content": "ok",
            },
            {"role": "assistant", "content": "Done."},
        ],
        max_messages=20,
    )
    loaded = load_history_messages(
        session_store=store,
        session_id="s1",
        max_history_messages=20,
    )
    assert [m["role"] for m in loaded] == ["user", "assistant", "tool", "assistant"]
    assert loaded[1]["tool_calls"][0]["id"] == "call_1"

    # When slicing cuts off the assistant tool_calls message, leading orphan tool row is dropped.
    sliced = load_history_messages(
        session_store=store,
        session_id="s1",
        max_history_messages=2,
    )
    assert [m["role"] for m in sliced] == ["assistant"]


def test_channel_allowlist_blocks_substring_and_blank_entries() -> None:
    class _DummyChannel(BaseChannel):
        name = "dummy"

        async def start(self) -> None:
            return None

        async def stop(self) -> None:
            return None

        async def send(self, msg: Any) -> None:
            del msg

    ch = _DummyChannel({"allow_from": ["alice@example.com", ""]}, MessageBus())
    assert ch.is_allowed("alice@example.com") is True
    assert ch.is_allowed("Alice <alice@example.com>") is True
    assert ch.is_allowed("evil-alice@example.com.attacker.com") is False
    assert ch.is_allowed("bob@example.com") is False

    wa = _DummyChannel({"allow_from": ["15551234567"]}, MessageBus())
    assert wa.is_allowed("15551234567@s.whatsapp.net") is True
    assert wa.is_allowed("9155512345670@s.whatsapp.net") is False


def test_whatsapp_python_bridge_requires_auth_before_event_replay() -> None:
    class _FakeWS:
        def __init__(self, messages: list[str]) -> None:
            self._messages = list(messages)
            self.sent: list[str] = []
            self.closed: tuple[int, str] | None = None

        def __aiter__(self):
            return self

        async def __anext__(self) -> str:
            if not self._messages:
                raise StopAsyncIteration
            return self._messages.pop(0)

        async def send(self, payload: str) -> None:
            self.sent.append(payload)

        async def close(self, code: int = 1000, reason: str = "") -> None:
            self.closed = (code, reason)

    async def _run() -> None:
        bridge = WhatsAppPythonBridge(token="secret-token")
        bridge._provider_started = True
        bridge._recent_events = [{"type": "qr", "qr": "SENSITIVE_QR"}]

        bad_ws = _FakeWS([json.dumps({"type": "auth", "token": "wrong"})])
        await bridge._on_client(bad_ws)
        assert bad_ws.sent == []
        assert bad_ws.closed == (4003, "Invalid bridge token")
        assert bad_ws not in bridge._clients

        good_ws = _FakeWS([json.dumps({"type": "auth", "token": "secret-token"})])
        await bridge._on_client(good_ws)
        assert len(good_ws.sent) == 1
        assert "SENSITIVE_QR" in good_ws.sent[0]

    asyncio.run(_run())


def test_git_cli_blocks_config_injection_and_diff_flag_injection(
    tmp_path: Path,
) -> None:
    git_cli = GitCliTool(allowed_read_roots=[str(tmp_path)])
    assert git_cli._is_read_only(["-c", "core.pager=sh -c id", "log"]) is False
    assert git_cli._is_read_only(["config", "user.name"]) is False
    assert git_cli._is_read_only(["status", "--short"]) is True

    res_raw = asyncio.run(
        git_cli.execute(
            args=["-c", "core.pager=sh -c id", "log"],
            repo_path=str(tmp_path),
            allow_mutation=True,
        )
    )
    res = json.loads(res_raw)
    assert "Blocked unsafe git option" in res["error"]

    git_diff = GitDiffTool(allowed_read_roots=[str(tmp_path)])
    diff_raw = asyncio.run(
        git_diff.execute(repo_path=str(tmp_path), target="--output=/tmp/pwn")
    )
    assert "not allowed" in json.loads(diff_raw)["error"]


def test_tool_validation_union_types_text_probe_and_policy_fail_closed(
    tmp_path: Path,
) -> None:
    class _UnionTool(FunctionTool):
        @property
        def name(self) -> str:
            return "union_test"

        @property
        def description(self) -> str:
            return "test"

        @property
        def parameters(self) -> dict[str, Any]:
            return {
                "type": "object",
                "properties": {
                    "note": {"type": ["string", "null"]},
                },
            }

        async def execute(self, **kwargs: Any) -> str:
            return json.dumps(kwargs)

    tool = _UnionTool()
    assert tool.validate_params({"note": None}) == []
    assert tool.validate_params({"note": "hello"}) == []
    assert tool.validate_params({"note": 123}) != []

    # _is_probably_text_file must work without calling Path.read_bytes().
    sample = tmp_path / "sample.txt"
    sample.write_text("hello world", encoding="utf-8")
    assert _is_probably_text_file(sample, probe_bytes=16) is True

    # Atomic overwrite in RenameFileTool preserves target if overwrite=False and replaces if True.
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("new", encoding="utf-8")
    dst.write_text("old", encoding="utf-8")
    rename_tool = RenameFileTool(allowed_dir=tmp_path)
    out = asyncio.run(
        rename_tool.execute(path=str(src), new_path=str(dst), overwrite=True)
    )
    assert out.startswith("Successfully renamed")
    assert dst.read_text(encoding="utf-8") == "new"

    # Unknown tool policy profile fails closed to minimal (not full), and
    # provider deny patterns flow into ToolPermissionContext.
    cfg = ToolsConfig(
        profile="typo_profile_name",
        by_provider={
            "ollama": ToolPolicyConfig(
                profile="minimal",
                allow=["read_file"],
                deny=["mcp_*"],
            )
        },
    )
    resolved = resolve_allowed_tools(
        all_tool_names=["gui_context", "read_file", "exec", "mcp_fs_read"],
        tools_cfg=cfg,
        provider="ollama",
        model="qwen3",
    )
    assert "exec" not in resolved.allowed_tools
    ctx = build_tool_permission_context(
        all_tool_names=["gui_context", "read_file", "exec", "mcp_fs_read"],
        resolved_policy=resolved,
    )
    assert ctx.blocks("mcp_github_list") is True


def test_streaming_chat_task_inbound_fallback_model_and_behavior_catalog_error() -> (
    None
):
    task = StreamingChatTask("hi", widget=None)
    assert task.inbound is None
    assert task._resolve_gui_track_fallback_model_name() == "Cutie"

    task._invoke_widget_json_slot = lambda *_a, **_kw: {
        "ok": False,
        "error": "Failed to run GUI action",
        "transport_error": True,
    }  # type: ignore[method-assign]
    res = task._tool_gui_behavior_catalog(action="list")
    assert res["ok"] is False


def test_sync_dream_cron_job_updates_interval_hours(tmp_path: Path) -> None:
    svc = CronService(store_path=tmp_path / "jobs.json")
    job_id = _sync_dream_cron_job(cron_service=svc, enabled=True, interval_hours=6)
    assert job_id is not None
    jobs = svc.list_jobs(include_disabled=True)
    assert jobs[0].schedule.every_ms == 6 * 3600 * 1000

    new_job_id = _sync_dream_cron_job(cron_service=svc, enabled=True, interval_hours=12)
    assert new_job_id is not None
    jobs = svc.list_jobs(include_disabled=True)
    assert len(jobs) == 1
    assert jobs[0].schedule.every_ms == 12 * 3600 * 1000


def test_agent_loop_mcp_disconnect_unregisters_tools_and_consolidation_failure_preserves_history(
    tmp_path: Path,
) -> None:
    class _DummyTool(FunctionTool):
        def __init__(self, tool_name: str) -> None:
            self._name = tool_name

        @property
        def name(self) -> str:
            return self._name

        @property
        def description(self) -> str:
            return "dummy"

        @property
        def parameters(self) -> dict[str, Any]:
            return {"type": "object", "properties": {}}

        async def execute(self, **kwargs: Any) -> str:
            del kwargs
            return "ok"

    from annolid.core.agent.loop import AgentMemoryConfig

    registry = FunctionToolRegistry()
    loop = AgentLoop(
        tools=registry,
        workspace=tmp_path,
        memory_config=AgentMemoryConfig(max_history_messages=20, memory_window=4),
        llm_callable=lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    registry.register(_DummyTool("mcp_demo_tool"))
    loop._finalize_mcp_registration(set())
    assert registry.has("mcp_demo_tool")
    asyncio.run(loop.close())
    assert not registry.has("mcp_demo_tool")

    # Consolidation failure must keep unconsolidated session history intact.
    history = [
        {"role": "user", "content": f"message number {i} with detailed context " * 20}
        for i in range(6)
    ]
    loop._memory_store.append_history("s_keep", history, max_messages=20)
    kept = asyncio.run(
        loop._consolidate_memory(
            session_id="s_keep",
            history=history,
        )
    )
    assert len(kept) == len(history)
    assert len(loop._memory_store.get_history("s_keep")) == len(history)


def test_admin_tools_enforce_workspace_bounds_and_do_not_leak_update_consent(
    tmp_path: Path, monkeypatch
) -> None:
    from annolid.core.agent.tools.function_admin import (
        AdminEvalRunTool,
        AdminMemoryFlushTool,
        AdminUpdateRunTool,
    )

    ws = tmp_path / "workspace"
    ws.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    traces = ws / "traces.jsonl"
    traces.write_text(
        json.dumps({"trace_id": "t1", "user_message": "hi", "expected_substring": "ok"})
        + "\n",
        encoding="utf-8",
    )
    cand = ws / "cand.jsonl"
    cand.write_text(
        json.dumps({"trace_id": "t1", "content": "ok"}) + "\n", encoding="utf-8"
    )

    eval_tool = AdminEvalRunTool(allowed_dir=ws)
    try:
        asyncio.run(
            eval_tool.execute(
                traces_path=str(traces),
                candidate_responses_path=str(cand),
                output_path=str(outside / "escape.json"),
            )
        )
        raise AssertionError(
            "Expected PermissionError for eval output outside workspace"
        )
    except PermissionError:
        pass
    assert not (outside / "escape.json").exists()

    flush_tool = AdminMemoryFlushTool(allowed_dir=ws)
    try:
        asyncio.run(flush_tool.execute(workspace=str(outside), note="escape"))
        raise AssertionError(
            "Expected PermissionError for memory flush outside workspace"
        )
    except PermissionError:
        pass

    monkeypatch.setenv("ANNOLID_BOT_UPDATE_REQUIRE_CONSENT", "1")
    monkeypatch.setenv("ANNOLID_OPERATOR_UPDATE_CONSENT_PHRASE", "SECRET_PHRASE_99")
    update_tool = AdminUpdateRunTool(project="annolid")
    raw = asyncio.run(update_tool.execute(channel="stable", execute=True))
    payload = json.loads(raw)
    assert payload["ok"] is False
    assert payload["reason"] == "operator_consent_required"
    assert "required_phrase" not in payload
    assert "SECRET_PHRASE_99" not in raw


def test_admin_tools_expose_provider_safe_names_and_resolve_legacy_dotted_aliases(
    tmp_path: Path,
) -> None:
    from annolid.core.agent.tool_call_utils import sanitize_tool_name
    from annolid.core.agent.tools.function_admin import (
        AdminEvalRunTool,
        AdminMemoryFlushTool,
        AdminSkillsRefreshTool,
        AdminUpdateRunTool,
    )
    from annolid.core.agent.tools.function_registry import FunctionToolRegistry
    from annolid.core.agent.tools.policy import (
        build_tool_permission_context,
        resolve_allowed_tools,
    )

    registry = FunctionToolRegistry()
    registry.register(AdminSkillsRefreshTool(allowed_dir=tmp_path))
    registry.register(AdminMemoryFlushTool(allowed_dir=tmp_path))
    registry.register(AdminEvalRunTool(allowed_dir=tmp_path))
    registry.register(AdminUpdateRunTool())

    definitions = registry.get_definitions()
    schema_names = [item["function"]["name"] for item in definitions]
    assert set(schema_names) == {
        "admin_skills_refresh",
        "admin_memory_flush",
        "admin_eval_run",
        "admin_update_run",
    }
    assert all(sanitize_tool_name(name) == name for name in schema_names)
    assert all("." not in name for name in schema_names)

    # Legacy dotted aliases still resolve via registry lookup and execution.
    for dotted, canonical in (
        ("skills.refresh", "admin_skills_refresh"),
        ("memory.flush", "admin_memory_flush"),
        ("eval.run", "admin_eval_run"),
        ("update.run", "admin_update_run"),
    ):
        assert registry.has(dotted)
        assert dotted in registry
        assert registry.get(dotted) is registry.get(canonical)

    flush_res = json.loads(
        asyncio.run(
            registry.execute("memory.flush", {"session_id": "s1", "note": "compat"})
        )
    )
    assert flush_res["ok"] is True

    # Policy denying legacy dotted name also blocks canonical underscored name.
    resolved = resolve_allowed_tools(
        all_tool_names=registry.tool_names,
        tools_cfg=ToolsConfig(profile="full", deny=["update.run"]),
        provider="openai",
        model="gpt-4o",
    )
    assert "admin_update_run" not in resolved.allowed_tools
    perm_ctx = build_tool_permission_context(
        all_tool_names=registry.tool_names,
        resolved_policy=resolved,
    )
    assert perm_ctx.blocks("admin_update_run") is True
    assert perm_ctx.blocks("update.run") is True


def test_agent_loop_destructive_tool_approval_callback_blocks_and_allows(
    tmp_path: Path,
) -> None:
    from annolid.core.agent.loop import AgentLoop
    from annolid.core.agent.tools.function_base import FunctionTool
    from annolid.core.agent.tools.function_registry import FunctionToolRegistry
    from annolid.core.agent.tools.policy import classify_destructive_tool_call

    assert classify_destructive_tool_call("exec", {"command": "rm -rf foo"}) is not None
    assert classify_destructive_tool_call("read_file", {"path": "a.txt"}) is None
    assert classify_destructive_tool_call("git_cli", {"allow_mutation": False}) is None
    assert (
        classify_destructive_tool_call("git_cli", {"allow_mutation": True}) is not None
    )
    assert classify_destructive_tool_call("gui_self_update", {"execute": False}) is None
    assert (
        classify_destructive_tool_call("gui_self_update", {"execute": True}) is not None
    )

    executed: list[str] = []

    class _FakeExecTool(FunctionTool):
        @property
        def name(self) -> str:
            return "exec"

        @property
        def description(self) -> str:
            return "fake exec"

        @property
        def parameters(self) -> dict[str, Any]:
            return {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            }

        async def execute(self, command: str = "") -> str:
            executed.append(command)
            return f"ran:{command}"

    def _make_llm(command: str):
        calls = 0

        async def _llm(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "name": "exec",
                            "arguments": {"command": command},
                        }
                    ],
                }
            return {"content": "done", "tool_calls": []}

        return _llm

    reg = FunctionToolRegistry()
    reg.register(_FakeExecTool())

    # 1. Denied by user callback -> tool must NOT run.
    seen_prompts: list[tuple[str, str]] = []

    async def _deny(name: str, args: dict[str, Any], reason: str) -> bool:
        del args
        seen_prompts.append((name, reason))
        return False

    loop_deny = AgentLoop(
        tools=reg,
        workspace=str(tmp_path),
        llm_callable=_make_llm("echo denied"),
        tool_approval_callback=_deny,
    )
    res_deny = asyncio.run(loop_deny.run("run echo", session_id="s-deny"))
    assert executed == []
    assert seen_prompts and seen_prompts[0][0] == "exec"
    assert "denied by user confirmation" in res_deny.tool_runs[0].result.lower()

    # 2. Callback raises -> fail closed, tool must NOT run.
    def _boom(_name: str, _args: dict[str, Any], _reason: str) -> bool:
        raise RuntimeError("modal crashed")

    loop_err = AgentLoop(
        tools=reg,
        workspace=str(tmp_path),
        llm_callable=_make_llm("echo boom"),
        tool_approval_callback=_boom,
    )
    res_err = asyncio.run(loop_err.run("run echo", session_id="s-err"))
    assert executed == []
    assert "denied by user confirmation" in res_err.tool_runs[0].result.lower()

    # 3. Approved by user callback -> tool executes.
    loop_allow = AgentLoop(
        tools=reg,
        workspace=str(tmp_path),
        llm_callable=_make_llm("echo allowed"),
        tool_approval_callback=lambda _n, _a, _r: True,
    )
    res_allow = asyncio.run(loop_allow.run("run echo", session_id="s-allow"))
    assert executed == ["echo allowed"]
    assert res_allow.tool_runs[0].result == "ran:echo allowed"


def test_gui_tool_approval_modal_secret_redaction_and_mixin_slot(monkeypatch) -> None:
    from qtpy import QtCore
    from annolid.gui.widgets import ai_chat_actions_mixin as actions_mod
    from annolid.gui.widgets import ai_chat_backend as backend_mod
    from annolid.gui.widgets.ai_chat_backend import StreamingChatTask
    from annolid.gui.widgets.ai_chat_widget import AIChatWidget

    _ensure_qapp()

    # Verify parameter preview redacts sensitive keys.
    preview = StreamingChatTask._format_tool_approval_args_preview(
        {
            "command": "rm temp.txt",
            "api_key": "sk-secret-123",
            "operator_consent": "APPROVE_ME",
            "token": "tok-xyz",
        }
    )
    assert "sk-secret-123" not in preview
    assert "APPROVE_ME" not in preview
    assert "tok-xyz" not in preview
    assert "[REDACTED]" in preview
    assert "rm temp.txt" in preview

    widget = AIChatWidget()
    try:
        # Mock QMessageBox in ai_chat_actions_mixin so we deterministically test
        # Deny, Approve Once, Approve for Session, and session-clear reset.
        mode = {"choice": "deny"}
        dialog_calls: list[str] = []

        class _FakeMsgBox:
            Warning = QtWidgets.QMessageBox.Warning
            AcceptRole = QtWidgets.QMessageBox.AcceptRole
            YesRole = QtWidgets.QMessageBox.YesRole
            RejectRole = QtWidgets.QMessageBox.RejectRole

            def __init__(self, parent=None) -> None:
                del parent
                self._buttons: dict[str, object] = {}
                self._clicked: object | None = None

            def setIcon(self, _icon) -> None:
                return None

            def setWindowTitle(self, _title: str) -> None:
                return None

            def setText(self, text: str) -> None:
                dialog_calls.append(text)

            def setInformativeText(self, _text: str) -> None:
                return None

            def addButton(self, label: str, _role) -> object:
                btn = object()
                self._buttons[label] = btn
                return btn

            def setDefaultButton(self, _btn) -> None:
                return None

            def setEscapeButton(self, _btn) -> None:
                return None

            def exec(self) -> int:
                choice = mode["choice"]
                if choice == "once":
                    self._clicked = self._buttons["Approve Once"]
                elif choice == "session":
                    self._clicked = self._buttons["Approve for Session"]
                else:
                    self._clicked = self._buttons["Deny"]
                return 0

            def clickedButton(self) -> object | None:
                return self._clicked

        monkeypatch.setattr(actions_mod.QtWidgets, "QMessageBox", _FakeMsgBox)

        payload_json = json.dumps(
            {
                "tool_name": "exec",
                "reason": "Execute a shell command on this machine",
                "args_preview": '{"command": "echo hi"}',
            }
        )
        # Invoke via Qt meta-object to verify @Slot registration across mixins.
        QtCore.QMetaObject.invokeMethod(
            widget,
            "request_tool_approval_json",
            QtCore.Qt.DirectConnection,
            QtCore.Q_ARG(str, payload_json),
        )
        res = widget.get_bot_action_result("tool_approval")
        assert res["approved"] is False
        assert res["scope"] == "denied"
        assert len(dialog_calls) == 1

        # Approve for Session caches approval and skips modal on 2nd call.
        mode["choice"] = "session"
        widget.request_tool_approval_json(payload_json)
        assert widget.get_bot_action_result("tool_approval")["approved"] is True
        assert len(dialog_calls) == 2

        mode["choice"] = "deny"
        widget.request_tool_approval_json(payload_json)
        assert widget.get_bot_action_result("tool_approval")["approved"] is True
        assert widget.get_bot_action_result("tool_approval")["scope"] == "session"
        assert len(dialog_calls) == 2  # cached, modal not reopened

        # Clearing conversation resets session-level approvals.
        widget.clear_chat_conversation()
        widget.request_tool_approval_json(payload_json)
        assert widget.get_bot_action_result("tool_approval")["approved"] is False
        assert len(dialog_calls) == 3

        # StreamingChatTask wires approval callback + permission context when enabled.
        captured_loop_kwargs: dict[str, Any] = {}
        monkeypatch.setattr(
            backend_mod,
            "build_chat_agent_loop",
            lambda **kwargs: captured_loop_kwargs.update(kwargs) or object(),
        )
        task = StreamingChatTask(
            prompt="delete shapes",
            image_path=None,
            model="gpt-4o",
            provider="openai",
            settings={"confirm_destructive_tools": True},
            widget=widget,
        )
        ctx = backend_mod._AgentExecutionContext(
            workspace=Path("/tmp"),
            allowed_read_roots=[],
            tools=backend_mod.FunctionToolRegistry(),
            system_prompt="sys",
            strict_runtime_tool_guard=True,
            tool_permission_context="perm-ctx-sentinel",
        )
        monkeypatch.setattr(task, "_resolve_loop_llm_callable", lambda: None)
        task._build_agent_loop_instance(
            context=ctx,
            prompt_needs_tools=True,
            mcp_servers={},
        )
        assert captured_loop_kwargs["tool_permission_context"] == "perm-ctx-sentinel"
        assert callable(captured_loop_kwargs["tool_approval_callback"])
    finally:
        widget.shutdown(wait_ms=100)
        widget.close()
