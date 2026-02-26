from __future__ import annotations

from pathlib import Path

from annolid.core.agent.gui_backend.tool_handlers_filesystem import rename_file_tool
from annolid.core.agent.gui_backend.tool_handlers_realtime import (
    check_stream_source_tool,
    get_realtime_status_tool,
    list_logs_tool,
    list_log_files_tool,
    read_log_file_tool,
    search_logs_tool,
    open_log_folder_tool,
    remove_log_folder_tool,
    list_realtime_logs_tool,
    list_realtime_models_tool,
    start_realtime_stream_tool,
    stop_realtime_stream_tool,
)


def test_start_realtime_stream_tool_rejects_bad_threshold() -> None:
    payload = start_realtime_stream_tool(
        camera_source="0",
        model_name="model",
        target_behaviors=[],
        confidence_threshold="bad",  # type: ignore[arg-type]
        viewer_type="threejs",
        classify_eye_blinks=False,
        blink_ear_threshold=None,
        blink_min_consecutive_frames=None,
        invoke_start=lambda *args: True,
        get_action_result=lambda _name: {},
    )
    assert payload["ok"] is False
    assert "confidence_threshold" in str(payload.get("error", ""))


def test_stop_realtime_stream_tool_success() -> None:
    payload = stop_realtime_stream_tool(invoke_stop=lambda: True)
    assert payload == {"ok": True, "queued": True}


def test_start_realtime_stream_tool_normalizes_rtsp_transport() -> None:
    captured: dict[str, object] = {}

    payload = start_realtime_stream_tool(
        camera_source="rtsp://10.0.0.2:554/stream1",
        model_name="model",
        target_behaviors=[],
        confidence_threshold=0.4,
        viewer_type="threejs",
        rtsp_transport="TCP",
        classify_eye_blinks=False,
        blink_ear_threshold=None,
        blink_min_consecutive_frames=None,
        invoke_start=lambda *args: bool(captured.setdefault("args", args) or True),
        get_action_result=lambda _name: {},
    )
    assert payload["ok"] is True
    assert payload["rtsp_transport"] == "tcp"
    assert isinstance(captured.get("args"), tuple)
    assert len(captured["args"]) == 10
    assert captured["args"][5] == "tcp"


def test_realtime_status_and_inventory_tools_proxy_widget_payloads() -> None:
    status = get_realtime_status_tool(
        invoke_widget_json_slot=lambda *_a, **_k: {
            "ok": True,
            "running": True,
            "camera_source": "rtsp://cam/stream",
        }
    )
    assert status["ok"] is True
    assert status["running"] is True

    models = list_realtime_models_tool(
        invoke_widget_json_slot=lambda *_a, **_k: {
            "ok": True,
            "count": 2,
            "models": [{"id": "yolo11n"}, {"id": "mediapipe_face"}],
        }
    )
    assert models["count"] == 2

    logs = list_realtime_logs_tool(
        invoke_widget_json_slot=lambda *_a, **_k: {
            "ok": True,
            "detections_log_path": "/tmp/a.ndjson",
            "bot_event_log_path": "/tmp/b.ndjson",
        }
    )
    assert str(logs["detections_log_path"]).endswith(".ndjson")

    app_logs = list_logs_tool(
        invoke_widget_json_slot=lambda *_a, **_k: {
            "ok": True,
            "count": 3,
            "logs": [{"target": "logs", "path": "/tmp/logs"}],
        }
    )
    assert app_logs["ok"] is True
    opened = open_log_folder_tool(
        target="logs",
        invoke_widget_json_slot=lambda *_a, **_k: {
            "ok": True,
            "target": "logs",
            "path": "/tmp/logs",
        },
    )
    assert opened["ok"] is True
    removed = remove_log_folder_tool(
        target="realtime",
        invoke_widget_json_slot=lambda *_a, **_k: {
            "ok": True,
            "target": "realtime",
            "removed": True,
        },
    )
    assert removed["ok"] is True

    files = list_log_files_tool(
        target="logs",
        pattern="*.log",
        limit=20,
        recursive=True,
        sort_by="mtime",
        descending=True,
        invoke_widget_json_slot=lambda *_a, **_k: {
            "ok": True,
            "target": "logs",
            "count": 1,
            "files": [{"path": "/tmp/logs/app/a.log"}],
        },
    )
    assert files["ok"] is True

    content = read_log_file_tool(
        path="/tmp/logs/app/a.log",
        max_chars=2000,
        tail_lines=50,
        invoke_widget_json_slot=lambda *_a, **_k: {
            "ok": True,
            "path": "/tmp/logs/app/a.log",
            "content": "hello",
        },
    )
    assert content["ok"] is True

    search = search_logs_tool(
        query="error",
        target="logs",
        pattern="*.log",
        case_sensitive=False,
        use_regex=False,
        max_matches=10,
        max_files=5,
        invoke_widget_json_slot=lambda *_a, **_k: {
            "ok": True,
            "match_count": 1,
            "matches": [{"path": "/tmp/logs/app/a.log", "line": 1, "text": "error"}],
        },
    )
    assert search["ok"] is True


def test_check_stream_source_tool_normalizes_probe_args() -> None:
    seen: dict[str, object] = {}

    payload = check_stream_source_tool(
        camera_source="rtsp://10.0.0.2:554/stream1",
        rtsp_transport="TCP",
        timeout_sec=99.0,
        probe_frames=0,
        save_snapshot=True,
        invoke_check=lambda source,
        transport,
        timeout_sec,
        probe_frames,
        save_snapshot: (
            seen.update(
                {
                    "source": source,
                    "transport": transport,
                    "timeout_sec": timeout_sec,
                    "probe_frames": probe_frames,
                    "save_snapshot": bool(save_snapshot),
                }
            )
            or {"ok": True}
        ),
    )
    assert payload["ok"] is True
    assert seen["source"] == "rtsp://10.0.0.2:554/stream1"
    assert seen["transport"] == "tcp"
    assert float(seen["timeout_sec"]) == 30.0
    assert int(seen["probe_frames"]) == 1
    assert seen["save_snapshot"] is True


def test_check_stream_source_tool_rejects_unsafe_source_shapes() -> None:
    payload_local_path = check_stream_source_tool(
        camera_source="../../etc/passwd",
        invoke_check=lambda *_args: {"ok": True},
    )
    assert payload_local_path["ok"] is False
    assert "camera index or a stream URL" in str(payload_local_path.get("error", ""))

    payload_bad_scheme = check_stream_source_tool(
        camera_source="javascript://alert(1)",
        invoke_check=lambda *_args: {"ok": True},
    )
    assert payload_bad_scheme["ok"] is False
    assert "Unsupported stream URL scheme" in str(payload_bad_scheme.get("error", ""))

    payload_long = check_stream_source_tool(
        camera_source="r" * 3000,
        invoke_check=lambda *_args: {"ok": True},
    )
    assert payload_long["ok"] is False
    assert "too long" in str(payload_long.get("error", ""))


def test_check_stream_source_tool_requires_valid_email_address() -> None:
    payload = check_stream_source_tool(
        camera_source="0",
        email_to="not-an-email",
        invoke_check=lambda *_args: {"ok": True},
    )
    assert payload["ok"] is False
    assert "valid email address" in str(payload.get("error", ""))


def test_check_stream_source_tool_forces_snapshot_when_email_requested() -> None:
    seen: dict[str, object] = {}
    payload = check_stream_source_tool(
        camera_source="0",
        save_snapshot=False,
        email_to="user@example.com",
        invoke_check=lambda source,
        transport,
        timeout_sec,
        probe_frames,
        save_snapshot: (
            seen.update(
                {
                    "source": source,
                    "transport": transport,
                    "timeout_sec": timeout_sec,
                    "probe_frames": probe_frames,
                    "save_snapshot": bool(save_snapshot),
                }
            )
            or {"ok": True}
        ),
    )
    assert payload["ok"] is True
    assert seen["save_snapshot"] is True
    assert payload["save_snapshot"] is True
    assert payload["email_requested"] is True
    assert payload["email_to"] == "user@example.com"


def test_rename_file_tool_requires_source_or_active() -> None:
    payload = rename_file_tool(
        source_path="",
        new_name="renamed.pdf",
        new_path="",
        use_active_file=False,
        overwrite=False,
        get_pdf_state=lambda: {},
        get_active_video_path=lambda: "",
        workspace=Path("/tmp"),
        run_rename=lambda *args: "",
        reopen_pdf=lambda _path: True,
    )
    assert payload["ok"] is False
    assert "No source file provided" in str(payload.get("error", ""))


def test_rename_file_tool_resolves_relative_source_with_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path
    source = workspace / "downloads" / "2509.21965v2.pdf"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"%PDF-1.4 test")
    seen: dict[str, str] = {}

    payload = rename_file_tool(
        source_path="downloads/2509.21965v2.pdf",
        new_name="paper-title.pdf",
        new_path="",
        use_active_file=False,
        overwrite=False,
        get_pdf_state=lambda: {},
        get_active_video_path=lambda: "",
        workspace=workspace,
        run_rename=lambda current, target_name, target_path, overwrite_flag: (
            seen.update(
                {
                    "current": current,
                    "target_name": target_name,
                    "target_path": target_path,
                    "overwrite": str(bool(overwrite_flag)),
                }
            )
            or f"Successfully renamed {current} -> {Path(current).with_name(target_name)}"
        ),
        reopen_pdf=lambda _path: False,
    )

    assert payload["ok"] is True
    assert seen["current"] == str(source)
    assert seen["target_name"] == "paper-title.pdf"
