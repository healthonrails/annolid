from __future__ import annotations

from pathlib import Path

from annolid.core.agent.gui_backend.tool_handlers_filesystem import rename_file_tool
from annolid.core.agent.gui_backend.tool_handlers_realtime import (
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
