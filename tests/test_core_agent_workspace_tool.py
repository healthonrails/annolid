"""Tests for GoogleDriveTool and shared Google auth/tool config."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from annolid.core.agent.config.schema import GoogleAuthConfig, ToolsConfig
from annolid.core.agent.tools.google_drive import GoogleDriveTool


def test_google_drive_tool_name_and_description():
    tool = GoogleDriveTool()
    assert tool.name == "google_drive"
    assert "google drive" in tool.description.lower()
    schema = tool.to_schema()
    assert schema["function"]["name"] == "google_drive"


def test_google_drive_tool_parameters_contain_action():
    tool = GoogleDriveTool()
    params = tool.parameters
    assert "action" in params["properties"]


def test_google_drive_tool_rejects_missing_required_action_fields():
    tool = GoogleDriveTool()
    result = asyncio.run(tool.execute(action="get_file"))
    assert "requires `file_id`" in result


def test_google_drive_tool_preflight_without_files():
    ready, reason = GoogleDriveTool.preflight(
        credentials_file="/tmp/missing_credentials.json",
        token_file="/tmp/missing_token.json",
        allow_interactive_auth=False,
    )
    assert ready is False
    assert "missing" in reason.lower()


def test_google_auth_config_from_dict_and_roundtrip():
    cfg = GoogleAuthConfig.from_dict(
        {
            "credentialsFile": "~/.annolid/agent/client.json",
            "tokenFile": "~/.annolid/agent/token.json",
            "allowInteractiveAuth": True,
        }
    )
    assert cfg.credentials_file.endswith("client.json")
    assert cfg.token_file.endswith("token.json")
    assert cfg.allow_interactive_auth is True

    serialized = cfg.to_dict()
    restored = GoogleAuthConfig.from_dict(serialized)
    assert restored == cfg


def test_google_drive_and_google_auth_config_in_tools_config():
    cfg = ToolsConfig.from_dict(
        {
            "google_auth": {
                "credentials_file": "~/.annolid/agent/google_client.json",
                "token_file": "~/.annolid/agent/google_token.json",
                "allow_interactive_auth": True,
            },
            "google_drive": {"enabled": True},
        }
    )
    assert cfg.google_drive_enabled is True
    assert cfg.google_auth.credentials_file.endswith("google_client.json")

    serialized = cfg.to_dict()
    assert serialized["google_drive"]["enabled"] is True
    assert serialized["google_auth"]["allow_interactive_auth"] is True


def test_google_drive_tool_list_files_happy_path(monkeypatch):
    class _FakeFiles:
        def list(self, **kwargs):
            class _Req:
                def execute(self_nonlocal):
                    return {"files": [{"id": "1", "name": "a.txt"}]}

            return _Req()

    class _FakeService:
        def files(self):
            return _FakeFiles()

    tool = GoogleDriveTool()
    monkeypatch.setattr(tool, "_get_service", lambda: _FakeService())
    result = asyncio.run(tool.execute(action="list_files", max_results=5))
    payload = json.loads(result)
    assert payload["files"][0]["name"] == "a.txt"


def test_google_drive_tool_upload_saved_videos_batches_and_skips_realtime(
    tmp_path: Path, monkeypatch
):
    saved_video = tmp_path / "session_saved.mp4"
    saved_video.write_bytes(b"saved")
    realtime_video = tmp_path / "realtime_detect_clip.mp4"
    realtime_video.write_bytes(b"rt")

    tool = GoogleDriveTool(allowed_local_roots=[tmp_path])
    monkeypatch.setattr(tool, "_get_service", lambda: object())
    monkeypatch.setattr(
        tool,
        "_find_existing_file_by_name_and_size",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tool,
        "_upload_file_resumable",
        lambda *args, **kwargs: {"id": "up-1", "name": kwargs["local_path"].name},
    )

    result = asyncio.run(
        tool.execute(
            action="upload_saved_videos",
            source_dir=str(tmp_path),
            max_files=10,
            modified_within_hours=24,
        )
    )
    payload = json.loads(result)
    assert payload["action"] == "upload_saved_videos"
    assert payload["uploaded_count"] == 1
    assert payload["uploaded"][0]["local_path"].endswith("session_saved.mp4")


def test_google_drive_tool_upload_realtime_videos_only_realtime(
    tmp_path: Path, monkeypatch
):
    saved_video = tmp_path / "session_saved.mp4"
    saved_video.write_bytes(b"saved")
    realtime_dir = tmp_path / "realtime"
    realtime_dir.mkdir()
    realtime_video = realtime_dir / "clip.mp4"
    realtime_video.write_bytes(b"rt")

    tool = GoogleDriveTool(allowed_local_roots=[tmp_path])
    monkeypatch.setattr(tool, "_get_service", lambda: object())
    monkeypatch.setattr(
        tool,
        "_find_existing_file_by_name_and_size",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tool,
        "_upload_file_resumable",
        lambda *args, **kwargs: {"id": "up-2", "name": kwargs["local_path"].name},
    )

    result = asyncio.run(
        tool.execute(
            action="upload_realtime_videos",
            source_dir=str(tmp_path),
            max_files=10,
            modified_within_hours=24,
        )
    )
    payload = json.loads(result)
    assert payload["action"] == "upload_realtime_videos"
    assert payload["uploaded_count"] == 1
    assert payload["uploaded"][0]["local_path"].endswith("realtime/clip.mp4")


def test_google_drive_tool_upload_rejects_path_outside_allowed_roots(
    tmp_path: Path, monkeypatch
):
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    outside = tmp_path / "outside.mp4"
    outside.write_bytes(b"x")
    tool = GoogleDriveTool(allowed_local_roots=[allowed_root])
    monkeypatch.setattr(tool, "_get_service", lambda: object())

    result = asyncio.run(
        tool.execute(
            action="upload_file",
            local_path=str(outside),
        )
    )
    assert "outside allowed roots" in result.lower()
