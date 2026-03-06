"""Tests for GoogleWorkspaceTool, GWSSetupTool, and GWSToolConfig."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

from annolid.core.agent.config.schema import GWSToolConfig, ToolsConfig
from annolid.core.agent.tools.gws_setup import GWSSetupTool
from annolid.core.agent.tools.workspace import GoogleWorkspaceTool


# ---------------------------------------------------------------------------
# GoogleWorkspaceTool tests
# ---------------------------------------------------------------------------


def test_workspace_tool_name_and_description():
    tool = GoogleWorkspaceTool()
    assert tool.name == "google_workspace"
    assert "gws" in tool.description.lower()
    schema = tool.to_schema()
    assert schema["function"]["name"] == "google_workspace"


def test_workspace_tool_parameters_contain_service():
    tool = GoogleWorkspaceTool()
    params = tool.parameters
    assert "service" in params["properties"]
    assert "resource" in params["properties"]
    assert "method" in params["properties"]


def test_workspace_tool_is_available_with_gws(monkeypatch):
    monkeypatch.setattr(
        "annolid.core.agent.tools.workspace.resolve_command",
        lambda name: "/usr/local/bin/gws" if name == "gws" else None,
    )
    assert GoogleWorkspaceTool.is_available() is True


def test_workspace_tool_is_available_without_gws(monkeypatch):
    monkeypatch.setattr(
        "annolid.core.agent.tools.workspace.resolve_command", lambda name: None
    )
    assert GoogleWorkspaceTool.is_available() is False


def test_workspace_tool_rejects_missing_params():
    tool = GoogleWorkspaceTool()
    result = asyncio.run(tool.execute(service="drive"))
    payload = json.loads(result)
    assert "error" in payload


def test_workspace_tool_rejects_disallowed_service():
    tool = GoogleWorkspaceTool(allowed_services=["drive"])
    result = asyncio.run(tool.execute(service="gmail", resource="users", method="list"))
    payload = json.loads(result)
    assert "error" in payload
    assert "not allowed" in payload["error"]


def test_workspace_tool_drive_list():
    """Mock subprocess to simulate a successful gws drive files list."""

    async def _run():
        tool = GoogleWorkspaceTool()

        async def _fake_exec(*args, **kwargs):
            proc = AsyncMock()
            proc.communicate = AsyncMock(
                return_value=(
                    json.dumps({"files": [{"name": "test.txt"}]}).encode(),
                    b"",
                )
            )
            proc.returncode = 0
            return proc

        with (
            patch(
                "annolid.core.agent.tools.workspace.resolve_command",
                lambda name: "/usr/local/bin/gws" if name == "gws" else None,
            ),
            patch("asyncio.create_subprocess_exec", side_effect=_fake_exec),
        ):
            result = await tool.execute(
                service="drive",
                resource="files",
                method="list",
                params='{"pageSize": 5}',
            )
        payload = json.loads(result)
        assert "files" in payload

    asyncio.run(_run())


def test_workspace_tool_handles_missing_binary():
    """When gws is not found, return a clear error."""

    async def _run():
        tool = GoogleWorkspaceTool()

        async def _raise_fnf(*args, **kwargs):
            raise FileNotFoundError("gws not found")

        with (
            patch(
                "annolid.core.agent.tools.workspace.resolve_command",
                lambda name: "/usr/local/bin/gws" if name == "gws" else None,
            ),
            patch("asyncio.create_subprocess_exec", side_effect=_raise_fnf),
        ):
            result = await tool.execute(
                service="drive", resource="files", method="list"
            )
        payload = json.loads(result)
        assert "error" in payload
        assert "not found" in payload["error"]

    asyncio.run(_run())


def test_workspace_tool_handles_nonzero_exit():
    """Non-zero exit code returns stderr."""

    async def _run():
        tool = GoogleWorkspaceTool()

        async def _fake_exec(*args, **kwargs):
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b"", b"auth error"))
            proc.returncode = 1
            return proc

        with (
            patch(
                "annolid.core.agent.tools.workspace.resolve_command",
                lambda name: "/usr/local/bin/gws" if name == "gws" else None,
            ),
            patch("asyncio.create_subprocess_exec", side_effect=_fake_exec),
        ):
            result = await tool.execute(
                service="drive", resource="files", method="list"
            )
        payload = json.loads(result)
        assert payload["error"].startswith("gws exited")
        assert payload["stderr"] == "auth error"

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# GWSSetupTool tests
# ---------------------------------------------------------------------------


def test_gws_setup_tool_name():
    tool = GWSSetupTool()
    assert tool.name == "gws_setup"
    params = tool.parameters
    assert "action" in params["properties"]


def test_gws_setup_check(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "annolid.core.agent.tools.gws_setup.resolve_command",
        lambda name: "/usr/bin/gws"
        if name == "gws"
        else "/usr/bin/npm"
        if name == "npm"
        else None,
    )
    tool = GWSSetupTool(
        clone_dir=tmp_path / "clone", managed_skills_dir=tmp_path / "skills"
    )
    result = asyncio.run(tool.execute(action="check"))
    payload = json.loads(result)
    assert payload["gws_installed"] is True
    assert payload["npm_available"] is True
    assert payload["clone_exists"] is False


def test_gws_setup_check_no_gws(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "annolid.core.agent.tools.gws_setup.resolve_command", lambda name: None
    )
    tool = GWSSetupTool(
        clone_dir=tmp_path / "clone", managed_skills_dir=tmp_path / "skills"
    )
    result = asyncio.run(tool.execute(action="check"))
    payload = json.loads(result)
    assert payload["gws_installed"] is False


def test_gws_setup_invalid_action(tmp_path):
    tool = GWSSetupTool(clone_dir=tmp_path, managed_skills_dir=tmp_path)
    result = asyncio.run(tool.execute(action="bogus"))
    payload = json.loads(result)
    assert "error" in payload


def test_gws_setup_link_skills_creates_symlinks(tmp_path):
    """Simulates link_skills with a pre-existing clone directory."""
    clone_dir = tmp_path / "clone"
    skills_src = clone_dir / "skills"
    for skill_name in ["gws-shared", "gws-drive", "gws-gmail"]:
        skill_dir = skills_src / skill_name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(f"---\nname: {skill_name}\n---\n")

    managed = tmp_path / "managed"
    tool = GWSSetupTool(clone_dir=clone_dir, managed_skills_dir=managed)

    result = asyncio.run(
        tool.execute(action="link_skills", skills=["gws-shared", "gws-drive"])
    )
    payload = json.loads(result)
    assert payload["ok"] is True
    assert "gws-shared" in payload["linked"]
    assert "gws-drive" in payload["linked"]

    # Symlinks should exist
    assert (managed / "gws-shared").is_symlink()
    assert (managed / "gws-drive").is_symlink()

    # Running again should skip
    result2 = asyncio.run(
        tool.execute(action="link_skills", skills=["gws-shared", "gws-drive"])
    )
    payload2 = json.loads(result2)
    assert "gws-shared" in payload2["skipped"]


def test_gws_setup_link_skills_all(tmp_path):
    clone_dir = tmp_path / "clone"
    skills_src = clone_dir / "skills"
    for skill_name in ["gws-shared", "gws-drive", "gws-gmail", "gws-calendar"]:
        skill_dir = skills_src / skill_name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(f"---\nname: {skill_name}\n---\n")
    # Non-gws dir should be ignored
    (skills_src / "other-skill").mkdir()

    managed = tmp_path / "managed"
    tool = GWSSetupTool(clone_dir=clone_dir, managed_skills_dir=managed)

    result = asyncio.run(tool.execute(action="link_skills", skills=["all"]))
    payload = json.loads(result)
    assert payload["ok"] is True
    assert len(payload["linked"]) == 4
    assert "gws-shared" in payload["linked"]


def test_gws_setup_link_skills_missing_source(tmp_path):
    clone_dir = tmp_path / "clone"
    (clone_dir / "skills").mkdir(parents=True)
    managed = tmp_path / "managed"
    tool = GWSSetupTool(clone_dir=clone_dir, managed_skills_dir=managed)

    result = asyncio.run(tool.execute(action="link_skills", skills=["gws-nonexistent"]))
    payload = json.loads(result)
    assert payload["ok"] is False
    assert len(payload["errors"]) == 1


# ---------------------------------------------------------------------------
# GWSToolConfig tests
# ---------------------------------------------------------------------------


def test_gws_tool_config_default():
    cfg = GWSToolConfig()
    assert cfg.enabled is False
    assert cfg.auto_install is False
    assert "drive" in cfg.services


def test_gws_tool_config_from_dict():
    cfg = GWSToolConfig.from_dict(
        {
            "enabled": True,
            "autoInstall": True,
            "services": ["drive", "sheets"],
        }
    )
    assert cfg.enabled is True
    assert cfg.auto_install is True
    assert cfg.services == ["drive", "sheets"]


def test_gws_tool_config_roundtrip():
    original = GWSToolConfig(
        enabled=True,
        auto_install=True,
        services=["drive", "gmail"],
    )
    serialized = original.to_dict()
    restored = GWSToolConfig.from_dict(serialized)
    assert restored.enabled == original.enabled
    assert restored.auto_install == original.auto_install
    assert restored.services == original.services


def test_gws_tool_config_in_tools_config():
    cfg = ToolsConfig.from_dict(
        {
            "gws": {
                "enabled": True,
                "services": ["calendar"],
            }
        }
    )
    assert cfg.gws.enabled is True
    assert cfg.gws.services == ["calendar"]

    serialized = cfg.to_dict()
    assert "gws" in serialized
    assert serialized["gws"]["enabled"] is True
