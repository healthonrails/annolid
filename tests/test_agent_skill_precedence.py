from __future__ import annotations

import json
from pathlib import Path

from annolid.core.agent.skills import AgentSkillsLoader
from annolid.engine.cli import main as annolid_run


def _write_skill(root: Path, name: str, description: str) -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\ndescription: {description}\n---\n{description}\n",
        encoding="utf-8",
    )


def test_agent_skill_precedence_workspace_over_managed_over_builtin(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    builtin_root = tmp_path / "builtin" / "skills"
    managed_root = tmp_path / "managed" / "skills"
    workspace_root = workspace / "skills"

    _write_skill(builtin_root, "same", "builtin")
    _write_skill(managed_root, "same", "managed")
    _write_skill(workspace_root, "same", "workspace")

    loader = AgentSkillsLoader(
        workspace=workspace,
        builtin_skills_dir=builtin_root,
        managed_skills_dir=managed_root,
    )
    row = next(
        s for s in loader.list_skills(filter_unavailable=False) if s["name"] == "same"
    )
    assert row["source"] == "workspace"
    assert row["description"] == "workspace"
    assert row["path"] == str(workspace_root / "same" / "SKILL.md")


def test_agent_skill_refresh_command_reports_names(tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "workspace"
    _write_skill(workspace / "skills", "demo", "demo skill")
    rc = annolid_run(["agent", "skills", "refresh", "--workspace", str(workspace)])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["refreshed"] is True
    assert "demo" in payload["names"]
