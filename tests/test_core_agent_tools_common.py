from pathlib import Path

import pytest

from annolid.core.agent.tools.common import _resolve_read_path, _resolve_write_path


def test_resolve_read_path_rewrites_relative_annolid_workspace_alias() -> None:
    home_workspace = (Path.home() / ".annolid" / "workspace").resolve()
    resolved = _resolve_read_path(
        ".annolid/workspace/skills/weather/SKILL.md",
        allowed_read_roots=[str(home_workspace)],
    )
    assert resolved == (home_workspace / "skills" / "weather" / "SKILL.md").resolve()


def test_resolve_read_path_rewrites_cwd_prefixed_annolid_workspace_alias() -> None:
    home_workspace = (Path.home() / ".annolid" / "workspace").resolve()
    resolved = _resolve_read_path(
        "/tmp/project/.annolid/workspace/skills/weather/SKILL.md",
        allowed_read_roots=[str(home_workspace)],
    )
    assert resolved == (home_workspace / "skills" / "weather" / "SKILL.md").resolve()


def test_resolve_write_path_reports_hard_workspace_boundary(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}_outside.txt"

    with pytest.raises(PermissionError) as excinfo:
        _resolve_write_path(str(outside), allowed_dir=tmp_path)

    message = str(excinfo.value)
    assert "outside allowed directory" in message
    assert "hard policy boundary" in message
    assert "working_dir overrides" in message
