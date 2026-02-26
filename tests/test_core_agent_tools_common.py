from pathlib import Path

from annolid.core.agent.tools.common import _resolve_read_path


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
