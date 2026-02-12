from __future__ import annotations

from pathlib import Path

from annolid.core.agent.workspace_bootstrap import bootstrap_workspace


def test_bootstrap_workspace_creates_templates(tmp_path: Path) -> None:
    out = bootstrap_workspace(tmp_path)
    assert out["AGENTS.md"] == "created"
    assert out["SOUL.md"] == "created"
    assert out["USER.md"] == "created"
    assert out["TOOLS.md"] == "created"
    assert out["HEARTBEAT.md"] == "created"
    assert out["memory/MEMORY.md"] == "created"
    assert out["memory/HISTORY.md"] == "created"

    assert (tmp_path / "AGENTS.md").exists()
    assert (tmp_path / "memory" / "MEMORY.md").exists()
    assert (tmp_path / "memory" / "HISTORY.md").exists()


def test_bootstrap_workspace_skips_without_overwrite(tmp_path: Path) -> None:
    _ = bootstrap_workspace(tmp_path)
    agents = tmp_path / "AGENTS.md"
    agents.write_text("custom", encoding="utf-8")

    out = bootstrap_workspace(tmp_path, overwrite=False)
    assert out["AGENTS.md"] == "skipped"
    assert agents.read_text(encoding="utf-8") == "custom"

    out2 = bootstrap_workspace(tmp_path, overwrite=True)
    assert out2["AGENTS.md"] == "overwritten"
    assert agents.read_text(encoding="utf-8") != "custom"
