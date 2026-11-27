from __future__ import annotations

import json
from pathlib import Path

from annolid.core.agent.workspace_bootstrap import (
    bootstrap_workspace,
    prune_bootstrap_workspace,
)


def test_bootstrap_workspace_creates_templates(tmp_path: Path) -> None:
    out = bootstrap_workspace(tmp_path)
    assert out["AGENTS.md"] == "created"
    assert out["SOUL.md"] == "created"
    assert out["USER.md"] == "created"
    assert out["IDENTITY.md"] == "created"
    assert out["TOOLS.md"] == "created"
    assert out["HEARTBEAT.md"] == "created"
    assert out["BOOT.md"] == "created"
    assert out["BOOTSTRAP.md"] == "created"
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


def test_bootstrap_workspace_dry_run_does_not_write(tmp_path: Path) -> None:
    out = bootstrap_workspace(tmp_path, dry_run=True)
    assert out["AGENTS.md"] == "would_created"
    assert not (tmp_path / "AGENTS.md").exists()


def test_bootstrap_workspace_overwrite_with_backup(tmp_path: Path) -> None:
    _ = bootstrap_workspace(tmp_path)
    agents = tmp_path / "AGENTS.md"
    agents.write_text("custom", encoding="utf-8")

    backup_root = tmp_path / "backup"
    out = bootstrap_workspace(tmp_path, overwrite=True, backup_root=backup_root)
    assert out["AGENTS.md"] == "overwritten"
    assert (backup_root / "AGENTS.md").exists()
    assert (backup_root / "AGENTS.md").read_text(encoding="utf-8") == "custom"


def test_prune_bootstrap_workspace_removes_stale_manifest_files(tmp_path: Path) -> None:
    _ = bootstrap_workspace(tmp_path)
    stale_rel = "legacy/OLD.md"
    stale_path = tmp_path / stale_rel
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    stale_path.write_text("legacy", encoding="utf-8")

    manifest_path = tmp_path / ".annolid" / "bootstrap-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    managed = list(manifest.get("managed_files") or [])
    managed.append(stale_rel)
    manifest["managed_files"] = managed
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    out = prune_bootstrap_workspace(tmp_path)
    assert out[stale_rel] == "removed"
    assert not stale_path.exists()


def test_prune_bootstrap_workspace_dry_run_keeps_files(tmp_path: Path) -> None:
    _ = bootstrap_workspace(tmp_path)
    stale_rel = "legacy/OLD.md"
    stale_path = tmp_path / stale_rel
    stale_path.parent.mkdir(parents=True, exist_ok=True)
    stale_path.write_text("legacy", encoding="utf-8")

    manifest_path = tmp_path / ".annolid" / "bootstrap-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    managed = list(manifest.get("managed_files") or [])
    managed.append(stale_rel)
    manifest["managed_files"] = managed
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    out = prune_bootstrap_workspace(tmp_path, dry_run=True)
    assert out[stale_rel] == "would_removed"
    assert stale_path.exists()


def test_bootstrap_workspace_upgrades_unmodified_managed_template_and_preserves_custom(
    tmp_path: Path,
) -> None:
    _ = bootstrap_workspace(tmp_path)
    tools_path = tmp_path / "TOOLS.md"
    agents_path = tmp_path / "AGENTS.md"
    manifest_path = tmp_path / ".annolid" / "bootstrap-manifest.json"

    # Simulate an older untouched bundled TOOLS.md recorded in managed_hashes,
    # alongside a user-customized AGENTS.md that does NOT match managed_hashes.
    old_tools_bytes = b"# Old Bundled TOOLS\n- legacy_tool()\n"
    tools_path.write_bytes(old_tools_bytes)
    agents_path.write_text("user customized agents instructions", encoding="utf-8")

    import hashlib

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("managed_hashes", {})["TOOLS.md"] = hashlib.sha256(
        old_tools_bytes
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    dry = bootstrap_workspace(tmp_path, overwrite=False, dry_run=True)
    assert dry["TOOLS.md"] == "would_updated"
    assert dry["AGENTS.md"] == "would_skipped"
    assert tools_path.read_bytes() == old_tools_bytes

    backup_root = tmp_path / "backup"
    applied = bootstrap_workspace(tmp_path, overwrite=False, backup_root=backup_root)
    assert applied["TOOLS.md"] == "updated"
    assert applied["AGENTS.md"] == "skipped"
    assert "google_colab" in tools_path.read_text(encoding="utf-8")
    assert (backup_root / "TOOLS.md").read_bytes() == old_tools_bytes
    assert (
        agents_path.read_text(encoding="utf-8") == "user customized agents instructions"
    )


def test_context_builder_upgrades_unmodified_managed_tools_without_touching_custom(
    tmp_path: Path,
) -> None:
    import hashlib

    from annolid.core.agent.context import AgentContextBuilder

    _ = bootstrap_workspace(tmp_path)
    tools_path = tmp_path / "TOOLS.md"
    manifest_path = tmp_path / ".annolid" / "bootstrap-manifest.json"

    old_tools_bytes = b"# Legacy Bundled TOOLS\nvideo_ffmpeg_process(operation?)\n"
    tools_path.write_bytes(old_tools_bytes)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.setdefault("managed_hashes", {})["TOOLS.md"] = hashlib.sha256(
        old_tools_bytes
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    prompt = AgentContextBuilder(tmp_path).build_system_prompt()
    assert "google_colab" in prompt
    assert "google_colab" in tools_path.read_text(encoding="utf-8")

    # Now customize TOOLS.md — AgentContextBuilder must preserve custom text verbatim.
    tools_path.write_text("Custom lab TOOLS instructions", encoding="utf-8")
    prompt2 = AgentContextBuilder(tmp_path).build_system_prompt()
    assert "Custom lab TOOLS instructions" in prompt2
    assert tools_path.read_text(encoding="utf-8") == "Custom lab TOOLS instructions"
