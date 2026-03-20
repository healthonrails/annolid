from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from annolid.core.agent.gui_backend.context_setup import load_execution_prerequisites


def test_load_execution_prerequisites_includes_default_repo_root(
    monkeypatch, tmp_path: Path
) -> None:
    import annolid.core.agent.gui_backend.context_setup as context_setup_mod

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    repo_root = tmp_path / "annolid_repo"
    repo_root.mkdir()

    cfg = SimpleNamespace(
        tools=SimpleNamespace(
            allowed_read_roots=[
                str(tmp_path / "data_a"),
                str(tmp_path / "data_a"),
            ]
        )
    )

    monkeypatch.setattr(
        context_setup_mod, "get_agent_workspace_path", lambda: workspace
    )
    monkeypatch.setattr(context_setup_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(
        context_setup_mod, "_detect_annolid_repo_root", lambda: repo_root
    )

    prepared = load_execution_prerequisites()
    assert prepared.workspace == workspace
    assert prepared.allowed_read_roots == [str(repo_root), str(tmp_path / "data_a")]
