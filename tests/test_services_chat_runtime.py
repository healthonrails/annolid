from __future__ import annotations

from pathlib import Path

from annolid.services.chat_runtime import (
    build_chat_pdf_search_roots,
    build_chat_vcs_read_roots,
    build_chat_workspace_roots,
    get_chat_attachment_roots,
    get_chat_allowed_read_roots,
    get_chat_camera_snapshots_dir,
    get_chat_email_defaults,
    get_chat_realtime_defaults,
    get_chat_tutorials_dir,
    get_chat_workspace,
    read_chat_memory_text,
    resolve_chat_pdf_path,
)


def test_chat_runtime_workspace_and_roots(monkeypatch, tmp_path: Path) -> None:
    import annolid.services.chat_runtime as chat_runtime_mod

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class _Cfg:
        class tools:
            allowed_read_roots = ["/tmp/alpha", "/tmp/beta"]

    monkeypatch.setattr(chat_runtime_mod, "get_agent_workspace_path", lambda: workspace)
    monkeypatch.setattr(chat_runtime_mod, "load_config", lambda: _Cfg())
    monkeypatch.setattr(
        chat_runtime_mod,
        "build_workspace_roots",
        lambda workspace_path, read_roots: [workspace_path, Path(read_roots[0])],
    )
    monkeypatch.setattr(
        chat_runtime_mod,
        "build_pdf_search_roots",
        lambda workspace_path, read_roots: [
            workspace_path / "pdfs",
            Path(read_roots[1]),
        ],
    )

    assert get_chat_workspace() == workspace
    assert get_chat_allowed_read_roots() == ["/tmp/alpha", "/tmp/beta"]
    assert build_chat_workspace_roots() == [workspace, Path("/tmp/alpha")]
    assert build_chat_pdf_search_roots() == [workspace / "pdfs", Path("/tmp/beta")]
    assert build_chat_vcs_read_roots() == [str(workspace), "/tmp/alpha"]
    assert get_chat_attachment_roots() == [workspace, "/tmp/alpha", "/tmp/beta"]


def test_resolve_chat_pdf_path_uses_workspace_roots(
    monkeypatch, tmp_path: Path
) -> None:
    import annolid.services.chat_runtime as chat_runtime_mod

    roots = [tmp_path / "workspace", tmp_path / "shared"]
    monkeypatch.setattr(chat_runtime_mod, "build_chat_workspace_roots", lambda: roots)
    monkeypatch.setattr(
        chat_runtime_mod,
        "resolve_pdf_path_for_roots",
        lambda raw_path, resolved_roots: resolved_roots[1] / raw_path,
    )

    result = resolve_chat_pdf_path("paper.pdf")

    assert result == roots[1] / "paper.pdf"


def test_chat_runtime_config_defaults_and_dirs(monkeypatch, tmp_path: Path) -> None:
    import annolid.services.chat_runtime as chat_runtime_mod

    workspace = tmp_path / "workspace"
    workspace.mkdir()

    class _EmailCfg:
        enabled = True
        default_to = "bot@example.com"
        smtp_host = "smtp.example.com"
        smtp_port = 2525
        imap_host = "imap.example.com"
        imap_port = 993
        user = "bot"
        password = "secret"

    class _Cfg:
        class tools:
            realtime = {
                "camera_index": "rtsp://camera",
                "bot_email_to": "rt@example.com",
            }
            email = _EmailCfg()
            allowed_read_roots = ["/tmp/shared"]

    monkeypatch.setattr(chat_runtime_mod, "get_agent_workspace_path", lambda: workspace)
    monkeypatch.setattr(chat_runtime_mod, "load_config", lambda: _Cfg())

    memory_dir = workspace / "memory"
    memory_dir.mkdir()
    memory_file = memory_dir / "MEMORY.md"
    memory_file.write_text("remembered source", encoding="utf-8")

    assert get_chat_realtime_defaults()["camera_index"] == "rtsp://camera"
    assert get_chat_email_defaults()["default_to"] == "bot@example.com"
    assert read_chat_memory_text("MEMORY.md") == "remembered source"
    assert get_chat_tutorials_dir() == workspace / "tutorials"
    assert get_chat_camera_snapshots_dir() == workspace / "camera_snapshots"
