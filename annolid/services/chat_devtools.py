"""Service wrappers for filesystem, shell, and git-oriented chat tools."""

from __future__ import annotations

import json
from typing import Any, Optional

from annolid.infrastructure.agent_workspace import get_agent_workspace_path


async def chat_list_dir(path: str) -> dict[str, Any]:
    from annolid.core.agent.tools.filesystem import ListDirTool

    workspace = get_agent_workspace_path()
    tool = ListDirTool(allowed_read_roots=[str(workspace)])
    result = await tool.execute(path=path)
    if result.startswith("Error:"):
        return {"ok": False, "error": result}
    return {"ok": True, "result": result}


async def chat_read_file(path: str) -> dict[str, Any]:
    from annolid.core.agent.tools.filesystem import ReadFileTool

    workspace = get_agent_workspace_path()
    tool = ReadFileTool(allowed_read_roots=[str(workspace)])
    result = await tool.execute(path=path)
    if result.startswith("Error:"):
        return {"ok": False, "error": result}
    return {"ok": True, "result": result}


async def chat_exec_command(command: str) -> dict[str, Any]:
    from annolid.core.agent.tools.shell import ExecTool

    workspace = get_agent_workspace_path()
    tool = ExecTool(working_dir=str(workspace))
    result = await tool.execute(command=command)
    if result.startswith("Error:") and "timed out" not in result:
        return {"ok": False, "error": result}
    return {"ok": True, "result": result}


def _parse_tool_payload(raw: str, invalid_message: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except Exception:
        payload = {"error": str(raw or invalid_message)}
    if not isinstance(payload, dict):
        return {"ok": False, "error": invalid_message}
    if payload.get("error"):
        return {"ok": False, "error": str(payload.get("error"))}
    return {
        "ok": True,
        "result": str(payload.get("output") or "").strip(),
        "raw": payload,
    }


async def chat_git_status(
    *,
    repo_path: str = ".",
    short: bool = True,
    allowed_read_roots: list[str] | None = None,
) -> dict[str, Any]:
    from annolid.core.agent.tools.git import GitStatusTool

    workspace = get_agent_workspace_path()
    tool = GitStatusTool(
        allowed_dir=workspace,
        allowed_read_roots=allowed_read_roots,
    )
    raw = await tool.execute(repo_path=repo_path, short=short)
    return _parse_tool_payload(raw, "invalid_git_status_result")


async def chat_git_diff(
    *,
    repo_path: str = ".",
    cached: bool = False,
    target: Optional[str] = None,
    name_only: bool = False,
    allowed_read_roots: list[str] | None = None,
) -> dict[str, Any]:
    from annolid.core.agent.tools.git import GitDiffTool

    workspace = get_agent_workspace_path()
    tool = GitDiffTool(
        allowed_dir=workspace,
        allowed_read_roots=allowed_read_roots,
    )
    raw = await tool.execute(
        repo_path=repo_path,
        cached=cached,
        target=target,
        name_only=name_only,
    )
    return _parse_tool_payload(raw, "invalid_git_diff_result")


async def chat_git_log(
    *,
    repo_path: str = ".",
    max_count: int = 20,
    oneline: bool = True,
    allowed_read_roots: list[str] | None = None,
) -> dict[str, Any]:
    from annolid.core.agent.tools.git import GitLogTool

    workspace = get_agent_workspace_path()
    tool = GitLogTool(
        allowed_dir=workspace,
        allowed_read_roots=allowed_read_roots,
    )
    raw = await tool.execute(
        repo_path=repo_path,
        max_count=max_count,
        oneline=oneline,
    )
    return _parse_tool_payload(raw, "invalid_git_log_result")


async def chat_github_pr_status(
    *,
    repo_path: str = ".",
    allowed_read_roots: list[str] | None = None,
) -> dict[str, Any]:
    from annolid.core.agent.tools.git import GitHubPrStatusTool

    workspace = get_agent_workspace_path()
    tool = GitHubPrStatusTool(
        allowed_dir=workspace,
        allowed_read_roots=allowed_read_roots,
    )
    raw = await tool.execute(repo_path=repo_path)
    return _parse_tool_payload(raw, "invalid_github_pr_status_result")


async def chat_github_pr_checks(
    *,
    repo_path: str = ".",
    allowed_read_roots: list[str] | None = None,
) -> dict[str, Any]:
    from annolid.core.agent.tools.git import GitHubPrChecksTool

    workspace = get_agent_workspace_path()
    tool = GitHubPrChecksTool(
        allowed_dir=workspace,
        allowed_read_roots=allowed_read_roots,
    )
    raw = await tool.execute(repo_path=repo_path)
    return _parse_tool_payload(raw, "invalid_github_pr_checks_result")


async def chat_exec_start(
    *,
    command: str,
    working_dir: str = "",
    background: bool = True,
    timeout_s: float = 0.0,
    pty: bool = False,
) -> dict[str, Any]:
    from annolid.core.agent.tools.shell_sessions import ExecStartTool

    tool = ExecStartTool()
    raw = await tool.execute(
        command=str(command or ""),
        working_dir=str(working_dir or ""),
        background=bool(background),
        timeout_s=float(timeout_s or 0.0),
        pty=bool(pty),
    )
    try:
        payload = json.loads(raw)
    except Exception:
        payload = {"ok": False, "error": str(raw or "invalid_exec_start_result")}
    return (
        payload
        if isinstance(payload, dict)
        else {"ok": False, "error": "invalid_exec_start_result"}
    )


async def chat_exec_process(
    *,
    action: str,
    session_id: str = "",
    wait_ms: int = 0,
    tail_lines: int = 200,
    text: str = "",
    submit: bool = False,
) -> dict[str, Any]:
    from annolid.core.agent.tools.shell_sessions import ExecProcessTool

    tool = ExecProcessTool()
    raw = await tool.execute(
        action=str(action or ""),
        session_id=str(session_id or ""),
        wait_ms=int(wait_ms or 0),
        tail_lines=int(tail_lines or 200),
        text=str(text or ""),
        submit=bool(submit),
    )
    try:
        payload = json.loads(raw)
    except Exception:
        payload = {"ok": False, "error": str(raw or "invalid_exec_process_result")}
    return (
        payload
        if isinstance(payload, dict)
        else {"ok": False, "error": "invalid_exec_process_result"}
    )


__all__ = [
    "chat_exec_command",
    "chat_exec_process",
    "chat_exec_start",
    "chat_git_diff",
    "chat_git_log",
    "chat_git_status",
    "chat_github_pr_checks",
    "chat_github_pr_status",
    "chat_list_dir",
    "chat_read_file",
]
