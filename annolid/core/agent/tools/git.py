from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Sequence

from .function_base import FunctionTool
from .common import _resolve_read_path


class _RepoCliTool(FunctionTool):
    def __init__(
        self,
        *,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
        timeout: int = 20,
        max_chars: int = 20000,
    ):
        self._allowed_dir = allowed_dir
        self._allowed_read_roots = tuple(allowed_read_roots or ())
        self._timeout = int(timeout)
        self._max_chars = int(max_chars)

    def _resolve_repo_path(self, repo_path: str | None) -> Path:
        candidate = str(repo_path or ".")
        return _resolve_read_path(
            candidate,
            allowed_dir=self._allowed_dir,
            allowed_read_roots=self._allowed_read_roots,
        )

    async def _run_command(self, args: Sequence[str], *, repo_path: Path) -> str:
        payload: dict[str, Any] = {
            "command": list(args),
            "repo_path": str(repo_path),
        }
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                cwd=str(repo_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self._timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                payload["error"] = (
                    f"Command timed out after {self._timeout} seconds: {' '.join(args)}"
                )
                return json.dumps(payload)
        except FileNotFoundError:
            payload["error"] = f"Command not found: {args[0]}"
            return json.dumps(payload)
        except Exception as exc:
            payload["error"] = str(exc)
            return json.dumps(payload)

        stdout_text = (stdout or b"").decode("utf-8", errors="replace")
        stderr_text = (stderr or b"").decode("utf-8", errors="replace")
        combined = stdout_text
        if stderr_text.strip():
            combined = (
                f"{stdout_text}\nSTDERR:\n{stderr_text}" if stdout_text else stderr_text
            )
        truncated = False
        if len(combined) > self._max_chars:
            combined = combined[: self._max_chars]
            truncated = True

        payload.update(
            {
                "exit_code": int(proc.returncode or 0),
                "truncated": truncated,
                "output": combined,
            }
        )
        return json.dumps(payload)


class GitStatusTool(_RepoCliTool):
    @property
    def name(self) -> str:
        return "git_status"

    @property
    def description(self) -> str:
        return "Show git working tree status for a local repository."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string"},
                "short": {"type": "boolean"},
            },
            "required": [],
        }

    async def execute(
        self, repo_path: str = ".", short: bool = True, **kwargs: Any
    ) -> str:
        del kwargs
        try:
            repo = self._resolve_repo_path(repo_path)
        except PermissionError as exc:
            return json.dumps({"error": str(exc), "repo_path": str(repo_path)})
        args = ["git", "status"]
        if bool(short):
            args.extend(["--short", "--branch"])
        return await self._run_command(args, repo_path=repo)


class GitDiffTool(_RepoCliTool):
    @property
    def name(self) -> str:
        return "git_diff"

    @property
    def description(self) -> str:
        return "Show git diff for local changes or a target revision/range."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string"},
                "cached": {"type": "boolean"},
                "target": {"type": "string"},
                "name_only": {"type": "boolean"},
            },
            "required": [],
        }

    async def execute(
        self,
        repo_path: str = ".",
        cached: bool = False,
        target: str | None = None,
        name_only: bool = False,
        **kwargs: Any,
    ) -> str:
        del kwargs
        try:
            repo = self._resolve_repo_path(repo_path)
        except PermissionError as exc:
            return json.dumps({"error": str(exc), "repo_path": str(repo_path)})
        args = ["git", "diff"]
        if bool(cached):
            args.append("--cached")
        if bool(name_only):
            args.append("--name-only")
        target_text = str(target or "").strip()
        if target_text:
            args.append(target_text)
        return await self._run_command(args, repo_path=repo)


class GitLogTool(_RepoCliTool):
    @property
    def name(self) -> str:
        return "git_log"

    @property
    def description(self) -> str:
        return "Show recent git commit history."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "repo_path": {"type": "string"},
                "max_count": {"type": "integer", "minimum": 1, "maximum": 200},
                "oneline": {"type": "boolean"},
            },
            "required": [],
        }

    async def execute(
        self,
        repo_path: str = ".",
        max_count: int = 10,
        oneline: bool = True,
        **kwargs: Any,
    ) -> str:
        del kwargs
        try:
            repo = self._resolve_repo_path(repo_path)
        except PermissionError as exc:
            return json.dumps({"error": str(exc), "repo_path": str(repo_path)})
        count = max(1, min(int(max_count), 200))
        args = ["git", "log", f"--max-count={count}"]
        if bool(oneline):
            args.append("--oneline")
        return await self._run_command(args, repo_path=repo)


class GitHubPrStatusTool(_RepoCliTool):
    @property
    def name(self) -> str:
        return "github_pr_status"

    @property
    def description(self) -> str:
        return "Show GitHub pull request status for the current branch using gh CLI."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"repo_path": {"type": "string"}},
            "required": [],
        }

    async def execute(self, repo_path: str = ".", **kwargs: Any) -> str:
        del kwargs
        try:
            repo = self._resolve_repo_path(repo_path)
        except PermissionError as exc:
            return json.dumps({"error": str(exc), "repo_path": str(repo_path)})
        return await self._run_command(["gh", "pr", "status"], repo_path=repo)


class GitHubPrChecksTool(_RepoCliTool):
    @property
    def name(self) -> str:
        return "github_pr_checks"

    @property
    def description(self) -> str:
        return "Show GitHub pull request checks for the current branch using gh CLI."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"repo_path": {"type": "string"}},
            "required": [],
        }

    async def execute(self, repo_path: str = ".", **kwargs: Any) -> str:
        del kwargs
        try:
            repo = self._resolve_repo_path(repo_path)
        except PermissionError as exc:
            return json.dumps({"error": str(exc), "repo_path": str(repo_path)})
        return await self._run_command(["gh", "pr", "checks"], repo_path=repo)


__all__ = [
    "_RepoCliTool",
    "GitStatusTool",
    "GitDiffTool",
    "GitLogTool",
    "GitHubPrStatusTool",
    "GitHubPrChecksTool",
]
