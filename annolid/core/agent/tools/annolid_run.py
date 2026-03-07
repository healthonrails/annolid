from __future__ import annotations

import asyncio
import io
import json
import os
import shlex
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any, Sequence

from annolid.engine.cli import main as annolid_run_main

from .common import _normalize_allowed_read_roots, _resolve_read_path
from .function_base import FunctionTool


class AnnolidRunTool(FunctionTool):
    """Run `annolid-run` commands through the in-process CLI entrypoint."""

    _MUTATING_TOKENS = frozenset(
        {
            "--apply",
            "--fix",
            "add",
            "build-regression",
            "flush",
            "gate",
            "migrate",
            "predict",
            "refresh",
            "remove",
            "rollback",
            "run",
            "set",
            "shadow",
            "train",
            "upsert",
            "write",
        }
    )
    _PREFIXES = (
        ("annolid-run",),
        ("python", "-m", "annolid.engine.cli"),
        ("python3", "-m", "annolid.engine.cli"),
        ("py", "-m", "annolid.engine.cli"),
    )

    def __init__(
        self,
        *,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ) -> None:
        self._allowed_dir = (
            Path(allowed_dir).expanduser().resolve()
            if allowed_dir is not None
            else None
        )
        self._allowed_read_roots = _normalize_allowed_read_roots(
            self._allowed_dir, allowed_read_roots
        )

    @property
    def name(self) -> str:
        return "annolid_run"

    @property
    def description(self) -> str:
        return (
            "Run an `annolid-run` CLI command through Annolid's in-process engine "
            "entrypoint. Use this for agent, memory, update, predict, train, and "
            "other supported `annolid-run` subcommands. Mutating commands require "
            "`allow_mutation=true`."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "minLength": 1},
                "argv": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "working_dir": {"type": "string"},
                "allow_mutation": {"type": "boolean"},
            },
        }

    async def execute(self, **kwargs: Any) -> str:
        argv = self._normalize_argv(kwargs.get("command"), kwargs.get("argv"))
        allow_mutation = bool(kwargs.get("allow_mutation", False))
        if self._is_mutating(argv) and not allow_mutation:
            return json.dumps(
                {
                    "ok": False,
                    "argv": argv,
                    "error": (
                        "This `annolid-run` command may modify state. "
                        "Retry with allow_mutation=true only when that is intended."
                    ),
                },
                ensure_ascii=False,
            )
        working_dir = self._resolve_working_dir(kwargs.get("working_dir"))
        return await asyncio.to_thread(
            self._run_cli,
            argv,
            working_dir,
            allow_mutation,
        )

    def _normalize_argv(self, command: Any, argv: Any) -> list[str]:
        parsed: list[str]
        if isinstance(argv, list) and argv:
            parsed = [str(item) for item in argv if str(item).strip()]
        else:
            parsed = shlex.split(str(command or "").strip())
        if not parsed:
            raise ValueError("Provide `command` or `argv` for annolid_run.")
        lowered = [part.strip() for part in parsed]
        for prefix in self._PREFIXES:
            if len(lowered) >= len(prefix) and tuple(lowered[: len(prefix)]) == prefix:
                parsed = parsed[len(prefix) :]
                break
        if parsed and str(parsed[0] or "").strip().lower() == "help":
            topic = [str(part) for part in parsed[1:] if str(part).strip()]
            if topic and topic[0].lower() in {"annolid-run", "annolid", "cli"}:
                topic = topic[1:]
            if len(topic) >= 2 and str(topic[0]).strip().lower() in {
                "train",
                "predict",
            }:
                parsed = [topic[0], topic[1], "--help-model"]
            else:
                parsed = [*topic, "--help"] if topic else ["--help"]
        if not parsed:
            raise ValueError("No annolid-run subcommand was provided.")
        return parsed

    def _resolve_working_dir(self, raw_path: Any) -> Path | None:
        text = str(raw_path or "").strip()
        if not text:
            return self._allowed_dir
        return _resolve_read_path(
            text,
            allowed_dir=self._allowed_dir,
            allowed_read_roots=self._allowed_read_roots,
        )

    @classmethod
    def _is_mutating(cls, argv: Sequence[str]) -> bool:
        tokens = {str(part or "").strip().lower() for part in argv}
        return bool(tokens.intersection(cls._MUTATING_TOKENS))

    @staticmethod
    def _run_cli(
        argv: Sequence[str],
        working_dir: Path | None,
        allow_mutation: bool,
    ) -> str:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        prev_cwd = Path.cwd()
        try:
            if working_dir is not None:
                os.chdir(working_dir)
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exit_code = annolid_run_main(list(argv))
        finally:
            os.chdir(prev_cwd)
        stdout_text = stdout_buffer.getvalue()
        stderr_text = stderr_buffer.getvalue()
        return json.dumps(
            {
                "ok": int(exit_code or 0) == 0,
                "exit_code": int(exit_code or 0),
                "argv": list(argv),
                "working_dir": str(working_dir) if working_dir is not None else "",
                "allow_mutation": bool(allow_mutation),
                "stdout": stdout_text,
                "stderr": stderr_text,
            },
            ensure_ascii=False,
        )
