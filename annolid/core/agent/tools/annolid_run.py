from __future__ import annotations

import asyncio
import io
import json
import os
import shlex
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from threading import Lock
from typing import Any, Sequence

from annolid.engine.cli import main as annolid_run_main

from .common import _normalize_allowed_read_roots, _resolve_read_path
from .function_base import FunctionTool


class AnnolidRunTool(FunctionTool):
    """Run `annolid-run` commands through the in-process CLI entrypoint."""

    _CLI_LOCK = Lock()

    _MUTATING_SUBCOMMANDS = frozenset(
        {
            "agent",
            "agent-behavior",
            "agent-config-migrate",
            "agent-cron-add",
            "agent-cron-enable",
            "agent-cron-remove",
            "agent-cron-run",
            "agent-eval",
            "agent-meta-learning-maintenance-run",
            "agent-onboard",
            "agent-secrets-migrate",
            "agent-secrets-remove",
            "agent-secrets-set",
            "agent-skills-import",
            "agent-update",
            "citations-format",
            "citations-remove",
            "citations-upsert",
            "collect-labels",
            "dino-kpseg-embeddings",
            "dino-kpseg-precompute",
            "dino-kpseg-split",
            "import-deeplabcut-training-data",
            "index-to-yolo",
            "predict",
            "repair-tracks",
            "report",
            "segment-body-parts",
            "tracking-stats",
            "train",
            "zone-analysis",
        }
    )
    _MUTATING_OPERATOR_VERBS = frozenset(
        {
            "add",
            "build-regression",
            "bundle",
            "flush",
            "gate",
            "install",
            "migrate",
            "refresh",
            "remove",
            "rollback",
            "run",
            "set",
            "shadow",
            "upsert",
            "write",
        }
    )
    _MUTATING_FLAGS = frozenset(
        {
            "-o",
            "--apply",
            "--comparisons-csv",
            "--csv-out",
            "--disagreements-csv",
            "--events-csv",
            "--execute",
            "--fix",
            "--issues-csv",
            "--json-out",
            "--out",
            "--out-json",
            "--output",
            "--output-dir",
            "--output-prefix",
            "--overwrite",
            "--persist",
            "--persist-json",
            "--report-out",
            "--review-csv",
            "--subjects-csv",
            "--summary-csv",
            "--transitions-csv",
            "--update",
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
        cleaned = [str(part or "").strip() for part in argv if str(part or "").strip()]
        if not cleaned:
            return False
        lowered = [part.lower() for part in cleaned]
        if any(flag in {"-h", "--help", "--help-model"} for flag in lowered):
            return False

        subcmd = lowered[0]
        if subcmd in cls._MUTATING_SUBCOMMANDS:
            return True
        if subcmd == "gap-report" and "--no-write-reports" not in lowered:
            return True

        # Check multi-word operator verbs (e.g. `update run`, `agent skills refresh`, `memory flush`)
        # only among leading non-flag command tokens that are not file paths.
        for token in lowered[:4]:
            if token.startswith("-"):
                break
            if "/" in token or "\\" in token or "." in token:
                continue
            if token in cls._MUTATING_OPERATOR_VERBS:
                return True

        for token in lowered:
            if not token.startswith("-"):
                continue
            flag_name = token.split("=", 1)[0]
            if flag_name in cls._MUTATING_FLAGS:
                return True
        return False

    @classmethod
    def _run_cli(
        cls,
        argv: Sequence[str],
        working_dir: Path | None,
        allow_mutation: bool,
    ) -> str:
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with cls._CLI_LOCK:
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
