from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from annolid.core.agent.eval.gate import evaluate_report_gate
from annolid.core.agent.eval.run_agent_eval import run_eval
from annolid.core.agent.memory import AgentMemoryStore
from annolid.core.agent.observability import emit_governance_event
from annolid.core.agent.security_policy import (
    bot_update_requires_operator_consent,
    has_operator_consent,
    require_signed_updates,
)
from annolid.core.agent.skills import AgentSkillsLoader
from annolid.core.agent.update_manager.rollback import (
    build_rollback_plan,
    execute_rollback,
)
from annolid.core.agent.update_manager.service import UpdateManagerService
from annolid.core.agent.utils import get_agent_workspace_path

from .common import _resolve_read_path, _resolve_write_path
from .function_base import FunctionTool


class AdminSkillsRefreshTool(FunctionTool):
    def __init__(self, *, allowed_dir: Optional[Path] = None) -> None:
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "admin_skills_refresh"

    @property
    def aliases(self) -> tuple[str, ...]:
        return ("skills.refresh",)

    @property
    def description(self) -> str:
        return "Refresh skills snapshot using workspace > managed > bundled precedence."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "workspace": {"type": "string"},
            },
            "additionalProperties": False,
        }

    async def execute(self, workspace: str = "") -> str:
        ws_arg = str(workspace or "").strip()
        if self._allowed_dir is not None:
            resolved = (
                _resolve_read_path(ws_arg, allowed_dir=self._allowed_dir)
                if ws_arg
                else Path(self._allowed_dir).expanduser().resolve()
            )
        else:
            resolved = get_agent_workspace_path(ws_arg or None)
        loader = AgentSkillsLoader(workspace=resolved)
        before = loader.list_skills(filter_unavailable=False)
        before_names = sorted(str(s.get("name") or "") for s in before)
        loader.refresh_snapshot()
        after = loader.list_skills(filter_unavailable=False)
        after_names = sorted(str(s.get("name") or "") for s in after)
        added = [name for name in after_names if name not in set(before_names)]
        removed = [name for name in before_names if name not in set(after_names)]
        payload = {
            "ok": True,
            "workspace": str(resolved),
            "count": len(after),
            "names": [str(s.get("name") or "") for s in after],
            "added": added,
            "removed": removed,
        }
        emit_governance_event(
            event_type="skills",
            action="refresh",
            outcome="ok",
            actor="operator",
            details={
                "workspace": str(resolved),
                "count_before": len(before_names),
                "count_after": len(after_names),
                "added": added,
                "removed": removed,
            },
        )
        return json.dumps(payload, ensure_ascii=False)


class AdminMemoryFlushTool(FunctionTool):
    def __init__(self, *, allowed_dir: Optional[Path] = None) -> None:
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "admin_memory_flush"

    @property
    def aliases(self) -> tuple[str, ...]:
        return ("memory.flush",)

    @property
    def description(self) -> str:
        return "Append a memory flush entry into workspace daily and history files."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "workspace": {"type": "string"},
                "session_id": {"type": "string"},
                "note": {"type": "string"},
            },
            "additionalProperties": False,
        }

    async def execute(
        self,
        workspace: str = "",
        session_id: str = "",
        note: str = "",
    ) -> str:
        ws_arg = str(workspace or "").strip()
        if self._allowed_dir is not None:
            resolved = (
                _resolve_write_path(ws_arg, allowed_dir=self._allowed_dir)
                if ws_arg
                else Path(self._allowed_dir).expanduser().resolve()
            )
        else:
            resolved = get_agent_workspace_path(ws_arg or None)
        store = AgentMemoryStore(resolved)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        text = str(note or "").strip() or "operator memory flush"
        sid = str(session_id or "").strip()
        entry = (
            f"[{stamp}] {text}" if not sid else f"[{stamp}] {text} (session_id={sid})"
        )
        store.append_today(entry)
        store.append_history(entry)
        payload = {
            "ok": True,
            "workspace": str(resolved),
            "entry": entry,
            "today_file": str(store.get_today_file()),
            "history_file": str(store.history_file),
            "retrieval_plugin": store.retrieval_plugin_name,
        }
        emit_governance_event(
            event_type="memory",
            action="operator_flush",
            outcome="ok",
            actor="operator",
            details={
                "workspace": str(resolved),
                "session_id": sid,
                "entry_chars": len(entry),
            },
        )
        return json.dumps(payload, ensure_ascii=False)


class AdminEvalRunTool(FunctionTool):
    def __init__(
        self,
        *,
        allowed_dir: Optional[Path] = None,
        allowed_read_roots: Optional[Sequence[str | Path]] = None,
    ) -> None:
        self._allowed_dir = allowed_dir
        self._allowed_read_roots = allowed_read_roots

    @property
    def name(self) -> str:
        return "admin_eval_run"

    @property
    def aliases(self) -> tuple[str, ...]:
        return ("eval.run",)

    @property
    def description(self) -> str:
        return (
            "Run evaluation report and optional regression gate for candidate outputs."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "traces_path": {"type": "string"},
                "candidate_responses_path": {"type": "string"},
                "baseline_responses_path": {"type": "string"},
                "output_path": {"type": "string"},
                "max_regressions": {"type": "integer", "minimum": 0},
                "min_pass_rate": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            },
            "required": ["traces_path", "candidate_responses_path"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        traces_path: str,
        candidate_responses_path: str,
        baseline_responses_path: str = "",
        output_path: str = "",
        max_regressions: int = 0,
        min_pass_rate: float = 0.0,
    ) -> str:
        resolved_traces = _resolve_read_path(
            traces_path,
            allowed_dir=self._allowed_dir,
            allowed_read_roots=self._allowed_read_roots,
        )
        resolved_candidate = _resolve_read_path(
            candidate_responses_path,
            allowed_dir=self._allowed_dir,
            allowed_read_roots=self._allowed_read_roots,
        )
        resolved_baseline = (
            _resolve_read_path(
                baseline_responses_path,
                allowed_dir=self._allowed_dir,
                allowed_read_roots=self._allowed_read_roots,
            )
            if str(baseline_responses_path or "").strip()
            else None
        )
        out = str(output_path or "").strip()
        resolved_out = (
            _resolve_write_path(out, allowed_dir=self._allowed_dir) if out else None
        )
        report = run_eval(
            traces_path=resolved_traces,
            candidate_responses_path=resolved_candidate,
            baseline_responses_path=resolved_baseline,
        )
        gate = evaluate_report_gate(
            report,
            max_regressions=max(0, int(max_regressions)),
            min_pass_rate=max(0.0, float(min_pass_rate)),
        )
        payload: Dict[str, Any] = {
            "ok": bool(gate.get("passed", False)),
            "report": report,
            "gate": gate,
        }
        if resolved_out is not None:
            resolved_out.parent.mkdir(parents=True, exist_ok=True)
            resolved_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            payload["output_path"] = str(resolved_out)
        return json.dumps(payload, ensure_ascii=False)


class AdminUpdateRunTool(FunctionTool):
    def __init__(self, *, project: str = "annolid") -> None:
        self._project = str(project or "annolid").strip() or "annolid"

    @property
    def name(self) -> str:
        return "admin_update_run"

    @property
    def aliases(self) -> tuple[str, ...]:
        return ("update.run",)

    @property
    def description(self) -> str:
        return (
            "Run update transaction (preflight/stage/verify/apply/restart/post-check) "
            "with optional execute mode."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "channel": {"type": "string", "enum": ["stable", "beta", "dev"]},
                "timeout_s": {"type": "number", "minimum": 1.0, "maximum": 120.0},
                "require_signature": {"type": "boolean"},
                "execute": {"type": "boolean"},
                "run_post_check": {"type": "boolean"},
                "previous_version": {"type": "string"},
                "rollback": {"type": "boolean"},
                "operator_consent": {"type": "string"},
            },
            "additionalProperties": False,
        }

    async def execute(
        self,
        channel: str = "stable",
        timeout_s: float = 4.0,
        require_signature: bool = False,
        execute: bool = False,
        run_post_check: bool = True,
        previous_version: str = "",
        rollback: bool = False,
        operator_consent: str = "",
    ) -> str:
        channel_name = str(channel or "stable").strip().lower()
        if channel_name not in {"stable", "beta", "dev"}:
            channel_name = "stable"
        effective_require_signature = bool(
            bool(require_signature) or require_signed_updates()
        )

        if bool(execute) and bot_update_requires_operator_consent():
            if not has_operator_consent(operator_consent):
                payload = {
                    "ok": False,
                    "status": "blocked",
                    "reason": "operator_consent_required",
                    "message": (
                        "Operator consent is required before applying an update. "
                        "Ask the human operator to supply the configured consent phrase."
                    ),
                }
                emit_governance_event(
                    event_type="update",
                    action="run_blocked",
                    outcome="blocked",
                    actor="bot",
                    details={
                        "channel": channel_name,
                        "execute": True,
                        "reason": "operator_consent_required",
                    },
                )
                return json.dumps(payload, ensure_ascii=False)

        if bool(rollback):
            prev = str(previous_version or "").strip()
            if not prev:
                return json.dumps(
                    {
                        "ok": False,
                        "status": "error",
                        "reason": "previous_version_required_for_rollback",
                    },
                    ensure_ascii=False,
                )
            service = UpdateManagerService(project=self._project)
            preflight = service.manager.preflight()
            plan = build_rollback_plan(
                install_mode=str(preflight.get("install_mode") or "package"),
                project=self._project,
                previous_version=prev,
            )
            payload = execute_rollback(plan, execute=bool(execute))
            payload["ok"] = bool(payload.get("ok", False))
            return json.dumps(payload, ensure_ascii=False)

        service = UpdateManagerService(project=self._project)
        tx = service.run_transaction(
            channel=channel_name,
            timeout_s=max(1.0, float(timeout_s)),
            require_signature=effective_require_signature,
            execute=bool(execute),
            run_post_check=bool(run_post_check),
        )
        tx["ok"] = str(tx.get("status") or "").lower() not in {
            "failed",
            "failed_post_check",
            "failed_stage_artifact",
            "failed_canary",
        }
        tx["require_signature"] = bool(effective_require_signature)
        return json.dumps(tx, ensure_ascii=False)
