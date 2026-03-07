"""Service-layer orchestration for agent evaluation workflows."""

from __future__ import annotations

import json
from pathlib import Path


def run_agent_eval(
    *,
    traces: str | Path,
    candidate_responses: str | Path,
    baseline_responses: str | Path | None = None,
    out: str | Path,
    max_regressions: int = 0,
) -> tuple[dict, int]:
    from annolid.core.agent.eval.run_agent_eval import run_eval

    report = run_eval(
        traces_path=Path(traces).expanduser().resolve(),
        candidate_responses_path=Path(candidate_responses).expanduser().resolve(),
        baseline_responses_path=(
            Path(baseline_responses).expanduser().resolve()
            if baseline_responses
            else None
        ),
    )
    out_path = Path(out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    regressions = report.get("regressions")
    regression_count = len(regressions) if isinstance(regressions, list) else 0
    gate_limit = int(max_regressions)
    gate_passed = regression_count <= gate_limit
    if isinstance(report, dict):
        report["regression_gate"] = {
            "max_regressions": gate_limit,
            "regression_count": regression_count,
            "passed": gate_passed,
        }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {"out": str(out_path), "report": report}, (0 if gate_passed else 1)


def build_agent_regression_eval(
    *,
    workspace: str | None = None,
    out: str | Path,
    min_abs_rating: int = 1,
) -> dict:
    from annolid.core.agent.eval.telemetry import (
        RunTraceStore,
        build_regression_eval_rows,
    )
    from annolid.core.agent.utils import get_agent_workspace_path

    resolved_workspace = get_agent_workspace_path(workspace)
    store = RunTraceStore(resolved_workspace)
    rows = build_regression_eval_rows(
        trace_rows=store.load_traces(),
        feedback_rows=store.load_feedback(),
        min_abs_rating=int(min_abs_rating),
    )
    out_path = Path(out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_text = "\n".join(json.dumps(row, ensure_ascii=True) for row in rows)
    out_path.write_text((out_text + "\n") if out_text else "", encoding="utf-8")
    return {
        "workspace": str(resolved_workspace),
        "out": str(out_path),
        "count": len(rows),
    }


def evaluate_agent_eval_gate(
    *,
    report: str | Path | None = None,
    changed_files: str | Path | None = None,
    max_regressions: int = 0,
    min_pass_rate: float = 0.0,
) -> tuple[dict, int]:
    from annolid.core.agent.eval.gate import (
        eval_gate_required,
        evaluate_report_gate,
        load_changed_files,
    )

    changed_file_rows: list[str] = []
    if changed_files:
        changed_file_rows = load_changed_files(Path(changed_files))
    required = eval_gate_required(changed_file_rows)

    report_payload: dict = {}
    if report:
        report_payload = json.loads(
            Path(report).expanduser().resolve().read_text(encoding="utf-8")
        )

    if report_payload:
        gate = evaluate_report_gate(
            report_payload,
            max_regressions=int(max_regressions),
            min_pass_rate=float(min_pass_rate),
        )
    else:
        gate = {
            "passed": not required,
            "max_regressions": int(max_regressions),
            "regression_count": 0,
            "min_pass_rate": float(min_pass_rate),
            "pass_rate": 0.0,
        }
        if required:
            gate["reason"] = "eval_required_report_missing"

    payload = {
        "required": bool(required),
        "changed_files_count": len(changed_file_rows),
        "gate": gate,
    }
    return payload, (0 if bool(gate.get("passed", False)) else 1)


__all__ = [
    "build_agent_regression_eval",
    "evaluate_agent_eval_gate",
    "run_agent_eval",
]
