from __future__ import annotations

from pathlib import Path

from annolid.core.agent.eval.gate import eval_gate_required, evaluate_report_gate
from annolid.core.agent.eval.telemetry import RunTraceStore, build_regression_eval_rows


def test_trace_capture_and_feedback_build_regression_rows(tmp_path: Path) -> None:
    store = RunTraceStore(tmp_path)
    trace = store.capture_run(
        session_id="session-1",
        channel="email",
        chat_id="user@example.com",
        user_message="Please summarize protocol A for user@example.com",
        assistant_response="Protocol A summary: sample prep and analysis.",
        tool_names=["read_file"],
    )
    _ = store.capture_feedback(
        session_id="session-1",
        trace_id=str(trace.get("trace_id") or ""),
        rating=1,
        comment="good answer",
    )
    rows = build_regression_eval_rows(
        trace_rows=store.load_traces(),
        feedback_rows=store.load_feedback(),
        min_abs_rating=1,
    )
    assert len(rows) == 1
    assert rows[0]["trace_id"] == trace["trace_id"]
    assert rows[0]["expected_substring"]


def test_eval_gate_requirement_and_report_decision() -> None:
    required = eval_gate_required(
        ["annolid/core/agent/skills.py", "docs/source/install.md"]
    )
    assert required is True

    gate = evaluate_report_gate(
        {"regressions": [], "candidate": {"pass_rate": 1.0}},
        max_regressions=0,
        min_pass_rate=0.5,
    )
    assert gate["passed"] is True
