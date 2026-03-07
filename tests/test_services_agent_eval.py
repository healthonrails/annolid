from __future__ import annotations

import json
from pathlib import Path

from annolid.services.agent_eval import (
    build_agent_regression_eval,
    evaluate_agent_eval_gate,
    run_agent_eval,
)


def test_run_agent_eval_writes_report(monkeypatch, tmp_path: Path) -> None:
    import annolid.core.agent.eval.run_agent_eval as eval_mod

    traces = tmp_path / "traces.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    out = tmp_path / "reports" / "eval.json"
    traces.write_text("{}", encoding="utf-8")
    candidate.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        eval_mod,
        "run_eval",
        lambda **kwargs: {
            "candidate": {"pass_rate": 1.0},
            "regressions": ["r1"],
        },
    )

    payload, exit_code = run_agent_eval(
        traces=traces,
        candidate_responses=candidate,
        out=out,
        max_regressions=0,
    )

    written = json.loads(out.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert payload["out"] == str(out.resolve())
    assert written["regression_gate"]["regression_count"] == 1
    assert written["regression_gate"]["passed"] is False


def test_build_agent_regression_eval_writes_ndjson(monkeypatch, tmp_path: Path) -> None:
    import annolid.core.agent.eval.telemetry as telemetry_mod
    import annolid.core.agent.utils as utils_mod

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    out = tmp_path / "regression.ndjson"
    monkeypatch.setattr(
        utils_mod, "get_agent_workspace_path", lambda value=None: workspace
    )

    class _Store:
        def __init__(self, workspace_path):
            self.workspace_path = workspace_path

        def load_traces(self):
            return [{"trace_id": "t1"}]

        def load_feedback(self):
            return [{"trace_id": "t1", "rating": 1}]

    monkeypatch.setattr(telemetry_mod, "RunTraceStore", _Store)
    monkeypatch.setattr(
        telemetry_mod,
        "build_regression_eval_rows",
        lambda **kwargs: [
            {"trace_id": "t1", "user_message": "hello", "expected_substring": "ok"}
        ],
    )

    payload = build_agent_regression_eval(
        workspace=str(workspace),
        out=out,
        min_abs_rating=1,
    )

    lines = out.read_text(encoding="utf-8").splitlines()
    assert payload["count"] == 1
    assert payload["workspace"] == str(workspace)
    assert json.loads(lines[0])["trace_id"] == "t1"


def test_evaluate_agent_eval_gate_uses_report_and_changed_files(
    monkeypatch, tmp_path: Path
) -> None:
    import annolid.core.agent.eval.gate as gate_mod

    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps({"candidate": {"pass_rate": 0.5}}), encoding="utf-8"
    )
    changed_files = tmp_path / "changed.txt"
    changed_files.write_text("annolid/core/agent/skills.py\n", encoding="utf-8")

    monkeypatch.setattr(
        gate_mod, "load_changed_files", lambda path: ["annolid/core/agent/skills.py"]
    )
    monkeypatch.setattr(gate_mod, "eval_gate_required", lambda rows: True)
    monkeypatch.setattr(
        gate_mod,
        "evaluate_report_gate",
        lambda report, *, max_regressions, min_pass_rate: {
            "passed": False,
            "regression_count": 2,
            "max_regressions": max_regressions,
            "min_pass_rate": min_pass_rate,
            "pass_rate": report["candidate"]["pass_rate"],
        },
    )

    payload, exit_code = evaluate_agent_eval_gate(
        report=report_path,
        changed_files=changed_files,
        max_regressions=0,
        min_pass_rate=0.9,
    )

    assert exit_code == 1
    assert payload["required"] is True
    assert payload["changed_files_count"] == 1
    assert payload["gate"]["regression_count"] == 2
