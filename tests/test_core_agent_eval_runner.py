from __future__ import annotations

import json
from pathlib import Path

from annolid.core.agent.eval.run_agent_eval import run_eval
from annolid.engine.cli import main as annolid_run


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_eval_runner_compare_reports(tmp_path: Path) -> None:
    traces = tmp_path / "traces.jsonl"
    base = tmp_path / "baseline.jsonl"
    cand = tmp_path / "candidate.jsonl"
    _write_jsonl(
        traces,
        [
            {"trace_id": "t1", "user_message": "u1", "expected_substring": "hello"},
            {"trace_id": "t2", "user_message": "u2", "expected_substring": "world"},
        ],
    )
    _write_jsonl(
        base,
        [
            {"trace_id": "t1", "content": "hello"},
            {"trace_id": "t2", "content": "world"},
        ],
    )
    _write_jsonl(
        cand,
        [{"trace_id": "t1", "content": "hello"}, {"trace_id": "t2", "content": "oops"}],
    )

    report = run_eval(
        traces_path=traces,
        baseline_responses_path=base,
        candidate_responses_path=cand,
    )
    assert report["delta"]["passed"] == -1
    assert "t2" in report["regressions"]


def test_agent_eval_cli_regression_gate(tmp_path: Path) -> None:
    traces = tmp_path / "traces.jsonl"
    base = tmp_path / "baseline.jsonl"
    cand = tmp_path / "candidate.jsonl"
    out = tmp_path / "report.json"
    _write_jsonl(
        traces,
        [
            {"trace_id": "t1", "user_message": "u1", "expected_substring": "hello"},
        ],
    )
    _write_jsonl(base, [{"trace_id": "t1", "content": "hello"}])
    _write_jsonl(cand, [{"trace_id": "t1", "content": "nope"}])

    rc = annolid_run(
        [
            "agent-eval",
            "--traces",
            str(traces),
            "--baseline-responses",
            str(base),
            "--candidate-responses",
            str(cand),
            "--out",
            str(out),
            "--max-regressions",
            "0",
        ]
    )
    assert rc == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["regressions"] == ["t1"]
