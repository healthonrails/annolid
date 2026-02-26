from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .dataset import load_eval_responses, load_eval_traces
from .runner import EvalRunner, compare_reports


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_agent_eval",
        description="Replay traces, score outcomes, compare baseline vs candidate.",
    )
    p.add_argument("--traces", required=True, help="Path to eval traces (.json/.jsonl)")
    p.add_argument(
        "--candidate-responses",
        required=True,
        help="Path to candidate responses (.json/.jsonl)",
    )
    p.add_argument(
        "--baseline-responses",
        default=None,
        help="Optional path to baseline responses (.json/.jsonl)",
    )
    p.add_argument("--out", required=True, help="Output report path (.json)")
    p.add_argument(
        "--max-regressions",
        type=int,
        default=0,
        help="Regression gate: fail if regressions > N (default: 0)",
    )
    return p


def run_eval(
    *,
    traces_path: Path,
    candidate_responses_path: Path,
    baseline_responses_path: Optional[Path],
) -> dict:
    traces = load_eval_traces(traces_path)
    candidate = load_eval_responses(candidate_responses_path)
    runner = EvalRunner()
    candidate_report = runner.run(name="candidate", traces=traces, responses=candidate)

    if baseline_responses_path is None:
        return {
            "candidate": candidate_report.to_dict(),
            "regressions": [],
            "improvements": [],
            "new_failures": [r.trace_id for r in candidate_report.rows if not r.passed],
        }

    baseline = load_eval_responses(baseline_responses_path)
    baseline_report = runner.run(name="baseline", traces=traces, responses=baseline)
    return compare_reports(baseline=baseline_report, candidate=candidate_report)


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    report = run_eval(
        traces_path=Path(args.traces).expanduser().resolve(),
        candidate_responses_path=Path(args.candidate_responses).expanduser().resolve(),
        baseline_responses_path=(
            Path(args.baseline_responses).expanduser().resolve()
            if args.baseline_responses
            else None
        ),
    )
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    regressions = report.get("regressions")
    regression_count = len(regressions) if isinstance(regressions, list) else 0
    gate_limit = int(args.max_regressions)
    gate_passed = regression_count <= gate_limit
    if isinstance(report, dict):
        report["regression_gate"] = {
            "max_regressions": gate_limit,
            "regression_count": regression_count,
            "passed": gate_passed,
        }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not gate_passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
