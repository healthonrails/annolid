from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Mapping

from .dataset import EvalResponse, EvalTrace


@dataclass(frozen=True)
class EvalRow:
    trace_id: str
    passed: bool
    score: float
    reason: str
    expected_substring: str
    content_preview: str


@dataclass(frozen=True)
class EvalReport:
    name: str
    generated_at: str
    total: int
    passed: int
    failed: int
    pass_rate: float
    avg_score: float
    rows: List[EvalRow]

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "generated_at": self.generated_at,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.pass_rate,
            "avg_score": self.avg_score,
            "rows": [
                {
                    "trace_id": r.trace_id,
                    "passed": r.passed,
                    "score": r.score,
                    "reason": r.reason,
                    "expected_substring": r.expected_substring,
                    "content_preview": r.content_preview,
                }
                for r in self.rows
            ],
        }


class EvalRunner:
    def run(
        self,
        *,
        name: str,
        traces: List[EvalTrace],
        responses: Mapping[str, EvalResponse],
    ) -> EvalReport:
        rows: List[EvalRow] = []
        for trace in traces:
            resp = responses.get(trace.trace_id)
            content = str(resp.content if resp is not None else "")
            expected = str(trace.expected_substring or "").strip()
            if not content:
                rows.append(
                    EvalRow(
                        trace_id=trace.trace_id,
                        passed=False,
                        score=0.0,
                        reason="missing_response",
                        expected_substring=expected,
                        content_preview="",
                    )
                )
                continue
            if not expected:
                rows.append(
                    EvalRow(
                        trace_id=trace.trace_id,
                        passed=True,
                        score=1.0,
                        reason="no_expectation",
                        expected_substring="",
                        content_preview=content[:200],
                    )
                )
                continue
            ok = expected.lower() in content.lower()
            rows.append(
                EvalRow(
                    trace_id=trace.trace_id,
                    passed=bool(ok),
                    score=1.0 if ok else 0.0,
                    reason="matched" if ok else "expected_substring_missing",
                    expected_substring=expected,
                    content_preview=content[:200],
                )
            )

        total = len(rows)
        passed = sum(1 for r in rows if r.passed)
        failed = total - passed
        pass_rate = (float(passed) / float(total)) if total > 0 else 0.0
        avg_score = (
            float(sum(float(r.score) for r in rows)) / float(total)
            if total > 0
            else 0.0
        )
        return EvalReport(
            name=name,
            generated_at=datetime.now().isoformat(timespec="seconds"),
            total=total,
            passed=passed,
            failed=failed,
            pass_rate=round(pass_rate, 6),
            avg_score=round(avg_score, 6),
            rows=rows,
        )


def compare_reports(
    *,
    baseline: EvalReport,
    candidate: EvalReport,
) -> Dict[str, object]:
    base = {r.trace_id: r for r in baseline.rows}
    cand = {r.trace_id: r for r in candidate.rows}
    regressions: List[str] = []
    improvements: List[str] = []
    new_failures: List[str] = []
    for trace_id, b_row in base.items():
        c_row = cand.get(trace_id)
        if c_row is None:
            new_failures.append(trace_id)
            continue
        if b_row.passed and not c_row.passed:
            regressions.append(trace_id)
        if not b_row.passed and c_row.passed:
            improvements.append(trace_id)
    for trace_id in cand.keys():
        if trace_id not in base and not cand[trace_id].passed:
            new_failures.append(trace_id)

    return {
        "baseline": baseline.to_dict(),
        "candidate": candidate.to_dict(),
        "delta": {
            "pass_rate": round(candidate.pass_rate - baseline.pass_rate, 6),
            "avg_score": round(candidate.avg_score - baseline.avg_score, 6),
            "passed": int(candidate.passed - baseline.passed),
            "failed": int(candidate.failed - baseline.failed),
        },
        "regressions": sorted(set(regressions)),
        "improvements": sorted(set(improvements)),
        "new_failures": sorted(set(new_failures)),
    }
