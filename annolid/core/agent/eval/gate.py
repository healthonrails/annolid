from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence


_DEFAULT_EVAL_GATE_PATTERNS = (
    "annolid/core/agent/skills.py",
    "annolid/core/agent/skill_registry/",
    "annolid/core/agent/loop.py",
    "annolid/core/agent/context.py",
    "annolid/core/agent/tools/",
)


def eval_gate_required(
    changed_files: Sequence[str],
    *,
    patterns: Sequence[str] = _DEFAULT_EVAL_GATE_PATTERNS,
) -> bool:
    files = [str(p or "").strip() for p in changed_files if str(p or "").strip()]
    pats = [str(p or "").strip() for p in patterns if str(p or "").strip()]
    for path in files:
        for pattern in pats:
            if pattern.endswith("/"):
                if path.startswith(pattern):
                    return True
            elif path == pattern:
                return True
    return False


def evaluate_report_gate(
    report: Mapping[str, Any],
    *,
    max_regressions: int = 0,
    min_pass_rate: float = 0.0,
) -> Dict[str, Any]:
    regressions = report.get("regressions")
    regression_count = len(regressions) if isinstance(regressions, list) else 0
    candidate = (
        report.get("candidate") if isinstance(report.get("candidate"), Mapping) else {}
    )
    pass_rate = float(candidate.get("pass_rate") or 0.0)
    passed = regression_count <= int(max_regressions) and pass_rate >= float(
        min_pass_rate
    )
    return {
        "passed": bool(passed),
        "max_regressions": int(max_regressions),
        "regression_count": int(regression_count),
        "min_pass_rate": float(min_pass_rate),
        "pass_rate": pass_rate,
    }


def load_changed_files(path: Path) -> List[str]:
    text = Path(path).expanduser().resolve().read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]
