from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class CanaryPolicy:
    min_samples: int = 20
    max_failure_rate: float = 0.05
    max_regressions: int = 0


@dataclass(frozen=True)
class CanaryResult:
    passed: bool
    reason: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": bool(self.passed),
            "reason": str(self.reason),
            "details": dict(self.details),
        }


def evaluate_canary(
    metrics: Mapping[str, Any],
    *,
    policy: CanaryPolicy,
) -> CanaryResult:
    sample_count = int(metrics.get("sample_count", metrics.get("requests", 0)) or 0)
    failure_count = int(metrics.get("failure_count", metrics.get("failures", 0)) or 0)
    regressions = int(metrics.get("regressions", 0) or 0)
    failure_rate = (
        (float(failure_count) / float(sample_count)) if sample_count > 0 else 0.0
    )
    details = {
        "sample_count": sample_count,
        "failure_count": failure_count,
        "failure_rate": round(float(failure_rate), 6),
        "regressions": regressions,
        "policy": {
            "min_samples": int(policy.min_samples),
            "max_failure_rate": float(policy.max_failure_rate),
            "max_regressions": int(policy.max_regressions),
        },
    }
    if sample_count < int(policy.min_samples):
        return CanaryResult(False, "insufficient_samples", details)
    if failure_rate > float(policy.max_failure_rate):
        return CanaryResult(False, "failure_rate_exceeded", details)
    if regressions > int(policy.max_regressions):
        return CanaryResult(False, "regressions_exceeded", details)
    return CanaryResult(True, "ok", details)
