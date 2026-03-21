"""Citation verification report helpers for research pipelines."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

_VERIFY_STATUSES = ("verified", "suspicious", "hallucinated", "skipped")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_name(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text or "").strip())
    return cleaned.strip("._") or "citation"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def classify_citation_status(
    *,
    fields: dict[str, str],
    validation: dict[str, Any],
) -> tuple[str, float, str]:
    checked = bool(validation.get("checked"))
    verified = bool(validation.get("verified"))
    score = _safe_float(validation.get("score"), 0.0)
    message = str(validation.get("message") or "").strip()
    has_candidate = bool(dict(validation.get("candidate") or {}))
    has_identity = bool(str(fields.get("title") or "").strip()) or bool(
        str(fields.get("doi") or "").strip()
    )

    if verified:
        return ("verified", max(0.7, score), message or "Metadata verified.")
    if not checked and not has_identity:
        return ("skipped", 0.2, message or "Skipped: missing title/doi metadata.")
    if checked and (score >= 0.45 or has_candidate):
        return ("suspicious", max(0.2, score), message or "Weak external match.")
    if checked:
        return ("hallucinated", 0.0, message or "No credible external match.")
    return ("skipped", 0.2, message or "Verification unavailable.")


def build_citation_verification_report(
    *,
    key: str,
    bib_file: str,
    source: str,
    fields: dict[str, str],
    validation: dict[str, Any],
) -> dict[str, Any]:
    status, integrity_score, reason = classify_citation_status(
        fields=fields, validation=validation
    )
    if status not in _VERIFY_STATUSES:
        status = "skipped"

    counts = {name: 0 for name in _VERIFY_STATUSES}
    counts[status] = 1
    verification = {
        "status": status,
        "integrity_score": round(float(max(0.0, min(1.0, integrity_score))), 3),
        "reason": reason,
        "checked": bool(validation.get("checked")),
        "verified": bool(validation.get("verified")),
        "provider": str(validation.get("provider") or ""),
        "score": round(_safe_float(validation.get("score"), 0.0), 3),
    }
    return {
        "schema_version": "1.0",
        "generated_at": _utc_now_iso(),
        "bib_file": str(bib_file or ""),
        "entry_key": str(key or "").strip(),
        "source": str(source or "").strip(),
        "summary": {
            "total": 1,
            "counts": counts,
            "integrity_score": verification["integrity_score"],
        },
        "verification": verification,
        "entry": {
            "key": str(key or "").strip(),
            "fields": dict(fields),
            "validation": dict(validation),
        },
    }


def write_citation_verification_report(
    report: dict[str, Any], *, reports_dir: Path, report_stem: str
) -> Path:
    target_dir = Path(reports_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output = target_dir / f"{_safe_name(report_stem)}.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output


def build_citation_batch_report(
    *,
    bib_file: str,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = {name: 0 for name in _VERIFY_STATUSES}
    score_values: list[float] = []
    for row in list(entries or []):
        status = str(row.get("status") or "").strip().lower()
        if status not in counts:
            status = "skipped"
        counts[status] += 1
        score_values.append(_safe_float(row.get("integrity_score"), 0.0))
    total = len(entries or [])
    avg_score = (sum(score_values) / total) if total > 0 else 0.0
    return {
        "schema_version": "1.0",
        "generated_at": _utc_now_iso(),
        "bib_file": str(bib_file or ""),
        "summary": {
            "total": total,
            "counts": counts,
            "integrity_score": round(float(max(0.0, min(1.0, avg_score))), 3),
        },
        "entries": list(entries or []),
    }
