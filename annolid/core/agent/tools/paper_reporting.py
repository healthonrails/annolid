from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from annolid.utils.atomic_io import atomic_write_text

from .common import (
    _normalize_allowed_read_roots,
    _resolve_read_path,
    _resolve_write_path,
)
from .function_base import FunctionTool

_STATUS_ORDER = {"pass": 0, "warn": 1, "fail": 2}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_status(value: object) -> str:
    normalized = str(value or "").strip().lower()
    if normalized in {"ok", "success"}:
        return "pass"
    if normalized not in _STATUS_ORDER:
        return "warn"
    return normalized


def _merge_quality_status(*statuses: object) -> str:
    best = "pass"
    for status in statuses:
        normalized = _normalize_status(status)
        if _STATUS_ORDER[normalized] > _STATUS_ORDER[best]:
            best = normalized
    return best


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _validate_unit_interval(name: str, value: float) -> float:
    numeric = _safe_float(value, 0.0)
    if numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0.")
    return float(numeric)


def _load_json_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _unwrap_eval_report(payload: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("report"), Mapping):
        return dict(payload.get("report") or {})
    return dict(payload)


def _extract_citation_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    summary = (
        payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
    )
    counts = summary.get("counts") if isinstance(summary.get("counts"), Mapping) else {}
    total = _safe_int(summary.get("total"), 0)
    if total <= 0:
        total = (
            _safe_int(counts.get("verified"), 0)
            + _safe_int(counts.get("suspicious"), 0)
            + _safe_int(counts.get("hallucinated"), 0)
            + _safe_int(counts.get("skipped"), 0)
        )
    integrity_score = _safe_float(summary.get("integrity_score"), 0.0)

    suspicious = _safe_int(counts.get("suspicious"), 0)
    hallucinated = _safe_int(counts.get("hallucinated"), 0)
    suspicious_rate = (float(suspicious) / float(total)) if total > 0 else 0.0

    status = "pass"
    if hallucinated > 0:
        status = "fail"
    elif suspicious_rate >= 0.2 or integrity_score < 0.6:
        status = "warn"

    return {
        "available": True,
        "quality_status": status,
        "total": total,
        "counts": {
            "verified": _safe_int(counts.get("verified"), 0),
            "suspicious": suspicious,
            "hallucinated": hallucinated,
            "skipped": _safe_int(counts.get("skipped"), 0),
        },
        "integrity_score": round(float(integrity_score), 3),
        "suspicious_rate": round(float(suspicious_rate), 4),
    }


def _extract_novelty_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    recommendation = (
        str(payload.get("recommendation") or "").strip().lower() or "differentiate"
    )
    coverage_quality = (
        str(payload.get("coverage_quality") or "").strip().lower() or "low"
    )
    scores = payload.get("scores") if isinstance(payload.get("scores"), Mapping) else {}

    max_overlap = _safe_float(scores.get("max_overlap"), 1.0)
    mean_top3 = _safe_float(scores.get("mean_top3_overlap"), 1.0)
    coverage = _safe_float(scores.get("idea_token_coverage"), 0.0)
    related_count = _safe_int(scores.get("related_work_count"), 0)

    status = "pass"
    if recommendation == "abort":
        status = "fail"
    elif recommendation == "differentiate" or coverage_quality == "low":
        status = "warn"

    return {
        "available": True,
        "quality_status": status,
        "recommendation": recommendation,
        "reason": str(payload.get("reason") or "").strip(),
        "coverage_quality": coverage_quality,
        "scores": {
            "max_overlap": round(float(max_overlap), 4),
            "mean_top3_overlap": round(float(mean_top3), 4),
            "idea_token_coverage": round(float(coverage), 4),
            "related_work_count": int(related_count),
        },
    }


def _build_reproducibility_checklist(
    *,
    eval_report: Mapping[str, Any],
    citation: Mapping[str, Any] | None,
    novelty: Mapping[str, Any] | None,
) -> list[dict[str, str]]:
    metadata = (
        eval_report.get("metadata")
        if isinstance(eval_report.get("metadata"), Mapping)
        else {}
    )
    quality_checks = (
        eval_report.get("quality_checks")
        if isinstance(eval_report.get("quality_checks"), Sequence)
        else []
    )
    has_fail_quality_check = any(
        isinstance(item, Mapping) and _normalize_status(item.get("status")) == "fail"
        for item in quality_checks
    )

    checklist = [
        {
            "id": "eval_artifacts",
            "status": (
                "pass" if str(metadata.get("source_path") or "").strip() else "warn"
            ),
            "message": (
                "Evaluation source path is recorded."
                if str(metadata.get("source_path") or "").strip()
                else "Evaluation source path is missing."
            ),
        },
        {
            "id": "eval_quality_checks",
            "status": "fail" if has_fail_quality_check else "pass",
            "message": (
                "Evaluation quality checks include failures."
                if has_fail_quality_check
                else "Evaluation quality checks have no failures."
            ),
        },
        {
            "id": "citation_integrity",
            "status": (
                str(citation.get("quality_status") or "warn")
                if isinstance(citation, Mapping) and citation.get("available")
                else "warn"
            ),
            "message": (
                "Citation verification summary is included."
                if isinstance(citation, Mapping) and citation.get("available")
                else "Citation verification summary is missing."
            ),
        },
        {
            "id": "novelty_preflight",
            "status": (
                str(novelty.get("quality_status") or "warn")
                if isinstance(novelty, Mapping) and novelty.get("available")
                else "warn"
            ),
            "message": (
                "Novelty preflight summary is included."
                if isinstance(novelty, Mapping) and novelty.get("available")
                else "Novelty preflight summary is missing."
            ),
        },
    ]
    return checklist


def _collect_warnings(
    *,
    eval_report: Mapping[str, Any],
    citation: Mapping[str, Any] | None,
    novelty: Mapping[str, Any] | None,
) -> list[str]:
    warnings: list[str] = []
    quality_checks = (
        eval_report.get("quality_checks")
        if isinstance(eval_report.get("quality_checks"), Sequence)
        else []
    )
    for item in quality_checks:
        if not isinstance(item, Mapping):
            continue
        status = _normalize_status(item.get("status"))
        if status in {"warn", "fail"}:
            message = str(item.get("message") or "").strip()
            if message:
                warnings.append(f"Eval check ({item.get('id', 'check')}): {message}")

    if isinstance(citation, Mapping) and citation.get("available"):
        status = _normalize_status(citation.get("quality_status"))
        if status in {"warn", "fail"}:
            warnings.append(
                "Citation integrity risk: "
                f"hallucinated={_safe_int(citation.get('counts', {}).get('hallucinated') if isinstance(citation.get('counts'), Mapping) else 0)}, "
                f"suspicious_rate={_safe_float(citation.get('suspicious_rate')):.3f}, "
                f"integrity_score={_safe_float(citation.get('integrity_score')):.3f}."
            )
    else:
        warnings.append("Citation verification summary is missing.")

    if isinstance(novelty, Mapping) and novelty.get("available"):
        status = _normalize_status(novelty.get("quality_status"))
        if status in {"warn", "fail"}:
            warnings.append(
                "Novelty preflight requires attention: "
                f"recommendation={str(novelty.get('recommendation') or '')}, "
                f"coverage_quality={str(novelty.get('coverage_quality') or '')}, "
                f"max_overlap={_safe_float(novelty.get('scores', {}).get('max_overlap') if isinstance(novelty.get('scores'), Mapping) else 0.0):.3f}."
            )
    else:
        warnings.append("Novelty preflight summary is missing.")

    return warnings


def _render_checklist_markdown(rows: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "| Check | Status | Details |",
        "|---|---|---|",
    ]
    for row in rows:
        check_id = str(row.get("id") or "check")
        status = _normalize_status(row.get("status")).upper()
        message = str(row.get("message") or "")
        lines.append(f"| `{check_id}` | {status} | {message} |")
    return "\n".join(lines)


def _build_paper_ready_gate(
    *,
    citation: Mapping[str, Any] | None,
    novelty: Mapping[str, Any] | None,
    enabled: bool,
    citation_integrity_floor: float,
    novelty_coverage_floor: float,
    require_citation_summary: bool,
    require_novelty_summary: bool,
) -> dict[str, Any]:
    gate: dict[str, Any] = {
        "enabled": bool(enabled),
        "paper_ready": True,
        "status": "pass",
        "thresholds": {
            "citation_integrity_floor": float(citation_integrity_floor),
            "novelty_coverage_floor": float(novelty_coverage_floor),
            "require_citation_summary": bool(require_citation_summary),
            "require_novelty_summary": bool(require_novelty_summary),
        },
        "checks": [],
    }
    if not bool(enabled):
        return gate

    checks: list[dict[str, str]] = []

    citation_available = bool(
        isinstance(citation, Mapping) and citation.get("available")
    )
    citation_integrity = (
        _safe_float(citation.get("integrity_score"), 0.0) if citation_available else 0.0
    )
    checks.append(
        {
            "id": "citation_summary_presence",
            "status": (
                "pass"
                if citation_available
                else ("fail" if require_citation_summary else "warn")
            ),
            "message": (
                "Citation summary provided."
                if citation_available
                else "Citation summary missing."
            ),
        }
    )
    if citation_available:
        checks.append(
            {
                "id": "citation_integrity_floor",
                "status": (
                    "pass"
                    if citation_integrity >= float(citation_integrity_floor)
                    else "fail"
                ),
                "message": (
                    f"Integrity score {citation_integrity:.3f} meets floor "
                    f"{float(citation_integrity_floor):.3f}."
                    if citation_integrity >= float(citation_integrity_floor)
                    else (
                        f"Integrity score {citation_integrity:.3f} is below floor "
                        f"{float(citation_integrity_floor):.3f}."
                    )
                ),
            }
        )

    novelty_available = bool(isinstance(novelty, Mapping) and novelty.get("available"))
    novelty_scores = novelty.get("scores") if novelty_available else {}
    novelty_coverage = (
        _safe_float(novelty_scores.get("idea_token_coverage"), 0.0)
        if isinstance(novelty_scores, Mapping)
        else 0.0
    )
    checks.append(
        {
            "id": "novelty_summary_presence",
            "status": (
                "pass"
                if novelty_available
                else ("fail" if require_novelty_summary else "warn")
            ),
            "message": (
                "Novelty summary provided."
                if novelty_available
                else "Novelty summary missing."
            ),
        }
    )
    if novelty_available:
        checks.append(
            {
                "id": "novelty_coverage_floor",
                "status": (
                    "pass"
                    if novelty_coverage >= float(novelty_coverage_floor)
                    else "fail"
                ),
                "message": (
                    f"Novelty coverage {novelty_coverage:.3f} meets floor "
                    f"{float(novelty_coverage_floor):.3f}."
                    if novelty_coverage >= float(novelty_coverage_floor)
                    else (
                        f"Novelty coverage {novelty_coverage:.3f} is below floor "
                        f"{float(novelty_coverage_floor):.3f}."
                    )
                ),
            }
        )

    status = _merge_quality_status(*[item.get("status") for item in checks])
    gate["checks"] = checks
    gate["status"] = status
    gate["paper_ready"] = status != "fail"
    return gate


def _paper_ready_gate_markdown(gate: Mapping[str, Any]) -> str:
    enabled = bool(gate.get("enabled"))
    if not enabled:
        return "Paper-ready gate disabled."
    checks = gate.get("checks") if isinstance(gate.get("checks"), Sequence) else []
    lines = [
        f"- Gate status: `{_normalize_status(gate.get('status')).upper()}`",
        f"- Paper ready: `{str(bool(gate.get('paper_ready'))).lower()}`",
        "- Thresholds:",
    ]
    thresholds = (
        gate.get("thresholds") if isinstance(gate.get("thresholds"), Mapping) else {}
    )
    lines.extend(
        [
            "  - citation_integrity_floor="
            f"{_safe_float(thresholds.get('citation_integrity_floor'), 0.0):.3f}",
            "  - novelty_coverage_floor="
            f"{_safe_float(thresholds.get('novelty_coverage_floor'), 0.0):.3f}",
            "  - require_citation_summary="
            f"{str(bool(thresholds.get('require_citation_summary'))).lower()}",
            "  - require_novelty_summary="
            f"{str(bool(thresholds.get('require_novelty_summary'))).lower()}",
            "",
            _render_checklist_markdown(
                [item for item in checks if isinstance(item, Mapping)]
            ),
        ]
    )
    return "\n".join(lines)


def _build_markdown(
    *,
    eval_report: Mapping[str, Any],
    citation: Mapping[str, Any] | None,
    novelty: Mapping[str, Any] | None,
    draft_quality: Mapping[str, Any] | None = None,
    warnings: Sequence[str],
    checklist: Sequence[Mapping[str, object]],
    paper_ready_gate: Mapping[str, Any],
    generated_at_utc: str,
    quality_status: str,
) -> str:
    metadata = (
        eval_report.get("metadata")
        if isinstance(eval_report.get("metadata"), Mapping)
        else {}
    )
    paper_table = (
        eval_report.get("paper_table")
        if isinstance(eval_report.get("paper_table"), Mapping)
        else {}
    )
    eval_table_md = str(paper_table.get("markdown") or "").strip()
    if not eval_table_md:
        eval_table_md = "No evaluation table available."

    lines = [
        "# Unified Paper-Run Report",
        "",
        f"- Generated (UTC): `{generated_at_utc}`",
        f"- Model: `{metadata.get('model', '')}`",
        f"- Dataset: `{metadata.get('dataset', '')}`",
        f"- Split: `{metadata.get('split', '')}`",
        f"- Model family: `{metadata.get('model_family', '')}`",
        f"- Overall quality status: `{_normalize_status(quality_status).upper()}`",
        "",
        "## Model Evaluation (annolid_eval_report)",
        "",
        eval_table_md,
        "",
        "## Citation Verification Summary",
        "",
    ]

    if isinstance(citation, Mapping) and citation.get("available"):
        counts = (
            citation.get("counts")
            if isinstance(citation.get("counts"), Mapping)
            else {}
        )
        lines.extend(
            [
                f"- Status: `{_normalize_status(citation.get('quality_status')).upper()}`",
                f"- Total citations: `{_safe_int(citation.get('total'))}`",
                (
                    "- Counts: "
                    f"verified={_safe_int(counts.get('verified'))}, "
                    f"suspicious={_safe_int(counts.get('suspicious'))}, "
                    f"hallucinated={_safe_int(counts.get('hallucinated'))}, "
                    f"skipped={_safe_int(counts.get('skipped'))}"
                ),
                f"- Integrity score: `{_safe_float(citation.get('integrity_score')):.3f}`",
                f"- Suspicious rate: `{_safe_float(citation.get('suspicious_rate')):.3f}`",
            ]
        )
    else:
        lines.append("Citation summary not provided.")

    lines.extend(["", "## Novelty Summary", ""])
    if isinstance(novelty, Mapping) and novelty.get("available"):
        scores = (
            novelty.get("scores") if isinstance(novelty.get("scores"), Mapping) else {}
        )
        lines.extend(
            [
                f"- Status: `{_normalize_status(novelty.get('quality_status')).upper()}`",
                f"- Recommendation: `{str(novelty.get('recommendation') or '').upper()}`",
                f"- Reason: {str(novelty.get('reason') or '').strip() or 'No reason provided.'}",
                f"- Coverage quality: `{str(novelty.get('coverage_quality') or '').upper()}`",
                (
                    "- Scores: "
                    f"max_overlap={_safe_float(scores.get('max_overlap')):.3f}, "
                    f"mean_top3_overlap={_safe_float(scores.get('mean_top3_overlap')):.3f}, "
                    f"idea_token_coverage={_safe_float(scores.get('idea_token_coverage')):.3f}, "
                    f"related_work_count={_safe_int(scores.get('related_work_count'))}"
                ),
            ]
        )
    else:
        lines.append("Novelty summary not provided.")

    if isinstance(draft_quality, Mapping) and draft_quality.get("available"):
        missing = list(draft_quality.get("missing_sections") or [])
        lines.extend(
            [
                "",
                "## Manuscript Draft Quality",
                "",
                f"- Status: `{_normalize_status(draft_quality.get('quality_status')).upper()}`",
                f"- Words: `{_safe_int(draft_quality.get('word_count'))}`",
                f"- Inline citations: `{_safe_int(draft_quality.get('citation_count'))}`",
                f"- Placeholders: `{_safe_int(draft_quality.get('placeholder_count'))}`",
                f"- Unresolved citation keys: `{_safe_int(draft_quality.get('unresolved_citation_count'))}`",
                f"- Missing sections: `{', '.join(missing) if missing else 'none'}`",
            ]
        )

    lines.extend(["", "## Warnings", ""])
    if warnings:
        lines.extend(f"- {item}" for item in warnings)
    else:
        lines.append("- No warnings detected.")

    lines.extend(
        [
            "",
            "## Paper-Ready Gate",
            "",
            _paper_ready_gate_markdown(paper_ready_gate),
        ]
    )
    lines.extend(
        ["", "## Reproducibility Checklist", "", _render_checklist_markdown(checklist)]
    )
    return "\n".join(lines).strip() + "\n"


def _sanitize_report_basename(
    base_name: str, *, default: str = "paper_run_report"
) -> str:
    raw = str(base_name or "").strip()
    if not raw:
        return default
    if "/" in raw or "\\" in raw or ".." in raw:
        raise ValueError(
            "report_basename must be a plain filename stem without path separators or '..'."
        )
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw).strip("._")
    if not cleaned:
        raise ValueError("report_basename must contain alphanumeric characters.")
    return cleaned


def _extract_draft_quality_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics = (
        payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
    )
    status = _normalize_status(payload.get("quality_status", "warn"))
    return {
        "available": True,
        "quality_status": status,
        "word_count": _safe_int(metrics.get("word_count"), 0),
        "citation_count": _safe_int(metrics.get("citation_count"), 0),
        "placeholder_count": _safe_int(metrics.get("placeholder_count"), 0),
        "unresolved_citation_count": _safe_int(
            metrics.get("unresolved_citation_count"), 0
        ),
        "missing_sections": list(payload.get("missing_sections") or []),
        "warnings": [str(w) for w in (payload.get("warnings") or []) if str(w).strip()],
    }


def _write_unified_report_files(
    *,
    report: Mapping[str, Any],
    report_dir: Path,
    base_name: str,
) -> dict[str, str]:
    out_dir = Path(report_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _sanitize_report_basename(base_name, default="paper_run_report")
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    atomic_write_text(json_path, json.dumps(dict(report), indent=2))
    atomic_write_text(md_path, str(report.get("report_markdown") or ""))
    written = {
        "json": str(json_path),
        "markdown": str(md_path),
    }
    eval_block = report.get("evaluation")
    paper_table = (
        eval_block.get("paper_table")
        if isinstance(eval_block, Mapping)
        and isinstance(eval_block.get("paper_table"), Mapping)
        else {}
    )
    latex_table = str(paper_table.get("latex") or "").strip()
    if latex_table:
        tex_path = out_dir / f"{stem}_table.tex"
        atomic_write_text(tex_path, latex_table + "\n")
        written["latex_table"] = str(tex_path)
    return written


class AnnolidPaperRunReportTool(FunctionTool):
    def __init__(
        self,
        *,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ) -> None:
        self._allowed_dir = (
            Path(allowed_dir).expanduser().resolve()
            if allowed_dir is not None
            else None
        )
        self._allowed_read_roots = _normalize_allowed_read_roots(
            self._allowed_dir,
            allowed_read_roots,
        )

    @property
    def name(self) -> str:
        return "annolid_paper_run_report"

    @property
    def description(self) -> str:
        return (
            "Compose a unified paper-run report by merging annolid_eval_report output "
            "with citation verification and novelty preflight summaries."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "eval_report": {"type": "object"},
                "eval_report_json_path": {"type": "string"},
                "citation_report_path": {"type": "string"},
                "novelty_report_path": {"type": "string"},
                "draft_quality_report_path": {"type": "string"},
                "paper_draft_path": {"type": "string"},
                "bib_path": {"type": "string"},
                "report_dir": {"type": "string"},
                "report_basename": {"type": "string"},
                "paper_ready_gate": {"type": "boolean"},
                "citation_integrity_floor": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "novelty_coverage_floor": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "require_citation_summary": {"type": "boolean"},
                "require_novelty_summary": {"type": "boolean"},
                "allow_mutation": {"type": "boolean"},
            },
        }

    async def execute(
        self,
        eval_report: dict[str, Any] | None = None,
        eval_report_json_path: str = "",
        citation_report_path: str = "",
        novelty_report_path: str = "",
        draft_quality_report_path: str = "",
        paper_draft_path: str = "",
        bib_path: str = "",
        report_dir: str = "",
        report_basename: str = "paper_run_report",
        paper_ready_gate: bool = False,
        citation_integrity_floor: float = 0.6,
        novelty_coverage_floor: float = 0.25,
        require_citation_summary: bool = True,
        require_novelty_summary: bool = True,
        allow_mutation: bool = False,
        **kwargs: Any,
    ) -> str:
        del kwargs
        try:
            report_obj: dict[str, Any]
            if isinstance(eval_report, dict) and eval_report:
                report_obj = _unwrap_eval_report(eval_report)
            elif str(eval_report_json_path or "").strip():
                eval_path = _resolve_read_path(
                    eval_report_json_path,
                    allowed_dir=self._allowed_dir,
                    allowed_read_roots=self._allowed_read_roots,
                )
                report_obj = _unwrap_eval_report(_load_json_payload(eval_path))
            else:
                raise ValueError("Provide eval_report or eval_report_json_path.")

            if not isinstance(report_obj.get("paper_table"), Mapping):
                raise ValueError("eval_report is missing paper_table.")

            citation_summary: dict[str, Any] | None = None
            if str(citation_report_path or "").strip():
                citation_path = _resolve_read_path(
                    citation_report_path,
                    allowed_dir=self._allowed_dir,
                    allowed_read_roots=self._allowed_read_roots,
                )
                citation_summary = _extract_citation_summary(
                    _load_json_payload(citation_path)
                )

            novelty_summary: dict[str, Any] | None = None
            if str(novelty_report_path or "").strip():
                novelty_path = _resolve_read_path(
                    novelty_report_path,
                    allowed_dir=self._allowed_dir,
                    allowed_read_roots=self._allowed_read_roots,
                )
                novelty_summary = _extract_novelty_summary(
                    _load_json_payload(novelty_path)
                )

            draft_quality_summary: dict[str, Any] | None = None
            if str(draft_quality_report_path or "").strip():
                dq_path = _resolve_read_path(
                    draft_quality_report_path,
                    allowed_dir=self._allowed_dir,
                    allowed_read_roots=self._allowed_read_roots,
                )
                draft_quality_summary = _extract_draft_quality_summary(
                    _load_json_payload(dq_path)
                )
            elif str(paper_draft_path or "").strip():
                from annolid.services.paper_writer import assess_paper_draft_quality

                draft_path = _resolve_read_path(
                    paper_draft_path,
                    allowed_dir=self._allowed_dir,
                    allowed_read_roots=self._allowed_read_roots,
                )
                bib_refs: list[dict[str, object]] = []
                if str(bib_path or "").strip():
                    from annolid.utils.citations import load_bibtex

                    resolved_bib = _resolve_read_path(
                        bib_path,
                        allowed_dir=self._allowed_dir,
                        allowed_read_roots=self._allowed_read_roots,
                    )
                    for entry in load_bibtex(resolved_bib):
                        bib_refs.append(
                            {
                                "key": entry.key,
                                "title": entry.fields.get("title") or entry.key,
                                "author": entry.fields.get("author") or "",
                                "year": entry.fields.get("year") or "",
                                "doi": entry.fields.get("doi") or "",
                                "venue": entry.fields.get("journal")
                                or entry.fields.get("booktitle")
                                or "",
                                "source": "bibtex",
                            }
                        )
                draft_quality_summary = _extract_draft_quality_summary(
                    assess_paper_draft_quality(
                        draft_path.read_text(encoding="utf-8"),
                        references=bib_refs,
                    )
                )

            checklist = _build_reproducibility_checklist(
                eval_report=report_obj,
                citation=citation_summary,
                novelty=novelty_summary,
            )
            if isinstance(draft_quality_summary, Mapping) and draft_quality_summary.get(
                "available"
            ):
                checklist.append(
                    {
                        "id": "manuscript_draft_quality",
                        "status": str(
                            draft_quality_summary.get("quality_status") or "warn"
                        ),
                        "message": (
                            f"Manuscript draft audit: words={_safe_int(draft_quality_summary.get('word_count'))}, "
                            f"citations={_safe_int(draft_quality_summary.get('citation_count'))}, "
                            f"placeholders={_safe_int(draft_quality_summary.get('placeholder_count'))}."
                        ),
                    }
                )
            checklist_status = _merge_quality_status(
                *[item.get("status") for item in checklist]
            )
            citation_floor = _validate_unit_interval(
                "citation_integrity_floor",
                float(citation_integrity_floor),
            )
            novelty_floor = _validate_unit_interval(
                "novelty_coverage_floor",
                float(novelty_coverage_floor),
            )
            gate = _build_paper_ready_gate(
                citation=citation_summary,
                novelty=novelty_summary,
                enabled=bool(paper_ready_gate),
                citation_integrity_floor=citation_floor,
                novelty_coverage_floor=novelty_floor,
                require_citation_summary=bool(require_citation_summary),
                require_novelty_summary=bool(require_novelty_summary),
            )
            quality_status = _merge_quality_status(
                report_obj.get("quality_status", "warn"),
                citation_summary.get("quality_status", "warn")
                if isinstance(citation_summary, Mapping)
                else "warn",
                novelty_summary.get("quality_status", "warn")
                if isinstance(novelty_summary, Mapping)
                else "warn",
                draft_quality_summary.get("quality_status", "pass")
                if isinstance(draft_quality_summary, Mapping)
                else "pass",
                checklist_status,
                gate.get("status", "pass"),
            )
            warnings = _collect_warnings(
                eval_report=report_obj,
                citation=citation_summary,
                novelty=novelty_summary,
            )
            if isinstance(draft_quality_summary, Mapping) and draft_quality_summary.get(
                "available"
            ):
                for item_msg in draft_quality_summary.get("warnings") or []:
                    warnings.append(f"Manuscript draft: {item_msg}")
            for item in gate.get("checks", []):
                if not isinstance(item, Mapping):
                    continue
                status = _normalize_status(item.get("status"))
                if status in {"warn", "fail"}:
                    message = str(item.get("message") or "").strip()
                    if message:
                        warnings.append(
                            f"Paper-ready gate ({item.get('id', 'check')}): {message}"
                        )
            generated_at_utc = _utc_now_iso()
            markdown = _build_markdown(
                eval_report=report_obj,
                citation=citation_summary,
                novelty=novelty_summary,
                draft_quality=draft_quality_summary,
                warnings=warnings,
                checklist=checklist,
                paper_ready_gate=gate,
                generated_at_utc=generated_at_utc,
                quality_status=quality_status,
            )

            report: dict[str, Any] = {
                "metadata": {
                    "generated_at_utc": generated_at_utc,
                    "model": (
                        report_obj.get("metadata", {}).get("model")
                        if isinstance(report_obj.get("metadata"), Mapping)
                        else ""
                    ),
                    "dataset": (
                        report_obj.get("metadata", {}).get("dataset")
                        if isinstance(report_obj.get("metadata"), Mapping)
                        else ""
                    ),
                    "split": (
                        report_obj.get("metadata", {}).get("split")
                        if isinstance(report_obj.get("metadata"), Mapping)
                        else ""
                    ),
                    "model_family": (
                        report_obj.get("metadata", {}).get("model_family")
                        if isinstance(report_obj.get("metadata"), Mapping)
                        else ""
                    ),
                },
                "evaluation": {
                    "quality_status": _normalize_status(
                        report_obj.get("quality_status")
                    ),
                    "paper_table": dict(report_obj.get("paper_table") or {}),
                    "summary": dict(report_obj.get("summary") or {}),
                    "quality_checks": list(report_obj.get("quality_checks") or []),
                },
                "citation": citation_summary or {"available": False},
                "novelty": novelty_summary or {"available": False},
                "draft_quality": draft_quality_summary or {"available": False},
                "inputs": {
                    "eval_report_json_path": str(eval_report_json_path or ""),
                    "citation_report_path": str(citation_report_path or ""),
                    "novelty_report_path": str(novelty_report_path or ""),
                    "draft_quality_report_path": str(draft_quality_report_path or ""),
                    "paper_draft_path": str(paper_draft_path or ""),
                },
                "warnings": warnings,
                "reproducibility_checklist": checklist,
                "paper_ready_gate": gate,
                "paper_ready": bool(gate.get("paper_ready")),
                "quality_status": quality_status,
                "report_markdown": markdown,
            }

            if str(report_dir or "").strip():
                if bool(paper_ready_gate) and not bool(gate.get("paper_ready")):
                    return json.dumps(
                        {
                            "ok": False,
                            "error": (
                                "Paper-ready gate failed; export was blocked. "
                                "Adjust thresholds or improve citation/novelty quality."
                            ),
                            "report": report,
                        },
                        ensure_ascii=False,
                    )
                if not allow_mutation:
                    return json.dumps(
                        {
                            "ok": False,
                            "error": (
                                "Writing report files modifies state. Retry with "
                                "allow_mutation=true only when that is intended."
                            ),
                        },
                        ensure_ascii=False,
                    )
                report["written_files"] = _write_unified_report_files(
                    report=report,
                    report_dir=_resolve_write_path(
                        report_dir, allowed_dir=self._allowed_dir
                    ),
                    base_name=report_basename,
                )

            return json.dumps({"ok": True, "report": report}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                },
                ensure_ascii=False,
            )


__all__ = ["AnnolidPaperRunReportTool"]
