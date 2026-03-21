"""Novelty preflight scoring for research drafting workflows."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
}


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return [tok for tok in tokens if tok and tok not in _STOPWORDS]


def _to_reference_text(item: object) -> tuple[str, str]:
    if isinstance(item, Mapping):
        title = str(item.get("title") or "").strip()
        abstract = str(item.get("abstract") or item.get("summary") or "").strip()
        keywords = item.get("keywords")
        kw_text = ""
        if isinstance(keywords, Sequence) and not isinstance(keywords, (str, bytes)):
            kw_text = " ".join(
                str(v or "").strip() for v in keywords if str(v or "").strip()
            )
        text = " ".join(part for part in [title, abstract, kw_text] if part)
        return (title, text)
    text = str(item or "").strip()
    return (text[:120], text)


def novelty_preflight_check(
    *,
    idea_summary: str,
    related_work: Sequence[object],
    idea_title: str = "",
    top_k: int = 5,
    abort_overlap_threshold: float = 0.72,
    differentiate_overlap_threshold: float = 0.45,
) -> dict[str, Any]:
    idea_text = " ".join(
        part
        for part in [str(idea_title or "").strip(), str(idea_summary or "").strip()]
        if part
    )
    idea_tokens = set(_tokenize(idea_text))
    if not idea_tokens:
        return {
            "ok": False,
            "error": "idea_summary or idea_title is required for novelty preflight.",
        }

    scored: list[dict[str, Any]] = []
    related_tokens_union: set[str] = set()
    for idx, item in enumerate(list(related_work or [])):
        title, ref_text = _to_reference_text(item)
        ref_tokens = set(_tokenize(ref_text))
        if not ref_tokens:
            continue
        inter = idea_tokens.intersection(ref_tokens)
        union = idea_tokens.union(ref_tokens)
        jaccard = float(len(inter) / len(union)) if union else 0.0
        containment = float(len(inter) / len(idea_tokens)) if idea_tokens else 0.0
        overlap_score = (0.6 * containment) + (0.4 * jaccard)
        related_tokens_union.update(ref_tokens)
        scored.append(
            {
                "index": idx,
                "title": title or f"reference_{idx + 1}",
                "overlap_score": round(float(overlap_score), 4),
                "jaccard": round(float(jaccard), 4),
                "containment": round(float(containment), 4),
                "shared_terms": sorted(inter)[:25],
            }
        )

    scored.sort(key=lambda row: float(row.get("overlap_score") or 0.0), reverse=True)
    top_items = scored[: max(1, int(top_k or 5))]
    max_overlap = float(top_items[0]["overlap_score"]) if top_items else 0.0
    mean_top3 = (
        sum(float(item.get("overlap_score") or 0.0) for item in scored[:3])
        / float(min(3, len(scored)))
        if scored
        else 0.0
    )
    coverage_ratio = (
        float(len(idea_tokens.intersection(related_tokens_union)) / len(idea_tokens))
        if idea_tokens
        else 0.0
    )
    related_count = int(len(scored))
    if related_count >= 6 and coverage_ratio >= 0.45:
        coverage_quality = "high"
    elif related_count >= 3 and coverage_ratio >= 0.25:
        coverage_quality = "medium"
    else:
        coverage_quality = "low"

    recommendation = "proceed"
    reason = "Low textual overlap with related work."
    if max_overlap >= float(abort_overlap_threshold):
        recommendation = "abort"
        reason = (
            f"Highest overlap score {max_overlap:.2f} exceeds abort threshold "
            f"{float(abort_overlap_threshold):.2f}."
        )
    elif max_overlap >= float(differentiate_overlap_threshold) or mean_top3 >= float(
        differentiate_overlap_threshold * 0.8
    ):
        recommendation = "differentiate"
        reason = (
            f"Overlap indicates adjacent prior art (max={max_overlap:.2f}, "
            f"mean_top3={mean_top3:.2f}). Strengthen differentiation."
        )
    if coverage_quality == "low" and recommendation == "proceed":
        recommendation = "differentiate"
        reason = (
            "Related-work coverage is low; expand literature before drafting claims."
        )

    return {
        "ok": True,
        "recommendation": recommendation,
        "reason": reason,
        "scores": {
            "max_overlap": round(max_overlap, 4),
            "mean_top3_overlap": round(mean_top3, 4),
            "idea_token_coverage": round(coverage_ratio, 4),
            "related_work_count": related_count,
        },
        "coverage_quality": coverage_quality,
        "thresholds": {
            "abort_overlap_threshold": float(abort_overlap_threshold),
            "differentiate_overlap_threshold": float(differentiate_overlap_threshold),
        },
        "top_overlaps": top_items,
    }


__all__ = ["novelty_preflight_check"]
