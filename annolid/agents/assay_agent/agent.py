"""Assay inference agent for coarse task identification."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AssayInferenceResult:
    assay_type: str
    confidence: float
    sampled_frame_indices: list[int] = field(default_factory=list)
    evidence: list[dict[str, object]] = field(default_factory=list)


class AssayInferenceAgent:
    """Infer assay type from lightweight context and sampled frame metadata."""

    def infer(
        self, video: str | Path, context: dict | None = None
    ) -> AssayInferenceResult:
        payload = dict(context or {})
        text = " ".join(
            [
                str(video),
                str(payload.get("prompt") or ""),
                str(payload.get("assay") or ""),
                str(payload.get("metadata") or ""),
            ]
        ).lower()

        total_frames = _to_int(payload.get("total_frames"), default=0)
        sampled = _sample_frame_indices(total_frames=total_frames, max_samples=8)

        assay = "unknown"
        confidence = 0.45
        if any(
            token in text for token in ("novel object", "nor", "object recognition")
        ):
            assay = "novel_object_recognition"
            confidence = 0.9
        elif any(
            token in text for token in ("open field", "open_field", "center periphery")
        ):
            assay = "open_field"
            confidence = 0.88
        elif any(
            token in text
            for token in (
                "social interaction",
                "nose-to-nose",
                "resident intruder",
                "social",
            )
        ):
            assay = "social_interaction"
            confidence = 0.86
        elif any(token in text for token in ("courtship", "mounting", "mating")):
            assay = "courtship"
            confidence = 0.84
        elif any(token in text for token in ("aggression", "fight", "attack")):
            assay = "aggression"
            confidence = 0.86

        evidence = [
            {
                "stage": "assay_inference",
                "signal": "keyword_match",
                "assay_type": assay,
                "confidence": confidence,
                "sampled_frame_indices": list(sampled),
            }
        ]
        return AssayInferenceResult(
            assay_type=assay,
            confidence=float(confidence),
            sampled_frame_indices=sampled,
            evidence=evidence,
        )


def _to_int(value: object, *, default: int) -> int:
    try:
        parsed = int(value)  # type: ignore[arg-type]
        return parsed if parsed >= 0 else default
    except Exception:
        return default


def _sample_frame_indices(*, total_frames: int, max_samples: int) -> list[int]:
    total = max(0, int(total_frames))
    k = max(1, int(max_samples))
    if total <= 0:
        return []
    if total <= k:
        return list(range(total))
    if k == 1:
        return [0]
    step = (total - 1) / float(k - 1)
    values = sorted({int(round(i * step)) for i in range(k)})
    return values[:k]


__all__ = ["AssayInferenceAgent", "AssayInferenceResult"]
