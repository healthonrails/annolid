"""Default implementations for behavior-agent interfaces.

These defaults are intentionally simple and deterministic so GUI/CLI can adopt
Phase 2 orchestration incrementally without contract breakage.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from annolid.domain.behavior_agent import (
    BehaviorSegment,
    BehaviorSubEvent,
    TaskPlan,
    TrackArtifact,
)
from annolid.services.behavior_agent.bout_scoring import (
    normalize_aggression_sub_event_type,
)
from annolid.services.behavior_agent.interfaces import (
    AnalysisRunner,
    BehaviorSegmenter,
    MemoryStore,
    PerceptionAdapter,
    TaskInferencer,
)


@dataclass(frozen=True)
class KeywordTaskInferencer(TaskInferencer):
    """Infer a coarse assay type from filename/context keywords."""

    default_assay: str = "unknown"

    def infer(self, video: str | Path, context: dict | None = None) -> TaskPlan:
        text = " ".join(
            [
                str(video),
                str((context or {}).get("prompt") or ""),
                str((context or {}).get("assay") or ""),
            ]
        ).lower()

        assay = self.default_assay
        confidence = 0.5
        features: list[str] = []

        if "open field" in text or "open_field" in text:
            assay = "open_field"
            confidence = 0.8
            features = ["centroid", "speed", "center_occupancy"]
        elif "novel object" in text or "nor" in text:
            assay = "novel_object_recognition"
            confidence = 0.8
            features = ["nose_tip", "object_centers", "investigation_bouts"]
        elif "social" in text:
            assay = "social_interaction"
            confidence = 0.8
            features = ["nose_to_nose", "distance", "orientation"]
        elif "aggression" in text or "fight" in text:
            assay = "aggression"
            confidence = 0.8
            features = ["attack_initiation", "retreat", "impact_events"]

        return TaskPlan(
            assay_type=assay,
            confidence=confidence,
            objectives=[f"analyze {assay} behaviors"],
            target_features=features,
            context=dict(context or {}),
        )


@dataclass(frozen=True)
class PassThroughPerceptionAdapter(PerceptionAdapter):
    """Deterministic adapter for precomputed artifacts in context.

    Useful for testing and incremental integration before wiring live trackers.
    """

    def run(self, video: str | Path, plan: TaskPlan) -> list[TrackArtifact]:
        _ = (video, plan)
        return []


@dataclass(frozen=True)
class NDJSONPerceptionAdapter(PerceptionAdapter):
    """Load TrackArtifact rows from NDJSON for replayable analysis."""

    ndjson_path: str | Path

    def run(self, video: str | Path, plan: TaskPlan) -> list[TrackArtifact]:
        _ = (video, plan)
        path = Path(self.ndjson_path).expanduser().resolve()
        rows: list[TrackArtifact] = []
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(handle, start=1):
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except Exception:
                    continue
                artifact = self._to_track_artifact(payload, line_no=idx)
                if artifact is not None:
                    rows.append(artifact)
        return rows

    @staticmethod
    def _to_track_artifact(
        payload: dict[str, Any], *, line_no: int
    ) -> TrackArtifact | None:
        if not isinstance(payload, dict):
            return None

        frame_index = payload.get("frame_index")
        if frame_index is None:
            frame_index = payload.get("frame")
        if frame_index is None:
            frame_index = payload.get("frame_id")
        try:
            frame_value = int(frame_index)
        except Exception:
            return None

        artifact_id = (
            str(payload.get("artifact_id") or "").strip() or f"artifact_line_{line_no}"
        )
        bbox = payload.get("bbox_xyxy")
        bbox_xyxy = (
            [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
            if isinstance(bbox, list) and len(bbox) == 4
            else None
        )
        keypoints_raw = payload.get("keypoints")
        keypoints: dict[str, list[float]] = {}
        if isinstance(keypoints_raw, dict):
            for key, value in keypoints_raw.items():
                if isinstance(value, list) and len(value) >= 2:
                    try:
                        keypoints[str(key)] = [float(value[0]), float(value[1])]
                    except Exception:
                        continue

        meta = payload.get("meta")
        if not isinstance(meta, dict):
            meta = {}

        return TrackArtifact(
            artifact_id=artifact_id,
            frame_index=frame_value,
            track_id=(
                str(payload.get("track_id")) if payload.get("track_id") else None
            ),
            label=(str(payload.get("label")) if payload.get("label") else None),
            bbox_xyxy=bbox_xyxy,
            keypoints=keypoints,
            score=(
                float(payload.get("score"))
                if payload.get("score") is not None
                else None
            ),
            source=str(payload.get("source") or "ndjson"),
            meta=meta,
        )


@dataclass(frozen=True)
class AggressionSubEventSegmenter(BehaviorSegmenter):
    """Group normalized aggression sub-events into bouts using frame-gap logic."""

    frame_gap_threshold: int = 20

    def segment(
        self,
        plan: TaskPlan,
        artifacts: list[TrackArtifact],
    ) -> list[BehaviorSegment]:
        _ = plan
        events = self._extract_sub_events(artifacts)
        if not events:
            return []

        bouts: list[list[BehaviorSubEvent]] = []
        current: list[BehaviorSubEvent] = []
        last_frame: int | None = None

        for event in sorted(
            events, key=lambda item: (item.frame_index, item.event_type)
        ):
            if not current:
                current = [event]
                last_frame = event.frame_index
                continue

            assert last_frame is not None
            if event.frame_index - last_frame <= int(self.frame_gap_threshold):
                current.append(event)
                last_frame = event.frame_index
                continue

            bouts.append(current)
            current = [event]
            last_frame = event.frame_index

        if current:
            bouts.append(current)

        segments: list[BehaviorSegment] = []
        for idx, bout in enumerate(bouts, start=1):
            start = min(event.frame_index for event in bout)
            end = max(event.frame_index for event in bout)
            segments.append(
                BehaviorSegment(
                    segment_id=f"aggression_seg_{idx:04d}",
                    label="aggression_bout",
                    start_frame=start,
                    end_frame=end,
                    confidence=1.0,
                    rationale="Grouped from counted aggression sub-events",
                    sub_events=list(bout),
                    meta={"assay_type": plan.assay_type},
                )
            )
        return segments

    @staticmethod
    def _extract_sub_events(artifacts: list[TrackArtifact]) -> list[BehaviorSubEvent]:
        events: list[BehaviorSubEvent] = []
        for artifact in artifacts:
            candidates = AggressionSubEventSegmenter._collect_event_candidates(artifact)
            for event_type, count in candidates:
                normalized = normalize_aggression_sub_event_type(event_type)
                if normalized is None:
                    continue
                if not isinstance(count, int) or count < 0:
                    continue
                events.append(
                    BehaviorSubEvent(
                        event_type=normalized,
                        frame_index=int(artifact.frame_index),
                        count=int(count),
                        actor_id=str(artifact.track_id) if artifact.track_id else None,
                        target_id=None,
                        meta={"artifact_id": artifact.artifact_id},
                    )
                )
        return events

    @staticmethod
    def _collect_event_candidates(artifact: TrackArtifact) -> list[tuple[str, int]]:
        out: list[tuple[str, int]] = []

        if artifact.label:
            out.append((str(artifact.label), int(artifact.meta.get("count", 1))))

        event_type = artifact.meta.get("event_type")
        if isinstance(event_type, str):
            out.append((event_type, int(artifact.meta.get("count", 1))))

        sub_events = artifact.meta.get("sub_events")
        if isinstance(sub_events, list):
            for row in sub_events:
                if not isinstance(row, dict):
                    continue
                event = row.get("event_type")
                count = row.get("count", 1)
                if isinstance(event, str) and isinstance(count, int):
                    out.append((event, count))

        return out


class InMemoryMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._records: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def upsert(self, namespace: str, records: list[dict]) -> None:
        target = self._records[str(namespace)]
        for record in records:
            key = str(record.get("key") or "")
            if not key:
                continue
            replaced = False
            for idx, existing in enumerate(target):
                if str(existing.get("key") or "") == key:
                    target[idx] = dict(record)
                    replaced = True
                    break
            if not replaced:
                target.append(dict(record))

    def search(self, namespace: str, query: str, top_k: int = 5) -> list[dict]:
        q = str(query or "").lower()
        rows = self._records.get(str(namespace), [])
        if not q:
            return list(rows[: int(top_k)])
        scored: list[tuple[int, dict]] = []
        for row in rows:
            text = f"{row.get('key', '')} {row.get('value', '')}".lower()
            score = 1 if q in text else 0
            if score > 0:
                scored.append((score, row))
        return [row for _, row in scored[: int(top_k)]]


class DeterministicAnalysisRunner(AnalysisRunner):
    def generate_code(self, plan: TaskPlan, inputs: dict) -> str:
        _ = inputs
        return (
            "# Auto-generated deterministic analysis stub\n"
            f"ASSAY_TYPE = {plan.assay_type!r}\n"
            "def run(inputs):\n"
            "    return {'assay_type': ASSAY_TYPE, 'status': 'ok'}\n"
        )

    def execute(self, code: str, inputs: dict) -> dict:
        _ = code
        return {
            "status": "ok",
            "artifact_count": int(inputs.get("artifact_count", 0)),
            "segment_count": int(inputs.get("segment_count", 0)),
        }


__all__ = [
    "KeywordTaskInferencer",
    "PassThroughPerceptionAdapter",
    "NDJSONPerceptionAdapter",
    "AggressionSubEventSegmenter",
    "InMemoryMemoryStore",
    "DeterministicAnalysisRunner",
]
