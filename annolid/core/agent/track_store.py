from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from annolid.core.types import BBoxGeometry

from .tools.base import Instance, Instances


@dataclass
class TrackState:
    track_id: str
    label: str
    last_frame: int
    last_bbox: Tuple[float, float, float, float]
    hits: int = 0
    misses: int = 0


class TrackStore:
    """Lightweight track ID manager based on bbox IoU."""

    def __init__(
        self,
        *,
        iou_threshold: float = 0.3,
        max_misses: int = 5,
    ) -> None:
        self._iou_threshold = float(iou_threshold)
        self._max_misses = int(max_misses)
        self._next_id = 1
        self._tracks: Dict[str, TrackState] = {}

    def update(self, instances: Instances) -> Instances:
        frame_idx = int(instances.frame.frame_index)
        updated: List[Instance] = []

        for inst in instances.instances:
            track_id = inst.track_id or inst.instance_id
            if track_id is None:
                track_id = self._match_or_create(inst, frame_idx)
            updated.append(
                Instance(
                    frame=inst.frame,
                    geometry=inst.geometry,
                    label=inst.label,
                    score=inst.score,
                    mask=inst.mask,
                    instance_id=inst.instance_id,
                    track_id=track_id,
                    meta=inst.meta,
                )
            )

        self._prune_old(frame_idx)
        return Instances(frame=instances.frame, instances=updated, meta=instances.meta)

    def active_tracks(self) -> Sequence[TrackState]:
        return list(self._tracks.values())

    def _match_or_create(self, inst: Instance, frame_idx: int) -> str:
        bbox = self._bbox_from_instance(inst)
        if bbox is None:
            return self._create_track(inst, frame_idx, (0.0, 0.0, 0.0, 0.0))

        best_id = None
        best_iou = 0.0
        for track_id, track in self._tracks.items():
            if inst.label and track.label and inst.label != track.label:
                continue
            iou = _bbox_iou(bbox, track.last_bbox)
            if iou > best_iou:
                best_iou = iou
                best_id = track_id

        if best_id is not None and best_iou >= self._iou_threshold:
            track = self._tracks[best_id]
            track.last_bbox = bbox
            track.last_frame = frame_idx
            track.hits += 1
            track.misses = 0
            return best_id

        return self._create_track(inst, frame_idx, bbox)

    def _create_track(
        self,
        inst: Instance,
        frame_idx: int,
        bbox: Tuple[float, float, float, float],
    ) -> str:
        track_id = f"track_{self._next_id:04d}"
        self._next_id += 1
        self._tracks[track_id] = TrackState(
            track_id=track_id,
            label=inst.label or "instance",
            last_frame=frame_idx,
            last_bbox=bbox,
            hits=1,
            misses=0,
        )
        return track_id

    def _prune_old(self, frame_idx: int) -> None:
        to_delete: List[str] = []
        for track_id, track in self._tracks.items():
            if track.last_frame < frame_idx:
                track.misses += 1
            if track.misses > self._max_misses:
                to_delete.append(track_id)
        for track_id in to_delete:
            self._tracks.pop(track_id, None)

    @staticmethod
    def _bbox_from_instance(
        inst: Instance,
    ) -> Optional[Tuple[float, float, float, float]]:
        geom = inst.geometry
        if isinstance(geom, BBoxGeometry):
            return tuple(geom.xyxy)
        return None


def _bbox_iou(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0:
        return 0.0
    return inter_area / denom
