from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Tuple, Union


@dataclass(frozen=True)
class BBoxGeometry:
    type: Literal["bbox"]
    xyxy: Tuple[float, float, float, float]

    def to_dict(self) -> Dict[str, object]:
        x1, y1, x2, y2 = self.xyxy
        return {"type": "bbox", "xyxy": [float(x1), float(y1), float(x2), float(y2)]}


@dataclass(frozen=True)
class PolygonGeometry:
    type: Literal["polygon"]
    points: Tuple[Tuple[float, float], ...]

    def to_dict(self) -> Dict[str, object]:
        return {
            "type": "polygon",
            "points": [[float(x), float(y)] for x, y in self.points],
        }


@dataclass(frozen=True)
class RLEGeometry:
    type: Literal["rle"]
    size: Tuple[int, int]
    counts: str

    def to_dict(self) -> Dict[str, object]:
        h, w = self.size
        return {"type": "rle", "size": [int(h), int(w)], "counts": str(self.counts)}


Geometry = Union[BBoxGeometry, PolygonGeometry, RLEGeometry]


def geometry_from_dict(payload: Dict[str, object]) -> Geometry:
    kind = str(payload.get("type", "")).strip()
    if kind == "bbox":
        raw = payload.get("xyxy")
        if not isinstance(raw, Sequence) or len(raw) != 4:
            raise ValueError("bbox geometry requires 'xyxy' with 4 numbers.")
        x1, y1, x2, y2 = raw
        return BBoxGeometry("bbox", (float(x1), float(y1), float(x2), float(y2)))

    if kind == "polygon":
        raw_points = payload.get("points")
        if not isinstance(raw_points, Sequence) or len(raw_points) < 3:
            raise ValueError("polygon geometry requires 'points' with >= 3 vertices.")
        points: List[Tuple[float, float]] = []
        for pt in raw_points:
            if not isinstance(pt, Sequence) or len(pt) != 2:
                raise ValueError("polygon point must be [x, y].")
            x, y = pt
            points.append((float(x), float(y)))
        return PolygonGeometry("polygon", tuple(points))

    if kind == "rle":
        raw_size = payload.get("size")
        if not isinstance(raw_size, Sequence) or len(raw_size) != 2:
            raise ValueError("rle geometry requires 'size' [h, w].")
        h, w = raw_size
        counts = payload.get("counts")
        if not isinstance(counts, str) or not counts:
            raise ValueError("rle geometry requires non-empty 'counts' string.")
        return RLEGeometry("rle", (int(h), int(w)), counts)

    raise ValueError(f"Unknown geometry type: {kind!r}")
