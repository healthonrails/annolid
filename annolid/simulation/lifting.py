from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
from PIL import Image

from annolid.simulation.types import Pose2DFrame, Pose3DFrame


def load_depth_records(path: str | Path) -> Dict[int, Dict[str, Any]]:
    depth_path = Path(path).expanduser()
    records: Dict[int, Dict[str, Any]] = {}
    for line in depth_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        frame_index = int(payload.get("frame_index") or 0)
        records[frame_index] = payload
    return records


def lift_pose_frames_with_depth(
    pose_frames: Sequence[Pose2DFrame],
    *,
    depth_records: Mapping[int, Dict[str, Any]],
    coordinate_system: Mapping[str, Any] | None = None,
) -> list[Pose3DFrame]:
    coordinate_system = dict(coordinate_system or {})
    intrinsics = dict(coordinate_system.get("camera_intrinsics") or {})
    lifted: list[Pose3DFrame] = []
    for frame in pose_frames:
        depth_record = depth_records.get(frame.frame_index)
        if depth_record is None:
            raise KeyError(f"Missing depth record for frame_index={frame.frame_index}")
        depth_map = _decode_depth_map(depth_record)
        points_3d = {}
        for label, point in frame.points.items():
            x_px, y_px = point
            z_val = _sample_depth(depth_map, x_px=x_px, y_px=y_px)
            points_3d[label] = _pixel_to_3d(
                x_px=x_px,
                y_px=y_px,
                z_val=z_val,
                intrinsics=intrinsics,
            )
        lifted.append(
            Pose3DFrame(
                frame_index=frame.frame_index,
                video_name=frame.video_name,
                timestamp_sec=frame.timestamp_sec,
                points=points_3d,
                scores=dict(frame.scores),
                source_record=dict(frame.source_record),
            )
        )
    return lifted


def _decode_depth_map(record: Mapping[str, Any]) -> np.ndarray:
    depth_map = dict((record.get("otherData") or {}).get("depth_map") or {})
    encoded = str(depth_map.get("image_data") or "")
    if not encoded:
        raise ValueError("Depth record is missing otherData.depth_map.image_data")
    scale = dict(depth_map.get("scale") or {})
    d_min = float(scale.get("min", 0.0))
    d_max = float(scale.get("max", 1.0))
    payload = base64.b64decode(encoded)
    with Image.open(io.BytesIO(payload)) as image:
        quantized = np.array(image, dtype=np.float32)
    if quantized.size == 0:
        raise ValueError("Decoded depth map is empty")
    depth = (quantized / 65535.0) * (d_max - d_min) + d_min
    return depth


def _sample_depth(depth_map: np.ndarray, *, x_px: float, y_px: float) -> float:
    height, width = depth_map.shape[:2]
    x_idx = min(max(int(round(float(x_px))), 0), max(width - 1, 0))
    y_idx = min(max(int(round(float(y_px))), 0), max(height - 1, 0))
    return float(depth_map[y_idx, x_idx])


def _pixel_to_3d(
    *,
    x_px: float,
    y_px: float,
    z_val: float,
    intrinsics: Mapping[str, Any],
) -> tuple[float, float, float]:
    fx = _as_positive_float(intrinsics.get("fx"))
    fy = _as_positive_float(intrinsics.get("fy"))
    cx = _as_float(intrinsics.get("cx"), default=0.0)
    cy = _as_float(intrinsics.get("cy"), default=0.0)
    if fx is None or fy is None:
        return float(x_px), float(y_px), float(z_val)
    x_3d = (float(x_px) - cx) * float(z_val) / fx
    y_3d = (float(y_px) - cy) * float(z_val) / fy
    return float(x_3d), float(y_3d), float(z_val)


def _as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed
