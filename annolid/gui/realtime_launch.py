from __future__ import annotations

from typing import Optional, Sequence, Tuple

from annolid.realtime.config import Config as RealtimeConfig


def resolve_realtime_model_weight(model_name: str) -> str:
    value = str(model_name or "").strip()
    if not value:
        return "yolo11n-seg.pt"
    key = value.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "mediapipe_face": "mediapipe_face",
        "face_landmarks": "mediapipe_face",
        "mediapipe_hands": "mediapipe_hands",
        "mediapipe_pose": "mediapipe_pose",
        "yolo11n": "yolo11n-seg.pt",
        "yolo11x": "yolo11x-seg.pt",
    }
    return aliases.get(key, value)


def parse_camera_source(camera_source: object) -> object:
    value = str(camera_source or "").strip()
    if not value:
        return 0
    if value.lower() in {"default", "camera", "webcam", "cam", "cam0"}:
        return 0
    try:
        return int(value)
    except Exception:
        return value


def parse_target_behaviors(
    *,
    behavior_csv: str = "",
    behavior_list: Optional[Sequence[str]] = None,
    include_eye_blink: bool = False,
) -> list[str]:
    values: list[str] = []
    if behavior_list is not None:
        values.extend([str(v).strip() for v in behavior_list if str(v).strip()])
    else:
        values.extend(
            [p.strip() for p in str(behavior_csv or "").split(",") if p.strip()]
        )
    seen = set()
    normalized: list[str] = []
    for value in values:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(value)
    if include_eye_blink and "eye_blink" not in seen:
        normalized.append("eye_blink")
    return normalized


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _to_int_or(default: int, value: object) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def build_realtime_launch_payload(
    *,
    camera_source: object = "",
    model_name: str = "",
    target_behaviors_csv: str = "",
    target_behaviors: Optional[Sequence[str]] = None,
    confidence_threshold: Optional[float] = None,
    viewer_type: str = "threejs",
    enable_eye_control: bool = False,
    enable_hand_control: bool = False,
    classify_eye_blinks: bool = False,
    blink_ear_threshold: Optional[float] = None,
    blink_min_consecutive_frames: Optional[int] = None,
    subscriber_address: str = "tcp://127.0.0.1:5555",
    suppress_control_dock: bool = False,
    log_enabled: bool = False,
    log_path: str = "",
    server_address: str = "localhost",
    server_port: int = 5002,
    publisher_address: str = "tcp://*:5555",
    frame_width: int = 1280,
    frame_height: int = 960,
    max_fps: float = 30.0,
    publish_frames: bool = True,
    publish_annotated_frames: bool = False,
) -> Tuple[RealtimeConfig, dict]:
    model_weight = resolve_realtime_model_weight(model_name)
    camera_value = parse_camera_source(camera_source)
    targets = parse_target_behaviors(
        behavior_csv=target_behaviors_csv,
        behavior_list=target_behaviors,
        include_eye_blink=bool(classify_eye_blinks),
    )
    if confidence_threshold is None or float(confidence_threshold) < 0:
        conf = 0.25
    else:
        conf = _clamp(float(confidence_threshold), 0.0, 1.0)
    config = RealtimeConfig(
        camera_index=camera_value,
        server_address=str(server_address or "localhost"),
        server_port=_to_int_or(5002, server_port),
        model_base_name=model_weight,
        publisher_address=str(publisher_address or "tcp://*:5555"),
        target_behaviors=targets,
        confidence_threshold=conf,
        frame_width=max(160, _to_int_or(1280, frame_width)),
        frame_height=max(120, _to_int_or(960, frame_height)),
        max_fps=max(1.0, float(max_fps)),
        visualize=False,
        pause_on_recording_stop=True,
        mask_encoding="rle",
        publish_frames=bool(publish_frames),
        publish_annotated_frames=bool(publish_annotated_frames),
        frame_encoding="jpg",
        frame_quality=80,
    )
    viewer = str(viewer_type or "threejs").strip().lower()
    if viewer not in {"pyqt", "threejs"}:
        viewer = "threejs"
    extras: dict = {
        "subscriber_address": str(subscriber_address or "tcp://127.0.0.1:5555"),
        "viewer_type": viewer,
        "enable_eye_control": bool(enable_eye_control),
        "enable_hand_control": bool(enable_hand_control),
        "classify_eye_blinks": bool(classify_eye_blinks),
        "suppress_control_dock": bool(suppress_control_dock),
        "log_enabled": bool(log_enabled),
        "log_path": str(log_path or ""),
    }
    if blink_ear_threshold is not None:
        try:
            threshold = float(blink_ear_threshold)
        except Exception:
            threshold = -1.0
        if threshold > 0:
            extras["blink_ear_threshold"] = _clamp(threshold, 0.05, 0.6)
    if blink_min_consecutive_frames is not None:
        try:
            min_frames = int(blink_min_consecutive_frames)
        except Exception:
            min_frames = -1
        if min_frames > 0:
            extras["blink_min_consecutive_frames"] = max(1, min(30, min_frames))
    return config, extras
