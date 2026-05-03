from __future__ import annotations

import re
from copy import deepcopy
from urllib.parse import parse_qs, urlsplit, urlunsplit
from typing import Any, Optional, Sequence, Tuple

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
    value = _normalize_http_camera_control_url(value)
    try:
        return int(value)
    except Exception:
        return value


def _normalize_http_camera_control_url(source: str) -> str:
    value = str(source or "").strip()
    if not value:
        return value
    lower = value.lower()
    if not (lower.startswith("http://") or lower.startswith("https://")):
        return value
    try:
        parts = urlsplit(value)
        path = (parts.path or "").lower()
        query = parse_qs(parts.query or "", keep_blank_values=True)
    except Exception:
        return value

    # Common camera admin page URL that is not a media endpoint.
    if path.endswith("/img/main.cgi") and (
        query.get("next_file", [""])[0].lower() == "main.htm" or not parts.query
    ):
        return urlunsplit((parts.scheme, parts.netloc, "/img/video.mjpeg", "", ""))
    return value


def _normalize_rtsp_source(source: object, rtsp_transport: str = "auto") -> object:
    value = str(source or "").strip()
    if not value:
        return source
    lower = value.lower()
    if not lower.startswith(("rtsp://", "rtsps://")):
        return source

    transport = str(rtsp_transport or "auto").strip().lower()
    if transport not in {"auto", "tcp", "udp"}:
        transport = "auto"
    if transport == "auto":
        return value

    if "rtsp_transport=" in lower:
        return value
    separator = "&" if "?" in value else "?"
    return f"{value}{separator}rtsp_transport={transport}"


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


def parse_label_csv(values: object) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        parts = [p.strip() for p in values.split(",") if p.strip()]
    elif isinstance(values, Sequence):
        parts = [str(v).strip() for v in values if str(v).strip()]
    else:
        parts = [str(values).strip()] if str(values).strip() else []
    seen: set[str] = set()
    normalized: list[str] = []
    for value in parts:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(value)
    return normalized


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _to_int_or(default: int, value: object) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _slug_camera_id(value: object, index: int) -> str:
    text = str(value or "").strip() or f"camera{index}"
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_.-")
    return slug or f"camera{index}"


def _split_tcp_address(address: str) -> tuple[str, int] | None:
    value = str(address or "").strip()
    if not value.startswith("tcp://"):
        return None
    host_port = value[len("tcp://") :].strip()
    if ":" not in host_port:
        return None
    host, port_text = host_port.rsplit(":", 1)
    try:
        port = int(port_text)
    except Exception:
        return None
    return host or "*", port


def _tcp_address_with_port(address: str, port: int, *, subscriber: bool) -> str:
    parsed = _split_tcp_address(address)
    if parsed is None:
        return str(address or "")
    host, _old_port = parsed
    if subscriber and host in {"", "*", "0.0.0.0"}:
        host = "127.0.0.1"
    return f"tcp://{host}:{int(port)}"


def _default_subscriber_for_publisher(publisher: str) -> str:
    parsed = _split_tcp_address(publisher)
    if parsed is None:
        return "tcp://127.0.0.1:5555"
    _host, port = parsed
    return _tcp_address_with_port(publisher, port, subscriber=True)


def _normalize_camera_entries(cameras: object) -> list[dict[str, Any]]:
    if cameras is None:
        return []
    raw_items = cameras.get("cameras", []) if isinstance(cameras, dict) else cameras
    if isinstance(raw_items, str):
        raw_items = [p.strip() for p in raw_items.split(",") if p.strip()]
    if not isinstance(raw_items, Sequence):
        return []

    entries: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_items):
        if isinstance(item, dict):
            entry = dict(item)
        else:
            entry = {"source": item}
        source = entry.get("source", entry.get("camera_source", ""))
        camera_id = _slug_camera_id(
            entry.get("id", entry.get("name", f"camera{idx}")), idx
        )
        entry["source"] = source
        entry["camera_id"] = camera_id
        entries.append(entry)
    return entries


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
    viewer_only: bool = False,
    rtsp_transport: str = "auto",
    bot_report_enabled: bool = False,
    bot_report_interval_sec: float = 5.0,
    bot_watch_labels: object = "",
    bot_email_report: bool = False,
    bot_email_to: str = "",
    bot_email_min_interval_sec: float = 60.0,
    save_detection_segments: bool = False,
    detection_segment_targets_csv: str = "",
    detection_segment_output_dir: str = "",
    detection_segment_prebuffer_sec: float = 2.0,
    detection_segment_postbuffer_sec: float = 3.0,
    detection_segment_min_duration_sec: float = 1.0,
    detection_segment_max_duration_sec: float = 120.0,
    gdrive_auto_upload_enabled: bool = False,
    gdrive_auto_upload_delay_sec: float = 5.0,
    gdrive_auto_upload_remote_folder: str = "annolid/realtime_detect",
    gdrive_auto_upload_skip_if_exists: bool = True,
) -> Tuple[RealtimeConfig, dict]:
    viewer_only_mode = bool(viewer_only)
    model_weight = (
        "viewer_only"
        if viewer_only_mode and not str(model_name or "").strip()
        else resolve_realtime_model_weight(model_name)
    )
    camera_value = parse_camera_source(camera_source)
    camera_value = _normalize_rtsp_source(camera_value, rtsp_transport=rtsp_transport)
    targets = parse_target_behaviors(
        behavior_csv=target_behaviors_csv,
        behavior_list=target_behaviors,
        include_eye_blink=bool(classify_eye_blinks),
    )
    if confidence_threshold is None or float(confidence_threshold) < 0:
        conf = 0.25
    else:
        conf = _clamp(float(confidence_threshold), 0.0, 1.0)
    seg_prebuffer = max(0.0, float(detection_segment_prebuffer_sec))
    seg_postbuffer = max(0.0, float(detection_segment_postbuffer_sec))
    seg_min = max(0.0, float(detection_segment_min_duration_sec))
    seg_max = max(seg_min + 1.0, float(detection_segment_max_duration_sec))
    config = RealtimeConfig(
        camera_index=camera_value,
        camera_id="camera0",
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
        publish_frames=True if viewer_only_mode else bool(publish_frames),
        publish_annotated_frames=(
            False if viewer_only_mode else bool(publish_annotated_frames)
        ),
        viewer_only=viewer_only_mode,
        frame_encoding="jpg",
        frame_quality=80,
        save_detection_segments=(
            False if viewer_only_mode else bool(save_detection_segments)
        ),
        detection_segment_targets=parse_target_behaviors(
            behavior_csv=detection_segment_targets_csv,
            behavior_list=None,
            include_eye_blink=False,
        )
        or ["animal", "car", "person"],
        detection_segment_output_dir=str(detection_segment_output_dir or "").strip(),
        detection_segment_prebuffer_sec=seg_prebuffer,
        detection_segment_postbuffer_sec=seg_postbuffer,
        detection_segment_min_duration_sec=seg_min,
        detection_segment_max_duration_sec=seg_max,
    )
    viewer = str(viewer_type or "threejs").strip().lower()
    if viewer not in {"pyqt", "threejs"}:
        viewer = "threejs"
    rtsp_transport_value = str(rtsp_transport or "auto").strip().lower()
    if rtsp_transport_value not in {"auto", "tcp", "udp"}:
        rtsp_transport_value = "auto"
    try:
        bot_report_interval = max(
            1.0,
            float(bot_report_interval_sec)
            if bot_report_interval_sec is not None
            else 5.0,
        )
    except Exception:
        bot_report_interval = 5.0
    try:
        bot_email_min_interval = max(
            10.0,
            float(bot_email_min_interval_sec)
            if bot_email_min_interval_sec is not None
            else 60.0,
        )
    except Exception:
        bot_email_min_interval = 60.0

    extras: dict = {
        "subscriber_address": str(subscriber_address or "tcp://127.0.0.1:5555"),
        "viewer_type": viewer,
        "enable_eye_control": bool(enable_eye_control),
        "enable_hand_control": bool(enable_hand_control),
        "classify_eye_blinks": bool(classify_eye_blinks),
        "suppress_control_dock": bool(suppress_control_dock),
        "log_enabled": bool(log_enabled),
        "log_path": str(log_path or ""),
        "rtsp_transport": rtsp_transport_value,
        "viewer_only": viewer_only_mode,
        "bot_report_enabled": False if viewer_only_mode else bool(bot_report_enabled),
        "bot_report_interval_sec": bot_report_interval,
        "bot_watch_labels": parse_label_csv(bot_watch_labels),
        "bot_email_report": False if viewer_only_mode else bool(bot_email_report),
        "bot_email_to": str(bot_email_to or "").strip(),
        "bot_email_min_interval_sec": bot_email_min_interval,
        "save_detection_segments": bool(config.save_detection_segments),
        "detection_segment_targets": list(config.detection_segment_targets or []),
        "detection_segment_output_dir": str(config.detection_segment_output_dir or ""),
        "gdrive_auto_upload_enabled": False
        if viewer_only_mode
        else bool(gdrive_auto_upload_enabled),
        "gdrive_auto_upload_delay_sec": max(
            0.0, float(gdrive_auto_upload_delay_sec or 0.0)
        ),
        "gdrive_auto_upload_remote_folder": str(
            gdrive_auto_upload_remote_folder or "annolid/realtime_detect"
        ).strip(),
        "gdrive_auto_upload_skip_if_exists": bool(gdrive_auto_upload_skip_if_exists),
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


def build_multi_camera_realtime_launch_payloads(
    *,
    cameras: object,
    base_publisher_address: str = "tcp://*:5555",
    base_subscriber_address: str = "",
    output_root: str = "",
    **kwargs: Any,
) -> list[Tuple[RealtimeConfig, dict]]:
    """Build one isolated realtime launch payload per camera.

    Each camera entry may be a source string/int or a mapping with
    source/camera_source, id/name, publisher_address, subscriber_address,
    and detection_segment_output_dir.
    """
    entries = _normalize_camera_entries(cameras)
    if not entries:
        return []

    base_pub = str(base_publisher_address or "tcp://*:5555")
    pub_parts = _split_tcp_address(base_pub)
    base_pub_port = pub_parts[1] if pub_parts else 5555
    base_sub = str(base_subscriber_address or "")
    if not base_sub:
        base_sub = _default_subscriber_for_publisher(base_pub)
    sub_parts = _split_tcp_address(base_sub)
    base_sub_port = sub_parts[1] if sub_parts else base_pub_port

    sessions: list[Tuple[RealtimeConfig, dict]] = []
    seen_ids: set[str] = set()
    for idx, entry in enumerate(entries):
        camera_id = str(entry["camera_id"])
        if camera_id in seen_ids:
            camera_id = f"{camera_id}_{idx}"
        seen_ids.add(camera_id)

        publisher = str(entry.get("publisher_address") or "").strip()
        if not publisher:
            publisher = _tcp_address_with_port(
                base_pub, base_pub_port + idx, subscriber=False
            )
        subscriber = str(entry.get("subscriber_address") or "").strip()
        if not subscriber:
            subscriber = _tcp_address_with_port(
                base_sub, base_sub_port + idx, subscriber=True
            )

        camera_kwargs = deepcopy(kwargs)
        camera_kwargs["camera_source"] = entry.get("source", "")
        camera_kwargs["publisher_address"] = publisher
        camera_kwargs["subscriber_address"] = subscriber
        output_dir = str(entry.get("detection_segment_output_dir") or "").strip()
        if not output_dir and output_root:
            output_dir = str(output_root).rstrip("/\\") + "/" + camera_id
        if output_dir:
            camera_kwargs["detection_segment_output_dir"] = output_dir

        cfg, extras = build_realtime_launch_payload(**camera_kwargs)
        cfg.camera_id = camera_id
        extras["camera_id"] = camera_id
        extras["camera_source"] = str(cfg.camera_index)
        extras["multi_camera"] = True
        sessions.append((cfg, extras))

    return sessions
