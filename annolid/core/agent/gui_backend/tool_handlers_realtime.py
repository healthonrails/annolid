from __future__ import annotations

import json
import ipaddress
import re
from urllib.parse import urlsplit, urlunsplit
from email.utils import parseaddr
from typing import Any, Callable, Dict


def _redact_camera_source_for_bot_display(source: str) -> str:
    text = str(source or "").strip()
    if not text:
        return text
    if re.fullmatch(r"\d{1,4}", text):
        return text
    if "://" not in text:
        return text
    try:
        parts = urlsplit(text)
    except Exception:
        return text
    scheme = str(parts.scheme or "").lower()
    if scheme not in {"rtsp", "rtsps", "rtp", "http", "https", "srt", "tcp", "udp"}:
        return f"{scheme}://<redacted>" if scheme else "<redacted>"

    host = str(parts.hostname or "").strip()
    redact_host = False
    if host:
        lowered = host.lower()
        if lowered in {"localhost"}:
            redact_host = True
        else:
            try:
                ip_obj = ipaddress.ip_address(host)
                if (
                    ip_obj.is_private
                    or ip_obj.is_loopback
                    or ip_obj.is_link_local
                    or ip_obj.is_multicast
                ):
                    redact_host = True
            except Exception:
                # Hostname case: keep non-local names visible.
                redact_host = False

    if not parts.netloc:
        return text

    if not redact_host:
        # Strip credentials if present, keep host/port.
        safe_netloc = parts.netloc
        if "@" in safe_netloc:
            safe_netloc = safe_netloc.split("@", 1)[1]
        return urlunsplit(
            (parts.scheme, safe_netloc, parts.path, parts.query, parts.fragment)
        )

    port = f":{parts.port}" if parts.port else ""
    if scheme in {"rtp", "udp"} and host:
        replacement = f"@<private-host>{port}"
    else:
        replacement = f"<private-host>{port}"
    return urlunsplit(
        (parts.scheme, replacement, parts.path, parts.query, parts.fragment)
    )


def _normalize_email_recipient(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    _name, addr = parseaddr(text)
    normalized = str(addr or text).strip()
    if not normalized or "@" not in normalized:
        return ""
    if any(ch.isspace() for ch in normalized):
        return ""
    local, _, domain = normalized.rpartition("@")
    if not local or "." not in domain:
        return ""
    return normalized


def start_realtime_stream_tool(
    *,
    camera_source: str,
    model_name: str,
    target_behaviors: Any,
    confidence_threshold: float | None,
    viewer_type: str,
    rtsp_transport: str = "auto",
    classify_eye_blinks: bool,
    blink_ear_threshold: float | None,
    blink_min_consecutive_frames: int | None,
    bot_report_enabled: bool = False,
    bot_report_interval_sec: float | None = None,
    bot_watch_labels: Any = None,
    bot_email_report: bool = False,
    bot_email_to: str = "",
    invoke_start: Callable[
        [str, str, str, float, str, str, bool, float, int, str],
        bool,
    ],
    get_action_result: Callable[[str], Dict[str, Any]],
) -> Dict[str, Any]:
    model_text = str(model_name or "").strip()
    camera_text = str(camera_source or "").strip()
    viewer = str(viewer_type or "threejs").strip().lower()
    if viewer not in {"pyqt", "threejs"}:
        viewer = "threejs"
    transport = str(rtsp_transport or "auto").strip().lower()
    if transport not in {"auto", "tcp", "udp"}:
        transport = "auto"

    targets: list[str] = []
    if isinstance(target_behaviors, list):
        targets = [str(v).strip() for v in target_behaviors if str(v).strip()]
    elif isinstance(target_behaviors, str):
        targets = [p.strip() for p in target_behaviors.split(",") if p.strip()]

    threshold = None
    if confidence_threshold is not None:
        try:
            threshold = float(confidence_threshold)
        except Exception:
            return {
                "ok": False,
                "error": "confidence_threshold must be a float in [0, 1].",
            }
        threshold = max(0.0, min(1.0, threshold))

    ear_threshold = None
    if blink_ear_threshold is not None:
        try:
            ear_threshold = float(blink_ear_threshold)
        except Exception:
            return {"ok": False, "error": "blink_ear_threshold must be a float."}
        ear_threshold = max(0.05, min(0.6, ear_threshold))

    min_blink_frames = None
    if blink_min_consecutive_frames is not None:
        try:
            min_blink_frames = int(blink_min_consecutive_frames)
        except Exception:
            return {
                "ok": False,
                "error": "blink_min_consecutive_frames must be an integer.",
            }
        min_blink_frames = max(1, min(30, min_blink_frames))

    bot_interval = 5.0
    if bot_report_interval_sec is not None:
        try:
            bot_interval = float(bot_report_interval_sec)
        except Exception:
            return {"ok": False, "error": "bot_report_interval_sec must be a number."}
        bot_interval = max(1.0, min(3600.0, bot_interval))

    watch_labels: list[str] = []
    if isinstance(bot_watch_labels, list):
        watch_labels = [str(v).strip() for v in bot_watch_labels if str(v).strip()]
    elif isinstance(bot_watch_labels, str):
        watch_labels = [p.strip() for p in bot_watch_labels.split(",") if p.strip()]
    watch_labels_csv = ",".join(watch_labels)
    email_to = str(bot_email_to or "").strip()

    start_options_json = json.dumps(
        {
            "bot_report_enabled": bool(bot_report_enabled),
            "bot_report_interval_sec": float(bot_interval),
            "bot_watch_labels_csv": watch_labels_csv,
            "bot_email_report": bool(bot_email_report),
            "bot_email_to": email_to,
        },
        separators=(",", ":"),
    )

    ok = invoke_start(
        camera_text,
        model_text,
        ",".join(targets),
        threshold if threshold is not None else -1.0,
        viewer,
        transport,
        bool(classify_eye_blinks),
        ear_threshold if ear_threshold is not None else -1.0,
        min_blink_frames if min_blink_frames is not None else -1,
        start_options_json,
    )
    if not ok:
        return {"ok": False, "error": "Failed to queue realtime start action"}

    widget_result = get_action_result("start_realtime_stream")
    if widget_result:
        if not bool(widget_result.get("ok", False)):
            return {
                "ok": False,
                "error": str(
                    widget_result.get("error") or "Realtime stream failed to start."
                ),
            }
        return {
            "ok": True,
            "model_name": str(widget_result.get("model_name") or model_text),
            "camera_source": _redact_camera_source_for_bot_display(
                str(widget_result.get("camera_source") or camera_text or "0")
            ),
            "viewer_type": str(widget_result.get("viewer_type") or viewer),
            "rtsp_transport": str(widget_result.get("rtsp_transport") or transport),
            "classify_eye_blinks": bool(
                widget_result.get("classify_eye_blinks", classify_eye_blinks)
            ),
            "bot_report_enabled": bool(
                widget_result.get("bot_report_enabled", bot_report_enabled)
            ),
            "bot_report_interval_sec": float(
                widget_result.get("bot_report_interval_sec", bot_interval)
            ),
            "bot_watch_labels": widget_result.get("bot_watch_labels", watch_labels),
            "bot_email_report": bool(
                widget_result.get("bot_email_report", bot_email_report)
            ),
            "bot_email_to": str(widget_result.get("bot_email_to", email_to)),
        }

    return {
        "ok": True,
        "queued": True,
        "model_name": model_text,
        "camera_source": _redact_camera_source_for_bot_display(camera_text or "0"),
        "viewer_type": viewer,
        "rtsp_transport": transport,
        "classify_eye_blinks": bool(classify_eye_blinks),
        "bot_report_enabled": bool(bot_report_enabled),
        "bot_report_interval_sec": float(bot_interval),
        "bot_watch_labels": watch_labels,
        "bot_email_report": bool(bot_email_report),
        "bot_email_to": email_to,
    }


def stop_realtime_stream_tool(
    *,
    invoke_stop: Callable[[], bool],
) -> Dict[str, Any]:
    if not invoke_stop():
        return {"ok": False, "error": "Failed to queue realtime stop action"}
    return {"ok": True, "queued": True}


def get_realtime_status_tool(
    *,
    invoke_widget_json_slot: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    payload = invoke_widget_json_slot("bot_get_realtime_status")
    if "ok" not in payload:
        payload["ok"] = True
    return payload


def list_realtime_models_tool(
    *,
    invoke_widget_json_slot: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    payload = invoke_widget_json_slot("bot_list_realtime_models")
    if "ok" not in payload:
        payload["ok"] = True
    return payload


def list_realtime_logs_tool(
    *,
    invoke_widget_json_slot: Callable[..., Dict[str, Any]],
) -> Dict[str, Any]:
    payload = invoke_widget_json_slot("bot_list_realtime_logs")
    if "ok" not in payload:
        payload["ok"] = True
    return payload


def check_stream_source_tool(
    *,
    camera_source: str,
    rtsp_transport: str = "auto",
    timeout_sec: float = 3.0,
    probe_frames: int = 3,
    save_snapshot: bool = False,
    email_to: str = "",
    email_subject: str = "",
    email_content: str = "",
    invoke_check: Callable[[str, str, float, int, bool], Dict[str, Any]],
) -> Dict[str, Any]:
    source_text = str(camera_source or "").strip()
    if len(source_text) > 2048:
        return {"ok": False, "error": "camera_source is too long."}
    if source_text:
        is_index = bool(re.fullmatch(r"\d{1,4}", source_text))
        has_scheme = bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", source_text))
        if not is_index and not has_scheme:
            return {
                "ok": False,
                "error": (
                    "camera_source must be a camera index or a stream URL "
                    "(rtsp/rtsps/rtp/http/https/srt/tcp/udp)."
                ),
            }
        if has_scheme:
            scheme = source_text.split("://", 1)[0].strip().lower()
            if scheme not in {
                "rtsp",
                "rtsps",
                "rtp",
                "http",
                "https",
                "srt",
                "tcp",
                "udp",
            }:
                return {
                    "ok": False,
                    "error": f"Unsupported stream URL scheme: {scheme}",
                }

    transport = str(rtsp_transport or "auto").strip().lower()
    if transport not in {"auto", "tcp", "udp"}:
        transport = "auto"
    try:
        timeout_value = float(timeout_sec)
    except Exception:
        timeout_value = 3.0
    timeout_value = max(0.5, min(30.0, timeout_value))
    try:
        probe_value = int(probe_frames)
    except Exception:
        probe_value = 3
    probe_value = max(1, min(60, probe_value))
    requested_email_to = _normalize_email_recipient(str(email_to or ""))
    if str(email_to or "").strip() and not requested_email_to:
        return {"ok": False, "error": "email_to must be a valid email address."}
    effective_save_snapshot = bool(save_snapshot or requested_email_to)

    try:
        payload = invoke_check(
            source_text,
            transport,
            timeout_value,
            probe_value,
            effective_save_snapshot,
        )
    except Exception as exc:
        return {"ok": False, "error": f"Stream probe invocation failed: {exc}"}
    if not isinstance(payload, dict):
        return {"ok": False, "error": "Invalid stream probe response."}
    if "camera_source" not in payload:
        payload["camera_source"] = source_text or "0"
    payload["camera_source"] = _redact_camera_source_for_bot_display(
        str(payload.get("camera_source") or source_text or "0")
    )
    if payload.get("error"):
        raw_source = str(source_text or "").strip()
        if raw_source:
            payload["error"] = str(payload.get("error") or "").replace(
                raw_source, payload["camera_source"]
            )
    if "rtsp_transport" not in payload:
        payload["rtsp_transport"] = transport
    if "timeout_sec" not in payload:
        payload["timeout_sec"] = timeout_value
    if "probe_frames" not in payload:
        payload["probe_frames"] = probe_value
    if "save_snapshot" not in payload:
        payload["save_snapshot"] = effective_save_snapshot
    if requested_email_to:
        payload["email_to"] = requested_email_to
    payload["email_requested"] = bool(requested_email_to)
    if str(email_subject or "").strip():
        payload["email_subject"] = str(email_subject or "").strip()
    if str(email_content or "").strip():
        payload["email_content"] = str(email_content or "").strip()
    return payload
