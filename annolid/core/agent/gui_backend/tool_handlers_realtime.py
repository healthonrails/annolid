from __future__ import annotations

from typing import Any, Callable, Dict


def start_realtime_stream_tool(
    *,
    camera_source: str,
    model_name: str,
    target_behaviors: Any,
    confidence_threshold: float | None,
    viewer_type: str,
    classify_eye_blinks: bool,
    blink_ear_threshold: float | None,
    blink_min_consecutive_frames: int | None,
    invoke_start: Callable[[str, str, str, float, str, bool, float, int], bool],
    get_action_result: Callable[[str], Dict[str, Any]],
) -> Dict[str, Any]:
    model_text = str(model_name or "").strip()
    camera_text = str(camera_source or "").strip()
    viewer = str(viewer_type or "threejs").strip().lower()
    if viewer not in {"pyqt", "threejs"}:
        viewer = "threejs"

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

    ok = invoke_start(
        camera_text,
        model_text,
        ",".join(targets),
        threshold if threshold is not None else -1.0,
        viewer,
        bool(classify_eye_blinks),
        ear_threshold if ear_threshold is not None else -1.0,
        min_blink_frames if min_blink_frames is not None else -1,
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
            "camera_source": str(
                widget_result.get("camera_source") or camera_text or "0"
            ),
            "viewer_type": str(widget_result.get("viewer_type") or viewer),
            "classify_eye_blinks": bool(
                widget_result.get("classify_eye_blinks", classify_eye_blinks)
            ),
        }

    return {
        "ok": True,
        "queued": True,
        "model_name": model_text,
        "camera_source": camera_text or "0",
        "viewer_type": viewer,
        "classify_eye_blinks": bool(classify_eye_blinks),
    }


def stop_realtime_stream_tool(
    *,
    invoke_stop: Callable[[], bool],
) -> Dict[str, Any]:
    if not invoke_stop():
        return {"ok": False, "error": "Failed to queue realtime stop action"}
    return {"ok": True, "queued": True}
