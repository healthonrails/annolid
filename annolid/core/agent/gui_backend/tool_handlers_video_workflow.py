from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional


def segment_track_video_tool(
    *,
    path: str,
    text_prompt: str,
    mode: str,
    use_countgd: bool,
    model_name: str,
    to_frame: Optional[int],
    resolve_video_path: Callable[[str], Optional[Path]],
    invoke_segment_track: Callable[[str, str, str, bool, str, int], bool],
    get_action_result: Callable[[str], Dict[str, Any]],
) -> Dict[str, Any]:
    video_path = resolve_video_path(path)
    if video_path is None:
        return {
            "ok": False,
            "error": "Video not found from provided path/text.",
            "input": str(path or "").strip(),
        }

    prompt_text = str(text_prompt or "").strip()
    if not prompt_text:
        return {"ok": False, "error": "text_prompt is required"}
    mode_norm = str(mode or "track").strip().lower()
    if mode_norm not in {"segment", "track"}:
        return {"ok": False, "error": "mode must be 'segment' or 'track'"}
    target_frame = -1 if to_frame is None else int(to_frame)
    if target_frame != -1 and target_frame < 1:
        return {"ok": False, "error": "to_frame must be >= 1"}

    resolved_model = str(model_name or "").strip()
    if mode_norm == "track" and not resolved_model:
        resolved_model = "Cutie"

    ok = invoke_segment_track(
        str(video_path),
        prompt_text,
        mode_norm,
        bool(use_countgd),
        resolved_model,
        target_frame,
    )
    if not ok:
        return {"ok": False, "error": "Failed to queue segment/track workflow action"}

    widget_result = get_action_result("segment_track_video")
    if widget_result:
        if not bool(widget_result.get("ok", False)):
            return {
                "ok": False,
                "error": str(
                    widget_result.get("error")
                    or "Segment/track workflow failed in GUI."
                ),
                "path": str(video_path),
                "basename": Path(video_path).name,
                "text_prompt": prompt_text,
                "mode": mode_norm,
            }
        return {
            "ok": True,
            "path": str(video_path),
            "basename": Path(video_path).name,
            "text_prompt": prompt_text,
            "mode": str(widget_result.get("mode") or mode_norm),
            "use_countgd": bool(use_countgd),
            "model_name": str(widget_result.get("model_name") or resolved_model),
            "to_frame": (
                widget_result.get("to_frame")
                if widget_result.get("to_frame") is not None
                else (None if target_frame == -1 else target_frame)
            ),
        }

    return {
        "ok": True,
        "queued": True,
        "path": str(video_path),
        "basename": Path(video_path).name,
        "text_prompt": prompt_text,
        "mode": mode_norm,
        "use_countgd": bool(use_countgd),
        "model_name": resolved_model,
        "to_frame": None if target_frame == -1 else target_frame,
    }


def label_behavior_segments_tool(
    *,
    path: str,
    behavior_labels: Any,
    segment_mode: str,
    segment_frames: int,
    max_segments: int,
    subject: str,
    overwrite_existing: bool,
    llm_profile: str,
    llm_provider: str,
    llm_model: str,
    resolve_video_path: Callable[[str], Optional[Path]],
    invoke_label_behavior: Callable[
        [str, str, str, int, int, str, bool, str, str, str], bool
    ],
    get_action_result: Callable[[str], Dict[str, Any]],
) -> Dict[str, Any]:
    resolved_path = None
    if str(path or "").strip():
        resolved_path = resolve_video_path(str(path))
        if resolved_path is None:
            return {
                "ok": False,
                "error": "Video not found from provided path/text.",
                "input": str(path or "").strip(),
            }

    labels: list[str] = []
    if isinstance(behavior_labels, list):
        labels = [str(v).strip() for v in behavior_labels if str(v).strip()]
    elif isinstance(behavior_labels, str):
        labels = [p.strip() for p in behavior_labels.split(",") if p.strip()]

    mode_norm = str(segment_mode or "timeline").strip().lower()
    if mode_norm not in {"timeline", "uniform"}:
        return {"ok": False, "error": "segment_mode must be 'timeline' or 'uniform'"}
    frames = max(1, int(segment_frames))
    max_seg = max(1, int(max_segments))

    ok = invoke_label_behavior(
        str(resolved_path) if resolved_path else "",
        ",".join(labels),
        mode_norm,
        frames,
        max_seg,
        str(subject or "Agent"),
        bool(overwrite_existing),
        str(llm_profile or ""),
        str(llm_provider or ""),
        str(llm_model or ""),
    )
    if not ok:
        return {"ok": False, "error": "Failed to queue behavior labeling action"}

    widget_result = get_action_result("label_behavior_segments")
    if widget_result:
        if not bool(widget_result.get("ok", False)):
            return {
                "ok": False,
                "error": str(
                    widget_result.get("error")
                    or "Behavior segment labeling failed in GUI."
                ),
            }
        return {
            "ok": True,
            "mode": str(widget_result.get("mode") or mode_norm),
            "labeled_segments": int(widget_result.get("labeled_segments") or 0),
            "evaluated_segments": int(widget_result.get("evaluated_segments") or 0),
            "skipped_segments": int(widget_result.get("skipped_segments") or 0),
            "labels_used": list(widget_result.get("labels_used") or labels),
            "timestamps_csv": str(widget_result.get("timestamps_csv") or ""),
            "timestamps_rows": int(widget_result.get("timestamps_rows") or 0),
        }

    return {"ok": True, "queued": True, "mode": mode_norm}
