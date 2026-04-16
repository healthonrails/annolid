from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional


def _video_total_frames(path: Path) -> int:
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return 0
        try:
            return max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        finally:
            cap.release()
    except Exception:
        return 0


def _video_fps(path: Path) -> float:
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return 0.0
        try:
            return max(0.0, float(cap.get(cv2.CAP_PROP_FPS) or 0.0))
        finally:
            cap.release()
    except Exception:
        return 0.0


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
    use_defined_behavior_list: bool,
    segment_mode: str,
    segment_frames: int,
    segment_seconds: Optional[float],
    sample_frames_per_segment: int,
    max_segments: int,
    subject: str,
    overwrite_existing: bool,
    llm_profile: str,
    llm_provider: str,
    llm_model: str,
    video_description: str = "",
    instance_count: Optional[int] = None,
    experiment_context: str = "",
    behavior_definitions: str = "",
    focus_points: str = "",
    resolve_video_path: Callable[[str], Optional[Path]],
    invoke_label_behavior: Callable[
        [
            str,
            str,
            bool,
            str,
            int,
            float,
            int,
            int,
            str,
            bool,
            str,
            str,
            str,
            str,
            Optional[int],
            str,
            str,
            str,
        ],
        bool,
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
    seconds = float(segment_seconds) if segment_seconds is not None else 0.0
    if seconds < 0.0:
        seconds = 0.0
    try:
        instance_count_value = (
            int(instance_count) if instance_count is not None else None
        )
    except Exception:
        instance_count_value = None
    if instance_count_value is not None and instance_count_value <= 0:
        instance_count_value = None
    sample_frames = max(1, int(sample_frames_per_segment))
    max_seg = max(1, int(max_segments))
    if mode_norm == "uniform" and resolved_path is not None:
        resolved_video = Path(resolved_path)
        video_fps = _video_fps(resolved_video)
        if seconds > 0.0 and video_fps > 0.0:
            frames = max(1, int(round(seconds * video_fps)))
        total_frames = _video_total_frames(resolved_video)
        if total_frames > 1:
            # For short clips, avoid collapsing to a single uniform segment
            # when defaults (segment_frames=60) exceed video length.
            if frames >= total_frames:
                desired_segments = 4 if total_frames >= 16 else 2
                frames = max(1, int(total_frames // desired_segments))
            # Recover from accidental one-segment requests on longer videos.
            if max_seg == 1 and total_frames > frames:
                max_seg = max(2, int((total_frames + frames - 1) // frames))

    ok = invoke_label_behavior(
        str(resolved_path) if resolved_path else "",
        ",".join(labels),
        bool(use_defined_behavior_list),
        mode_norm,
        frames,
        seconds,
        sample_frames,
        max_seg,
        str(subject or "Agent"),
        bool(overwrite_existing),
        str(llm_profile or ""),
        str(llm_provider or ""),
        str(llm_model or ""),
        str(video_description or ""),
        instance_count_value,
        str(experiment_context or ""),
        str(behavior_definitions or ""),
        str(focus_points or ""),
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
            "segment_frames": int(widget_result.get("segment_frames") or frames),
            "segment_seconds": float(widget_result.get("segment_seconds") or seconds),
            "sample_frames_per_segment": int(
                widget_result.get("sample_frames_per_segment") or sample_frames
            ),
            "use_defined_behavior_list": bool(
                widget_result.get(
                    "use_defined_behavior_list", use_defined_behavior_list
                )
            ),
            "labels_used": list(widget_result.get("labels_used") or labels),
            "video_description": str(
                widget_result.get("video_description") or video_description or ""
            ),
            "instance_count": (
                widget_result.get("instance_count")
                if widget_result.get("instance_count") is not None
                else instance_count_value
            ),
            "experiment_context": str(
                widget_result.get("experiment_context") or experiment_context or ""
            ),
            "behavior_definitions": str(
                widget_result.get("behavior_definitions") or behavior_definitions or ""
            ),
            "focus_points": str(
                widget_result.get("focus_points") or focus_points or ""
            ),
            "timestamps_csv": str(widget_result.get("timestamps_csv") or ""),
            "timestamps_rows": int(widget_result.get("timestamps_rows") or 0),
            "behavior_log_json": str(widget_result.get("behavior_log_json") or ""),
            "behavior_log_rows": int(widget_result.get("behavior_log_rows") or 0),
        }

    return {
        "ok": True,
        "queued": True,
        "mode": mode_norm,
        "segment_frames": int(frames),
        "segment_seconds": float(seconds),
        "sample_frames_per_segment": int(sample_frames),
        "use_defined_behavior_list": bool(use_defined_behavior_list),
        "video_description": str(video_description or ""),
        "instance_count": instance_count_value,
        "experiment_context": str(experiment_context or ""),
        "behavior_definitions": str(behavior_definitions or ""),
        "focus_points": str(focus_points or ""),
    }


def process_video_behaviors_tool(
    *,
    path: str,
    text_prompt: str,
    mode: str,
    use_countgd: bool,
    model_name: str,
    to_frame: Optional[int],
    behavior_labels: Any,
    use_defined_behavior_list: bool,
    segment_mode: str,
    segment_frames: int,
    segment_seconds: Optional[float],
    sample_frames_per_segment: int,
    max_segments: int,
    subject: str,
    overwrite_existing: bool,
    llm_profile: str,
    llm_provider: str,
    llm_model: str,
    video_description: str = "",
    instance_count: Optional[int] = None,
    experiment_context: str = "",
    behavior_definitions: str = "",
    focus_points: str = "",
    run_tracking: bool,
    run_behavior_labeling: bool,
    resolve_video_path: Callable[[str], Optional[Path]],
    invoke_segment_track: Callable[[str, str, str, bool, str, int], bool],
    invoke_label_behavior: Callable[
        [
            str,
            str,
            bool,
            str,
            int,
            float,
            int,
            int,
            str,
            bool,
            str,
            str,
            str,
            str,
            Optional[int],
            str,
            str,
            str,
        ],
        bool,
    ],
    get_action_result: Callable[[str], Dict[str, Any]],
) -> Dict[str, Any]:
    resolved = resolve_video_path(path)
    if resolved is None:
        return {
            "ok": False,
            "error": "Video not found from provided path/text.",
            "input": str(path or "").strip(),
        }

    do_tracking = bool(run_tracking)
    do_labeling = bool(run_behavior_labeling)
    if not do_tracking and not do_labeling:
        return {
            "ok": False,
            "error": (
                "At least one stage must be enabled "
                "(run_tracking or run_behavior_labeling)."
            ),
        }

    resolved_path_str = str(resolved)
    stage_results: Dict[str, Dict[str, Any]] = {}
    resolve_once = lambda _raw: resolved  # noqa: E731

    if do_tracking:
        tracking_payload = segment_track_video_tool(
            path=resolved_path_str,
            text_prompt=text_prompt,
            mode=mode,
            use_countgd=use_countgd,
            model_name=model_name,
            to_frame=to_frame,
            resolve_video_path=resolve_once,
            invoke_segment_track=invoke_segment_track,
            get_action_result=get_action_result,
        )
        stage_results["tracking"] = dict(tracking_payload)
        if not bool(tracking_payload.get("ok", False)):
            return {
                "ok": False,
                "stage": "tracking",
                "error": str(
                    tracking_payload.get("error")
                    or "Video tracking stage failed in GUI workflow."
                ),
                "path": resolved_path_str,
                "basename": Path(resolved_path_str).name,
                "stages": stage_results,
            }

    if do_labeling:
        labeling_payload = label_behavior_segments_tool(
            path=resolved_path_str,
            behavior_labels=behavior_labels,
            use_defined_behavior_list=use_defined_behavior_list,
            segment_mode=segment_mode,
            segment_frames=segment_frames,
            segment_seconds=segment_seconds,
            sample_frames_per_segment=sample_frames_per_segment,
            max_segments=max_segments,
            subject=subject,
            overwrite_existing=overwrite_existing,
            llm_profile=llm_profile,
            llm_provider=llm_provider,
            llm_model=llm_model,
            video_description=video_description,
            instance_count=instance_count,
            experiment_context=experiment_context,
            behavior_definitions=behavior_definitions,
            focus_points=focus_points,
            resolve_video_path=resolve_once,
            invoke_label_behavior=invoke_label_behavior,
            get_action_result=get_action_result,
        )
        stage_results["behavior_labeling"] = dict(labeling_payload)
        if not bool(labeling_payload.get("ok", False)):
            return {
                "ok": False,
                "stage": "behavior_labeling",
                "error": str(
                    labeling_payload.get("error")
                    or "Behavior labeling stage failed in GUI workflow."
                ),
                "path": resolved_path_str,
                "basename": Path(resolved_path_str).name,
                "stages": stage_results,
            }

    return {
        "ok": True,
        "path": resolved_path_str,
        "basename": Path(resolved_path_str).name,
        "tracking_executed": do_tracking,
        "behavior_labeling_executed": do_labeling,
        "stages": stage_results,
    }


def behavior_catalog_tool(
    *,
    action: str,
    code: str = "",
    name: str = "",
    description: str = "",
    category_id: str = "",
    modifier_ids: Any = None,
    key_binding: str = "",
    is_state: Optional[bool] = None,
    exclusive_with: Any = None,
    save: bool = True,
    invoke_behavior_catalog: Callable[[str], bool],
    get_action_result: Callable[[str], Dict[str, Any]],
) -> Dict[str, Any]:
    action_norm = str(action or "").strip().lower()
    if action_norm not in {"list", "save", "create", "update", "delete"}:
        return {
            "ok": False,
            "error": f"Unsupported behavior catalog action: {action}",
        }
    payload = {
        "action": action_norm,
        "code": str(code or "").strip(),
        "name": str(name or "").strip(),
        "description": str(description or "").strip(),
        "category_id": str(category_id or "").strip(),
        "modifier_ids": [
            str(item).strip() for item in (modifier_ids or []) if str(item).strip()
        ],
        "key_binding": str(key_binding or "").strip(),
        "is_state": is_state,
        "exclusive_with": [
            str(item).strip() for item in (exclusive_with or []) if str(item).strip()
        ],
        "save": bool(save),
    }
    ok = invoke_behavior_catalog(json.dumps(payload))
    if not ok:
        return {"ok": False, "error": "Failed to queue behavior catalog action."}
    widget_result = get_action_result("behavior_catalog")
    if widget_result:
        if not bool(widget_result.get("ok", False)):
            return {
                "ok": False,
                "error": str(
                    widget_result.get("error")
                    or "Behavior catalog action failed in GUI."
                ),
            }
        return dict(widget_result)
    return {"ok": True, "queued": True, "action": action_norm}
