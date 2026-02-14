from __future__ import annotations

from typing import Any, Callable, Dict


def execute_direct_gui_command(
    command: Dict[str, Any],
    *,
    open_video: Callable[[str], Dict[str, Any]],
    open_url: Callable[[str], Dict[str, Any]],
    open_pdf: Callable[[str], Dict[str, Any]],
    set_frame: Callable[[int], Dict[str, Any]],
    track_next_frames: Callable[[int], Dict[str, Any]],
    segment_track_video: Callable[..., Dict[str, Any]],
    label_behavior_segments: Callable[..., Dict[str, Any]],
    start_realtime_stream: Callable[..., Dict[str, Any]],
    stop_realtime_stream: Callable[[], Dict[str, Any]],
    set_chat_model: Callable[[str, str], Dict[str, Any]],
) -> str:
    if not command:
        return ""
    name = str(command.get("name") or "")
    args = dict(command.get("args") or {})
    payload: Dict[str, Any]

    if name == "open_video":
        payload = open_video(str(args.get("path") or ""))
        if payload.get("ok"):
            return f"Opened video in Annolid: {payload.get('path')}"
        return str(payload.get("error") or "Failed to open video.")

    if name == "open_url":
        payload = open_url(str(args.get("url") or ""))
        if payload.get("ok"):
            resolved = str(payload.get("url") or "").strip()
            if resolved:
                return f"Opened URL in Annolid: {resolved}"
            return "Opened URL in Annolid."
        return str(payload.get("error") or "Failed to open URL.")

    if name == "open_pdf":
        payload = open_pdf(str(args.get("path") or ""))
        if payload.get("ok"):
            resolved = str(payload.get("path") or "").strip()
            if resolved:
                return f"Opened PDF in Annolid: {resolved}"
            return "Opened a PDF in Annolid."
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            lines = [
                "Multiple PDFs are available. Reply with the file name or full path to open one:",
            ]
            lines.extend(f"- {item}" for item in choices)
            return "\n".join(lines)
        return str(payload.get("error") or "Failed to open PDF.")

    if name == "set_frame":
        payload = set_frame(int(args.get("frame_index") or 0))
        if payload.get("ok"):
            return f"Moved to frame {payload.get('frame_index')}."
        return str(payload.get("error") or "Failed to set frame.")

    if name == "track_next_frames":
        payload = track_next_frames(int(args.get("to_frame") or 0))
        if payload.get("ok"):
            return f"Started tracking to frame {payload.get('to_frame')}."
        return str(payload.get("error") or "Failed to start tracking.")

    if name == "segment_track_video":
        payload = segment_track_video(
            path=str(args.get("path") or ""),
            text_prompt=str(args.get("text_prompt") or ""),
            mode=str(args.get("mode") or "track"),
            use_countgd=bool(args.get("use_countgd", False)),
            model_name=str(args.get("model_name") or ""),
            to_frame=(
                int(args.get("to_frame"))
                if args.get("to_frame") not in (None, "")
                else None
            ),
        )
        if payload.get("ok"):
            action = str(payload.get("mode") or "track")
            basename = str(payload.get("basename") or "")
            prompt = str(payload.get("text_prompt") or "")
            return (
                f"Started {action} workflow for '{prompt}' in {basename}. "
                "Opened video, segmented, and saved annotations."
            )
        return str(payload.get("error") or "Failed to start workflow.")

    if name == "label_behavior_segments":
        payload = label_behavior_segments(
            path=str(args.get("path") or ""),
            behavior_labels=args.get("behavior_labels"),
            segment_mode=str(args.get("segment_mode") or "timeline"),
            segment_frames=int(args.get("segment_frames") or 60),
            max_segments=int(args.get("max_segments") or 120),
            subject=str(args.get("subject") or "Agent"),
            overwrite_existing=bool(args.get("overwrite_existing", False)),
            llm_profile=str(args.get("llm_profile") or ""),
            llm_provider=str(args.get("llm_provider") or ""),
            llm_model=str(args.get("llm_model") or ""),
        )
        if payload.get("ok"):
            summary = (
                f"Labeled {payload.get('labeled_segments')} behavior segment(s) "
                f"using {payload.get('mode')} mode."
            )
            csv_path = str(payload.get("timestamps_csv") or "").strip()
            if csv_path:
                summary += f" Timestamps saved to {csv_path}."
            return summary
        return str(payload.get("error") or "Failed to label behavior segments.")

    if name == "start_realtime_stream":
        payload = start_realtime_stream(
            camera_source=str(args.get("camera_source") or ""),
            model_name=str(args.get("model_name") or ""),
            target_behaviors=args.get("target_behaviors"),
            confidence_threshold=args.get("confidence_threshold"),
            viewer_type=str(args.get("viewer_type") or ""),
            classify_eye_blinks=bool(args.get("classify_eye_blinks", False)),
            blink_ear_threshold=args.get("blink_ear_threshold"),
            blink_min_consecutive_frames=args.get("blink_min_consecutive_frames"),
        )
        if payload.get("ok"):
            model_name = str(payload.get("model_name") or "")
            return (
                f"Started realtime stream with model {model_name}."
                if model_name
                else "Started realtime stream."
            )
        return str(payload.get("error") or "Failed to start realtime stream.")

    if name == "stop_realtime_stream":
        payload = stop_realtime_stream()
        if payload.get("ok"):
            return "Stopped realtime stream."
        return str(payload.get("error") or "Failed to stop realtime stream.")

    if name == "set_chat_model":
        payload = set_chat_model(
            str(args.get("provider") or ""),
            str(args.get("model") or ""),
        )
        if payload.get("ok"):
            return (
                f"Updated chat model to {payload.get('provider')}/"
                f"{payload.get('model')}."
            )
        return str(payload.get("error") or "Failed to update chat model.")

    return ""
