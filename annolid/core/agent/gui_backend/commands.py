from __future__ import annotations

import re
from typing import Any, Dict

_DIRECT_GUI_REFUSAL_HINTS = (
    "cannot directly access",
    "can't directly access",
    "cannot access your local file system",
    "can't access your local file system",
    "i cannot open applications",
    "i can't open applications",
)


def parse_direct_gui_command(prompt: str) -> Dict[str, Any]:
    text = str(prompt or "").strip()
    if not text:
        return {}
    lower = text.lower()

    model_match = re.search(
        r"(?:set|switch)\s+(?:chat\s+)?model\s+"
        r"(ollama|openai|openrouter|gemini)\s*[:/]\s*([^\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    if model_match:
        return {
            "name": "set_chat_model",
            "args": {
                "provider": model_match.group(1).strip().lower(),
                "model": model_match.group(2).strip().strip("."),
            },
        }

    workflow_match = re.search(
        r"\b(segment|track)\b\s+(?P<prompt>.+?)\s+(?:in|on)\s+(?P<path>.+)",
        text,
        flags=re.IGNORECASE,
    )
    if workflow_match:
        mode = workflow_match.group(1).strip().lower()
        text_prompt = workflow_match.group("prompt").strip().strip("\"'")
        path_text = workflow_match.group("path").strip()
        if path_text.lower().startswith("video "):
            path_text = path_text[6:].strip()
        has_video_hint = bool(
            re.search(
                r"\.(?:mp4|avi|mov|mkv|m4v|wmv|flv)\b",
                path_text,
                flags=re.IGNORECASE,
            )
            or "video" in path_text.lower()
        )
        if text_prompt and path_text and has_video_hint:
            to_frame_match = re.search(
                r"\bto\s+frame\s+(\d+)\b",
                text,
                flags=re.IGNORECASE,
            )
            return {
                "name": "segment_track_video",
                "args": {
                    "path": path_text,
                    "text_prompt": text_prompt,
                    "mode": "track" if mode == "track" else "segment",
                    "use_countgd": "countgd" in lower,
                    "to_frame": (
                        int(to_frame_match.group(1))
                        if to_frame_match is not None
                        else None
                    ),
                },
            }

    segment_label_match = re.search(
        r"\b(?:segment|track)\b\s+(?P<path>.+?)\s+\bwith\s+labels?\b\s+(?P<labels>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if segment_label_match:
        path_text = segment_label_match.group("path").strip()
        labels_text = segment_label_match.group("labels").strip()
        if path_text.lower().startswith("video "):
            path_text = path_text[6:].strip()
        if re.search(
            r"\.(?:mp4|avi|mov|mkv|m4v|wmv|flv)\b",
            path_text,
            flags=re.IGNORECASE,
        ):
            labels = [
                p.strip().strip("\"'`").strip(" .")
                for p in re.split(r",|;|\band\b", labels_text, flags=re.IGNORECASE)
                if p.strip().strip("\"'`").strip(" .")
            ]
            return {
                "name": "label_behavior_segments",
                "args": {
                    "path": path_text,
                    "behavior_labels": labels,
                    "segment_mode": "uniform",
                    "overwrite_existing": False,
                },
            }

    label_match = re.search(
        r"\blabel\s+behaviors?\b.*\b(?:in|for)\b\s+(?P<path>.+)",
        text,
        flags=re.IGNORECASE,
    )
    if label_match:
        path_text = label_match.group("path").strip()
        if path_text.lower().startswith("video "):
            path_text = path_text[6:].strip()
        if re.search(
            r"\.(?:mp4|avi|mov|mkv|m4v|wmv|flv)\b",
            path_text,
            flags=re.IGNORECASE,
        ):
            mode = "timeline" if "timeline" in lower else "uniform"
            overwrite = "overwrite" in lower or "replace" in lower
            return {
                "name": "label_behavior_segments",
                "args": {
                    "path": path_text,
                    "segment_mode": mode,
                    "overwrite_existing": overwrite,
                },
            }

    stop_stream_match = re.search(
        r"\b(?:stop|end|close)\b\s+(?:realtime|real[-\s]?time|stream)\b",
        lower,
    )
    if stop_stream_match:
        return {"name": "stop_realtime_stream", "args": {}}

    if re.search(r"\b(?:realtime|real[-\s]?time|stream)\b", lower):
        start_stream_hint = re.search(
            r"\b(?:start|open|run|launch|begin)\b", lower
        ) or ("mediapipe" in lower)
        if start_stream_hint:
            model_name = ""
            if "mediapipe face" in lower or "face landmark" in lower:
                model_name = "mediapipe_face"
            elif "mediapipe hands" in lower:
                model_name = "mediapipe_hands"
            elif "mediapipe pose" in lower:
                model_name = "mediapipe_pose"
            camera_source = ""
            cam_match = re.search(
                r"\bcamera\s+(\d+)\b",
                lower,
            )
            if cam_match:
                camera_source = cam_match.group(1)
            elif "webcam" in lower:
                camera_source = "0"
            viewer_type = (
                "pyqt" if ("pyqt" in lower or "canvas" in lower) else "threejs"
            )
            classify_eye_blinks = bool(
                ("blink" in lower or "eye blink" in lower)
                and model_name == "mediapipe_face"
            )
            return {
                "name": "start_realtime_stream",
                "args": {
                    "camera_source": camera_source,
                    "model_name": model_name,
                    "viewer_type": viewer_type,
                    "classify_eye_blinks": classify_eye_blinks,
                },
            }

    track_match = re.search(
        r"(?:track|predict)(?:\s+from\s+current)?\s+"
        r"(?:to|until)?\s*frame\s+(\d+)",
        lower,
    )
    if track_match:
        return {
            "name": "track_next_frames",
            "args": {"to_frame": int(track_match.group(1))},
        }

    frame_match = re.search(
        r"(?:go\s+to|jump\s+to|set)\s+frame\s+(\d+)",
        lower,
    )
    if frame_match:
        return {
            "name": "set_frame",
            "args": {"frame_index": int(frame_match.group(1))},
        }

    open_pdf_hint = (
        "open pdf" in lower
        or "load pdf" in lower
        or "open a pdf" in lower
        or "open the pdf" in lower
        or "gui_open_pdf(" in lower
    )
    open_pdf_path_hint = re.match(
        r"\s*(?:open|load)\s+[^\n]+?\.pdf\b",
        text,
        flags=re.IGNORECASE,
    )
    if (
        open_pdf_hint
        or open_pdf_path_hint
        or re.fullmatch(
            r"(?:pdf\s+)?[^\n]+?\.pdf",
            text,
            flags=re.IGNORECASE,
        )
    ):
        return {"name": "open_pdf", "args": {"path": text}}

    open_video_hint = (
        "open video" in lower
        or "load video" in lower
        or "open this video" in lower
        or "open the video" in lower
        or "gui_open_video(" in lower
    )
    open_path_hint = re.match(
        r"\s*(?:open|load)\s+[^\n]+?\.(?:mp4|avi|mov|mkv|m4v|wmv|flv)\b",
        text,
        flags=re.IGNORECASE,
    )
    if (
        open_video_hint
        or open_path_hint
        or re.fullmatch(
            r"(?:video\s+)?[^\n]+?\.(?:mp4|avi|mov|mkv|m4v|wmv|flv)",
            text,
            flags=re.IGNORECASE,
        )
    ):
        return {"name": "open_video", "args": {"path": text}}

    return {}


def looks_like_local_access_refusal(text: str) -> bool:
    value = str(text or "").lower()
    if not value:
        return False
    return any(hint in value for hint in _DIRECT_GUI_REFUSAL_HINTS)


def prompt_may_need_tools(prompt: str) -> bool:
    text = str(prompt or "").lower()
    if not text:
        return False
    hints = (
        "tool",
        "search",
        "read",
        "list",
        "open",
        "download",
        "fetch",
        "extract",
        "video",
        "frame",
        "track",
        "segment",
        "prompt",
        "label",
        "workspace",
        "file",
        "gui_",
        "use ",
    )
    return any(token in text for token in hints)
