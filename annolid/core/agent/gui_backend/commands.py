from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict

from annolid.core.agent.web_intents import LIVE_WEB_INTENT_HINTS
from annolid.core.agent.gui_backend.command_registry import (
    parse_direct_slash_command,
)

_DIRECT_GUI_REFUSAL_HINTS = (
    "cannot directly access",
    "can't directly access",
    "cannot access your local file system",
    "can't access your local file system",
    "i cannot open applications",
    "i can't open applications",
    "don't have access to shell execution tools",
    "do not have access to shell execution tools",
    "don't have access to git commands",
    "do not have access to git commands",
    "no git tools are currently available",
    "don't have file rename capabilities available",
    "do not have file rename capabilities available",
    "don't have file operation capabilities available",
    "do not have file operation capabilities available",
    "don't have shell capabilities available",
    "do not have shell capabilities available",
)

_ACTIVE_FILE_HINTS = {
    "this file",
    "this pdf",
    "active file",
    "active pdf",
    "current file",
    "current pdf",
    "opened file",
    "opened pdf",
    "open file",
    "open pdf",
    "document",
    "this document",
    "current document",
}

_THREEJS_EXAMPLE_ALIASES = {
    "helix": "helix_points_csv",
    "helix_points_csv": "helix_points_csv",
    "wave": "wave_surface_obj",
    "wave_surface": "wave_surface_obj",
    "wave_surface_obj": "wave_surface_obj",
    "sphere": "sphere_points_ply",
    "sphere_points_ply": "sphere_points_ply",
    "brain": "brain_viewer_html",
    "brain_viewer": "brain_viewer_html",
    "brain_viewer_html": "brain_viewer_html",
    "two_mice": "two_mice_html",
    "two_mice_html": "two_mice_html",
}


def _strip_wrapping_quotes(text: str) -> str:
    value = str(text or "").strip()
    if len(value) >= 2 and (
        (value[0] == "'" and value[-1] == "'")
        or (value[0] == '"' and value[-1] == '"')
        or (value[0] == "`" and value[-1] == "`")
    ):
        return value[1:-1].strip()
    return value


def _strip_trailing_punctuation(text: str) -> str:
    return str(text or "").strip().rstrip(").,;!?")


def _looks_like_path(value: str) -> bool:
    token = str(value or "").strip()
    if not token:
        return False
    if token.startswith(("~", "/", "./", "../")):
        return True
    if "\\" in token or "/" in token:
        return True
    if re.match(r"^[a-zA-Z]:\\", token):
        return True
    return False


def _extract_bibtex_payload(text: str) -> str:
    raw = str(text or "")
    if not raw.strip():
        return ""
    code_blocks = re.findall(
        r"```(?:\s*(?:bibtex|bib|tex))?\s*([\s\S]*?)```",
        raw,
        flags=re.IGNORECASE,
    )
    entry_re = re.compile(r"@[a-zA-Z][a-zA-Z0-9_-]*\s*[\{\(]")
    candidates = [blk.strip() for blk in code_blocks if entry_re.search(blk or "")]
    if candidates:
        return "\n\n".join(candidates).strip()
    marker = entry_re.search(raw)
    if marker:
        return raw[marker.start() :].strip()
    return raw.strip()


def _extract_segment_seconds(text: str) -> float | None:
    raw = str(text or "")
    if not raw:
        return None
    match = re.search(
        r"\b(?P<value>\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b",
        raw,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    try:
        value = float(match.group("value"))
    except Exception:
        return None
    if value <= 0.0:
        return None
    return value


def _mentions_defined_behavior_list(text: str) -> bool:
    lowered = str(text or "").lower()
    hints = (
        "defined list",
        "behavior list",
        "behaviour list",
        "from flags",
        "from the flags",
        "from schema",
        "from the schema",
    )
    return any(hint in lowered for hint in hints)


def _extract_instance_count(text: str) -> int | None:
    raw = str(text or "")
    if not raw:
        return None
    patterns = (
        r"\binstances?\s*(?::|=)\s*(\d+)\b",
        r"\b(\d+)\s+(?:instances?|subjects?|animals?|mice|mice\s+instances?)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match is None:
            continue
        try:
            count = int(match.group(1))
        except Exception:
            continue
        if count > 0:
            return count
    return None


def _extract_context_value(text: str, keys: list[str]) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    key_expr = "|".join(re.escape(k) for k in keys)
    stop_tokens = (
        r"video(?:\s+description)?",
        r"instances?",
        r"experiment(?:s|al)?(?:\s+context)?",
        r"(?:behavior|behaviour)\s+definitions?",
        r"definitions?",
        r"focus(?:\s+points?)?",
        r"segment(?:_|\s+)?mode",
        r"segment(?:_|\s+)?seconds?",
        r"segment(?:_|\s+)?frames?",
        r"sample(?:_|\s+)?frames?",
        r"max(?:_|\s+)?segments?",
        r"subject",
        r"overwrite(?:_|\s+)?existing",
        r"llm(?:_|\s+)?profile",
        r"llm(?:_|\s+)?provider",
        r"llm(?:_|\s+)?model",
    )
    stop_expr = "|".join(stop_tokens)
    pattern = (
        rf"(?:^|[\s,;])(?:{key_expr})\s*(?::|=)\s*(?P<value>.+?)"
        rf"(?=(?:[\s,;]+(?:{stop_expr})\s*(?::|=))|$)"
    )
    match = re.search(pattern, raw, flags=re.IGNORECASE | re.DOTALL)
    if match is None:
        return ""
    return _strip_wrapping_quotes(
        _strip_trailing_punctuation(str(match.group("value") or "").strip())
    )


def _extract_behavior_definitions_fallback(text: str) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    patterns = (
        r"\b(?:behavior|behaviour)\s+[a-z0-9 _-]{1,40}\s+(?:is|means|defined\s+as|includes?|counts?\s+of)\b[^.;\n]*",
        r"\baggression\s+bout\b[^.;\n]*(?:defined\s+as|means|includes?|counts?\s+of)\b[^.;\n]*",
    )
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match is None:
            continue
        value = _strip_wrapping_quotes(_strip_trailing_punctuation(match.group(0)))
        if value:
            return value
    return ""


def _extract_focus_points_fallback(text: str) -> str:
    raw = str(text or "")
    if not raw:
        return ""
    focus_match = re.search(
        r"\bfocus\s+on\s+(?P<value>[^.;\n]+)",
        raw,
        flags=re.IGNORECASE,
    )
    if focus_match is not None:
        value = _strip_wrapping_quotes(
            _strip_trailing_punctuation(str(focus_match.group("value") or "").strip())
        )
        if value:
            return value
    action_match = re.search(
        r"\b(?:count|track|identify|monitor)\b[^.;\n]*(?:bouts?|initiator|responder|fight|aggression|slap|run\s+away)[^.;\n]*",
        raw,
        flags=re.IGNORECASE,
    )
    if action_match is not None:
        value = _strip_wrapping_quotes(
            _strip_trailing_punctuation(action_match.group(0))
        )
        if value:
            return value
    return ""


def _extract_behavior_context_args(text: str) -> Dict[str, Any]:
    video_description = _extract_context_value(
        text,
        keys=["video description", "video context", "video"],
    )
    experiment_context = _extract_context_value(
        text,
        keys=["experiment context", "experiment", "experiments"],
    )
    behavior_definitions = _extract_context_value(
        text,
        keys=["behavior definitions", "behaviour definitions", "definitions"],
    )
    focus_points = _extract_context_value(
        text,
        keys=["focus points", "focus", "things to focus", "focus on"],
    )
    if not behavior_definitions:
        behavior_definitions = _extract_behavior_definitions_fallback(text)
    if not focus_points:
        focus_points = _extract_focus_points_fallback(text)
    payload: Dict[str, Any] = {}
    if video_description:
        payload["video_description"] = video_description
    instance_count = _extract_instance_count(text)
    if instance_count is not None:
        payload["instance_count"] = instance_count
    if experiment_context:
        payload["experiment_context"] = experiment_context
    if behavior_definitions:
        payload["behavior_definitions"] = behavior_definitions
    if focus_points:
        payload["focus_points"] = focus_points
    return payload


def _split_behavior_labels(text: str) -> list[str]:
    labels_text = str(text or "").strip()
    labels_text = re.split(
        r"\b(?:every|timeline|uniform|overwrite|replace|from\s+defined\s+list|from\s+schema|from\s+flags|video(?:\s+description)?|instances?|experiment(?:s|al)?(?:\s+context)?|(?:behavior|behaviour)\s+definitions?|definitions?|focus(?:\s+points?)?)\b",
        labels_text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    return [
        p.strip().strip("\"'`").strip(" .")
        for p in re.split(r",|;|\band\b", labels_text, flags=re.IGNORECASE)
        if p.strip().strip("\"'`").strip(" .")
    ]


def _extract_behavior_labels_clause(text: str) -> list[str]:
    raw = str(text or "")
    if not raw:
        return []
    match = re.search(
        r"\b(?:with\s+labels?|labels?|behaviors?(?:\s+list)?)\s*(?::|=)\s*(?P<labels>.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if match is None:
        return []
    labels_text = str(match.group("labels") or "").strip()
    if not labels_text:
        return []
    labels_text = re.split(
        r"\b(?:every|timeline|uniform|overwrite|replace|from\s+defined\s+list|from\s+schema|from\s+flags)\b",
        labels_text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    return _split_behavior_labels(labels_text)


def _strip_behavior_labels_clause(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    cleaned = re.sub(
        r"\s+\b(?:with\s+labels?|labels?|behaviors?(?:\s+list)?)\s*(?::|=)\s*.+$",
        "",
        raw,
        flags=re.IGNORECASE,
    )
    return str(cleaned or "").strip()


def _trim_video_path_to_extension(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    ext_match = re.search(
        r"\.(?:mp4|avi|mov|mkv|m4v|wmv|flv)\b",
        raw,
        flags=re.IGNORECASE,
    )
    if ext_match is None:
        return raw
    return raw[: ext_match.end()].strip().rstrip(").,;!?")


def _normalize_threejs_example_id(value: str) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return ""
    normalized = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    if normalized in _THREEJS_EXAMPLE_ALIASES:
        return _THREEJS_EXAMPLE_ALIASES[normalized]
    if "two" in raw and "mice" in raw:
        return "two_mice_html"
    if "brain" in raw:
        return "brain_viewer_html"
    if "helix" in raw:
        return "helix_points_csv"
    if "wave" in raw:
        return "wave_surface_obj"
    if "sphere" in raw:
        return "sphere_points_ply"
    return ""


def _extract_email_address(text: str) -> str:
    match = re.search(
        r"\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b",
        str(text or ""),
    )
    if not match:
        return ""
    return str(match.group(0) or "").strip()


def _parse_interval_seconds(
    raw_value: str,
    raw_unit: str,
) -> float:
    value = float(raw_value or 0)
    if value <= 0:
        return 0.0
    unit = str(raw_unit or "").strip().lower()
    if unit.startswith(("hour", "hr")):
        return value * 3600.0
    if unit.startswith(("minute", "min")):
        return value * 60.0
    return value


def _extract_tutorial_level(text: str) -> str:
    lower = str(text or "").lower()
    if re.search(r"\b(?:beginner|basic|intro|introduction)\b", lower):
        return "beginner"
    if re.search(r"\b(?:advanced|expert|deep\s*dive)\b", lower):
        return "advanced"
    return "intermediate"


def parse_direct_gui_command(prompt: str) -> Dict[str, Any]:
    text = str(prompt or "").strip()
    if not text:
        return {}
    lower = text.lower()

    slash_command = parse_direct_slash_command(text)
    if slash_command:
        return slash_command

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

    tutorial_match = re.search(
        r"\b(?:create|generate|make|write|build)\b[\s\S]*?\b(?:on[-\s]?demand\s+)?"
        r"(?:tutorial|guide|walkthrough|how[-\s]?to)\b(?:\s+(?:for|on|about))?\s*(?P<topic>.+)?$",
        text,
        flags=re.IGNORECASE,
    )
    annolid_howto_match = re.search(
        r"\bhow\s+(?:do|can)\s+(?:i|we)\s+use\s+annolid\b(?:\s+(?:for|to))?\s*(?P<topic>.+)?$",
        text,
        flags=re.IGNORECASE,
    )
    tutorial_source = tutorial_match or annolid_howto_match
    if tutorial_source:
        raw_topic = str((tutorial_source.groupdict().get("topic") or "")).strip(" .")
        topic = _strip_wrapping_quotes(_strip_trailing_punctuation(raw_topic))
        if not topic:
            topic = "getting started"
        return {
            "name": "generate_annolid_tutorial",
            "args": {
                "topic": topic,
                "level": _extract_tutorial_level(text),
                "save_to_file": bool(
                    re.search(r"\b(?:save|export|write)\b", lower)
                    and re.search(r"\b(?:file|markdown|md)\b", lower)
                ),
                "include_code_refs": bool(
                    re.search(r"\b(?:code|api|implementation|source)\b", lower)
                ),
            },
        }

    rename_with_title_match = re.match(
        r"\s*rename\s+(?P<src>.+?)\s+with\s+title\s+(?P<dst>.+?)\s*$",
        text,
        flags=re.IGNORECASE,
    )
    rename_to_match = re.match(
        r"\s*rename\s+(?P<src>.+?)\s+(?:to|as|->)\s+(?P<dst>.+?)\s*$",
        text,
        flags=re.IGNORECASE,
    )
    move_to_match = re.match(
        r"\s*move\s+(?P<src>.+?)\s+(?:to|as|->)\s+(?P<dst>.+?)\s*$",
        text,
        flags=re.IGNORECASE,
    )
    rename_match = rename_with_title_match or rename_to_match or move_to_match
    if rename_match:
        src_raw = _strip_wrapping_quotes(
            _strip_trailing_punctuation(rename_match.group("src") or "")
        )
        dst_raw = _strip_wrapping_quotes(
            _strip_trailing_punctuation(rename_match.group("dst") or "")
        )
        if src_raw.lower().startswith("file "):
            src_raw = src_raw[5:].strip()
        if src_raw.lower().startswith("pdf "):
            src_raw = src_raw[4:].strip()

        if not dst_raw:
            return {}

        src_lower = src_raw.lower()
        use_active = src_lower in _ACTIVE_FILE_HINTS or src_lower.startswith("this ")
        source_path = "" if use_active else src_raw
        args: Dict[str, Any] = {
            "source_path": source_path,
            "use_active_file": bool(use_active),
            "overwrite": ("overwrite" in lower or "replace" in lower),
        }
        if _looks_like_path(dst_raw):
            args["new_path"] = dst_raw
            args["new_name"] = ""
        else:
            args["new_name"] = dst_raw
            args["new_path"] = ""
        return {"name": "rename_file", "args": args}

    add_bibtex_match = re.search(
        r"\b(?:add|save|store|insert|append|import)\b[\s\S]*@[a-zA-Z][a-zA-Z0-9_-]*\s*[\{\(]",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if add_bibtex_match:
        bib_file_match = re.search(
            r"\b(?:to|into|in)\s+([^\n]+?\.bib)\b",
            text,
            flags=re.IGNORECASE,
        )
        return {
            "name": "add_citation_raw",
            "args": {
                "bibtex": _extract_bibtex_payload(text),
                "bib_file": (
                    _strip_wrapping_quotes(bib_file_match.group(1).strip())
                    if bib_file_match
                    else ""
                ),
            },
        }

    list_citation_match = re.search(
        r"\b(?:list|show|display)\b.*\b(?:citations?|bib(?:tex)?\s+entries?)\b",
        lower,
    )
    if list_citation_match:
        bib_file_match = re.search(
            r"\bfrom\s+([^\n]+?\.bib)\b",
            text,
            flags=re.IGNORECASE,
        )
        query_match = re.search(
            r"\b(?:for|matching|about)\s+([^\n]+)$",
            text,
            flags=re.IGNORECASE,
        )
        return {
            "name": "list_citations",
            "args": {
                "bib_file": (
                    _strip_wrapping_quotes(bib_file_match.group(1).strip())
                    if bib_file_match
                    else ""
                ),
                "query": (
                    _strip_wrapping_quotes(query_match.group(1).strip())
                    if query_match
                    else ""
                ),
            },
        }

    verify_citation_match = re.search(
        r"\b(?:verify|validate|check)\b.*\b(?:citations?|bib(?:tex)?\s+entries?)\b",
        lower,
    )
    if verify_citation_match:
        bib_file_match = re.search(
            r"\bfrom\s+([^\n]+?\.bib)\b",
            text,
            flags=re.IGNORECASE,
        )
        limit_match = re.search(
            r"\b(?:limit|top|first)\s+(\d{1,4})\b",
            text,
            flags=re.IGNORECASE,
        )
        return {
            "name": "verify_citations",
            "args": {
                "bib_file": (
                    _strip_wrapping_quotes(bib_file_match.group(1).strip())
                    if bib_file_match
                    else ""
                ),
                "limit": int(limit_match.group(1)) if limit_match else 200,
            },
        }

    save_citation_match = re.search(
        r"\b(?:save|add|store|export)\b.*\b(?:citation|cite|bib(?:tex)?\b)\b",
        lower,
    )
    if save_citation_match:
        key_match = re.search(
            r"\b(?:as|key)\s+([a-z0-9][a-z0-9:_\-./]{1,127})\b",
            text,
            flags=re.IGNORECASE,
        )
        bib_file_match = re.search(
            r"\bto\s+([^\n]+?\.bib)\b",
            text,
            flags=re.IGNORECASE,
        )
        source = "auto"
        if re.search(r"\b(?:from|in)\s+(?:the\s+)?pdf\b", lower):
            source = "pdf"
        elif re.search(r"\b(?:from|in)\s+(?:the\s+)?web\b", lower) or re.search(
            r"\b(?:from|in)\s+(?:the\s+)?browser\b", lower
        ):
            source = "web"
        return {
            "name": "save_citation",
            "args": {
                "key": (key_match.group(1).strip() if key_match else ""),
                "bib_file": (
                    _strip_wrapping_quotes(bib_file_match.group(1).strip())
                    if bib_file_match
                    else ""
                ),
                "source": source,
                "validate_before_save": not bool(
                    re.search(
                        r"\b(?:without|skip|no)\s+(?:online\s+)?validation\b",
                        lower,
                    )
                ),
                "strict_validation": bool(
                    re.search(r"\bstrict(?:\s+validation)?\b", lower)
                ),
                "verify_after_save": bool(
                    re.search(r"\bverify(?:\s+after\s+save)?\b", lower)
                    or re.search(r"\bverification\b", lower)
                    or re.search(r"\bintegrity\s+report\b", lower)
                ),
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
            labels = _split_behavior_labels(labels_text)
            segment_seconds = _extract_segment_seconds(text)
            return {
                "name": "label_behavior_segments",
                "args": {
                    "path": path_text,
                    "behavior_labels": labels,
                    "use_defined_behavior_list": _mentions_defined_behavior_list(text),
                    "segment_mode": "uniform",
                    "segment_seconds": segment_seconds,
                    "overwrite_existing": False,
                    **_extract_behavior_context_args(text),
                },
            }

    label_match = re.search(
        r"\blabel\s+behaviors?\b.*?\b(?:in|for)\b\s+(?P<path>.+)",
        text,
        flags=re.IGNORECASE,
    )
    if label_match:
        path_text = label_match.group("path").strip()
        labels: list[str] = []
        with_labels_match = re.search(
            r"^(?P<path>.+?)\s+\bwith\s+labels?\b\s+(?P<labels>.+)$",
            path_text,
            flags=re.IGNORECASE,
        )
        if with_labels_match:
            path_text = with_labels_match.group("path").strip()
            labels_text = with_labels_match.group("labels").strip()
            labels = _split_behavior_labels(labels_text)
        if not labels:
            labels = _extract_behavior_labels_clause(
                path_text
            ) or _extract_behavior_labels_clause(text)
        path_text = _strip_behavior_labels_clause(path_text)
        path_text = _trim_video_path_to_extension(path_text)
        if path_text.lower().startswith("video "):
            path_text = path_text[6:].strip()
        if re.search(
            r"\.(?:mp4|avi|mov|mkv|m4v|wmv|flv)\b",
            path_text,
            flags=re.IGNORECASE,
        ):
            mode = "timeline" if "timeline" in lower else "uniform"
            overwrite = "overwrite" in lower or "replace" in lower
            segment_seconds = _extract_segment_seconds(text)
            return {
                "name": "label_behavior_segments",
                "args": {
                    "path": path_text,
                    "behavior_labels": labels if labels else None,
                    "use_defined_behavior_list": _mentions_defined_behavior_list(text),
                    "segment_mode": mode,
                    "segment_seconds": segment_seconds,
                    "overwrite_existing": overwrite,
                    **_extract_behavior_context_args(text),
                },
            }

    process_match = re.search(
        r"\b(?:process|analy[sz]e|run)\b.*?\bvideo\b.*?\bbehaviors?\b.*?\b(?:in|for|on)\b\s+(?P<path>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if process_match:
        path_text = _trim_video_path_to_extension(process_match.group("path").strip())
        if path_text.lower().startswith("video "):
            path_text = path_text[6:].strip()
        if re.search(
            r"\.(?:mp4|avi|mov|mkv|m4v|wmv|flv)\b",
            path_text,
            flags=re.IGNORECASE,
        ):
            labels = _extract_behavior_labels_clause(text)
            segment_seconds = _extract_segment_seconds(text)
            mode = "timeline" if "timeline" in lower else "uniform"
            return {
                "name": "process_video_behaviors",
                "args": {
                    "path": path_text,
                    "text_prompt": "animal",
                    "behavior_labels": labels if labels else None,
                    "use_defined_behavior_list": _mentions_defined_behavior_list(text),
                    "segment_mode": mode,
                    "segment_seconds": segment_seconds,
                    "run_tracking": True,
                    "run_behavior_labeling": True,
                    "overwrite_existing": ("overwrite" in lower or "replace" in lower),
                    **_extract_behavior_context_args(text),
                },
            }

    behavior_catalog_list_match = re.search(
        r"\b(?:list|show|display)\b\s+(?:the\s+)?(?:behavior|behaviour)s?\b"
        r"|\b(?:behavior|behaviour)s?\s+list\b",
        lower,
    )
    if behavior_catalog_list_match:
        return {"name": "behavior_catalog", "args": {"action": "list"}}

    behavior_catalog_save_match = re.search(
        r"\b(?:save|persist|store)\b\s+(?:the\s+)?(?:behavior|behaviour)\s+"
        r"(?:catalog|list|schema)\b",
        lower,
    )
    if behavior_catalog_save_match:
        return {"name": "behavior_catalog", "args": {"action": "save"}}

    behavior_catalog_delete_match = re.search(
        r"\b(?:delete|remove)\b\s+(?:the\s+)?(?:behavior|behaviour)\s+"
        r"(?P<code>[a-z0-9][a-z0-9._-]*)",
        lower,
    )
    if behavior_catalog_delete_match:
        return {
            "name": "behavior_catalog",
            "args": {
                "action": "delete",
                "code": behavior_catalog_delete_match.group("code").strip(),
            },
        }

    behavior_catalog_update_match = re.search(
        r"\bupdate\b\s+(?:the\s+)?(?:behavior|behaviour)\s+"
        r"(?P<code>[a-z0-9][a-z0-9._-]*)",
        lower,
    )
    if behavior_catalog_update_match:
        return {
            "name": "behavior_catalog",
            "args": {
                "action": "update",
                "code": behavior_catalog_update_match.group("code").strip(),
            },
        }

    behavior_catalog_create_match = re.search(
        r"\b(?:create|add|new)\b\s+(?:a\s+)?(?:behavior|behaviour)\s+"
        r"(?P<code>[a-z0-9][a-z0-9._-]*)",
        lower,
    )
    if behavior_catalog_create_match:
        return {
            "name": "behavior_catalog",
            "args": {
                "action": "create",
                "code": behavior_catalog_create_match.group("code").strip(),
            },
        }

    list_pdfs_match = re.search(
        r"\b(?:list|show|find|search)\b\s+(?:all\s+)?(?:the\s+)?(?:local\s+)?pdfs?\b",
        lower,
    )
    if list_pdfs_match:
        query_match = re.search(
            r"\b(?:by|for|containing|named)\s+(?P<query>.+)$",
            lower,
        )
        return {
            "name": "list_pdfs",
            "args": {
                "query": query_match.group("query").strip() if query_match else None
            },
        }

    summarize_pdf_match = re.search(
        r"^\s*(?:please\s+)?"
        r"(?:summarize|summarise|summarization|summarisation|summarzie|summary|tldr|tl;dr|overview|explain)\b"
        r"[\s\S]*\b(?:paper|pdf|document)\b",
        lower,
    )
    if summarize_pdf_match:
        # Keep this direct command narrowly scoped; source-page extraction/explanation
        # prompts are handled better by the normal tool loop + PDF fallback path.
        if "source:" in lower:
            return {}
        return {"name": "pdf_summarize", "args": {}}

    clawhub_install_match = re.search(
        r"\b(?:install|add)\s+(?:the\s+)?(?:skill\s+)?(?P<slug>[a-z0-9][a-z0-9._-]{0,127})\s+"
        r"(?:from\s+)?clawhub\b",
        lower,
    )
    if clawhub_install_match:
        return {
            "name": "clawhub_install_skill",
            "args": {"slug": clawhub_install_match.group("slug").strip()},
        }

    clawhub_search_match = re.search(
        r"\b(?:search|find|discover)\b.*\bskills?\b.*\b(?:on|in|from)\s+clawhub\b",
        lower,
    ) or re.search(
        r"\b(?:search|find|discover)\s+clawhub\s+(?:skills?\s+)?(?:for\s+)?(?P<q>.+)$",
        lower,
    )
    if clawhub_search_match:
        query = ""
        if clawhub_search_match.groupdict().get("q"):
            query = str(clawhub_search_match.group("q") or "").strip(" .")
        else:
            query_match = re.search(
                r"\b(?:for|about)\s+(?P<query>.+?)\s+(?:on|in|from)\s+clawhub\b",
                lower,
            )
            if query_match:
                query = str(query_match.group("query") or "").strip(" .")
            else:
                trailing_query_match = re.search(
                    r"\b(?:on|in|from)\s+clawhub\b\s+(?:for|about)\s+(?P<query>.+)$",
                    lower,
                )
                if trailing_query_match:
                    query = str(trailing_query_match.group("query") or "").strip(" .")
        if not query:
            query = "annolid"
        return {
            "name": "clawhub_search_skills",
            "args": {"query": query, "limit": 5},
        }

    stop_stream_match = re.search(
        r"\b(?:stop|end|close)\b\s+(?:realtime|real[-\s]?time|stream)\b",
        lower,
    )
    if stop_stream_match:
        return {"name": "stop_realtime_stream", "args": {}}

    schedule_add_match = re.search(
        r"\b(?:schedule|add)\b[\s\S]*?\b("
        r"camera\s*check|periodic\s*report|email\s*summary"
        r")\b[\s\S]*?\bevery\s+(\d+(?:\.\d+)?)\s*"
        r"(seconds?|secs?|minutes?|mins?|hours?|hrs?)?\b",
        lower,
        flags=re.IGNORECASE,
    )
    if schedule_add_match:
        task_type_text = str(schedule_add_match.group(1) or "").strip().lower()
        if "camera" in task_type_text:
            task_type = "camera_check"
        elif "periodic" in task_type_text:
            task_type = "periodic_report"
        else:
            task_type = "email_summary"
        every_seconds = _parse_interval_seconds(
            str(schedule_add_match.group(2) or "").strip(),
            str(schedule_add_match.group(3) or "").strip(),
        )
        camera_source = ""
        source_match = re.search(
            r"\b(?:rtsp|rtsps|rtp|udp|srt|tcp|https?|http)://[^\s\"'<>]+",
            text,
            flags=re.IGNORECASE,
        )
        if source_match:
            camera_source = _strip_trailing_punctuation(source_match.group(0))
        email_to = _extract_email_address(text)
        max_runs_match = re.search(
            r"\b(?:for|max)\s+(\d+)\s+(?:runs?|times?)\b",
            lower,
        )
        return {
            "name": "automation_schedule",
            "args": {
                "action": "add",
                "task_type": task_type,
                "name": task_type,
                "every_seconds": every_seconds,
                "camera_source": camera_source,
                "email_to": email_to,
                "run_immediately": bool(
                    re.search(r"\b(?:start now|run now|immediately)\b", lower)
                ),
                "max_runs": (
                    int(max_runs_match.group(1)) if max_runs_match is not None else None
                ),
            },
        }

    cron_check_match = re.search(
        r"\b(?:check|show|status)\b[\s\S]*\bcron\b[\s\S]*\bjob\b[\s#:=-]*([a-zA-Z0-9_-]{3,128})\b",
        text,
        flags=re.IGNORECASE,
    ) or re.search(
        r"\b(?:check|show|status)\b[\s\S]*\bscheduled\b[\s\S]*\bjob\b[\s#:=-]*([a-zA-Z0-9_-]{3,128})\b",
        text,
        flags=re.IGNORECASE,
    )
    if cron_check_match:
        return {
            "name": "cron",
            "args": {"action": "check", "job_id": cron_check_match.group(1).strip()},
        }

    cron_list_match = re.search(
        r"\b(?:list|show)\b[\s\S]*\b(?:cron|scheduled)\b[\s\S]*\bjobs?\b",
        lower,
    )
    if cron_list_match and "automation" not in lower:
        return {"name": "cron", "args": {"action": "list"}}

    cron_status_match = re.search(
        r"\bcron\b[\s\S]*\bstatus\b|\bstatus\b[\s\S]*\bcron\b",
        lower,
    ) or re.search(
        r"\b(?:scheduler|schedule)\b[\s\S]*\bstatus\b",
        lower,
    )
    if cron_status_match and "automation" not in lower:
        return {"name": "cron", "args": {"action": "status"}}

    schedule_list_match = re.search(
        r"\b(?:list|show)\b[\s\S]*\b(?:automation|scheduled)\b[\s\S]*\b(?:tasks?|jobs?)\b",
        lower,
    )
    if schedule_list_match:
        return {"name": "automation_schedule", "args": {"action": "list"}}

    schedule_status_match = re.search(
        r"\b(?:automation|scheduler)\b[\s\S]*\bstatus\b",
        lower,
    )
    if schedule_status_match:
        return {"name": "automation_schedule", "args": {"action": "status"}}

    schedule_run_match = re.search(
        r"\b(?:run|trigger)\b[\s\S]*\b(?:automation\s+)?(?:task|job)\s+([a-zA-Z0-9_-]+)\b",
        text,
        flags=re.IGNORECASE,
    )
    if schedule_run_match:
        return {
            "name": "automation_schedule",
            "args": {"action": "run", "task_id": schedule_run_match.group(1).strip()},
        }

    schedule_remove_match = re.search(
        r"\b(?:remove|delete|cancel)\b[\s\S]*\b(?:automation\s+)?(?:task|job)\s+([a-zA-Z0-9_-]+)\b",
        text,
        flags=re.IGNORECASE,
    )
    if schedule_remove_match:
        return {
            "name": "automation_schedule",
            "args": {
                "action": "remove",
                "task_id": schedule_remove_match.group(1).strip(),
            },
        }

    check_stream_health_match = re.search(
        r"\b(?:check|test|probe|verify|take|get|send|show)\b.*\b(?:camera|wireless\s+camera|stream(?:ing)?|rtsp|rtp|snapshot|photo|image|frame)\b",
        lower,
    )
    asks_for_email = bool(
        re.search(r"\b(?:email|e-mail|mail)\b", lower)
        and re.search(r"\b(?:send|forward|share)\b", lower)
    )
    parsed_email_to = _extract_email_address(text)
    is_detection_request = bool(
        "yolo11" in lower
        or "mediapipe" in lower
        or re.search(r"\b(?:detect|predict|track)\b", lower)
    )
    if (
        check_stream_health_match
        and (not asks_for_email or parsed_email_to)
        and not is_detection_request
    ):
        camera_source = ""
        stream_match = re.search(
            r"\b(?:rtsp|rtsps|rtp|udp|srt|tcp|https?|http)://[^\s\"'<>]+",
            text,
            flags=re.IGNORECASE,
        )
        if stream_match:
            camera_source = _strip_trailing_punctuation(stream_match.group(0))
        cam_match = re.search(r"\bcamera\s+(\d+)\b", lower)
        if cam_match:
            camera_source = cam_match.group(1)
        elif "webcam" in lower:
            camera_source = "0"
        rtsp_transport = "auto"
        if "rtsp" in lower or "rtsps" in lower:
            if re.search(
                r"\b(?:rtsp(?:\s+over)?\s+tcp|tcp\s+rtsp|using\s+tcp)\b", lower
            ):
                rtsp_transport = "tcp"
            elif re.search(
                r"\b(?:rtsp(?:\s+over)?\s+udp|udp\s+rtsp|using\s+udp)\b", lower
            ):
                rtsp_transport = "udp"
        save_snapshot = bool(
            re.search(
                r"\b(?:save|take|capture)\b.*\b(?:snapshot|photo|image|frame)\b", lower
            )
            or re.search(r"\bsnapshot\b", lower)
        )
        return {
            "name": "check_stream_source",
            "args": {
                "camera_source": camera_source,
                "rtsp_transport": rtsp_transport,
                "timeout_sec": 3.0,
                "probe_frames": 3,
                "save_snapshot": (True if parsed_email_to else save_snapshot),
                "email_to": parsed_email_to,
            },
        }

    if re.search(r"\b(?:realtime|real[-\s]?time)\s+(?:status|state)\b", lower):
        return {"name": "get_realtime_status", "args": {}}

    if re.search(
        r"\b(?:list|show|get)\b.*\b(?:realtime|real[-\s]?time)\b.*\bmodels?\b", lower
    ) or (
        re.search(r"\b(?:realtime|real[-\s]?time)\s+models?\b", lower)
        and not re.search(r"\b(?:start|open|run|launch|begin)\b", lower)
    ):
        return {"name": "list_realtime_models", "args": {}}

    if re.search(
        r"\b(?:list|show|get)\b.*\b(?:realtime|real[-\s]?time)\b.*\blogs?\b", lower
    ) or (
        re.search(r"\b(?:realtime|real[-\s]?time)\s+logs?\b", lower)
        and not re.search(r"\b(?:start|open|run|launch|begin)\b", lower)
    ):
        return {"name": "list_realtime_logs", "args": {}}

    if re.search(
        r"\b(?:realtime|real[-\s]?time|stream(?:ing)?|camera|webcam|device)\b", lower
    ):
        start_stream_hint = (
            re.search(r"\b(?:start|open|run|launch|begin|check|test|detect)\b", lower)
            or ("mediapipe" in lower)
            or ("yolo11" in lower)
        )
        if start_stream_hint:
            model_name = ""
            if "mediapipe face" in lower or "face landmark" in lower:
                model_name = "mediapipe_face"
            elif "mediapipe hands" in lower:
                model_name = "mediapipe_hands"
            elif "mediapipe pose" in lower:
                model_name = "mediapipe_pose"
            elif "yolo11x" in lower:
                model_name = "yolo11x"
            elif "yolo11n" in lower:
                model_name = "yolo11n"
            elif "yolo11" in lower:
                model_name = "yolo11n"
            camera_source = ""
            stream_match = re.search(
                r"\b(?:rtsp|rtsps|rtp|udp|srt|tcp)://[^\s\"'<>]+",
                text,
                flags=re.IGNORECASE,
            )
            if stream_match:
                camera_source = _strip_trailing_punctuation(stream_match.group(0))
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
            rtsp_transport = "auto"
            if "rtsp" in lower or "rtsps" in lower:
                if re.search(
                    r"\b(?:rtsp(?:\s+over)?\s+tcp|tcp\s+rtsp|using\s+tcp)\b", lower
                ):
                    rtsp_transport = "tcp"
                elif re.search(
                    r"\b(?:rtsp(?:\s+over)?\s+udp|udp\s+rtsp|using\s+udp)\b", lower
                ):
                    rtsp_transport = "udp"
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
                    "rtsp_transport": rtsp_transport,
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

    threejs_example_match = re.match(
        r"\s*(?:open|load|show)\s+(?:the\s+)?(?:threejs|three\.js|3d)"
        r"(?:\s+viewer)?\s+(?:an?\s+)?(?:examples?)\b"
        r"(?:\s+(?:called|named))?(?:\s+(?P<example>[^\n]+?))?\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if threejs_example_match:
        example_raw = _strip_wrapping_quotes(
            _strip_trailing_punctuation(threejs_example_match.group("example") or "")
        )
        example_id = _normalize_threejs_example_id(example_raw) or "two_mice_html"
        return {
            "name": "open_threejs_example",
            "args": {"example_id": example_id},
        }

    open_threejs_match = re.match(
        r"\s*(?:open|load|show)\s+(?:the\s+)?(?:threejs|three\.js|3d)"
        r"(?:\s+viewer)?(?:\s+(?P<target>[^\n]+?))?\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if open_threejs_match:
        target_raw = _strip_wrapping_quotes(
            _strip_trailing_punctuation(open_threejs_match.group("target") or "")
        )
        target_raw = re.sub(
            r"^(?:html?|url|file|page)\s+",
            "",
            target_raw,
            flags=re.IGNORECASE,
        ).strip()
        if not target_raw:
            return {
                "name": "open_threejs_example",
                "args": {"example_id": "two_mice_html"},
            }
        return {"name": "open_threejs", "args": {"path_or_url": target_raw}}

    browser_http_match = re.match(
        r"\s*(?:open|load|show)\s+(?:this\s+)?(?P<url>https?://[^\s<>\"]+)\s+"
        r"(?:in\s+(?:the\s+)?)?browser\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if browser_http_match:
        url_text = str(browser_http_match.group("url") or "").strip().rstrip(").,;!?")
        return {"name": "open_in_browser", "args": {"url": url_text}}

    browser_domain_match = re.match(
        r"\s*(?:open|load|show)\s+(?:this\s+)?"
        r"(?P<url>(?:www\.)?[a-z0-9][a-z0-9\-]{0,62}"
        r"(?:\.[a-z0-9][a-z0-9\-]{0,62})+(?::\d+)?(?:/[^\s<>\"]*)?)\s+"
        r"(?:in\s+(?:the\s+)?)?browser\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if browser_domain_match:
        url_text = str(browser_domain_match.group("url") or "").strip().rstrip(").,;!?")
        normalized = (
            url_text
            if url_text.lower().startswith(("http://", "https://"))
            else f"https://{url_text}"
        )
        return {"name": "open_in_browser", "args": {"url": normalized}}

    explicit_url_open = re.match(
        r"\s*(?:open|load|show)\s+(?:this\s+)?(?P<url>https?://[^\s<>\"]+)\s*$",
        text,
        flags=re.IGNORECASE,
    )
    bare_url = re.match(
        r"\s*(?P<url>https?://[^\s<>\"]+)\s*$",
        text,
        flags=re.IGNORECASE,
    )
    url_match = explicit_url_open or bare_url
    if url_match:
        url_text = str(url_match.group("url") or "").strip().rstrip(").,;!?")
        if re.search(r"\.pdf(?:\b|[?#])", url_text, flags=re.IGNORECASE):
            return {"name": "open_pdf", "args": {"path": url_text}}
        return {"name": "open_url", "args": {"url": url_text}}

    domain_pattern = (
        r"(?P<url>(?:www\.)?[a-z0-9][a-z0-9\-]{0,62}"
        r"(?:\.[a-z0-9][a-z0-9\-]{0,62})+(?::\d+)?(?:/[^\s<>\"]*)?)"
    )
    explicit_domain_open = re.match(
        r"\s*(?:open|load|show)\s+(?:this\s+)?" + domain_pattern + r"\s*$",
        text,
        flags=re.IGNORECASE,
    )
    bare_domain = re.match(
        r"\s*" + domain_pattern + r"\s*$",
        text,
        flags=re.IGNORECASE,
    )
    domain_match = explicit_domain_open or bare_domain
    if domain_match:
        url_text = str(domain_match.group("url") or "").strip().rstrip(").,;!?")
        local_candidate = Path(url_text).expanduser()
        if (
            url_text
            and not url_text.lower().startswith(("www.", "http://", "https://"))
            and local_candidate.exists()
            and local_candidate.is_file()
        ):
            return {
                "name": "open_url",
                "args": {"url": str(local_candidate.resolve())},
            }
        if re.search(
            r"\.(?:mp4|avi|mov|mkv|m4v|wmv|flv|pdf|png|jpe?g|gif|tiff?|bmp)\b",
            url_text,
            flags=re.IGNORECASE,
        ) and not url_text.lower().startswith(("www.", "http://", "https://")):
            pass
        else:
            normalized = (
                url_text
                if url_text.lower().startswith(("http://", "https://"))
                else f"https://{url_text}"
            )
            if re.search(r"\.pdf(?:\b|[?#])", normalized, flags=re.IGNORECASE):
                return {"name": "open_pdf", "args": {"path": normalized}}
            return {"name": "open_url", "args": {"url": normalized}}

    open_pdf_hint = (
        "open pdf" in lower
        or "load pdf" in lower
        or "open a pdf" in lower
        or "open the pdf" in lower
        or "gui_open_pdf(" in lower
    )
    open_url_hint = re.search(
        r"\b(?:open|load|show)\s+(?:this\s+)?(?:url|link|website|web\s+page)\b",
        lower,
    )
    url_in_text = re.search(r"https?://[^\s<>\"]+", text, flags=re.IGNORECASE)
    domain_in_text = re.search(
        r"\b(?:www\.)?[a-z0-9][a-z0-9\-]{0,62}"
        r"(?:\.[a-z0-9][a-z0-9\-]{0,62})+(?::\d+)?(?:/[^\s<>\"]*)?",
        text,
        flags=re.IGNORECASE,
    )
    if open_url_hint and (url_in_text or domain_in_text):
        raw_url = url_in_text.group(0) if url_in_text else domain_in_text.group(0)
        url_text = str(raw_url or "").strip().rstrip(").,;!?")
        if not url_text.lower().startswith(("http://", "https://")):
            url_text = f"https://{url_text}"
        if re.search(r"\.pdf(?:\b|[?#])", url_text, flags=re.IGNORECASE):
            return {"name": "open_pdf", "args": {"path": url_text}}
        return {"name": "open_url", "args": {"url": url_text}}
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

    open_local_html_hint = re.match(
        r"\s*(?:open|load|show)\s+[^\n]+?\.(?:html?|xhtml?)\b",
        text,
        flags=re.IGNORECASE,
    )
    if open_local_html_hint or re.fullmatch(
        r"[^\n]+?\.(?:html?|xhtml?)",
        text,
        flags=re.IGNORECASE,
    ):
        return {"name": "open_url", "args": {"url": text}}
    open_local_markdown_hint = re.match(
        r"\s*(?:open|load|show)\s+[^\n]+?\.(?:md|markdown|mdown|mkdn)\b",
        text,
        flags=re.IGNORECASE,
    )
    if open_local_markdown_hint:
        return {"name": "open_url", "args": {"url": text}}
    list_dir_match = re.match(
        r"\s*(?:list\s+directory|ls|dir)\s+(?P<path>[^\n]+?)\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if list_dir_match:
        return {
            "name": "list_dir",
            "args": {
                "path": _strip_wrapping_quotes(list_dir_match.group("path").strip())
            },
        }

    read_file_match = re.match(
        r"\s*(?:read\s+file|cat)\s+(?P<path>[^\n]+?)\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if read_file_match:
        return {
            "name": "read_file",
            "args": {
                "path": _strip_wrapping_quotes(read_file_match.group("path").strip())
            },
        }

    exec_start_match = re.match(
        r"\s*(?:start|run|open|launch)\s+(?:a\s+)?(?:shell|bash|terminal)\s+"
        r"(?:session\s+)?(?:for\s+)?(?P<cmd>.+?)\s*$",
        text,
        flags=re.IGNORECASE,
    ) or re.match(
        r"\s*(?:run|start)\s+(?:in\s+)?background\s*[: ]\s*(?P<cmd>.+?)\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if exec_start_match:
        return {
            "name": "exec_start",
            "args": {
                "command": exec_start_match.group("cmd").strip(),
                "background": True,
            },
        }

    if re.search(r"\b(?:list|show)\b[\s\S]*\bsessions\b", lower) or re.search(
        r"\bsessions\b\s+(?:list|show)\b", lower
    ):
        return {"name": "exec_process", "args": {"action": "list"}}

    sid_match = re.search(r"\b(sh_[a-z0-9]{6,64})\b", lower, flags=re.IGNORECASE)
    session_id = sid_match.group(1) if sid_match else ""
    if session_id and re.search(r"\b(?:poll|check|status)\b[\s\S]*\bsession\b", lower):
        return {
            "name": "exec_process",
            "args": {"action": "poll", "session_id": session_id, "wait_ms": 1500},
        }
    if (
        session_id
        and "session" in lower
        and re.search(r"\b(?:log|logs|tail|output)\b", lower)
    ):
        tail_match = re.search(r"\b(?:last|tail)\s+(\d+)\s+lines?\b", lower)
        return {
            "name": "exec_process",
            "args": {
                "action": "log",
                "session_id": session_id,
                "tail_lines": int(tail_match.group(1)) if tail_match else 200,
            },
        }
    if session_id and re.search(
        r"\b(?:kill|stop|terminate|cancel)\b[\s\S]*\bsession\b", lower
    ):
        return {
            "name": "exec_process",
            "args": {"action": "kill", "session_id": session_id},
        }

    write_match = re.match(
        r"\s*(?:write|send|type|submit)\s+(?:to\s+)?session\s+"
        r"(?P<sid>sh_[a-z0-9]{6,64})\s*(?::|\s)\s*(?P<txt>.+?)\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if write_match:
        action = (
            "submit"
            if re.match(r"^\s*submit\b", text, flags=re.IGNORECASE)
            else "write"
        )
        return {
            "name": "exec_process",
            "args": {
                "action": action,
                "session_id": write_match.group("sid").strip(),
                "text": write_match.group("txt").strip(),
                "submit": action == "submit",
            },
        }

    exec_match = re.match(
        r"\s*(?:run\s+command|exec|!)\s*(?P<cmd>.+?)\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if exec_match:
        return {
            "name": "exec_command",
            "args": {"command": exec_match.group("cmd").strip()},
        }

    annolid_update_match = re.search(
        r"\bannolid(?:-bot)?\b[\s\S]*\b(?:self[-\s]?update|upgrade|update)\b",
        lower,
    ) or re.search(
        r"\b(?:self[-\s]?update|upgrade|update)\b[\s\S]*\bannolid(?:-bot)?\b",
        lower,
    )
    explicit_annolid_run = re.match(
        r"\s*(?:/)?(?:run\s+)?annolid(?:-run|\s+run)\s+",
        text,
        flags=re.IGNORECASE,
    )
    if annolid_update_match and not explicit_annolid_run:
        channel = "stable"
        channel_arg_match = re.search(
            r"--channel\s+(stable|beta|dev)\b",
            lower,
        )
        channel_text_match = re.search(
            r"\b(?:channel\s+(stable|beta|dev)|(stable|beta|dev)\s+channel)\b",
            lower,
        )
        if channel_arg_match:
            channel = str(channel_arg_match.group(1) or "stable").strip().lower()
        elif channel_text_match:
            channel = (
                str(
                    channel_text_match.group(1)
                    or channel_text_match.group(2)
                    or "stable"
                )
                .strip()
                .lower()
            )
        if channel not in {"stable", "beta", "dev"}:
            channel = "stable"

        execute_update = bool(
            re.search(
                r"\b(?:update\s+now|run\s+update|apply\s+update|execute\s+update)\b",
                lower,
            )
            or re.search(r"\bgit\s+pull\b", lower)
            or re.search(r"\bpip\s+install\s+-e\b", lower)
            or re.search(r"\bupgrade\b", lower)
            or "--execute" in lower
        )
        run_post_check = not bool(
            re.search(
                r"\b(?:skip|without|no)\s+(?:post[-\s]?)?check(?:s)?\b",
                lower,
            )
            or "--skip-post-check" in lower
        )
        require_signature = bool(
            re.search(r"\b(?:signed|signature)\b", lower)
            or "--require-signature" in lower
        )
        return {
            "name": "self_update",
            "args": {
                "channel": channel,
                "execute": execute_update,
                "run_post_check": run_post_check,
                "require_signature": require_signature,
                "operator_consent": "approved_by_user" if execute_update else "",
            },
        }

    annolid_run_match = re.match(
        r"\s*(?:/)?(?:run\s+)?annolid(?:-run|\s+run)\s+(?P<cmd>.+?)\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if annolid_run_match:
        return {
            "name": "annolid_run",
            "args": {
                "command": annolid_run_match.group("cmd").strip(),
                "allow_mutation": False,
            },
        }

    annolid_help_match = re.match(
        r"\s*(?:/)?help\s+(?:for\s+)?annolid(?:-run|\s+run)(?:\s+(?P<cmd>.+?))?\s*$",
        text,
        flags=re.IGNORECASE,
    )
    if annolid_help_match:
        topic = str(annolid_help_match.group("cmd") or "").strip()
        return {
            "name": "annolid_run",
            "args": {
                "command": f"help {topic}".strip(),
                "allow_mutation": False,
            },
        }

    # Git/GitHub quick-action aliases: map natural language into dedicated
    # VCS tools first (avoid relying on generic shell-exec availability).
    if re.search(r"\bgit\s+status\b", lower) or re.search(
        r"\bcheck\b[\s\S]*\bgit\b[\s\S]*\bchanges?\b", lower
    ):
        return {"name": "git_status", "args": {"short": True}}
    if re.search(
        r"\b(?:check|show|list)\b[\s\S]*\bunstaged\b[\s\S]*\bchanges?\b", lower
    ):
        return {"name": "git_diff", "args": {"cached": False}}
    if re.search(r"\bstaged\b[\s\S]*\bchanges?\b", lower) or re.search(
        r"\bgit\s+diff\s+--cached\b", lower
    ):
        return {"name": "git_diff", "args": {"cached": True}}
    if re.search(r"\bgit\s+diff\b", lower):
        return {"name": "git_diff", "args": {}}
    if re.search(r"\bgit\s+log\b", lower) or re.search(
        r"\brecent\b[\s\S]*\bcommits?\b", lower
    ):
        return {"name": "git_log", "args": {"max_count": 20, "oneline": True}}
    if re.search(r"\bgh\s+pr\s+status\b", lower) or re.search(
        r"\bgithub\b[\s\S]*\bpr\b[\s\S]*\bstatus\b", lower
    ):
        return {"name": "github_pr_status", "args": {}}
    if re.search(r"\bgh\s+pr\s+checks\b", lower) or re.search(
        r"\bgithub\b[\s\S]*\bpr\b[\s\S]*\bchecks?\b", lower
    ):
        return {"name": "github_pr_checks", "args": {}}

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
        "list",
        "calendar",
        "meeting",
        "appointment",
        "event",
        "schedule",
        "cron",
        "automation",
        "task",
        "job",
        "ls",
        "dir",
        "cat",
        "exec",
        "shell",
        "git",
        "github",
        "gh",
        "commit",
        "diff",
        "staged",
        "unstaged",
        "pr",
        "session",
        "command",
        "pwd",
        "open",
        "email",
        "mail",
        "snapshot",
        "mjpeg",
        "rtsp",
        "rtp",
        "threejs",
        "three.js",
        "3d",
        "download",
        "fetch",
        "extract",
        "video",
        "frame",
        "track",
        "tracking stats",
        "tracking statistics",
        "abnormal segment",
        "abnormal segments",
        "manual frame",
        "manual frames",
        "bad shape",
        "bad shapes",
        "unresolved bad shapes",
        "segment",
        "swarm",
        "subagent",
        "agent",
        "roundtable",
        "discuss",
        "debate",
        "brainstorm",
        "consensus",
        "roadmap",
        "priorities",
        "pitches",
        "feature",
        "proposal",
        "prompt",
        "label",
        "workspace",
        "file",
        "citation",
        "bib",
        "paper",
        "gui_",
        "use ",
        *LIVE_WEB_INTENT_HINTS,
    )
    return any(token in text for token in hints)
