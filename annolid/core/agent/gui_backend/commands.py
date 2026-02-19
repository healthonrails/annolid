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
    rename_match = rename_with_title_match or rename_to_match
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
        r"\b(?:add|save|store)\s+citation\b.*@[a-zA-Z]+\s*[\{\(]",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if add_bibtex_match:
        bib_file_match = re.search(
            r"\bto\s+([^\n]+?\.bib)\b",
            text,
            flags=re.IGNORECASE,
        )
        return {
            "name": "add_citation_raw",
            "args": {
                "bibtex": text,
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
        "citation",
        "bib",
        "paper",
        "gui_",
        "use ",
        "weather",
        "forecast",
        "temperature",
        "news",
        "price",
        "stock",
        "latest",
        "current",
        "live",
        "today",
    )
    return any(token in text for token in hints)
