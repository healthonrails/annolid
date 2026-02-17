import asyncio
from typing import Any, Callable, Dict


async def execute_direct_gui_command(
    command: Dict[str, Any],
    *,
    open_video: Callable[[str], Any],
    open_url: Callable[[str], Any],
    open_in_browser: Callable[[str], Any],
    open_pdf: Callable[[str], Any],
    set_frame: Callable[[int], Any],
    track_next_frames: Callable[[int], Any],
    segment_track_video: Callable[..., Any],
    label_behavior_segments: Callable[..., Any],
    start_realtime_stream: Callable[..., Any],
    stop_realtime_stream: Callable[[], Any],
    list_pdfs: Callable[..., Any],
    clawhub_search_skills: Callable[..., Any],
    clawhub_install_skill: Callable[..., Any],
    set_chat_model: Callable[[str, str], Any],
    rename_file: Callable[..., Any],
    save_citation: Callable[..., Any],
) -> str:
    if not command:
        return ""
    name = str(command.get("name") or "")
    args = dict(command.get("args") or {})
    payload: Dict[str, Any]

    async def _run(func: Callable, *a, **kw):
        res = func(*a, **kw)
        if asyncio.iscoroutine(res):
            return await res
        return res

    if name == "open_video":
        payload = await _run(open_video, str(args.get("path") or ""))
        if payload.get("ok"):
            return f"Opened video in Annolid: {payload.get('path')}"
        return str(payload.get("error") or "Failed to open video.")

    if name == "open_url":
        payload = await _run(open_url, str(args.get("url") or ""))
        if payload.get("ok"):
            resolved = str(payload.get("url") or "").strip()
            if resolved:
                return f"Opened URL in Annolid: {resolved}"
            return "Opened URL in Annolid."
        return str(payload.get("error") or "Failed to open URL.")

    if name == "open_in_browser":
        payload = await _run(open_in_browser, str(args.get("url") or ""))
        if payload.get("ok"):
            resolved = str(payload.get("url") or "").strip()
            if resolved:
                return f"Opened URL in browser: {resolved}"
            return "Opened URL in browser."
        return str(payload.get("error") or "Failed to open URL in browser.")

    if name == "open_pdf":
        payload = await _run(open_pdf, str(args.get("path") or ""))
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
        payload = await _run(set_frame, int(args.get("frame_index") or 0))
        if payload.get("ok"):
            return f"Moved to frame {payload.get('frame_index')}."
        return str(payload.get("error") or "Failed to set frame.")

    if name == "track_next_frames":
        payload = await _run(track_next_frames, int(args.get("to_frame") or 0))
        if payload.get("ok"):
            return f"Started tracking to frame {payload.get('to_frame')}."
        return str(payload.get("error") or "Failed to start tracking.")

    if name == "segment_track_video":
        payload = await _run(
            segment_track_video,
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
        payload = await _run(
            label_behavior_segments,
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
        payload = await _run(
            start_realtime_stream,
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
        payload = await _run(stop_realtime_stream)
        if payload.get("ok"):
            return "Stopped realtime stream."
        return str(payload.get("error") or "Failed to stop realtime stream.")

    if name == "list_pdfs":
        payload = await _run(list_pdfs, query=args.get("query"))
        if payload.get("ok"):
            files = payload.get("files", [])
            if not files:
                return "No local PDF files found."
            lines = [f"Found {payload.get('count')} PDF(s):"]
            lines.extend(f"- {f}" for f in files)
            if payload.get("truncated"):
                lines.append("... (showing top results)")
            return "\n".join(lines)
        return str(payload.get("error") or "Failed to list PDF files.")

    if name == "clawhub_search_skills":
        payload = await _run(
            clawhub_search_skills,
            query=str(args.get("query") or ""),
            limit=int(args.get("limit") or 5),
        )
        if payload.get("ok"):
            output = str(payload.get("stdout") or "").strip()
            if output:
                return output
            results = payload.get("results")
            if isinstance(results, list) and results:
                query_text = str(payload.get("query") or "").strip()
                lines = [
                    (
                        f"ClawHub skills for '{query_text}':"
                        if query_text
                        else "ClawHub skills:"
                    )
                ]
                for item in results:
                    if not isinstance(item, dict):
                        continue
                    slug = str(item.get("slug") or "").strip()
                    name_text = str(item.get("name") or "").strip()
                    desc = str(item.get("description") or "").strip()
                    label = slug or name_text or "unknown-skill"
                    title = (
                        f"{label} - {name_text}"
                        if slug and name_text and slug != name_text
                        else label
                    )
                    lines.append(f"- {title}")
                    if desc:
                        lines.append(f"  {desc}")
                if len(lines) > 1:
                    return "\n".join(lines)
                return "ClawHub search completed but returned no parsable items."
            count = payload.get("count")
            if isinstance(count, int) and count == 0:
                query_text = str(payload.get("query") or "").strip()
                if query_text:
                    return f"No ClawHub skills found for '{query_text}'."
                return "No ClawHub skills found."
            return "ClawHub search completed."
        return str(payload.get("error") or "Failed to search ClawHub skills.")

    if name == "clawhub_install_skill":
        payload = await _run(
            clawhub_install_skill,
            slug=str(args.get("slug") or ""),
        )
        if payload.get("ok"):
            slug = str(payload.get("slug") or "").strip()
            if slug:
                return (
                    f"Installed skill '{slug}' from ClawHub. "
                    "Start a new Annolid Bot session to load it."
                )
            return "Installed ClawHub skill. Start a new session to load it."
        return str(payload.get("error") or "Failed to install ClawHub skill.")

    if name == "set_chat_model":
        payload = await _run(
            set_chat_model,
            str(args.get("provider") or ""),
            str(args.get("model") or ""),
        )
        if payload.get("ok"):
            return (
                f"Updated chat model to {payload.get('provider')}/"
                f"{payload.get('model')}."
            )
        return str(payload.get("error") or "Failed to update chat model.")

    if name == "rename_file":
        payload = await _run(
            rename_file,
            source_path=str(args.get("source_path") or ""),
            new_name=str(args.get("new_name") or ""),
            new_path=str(args.get("new_path") or ""),
            use_active_file=bool(args.get("use_active_file", False)),
            overwrite=bool(args.get("overwrite", False)),
        )
        if payload.get("ok"):
            old_path = str(payload.get("old_path") or "").strip()
            new_path = str(payload.get("new_path") or "").strip()
            if old_path and new_path:
                return f"Renamed file: {old_path} -> {new_path}"
            if new_path:
                return f"Renamed file to: {new_path}"
            return "Renamed file."
        return str(payload.get("error") or "Failed to rename file.")

    if name == "save_citation":
        payload = await _run(
            save_citation,
            key=str(args.get("key") or ""),
            bib_file=str(args.get("bib_file") or ""),
            source=str(args.get("source") or "auto"),
            validate_before_save=bool(args.get("validate_before_save", True)),
            strict_validation=bool(args.get("strict_validation", False)),
        )
        if payload.get("ok"):
            key = str(payload.get("key") or "").strip()
            bib_path = str(payload.get("bib_file") or "").strip()
            source_used = str(payload.get("source") or "auto").strip()
            action = "Created" if bool(payload.get("created")) else "Updated"
            validation = dict(payload.get("validation") or {})
            hint = ""
            if bool(validation.get("checked")):
                provider = str(validation.get("provider") or "").strip()
                verified = bool(validation.get("verified"))
                score = float(validation.get("score") or 0.0)
                state = "verified" if verified else "unverified"
                label = f"{provider} {state}" if provider else state
                hint = f" Validation: {label} ({score:.2f})."
            if key and bib_path:
                return (
                    f"{action} citation '{key}' in {bib_path} "
                    f"(source: {source_used}).{hint}"
                )
            if key:
                return f"{action} citation '{key}'.{hint}"
            return f"Saved citation.{hint}"
        return str(payload.get("error") or "Failed to save citation.")

    return ""

    return ""
