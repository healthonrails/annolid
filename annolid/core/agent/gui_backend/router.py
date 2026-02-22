import asyncio
import ipaddress
from urllib.parse import urlsplit, urlunsplit
from typing import Any, Callable, Dict


def _safe_source_for_display(source: str) -> str:
    text = str(source or "").strip()
    if not text or text.isdigit() or "://" not in text:
        return text
    try:
        parts = urlsplit(text)
    except Exception:
        return text
    host = str(parts.hostname or "").strip()
    if not host:
        return text
    redact = host.lower() == "localhost"
    if not redact:
        try:
            ip_obj = ipaddress.ip_address(host)
            redact = bool(
                ip_obj.is_private
                or ip_obj.is_loopback
                or ip_obj.is_link_local
                or ip_obj.is_multicast
            )
        except Exception:
            redact = False
    safe_netloc = parts.netloc
    if "@" in safe_netloc:
        safe_netloc = safe_netloc.split("@", 1)[1]
    if not redact:
        return urlunsplit(
            (parts.scheme, safe_netloc, parts.path, parts.query, parts.fragment)
        )
    port = f":{parts.port}" if parts.port else ""
    replacement = f"<private-host>{port}"
    if parts.scheme.lower() in {"rtp", "udp"}:
        replacement = f"@<private-host>{port}"
    return urlunsplit(
        (parts.scheme, replacement, parts.path, parts.query, parts.fragment)
    )


async def execute_direct_gui_command(
    command: Dict[str, Any],
    *,
    open_video: Callable[[str], Any],
    open_url: Callable[[str], Any],
    open_in_browser: Callable[[str], Any],
    open_threejs: Callable[[str], Any],
    open_threejs_example: Callable[[str], Any],
    open_pdf: Callable[[str], Any],
    set_frame: Callable[[int], Any],
    track_next_frames: Callable[[int], Any],
    segment_track_video: Callable[..., Any],
    label_behavior_segments: Callable[..., Any],
    start_realtime_stream: Callable[..., Any],
    stop_realtime_stream: Callable[[], Any],
    get_realtime_status: Callable[[], Any],
    list_realtime_models: Callable[[], Any],
    list_realtime_logs: Callable[[], Any],
    check_stream_source: Callable[..., Any],
    list_pdfs: Callable[..., Any],
    clawhub_search_skills: Callable[..., Any],
    clawhub_install_skill: Callable[..., Any],
    set_chat_model: Callable[[str, str], Any],
    rename_file: Callable[..., Any],
    list_citations: Callable[..., Any],
    add_citation_raw: Callable[..., Any],
    save_citation: Callable[..., Any],
    generate_annolid_tutorial: Callable[..., Any],
    automation_schedule: Callable[..., Any],
    list_dir: Callable[..., Any],
    read_file: Callable[..., Any],
    exec_command: Callable[..., Any],
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

    if name == "open_threejs":
        payload = await _run(open_threejs, str(args.get("path_or_url") or ""))
        if payload.get("ok"):
            target = str(payload.get("target") or "").strip()
            if target:
                return f"Opened Three.js view: {target}"
            return "Opened Three.js view."
        return str(payload.get("error") or "Failed to open Three.js view.")

    if name == "open_threejs_example":
        payload = await _run(open_threejs_example, str(args.get("example_id") or ""))
        if payload.get("ok"):
            example_id = str(payload.get("example_id") or "").strip()
            if example_id:
                return f"Opened Three.js example: {example_id}"
            return "Opened Three.js example."
        return str(payload.get("error") or "Failed to open Three.js example.")

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
            rtsp_transport=str(args.get("rtsp_transport") or ""),
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

    if name == "get_realtime_status":
        payload = await _run(get_realtime_status)
        if payload.get("ok"):
            running = bool(payload.get("running", False))
            model_name = str(payload.get("model_name") or "").strip()
            camera_source = _safe_source_for_display(
                str(payload.get("camera_source") or "").strip()
            )
            viewer_type = str(payload.get("viewer_type") or "").strip()
            state = "running" if running else "stopped"
            parts = [f"Realtime status: {state}."]
            if model_name:
                parts.append(f"Model: {model_name}.")
            if camera_source:
                parts.append(f"Source: {camera_source}.")
            if viewer_type:
                parts.append(f"Viewer: {viewer_type}.")
            return " ".join(parts)
        return str(payload.get("error") or "Failed to get realtime status.")

    if name == "list_realtime_models":
        payload = await _run(list_realtime_models)
        if payload.get("ok"):
            models = payload.get("models", [])
            if not isinstance(models, list) or not models:
                return "No realtime models found."
            lines = [f"Available realtime models ({len(models)}):"]
            for item in models:
                if not isinstance(item, dict):
                    continue
                display_name = str(item.get("display_name") or "").strip()
                weight = str(item.get("weight_file") or "").strip()
                if display_name and weight:
                    lines.append(f"- {display_name} ({weight})")
                elif display_name:
                    lines.append(f"- {display_name}")
                elif weight:
                    lines.append(f"- {weight}")
            return "\n".join(lines)
        return str(payload.get("error") or "Failed to list realtime models.")

    if name == "list_realtime_logs":
        payload = await _run(list_realtime_logs)
        if payload.get("ok"):
            detections = str(payload.get("detections_log_path") or "").strip()
            bot_events = str(payload.get("bot_event_log_path") or "").strip()
            if not detections and not bot_events:
                return "No realtime log files are currently available."
            lines = ["Realtime log files:"]
            if detections:
                lines.append(f"- detections: {detections}")
            if bot_events:
                lines.append(f"- bot-events: {bot_events}")
            return "\n".join(lines)
        return str(payload.get("error") or "Failed to list realtime logs.")

    if name == "check_stream_source":
        payload = await _run(
            check_stream_source,
            camera_source=str(args.get("camera_source") or ""),
            rtsp_transport=str(args.get("rtsp_transport") or "auto"),
            timeout_sec=float(args.get("timeout_sec") or 3.0),
            probe_frames=int(args.get("probe_frames") or 3),
            save_snapshot=bool(args.get("save_snapshot", False)),
            email_to=str(args.get("email_to") or ""),
            email_subject=str(args.get("email_subject") or ""),
            email_content=str(args.get("email_content") or ""),
        )
        if payload.get("ok"):
            source = str(payload.get("camera_source") or "").strip()
            source = _safe_source_for_display(source)
            size = ""
            width = int(payload.get("frame_width") or 0)
            height = int(payload.get("frame_height") or 0)
            if width > 0 and height > 0:
                size = f" {width}x{height}"
            message = f"Stream probe succeeded for {source or 'source'}.{size}".strip()
            snapshot_path = str(payload.get("snapshot_path") or "").strip()
            if snapshot_path:
                message += f" Snapshot saved: {snapshot_path}"
            if payload.get("snapshot_opened_on_canvas"):
                message += " Snapshot opened on canvas."
            if str(payload.get("email_to") or "").strip():
                if payload.get("email_sent"):
                    message += " Email sent successfully."
                else:
                    email_result = str(payload.get("email_result") or "").strip()
                    if email_result:
                        message += f" Email send failed: {email_result}"
                    else:
                        message += " Email send failed."
            return message
        return str(payload.get("error") or "Stream probe failed.")

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

    if name == "list_citations":
        payload = await _run(
            list_citations,
            bib_file=str(args.get("bib_file") or ""),
            query=str(args.get("query") or ""),
        )
        if payload.get("ok"):
            items = list(payload.get("entries") or [])
            if not items:
                return "No citations found."
            count = int(payload.get("count") or len(items))
            lines = [f"Found {count} citation(s):"]
            for entry in items[:20]:
                if not isinstance(entry, dict):
                    continue
                key = str(entry.get("key") or "").strip()
                title = str(entry.get("title") or "").strip()
                year = str(entry.get("year") or "").strip()
                label = key or "(no-key)"
                detail = f"{title} ({year})" if title and year else (title or year)
                lines.append(f"- {label}: {detail}".rstrip(": "))
            if count > len(items):
                lines.append("... (showing top results)")
            return "\n".join(lines)
        return str(payload.get("error") or "Failed to list citations.")

    if name == "add_citation_raw":
        payload = await _run(
            add_citation_raw,
            bibtex=str(args.get("bibtex") or ""),
            bib_file=str(args.get("bib_file") or ""),
        )
        if payload.get("ok"):
            key = str(payload.get("key") or "").strip()
            keys = [
                str(k).strip()
                for k in list(payload.get("keys") or [])
                if str(k).strip()
            ]
            bib_path = str(payload.get("bib_file") or "").strip()
            created_count = int(payload.get("created_count") or 0)
            updated_count = int(payload.get("updated_count") or 0)
            action = "Created" if bool(payload.get("created")) else "Updated"
            if keys and len(keys) > 1:
                if bib_path:
                    return (
                        f"Saved {len(keys)} citations in {bib_path} from provided BibTeX "
                        f"({created_count} created, {updated_count} updated)."
                    )
                return (
                    f"Saved {len(keys)} citations from provided BibTeX "
                    f"({created_count} created, {updated_count} updated)."
                )
            if key and bib_path:
                return f"{action} citation '{key}' in {bib_path} from provided BibTeX."
            if key:
                return f"{action} citation '{key}' from provided BibTeX."
            return "Saved citation from provided BibTeX."
        return str(payload.get("error") or "Failed to add citation from BibTeX.")

    if name == "generate_annolid_tutorial":
        payload = await _run(
            generate_annolid_tutorial,
            topic=str(args.get("topic") or ""),
            level=str(args.get("level") or "intermediate"),
            save_to_file=bool(args.get("save_to_file", False)),
            include_code_refs=bool(args.get("include_code_refs", True)),
            open_in_web_viewer=bool(args.get("open_in_web_viewer", True)),
        )
        if payload.get("ok"):
            tutorial_text = str(payload.get("tutorial") or "").strip()
            output_path = str(payload.get("output_path") or "").strip()
            generated_with_model = bool(payload.get("generated_with_model", False))
            model_error = str(payload.get("model_error") or "").strip()
            if output_path:
                note = ""
                if bool(payload.get("opened_in_web_viewer")):
                    note = " Opened in web viewer."
                else:
                    open_error = str(payload.get("open_viewer_error") or "").strip()
                    if open_error:
                        note = f" Could not open in web viewer: {open_error}"
                if not generated_with_model and model_error:
                    note += f" Tutorial fallback used: {model_error}"
                return (
                    f"Tutorial created and saved: {output_path}.{note}\n\n"
                    f"{tutorial_text}"
                )
            if not generated_with_model and model_error:
                return f"{tutorial_text}\n\n(Fallback used: {model_error})".strip()
            return tutorial_text or "Generated Annolid tutorial."
        return str(payload.get("error") or "Failed to generate Annolid tutorial.")

    if name == "automation_schedule":
        payload = await _run(
            automation_schedule,
            action=str(args.get("action") or ""),
            task_id=str(args.get("task_id") or ""),
            name=str(args.get("name") or ""),
            task_type=str(args.get("task_type") or ""),
            every_seconds=float(args.get("every_seconds") or 0),
            camera_source=str(args.get("camera_source") or ""),
            email_to=str(args.get("email_to") or ""),
            notes=str(args.get("notes") or ""),
            run_immediately=bool(args.get("run_immediately", True)),
            max_runs=(
                int(args.get("max_runs"))
                if args.get("max_runs") not in (None, "")
                else None
            ),
        )
        if payload.get("ok"):
            result = str(payload.get("result") or "").strip()
            return result or "Automation command completed."
        return str(payload.get("error") or "Failed to execute automation command.")

    if name == "list_dir":
        payload = await _run(list_dir, str(args.get("path") or ""))
        if payload.get("ok"):
            res = str(payload.get("result") or "").strip()
            return f"Directory contents:\n{res}" if res else "Directory is empty."
        return str(payload.get("error") or "Failed to list directory.")

    if name == "read_file":
        payload = await _run(read_file, str(args.get("path") or ""))
        if payload.get("ok"):
            res = str(payload.get("result") or "").strip()
            # Truncating directly in the router (though the tool might already do it)
            if len(res) > 2000:
                res = res[:2000] + "\n...[truncated]"
            return f"File contents:\n{res}" if res else "File is empty."
        return str(payload.get("error") or "Failed to read file.")

    if name == "exec_command":
        payload = await _run(exec_command, str(args.get("command") or ""))
        if payload.get("ok"):
            res = str(payload.get("result") or "").strip()
            return (
                f"Command output:\n{res}" if res else "Command executed with no output."
            )
        return str(payload.get("error") or "Failed to execute command.")

    return ""
