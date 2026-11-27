---
name: gui
description: Automate the live Annolid desktop GUI across canvas state, video tracking, shapes, PDFs, embedded web views, realtime streams, logs, and tutorials.
metadata: '{"annolid":{"always":false,"tags":["gui","ui","canvas","desktop","pdf","video","web","viewer","tutorial","chat"]}}'
---

# GUI Automation

Use Annolid GUI tools when a task depends on live desktop app state (currently opened video, annotations, PDF, embedded web view, 3D view, or chat panel).

## Core App & Chat State

- `gui_context`: Call first to inspect active mode, opened video/PDF/URL, current frame, and selected model.
- `gui_shared_image_path`: Check whether an image is currently shared to Annolid Bot.
- `gui_select_annotation_model`: Switch the main annotation/tracking model dropdown.
- `gui_set_chat_prompt`, `gui_send_chat_prompt`, `gui_set_chat_model`: Deterministic chat controls.
- `gui_generate_annolid_tutorial`: Build a grounded tutorial from local code/docs.
- `gui_self_update`: Check for Annolid updates (`execute=false` by default; `execute=true` only with explicit operator consent).

## Embedded Web Workflows

- `gui_open_url`, `gui_open_in_browser`, `gui_web_save_current`
- `gui_web_run_steps`: Preferred for multi-step web sequences (`open_url`, `wait`, `get_text`, `click`, `type`, `scroll`, `find_forms`, `capture_screenshot`, `describe_view`).
- Targeted actions: `gui_web_get_dom_text`, `gui_web_extract_structured`, `gui_web_click`, `gui_web_type`, `gui_web_scroll`, `gui_web_find_forms`, `gui_web_capture_screenshot`, `gui_web_describe_view`.
- Prefer indexed element selectors (`[12: ...]`) from `get_text` over guessed CSS selectors, and re-read text after state-changing actions.

## PDF & Citation Workflows

- `gui_open_pdf`, `gui_list_pdfs`, `gui_arxiv_search`
- `gui_pdf_get_state`, `gui_pdf_find_sections`, `gui_pdf_get_text`, `gui_pdf_summarize` (preferred for cached full-document summaries)
- `gui_save_citation` (supports `validate_before_save`, `verify_after_save`), `gui_verify_citations`
- Confirm a PDF is loaded via `gui_pdf_get_state` before extracting text; start with small `max_chars`/`pages`.

## Video, Tracking, Shapes & Behavior Workflows

- Video & tracking: `gui_open_video`, `gui_set_frame`, `gui_track_next_frames`, `gui_set_ai_text_prompt`, `gui_run_ai_text_segmentation`, `gui_segment_track_video`, `gui_correct_tracking_ndjson`, `gui_analyze_tracking_stats`.
  - Note: when `model_name="SAM3"` and `mode="track"`, `gui_segment_track_video` routes to `sam3_agent_video_track`.
- Shapes: `gui_list_shapes`, `gui_select_shapes`, `gui_set_selected_shape_label`, `gui_delete_selected_shapes` (live canvas) and `gui_list_shapes_in_annotation`, `gui_relabel_shapes_in_annotation`, `gui_delete_shapes_in_annotation` (JSON/NDJSON files).
- Behavior: `gui_behavior_catalog`, `gui_label_behavior_segments`, `gui_process_video_behaviors`, `gui_score_aggression_bouts`.
- Realtime & logs: `gui_list_realtime_models`, `gui_check_stream_source`, `gui_start_realtime_stream`, `gui_get_realtime_status`, `gui_stop_realtime_stream`, `gui_list_realtime_logs`, `gui_list_logs`, `gui_open_log_folder`, `gui_list_log_files`, `gui_read_log_file`, `gui_search_logs`, `gui_remove_log_folder`.

## Reliability Pattern

1. Inspect state first (`gui_context`, `gui_pdf_get_state`, etc.)—never assume UI state.
2. Execute one focused action and verify the returned JSON payload.
3. Retry with a narrower fallback if needed, and stop realtime streams explicitly when finished.
