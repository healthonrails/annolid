---
name: GUI Automation
description: Use Annolid GUI tools for robust automation across web, PDF, video, and chat controls.
---

# GUI Automation

Use Annolid GUI tools when tasks depend on current app state (opened video, PDF, or embedded web page).

## Core State

- `gui_context`: Always call this first to understand active mode, paths, frame, and model.
- `gui_shared_image_path`: Check whether image context is currently shared to the bot.
- `gui_set_chat_prompt`, `gui_send_chat_prompt`, `gui_set_chat_model`: Use these for deterministic bot UI control.

## Web Workflows

- `gui_open_url` / `gui_open_in_browser`: Open URL in embedded viewer or system browser.
- `gui_web_run_steps`: Preferred for multi-step web automation.
- `gui_web_get_dom_text`, `gui_web_click`, `gui_web_type`, `gui_web_scroll`, `gui_web_find_forms`: Use directly for targeted actions.

Best practices:
1. Start with `gui_web_run_steps` including `open_url`, `wait`, then `get_text`.
2. Prefer element index selectors from page text (`[12: ...]`) over guessed CSS selectors.
3. Add explicit `wait` after navigation/click on dynamic sites.
4. Re-run `get_text` after each state-changing action before next click/type.

## PDF Workflows

- `gui_open_pdf`: Open local path or URL-resolved PDF in Annolid viewer.
- `gui_pdf_get_state`: Check loaded PDF metadata and page position.
- `gui_pdf_get_text`: Extract current-page-centered content.
- `gui_pdf_find_sections`: Locate headings and page anchors.
- `gui_arxiv_search`: Resolve and open arXiv papers quickly.
- `gui_list_pdfs`: Discover available downloaded PDFs.
- `gui_save_citation`: Persist citation metadata to BibTeX.

Best practices:
1. Call `gui_pdf_get_state` before extraction to confirm a PDF is loaded.
2. Use smaller `max_chars`/`pages` first, then widen if needed.
3. When opening by URL fails, call `gui_list_pdfs` and retry with explicit local path.

## Video and Model Workflows

- `gui_open_video`, `gui_set_frame`, `gui_track_next_frames`
- `gui_set_ai_text_prompt`, `gui_run_ai_text_segmentation`
- `gui_segment_track_video`, `gui_label_behavior_segments`
- `gui_start_realtime_stream`, `gui_stop_realtime_stream`

Best practices:
1. Open video and set frame explicitly before segmentation/tracking.
2. Keep prompts short and object-specific for segmentation.
3. Use `gui_segment_track_video` for integrated open+segment(+track) flows.
4. Stop realtime stream explicitly when finished.

## Reliability Pattern

For robust execution:
1. Read state (`gui_context`, PDF/web state tools).
2. Execute one focused action.
3. Validate result from returned JSON.
4. Retry with a narrower fallback action if needed.

Do not assume the UI state; verify it before every multi-step automation sequence.
