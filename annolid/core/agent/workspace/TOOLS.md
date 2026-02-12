# Available Tools

This workspace is intended for the Annolid agent tool stack.

## File Tools

- `read_file(path)`
- `extract_pdf_text(path, start_page?, max_pages?, max_chars?)`
- `extract_pdf_images(path, output_dir?, start_page?, max_pages?, dpi?, overwrite?)`
- `write_file(path, content)`
- `edit_file(path, old_text, new_text)`
- `list_dir(path)`
- `memory_search(query, top_k?, max_snippet_chars?)`
- `memory_get(path, start_line?, end_line?, max_chars?)`
- `memory_set(key?, value?, note?)`

## Execution

- `exec(command, working_dir?)`

Safety notes:

- Prefer non-destructive commands.
- Keep command scope limited to relevant directories.
- Treat external downloads and scripts as untrusted until verified.

## Web

- `web_search(query, count?)`
- `web_fetch(url, extractMode?, maxChars?)`
- `download_url(url, output_path, max_bytes?, overwrite?, content_type_prefixes?)`

## Video

- `video_info(path)`
- `video_sample_frames(path, output_dir?, mode?, step?, target_fps?, indices?, start_frame?, max_frames?, overwrite?)`
- `video_segment(path, output_path?, start_frame?, end_frame?, start_sec?, end_sec?, overwrite?)`
- `video_process_segments(path, segments, output_dir?, overwrite?)`

## GUI (Annolid Bot)

- `gui_context()`
- `gui_shared_image_path()`
- `gui_open_video(path)`
- `gui_set_frame(frame_index)`
- `gui_set_chat_prompt(text)`
- `gui_send_chat_prompt()`
- `gui_set_chat_model(provider, model)`
- `gui_select_annotation_model(model_name)`
- `gui_track_next_frames(to_frame)`
- `gui_set_ai_text_prompt(text, use_countgd?)`
- `gui_run_ai_text_segmentation()`
- `gui_segment_track_video(path, text_prompt, mode?, use_countgd?, model_name?, to_frame?)`

## Communication and Delegation

- `message(content, channel?, chat_id?)`
- `spawn(task, label?)`

## Scheduling

- `cron` actions:
  - `add`, `list`, `remove`, `enable`, `disable`, `run`, `status`

Use cron for periodic reminders and routine checks.
