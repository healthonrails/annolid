# Available Tools

This workspace is intended for the Annolid agent tool stack.

## File Tools

- `read_file(path)`
- `extract_pdf_text(path, start_page?, max_pages?, max_chars?)`
- `open_pdf(path, start_page?, max_pages?, max_chars?)`
- `extract_pdf_images(path, output_dir?, start_page?, max_pages?, dpi?, overwrite?)`
- `write_file(path, content)`
- `edit_file(path, old_text, new_text)`
- `list_dir(path)`
- `code_search(query, path?, glob?, regex?, case_sensitive?, max_results?, context_lines?)`
- `code_explain(path, symbol?, include_source?, max_source_lines?)`
- `memory_search(query, top_k?, max_snippet_chars?)`
- `memory_get(path, start_line?, end_line?, max_chars?)`
- `memory_set(key?, value?, note?)`

## Execution

- `exec(command, working_dir?)`

Safety notes:

- Prefer non-destructive commands.
- Keep command scope limited to relevant directories.
- Treat external downloads and scripts as untrusted until verified.

## Git and GitHub

- `git_status(repo_path?, short?)`
- `git_diff(repo_path?, cached?, target?, name_only?)`
- `git_log(repo_path?, max_count?, oneline?)`
- `github_pr_status(repo_path?)`
- `github_pr_checks(repo_path?)`

## Web

- `web_search(query, count?)`
- `web_fetch(url, extractMode?, maxChars?)`
- `download_url(url, output_path, max_bytes?, overwrite?, content_type_prefixes?, request_headers?)`
- `download_pdf(url, output_path?, max_bytes?, overwrite?)`
- `clawhub_search_skills(query, limit?)`
- `clawhub_install_skill(slug)`

## Video

- `video_info(path)`
- `video_sample_frames(path, output_dir?, mode?, step?, target_fps?, indices?, start_frame?, max_frames?, overwrite?)`
- `video_segment(path, output_path?, start_frame?, end_frame?, start_sec?, end_sec?, overwrite?)`
- `video_process_segments(path, segments, output_dir?, overwrite?)`

## GUI (Annolid Bot)

- `gui_context()`
- `gui_shared_image_path()`
- `gui_open_video(path)`
- `gui_open_url(url)`
- `gui_open_in_browser(url)`
- `gui_web_get_dom_text(max_chars?)`
- `gui_web_click(selector)`
- `gui_web_type(selector, text, submit?)`
- `gui_web_scroll(delta_y?)`
- `gui_web_find_forms()`
- `gui_web_run_steps(steps, stop_on_error?, max_steps?)`
- `gui_open_pdf(path?)`
- `gui_pdf_get_state()`
- `gui_pdf_get_text(max_chars?, pages?)`
- `gui_pdf_find_sections(max_sections?, max_pages?)`
- `gui_set_frame(frame_index)`
- `gui_set_chat_prompt(text)`
- `gui_send_chat_prompt()`
- `gui_set_chat_model(provider, model)`
- `gui_select_annotation_model(model_name)`
- `gui_track_next_frames(to_frame)`
- `gui_set_ai_text_prompt(text, use_countgd?)`
- `gui_run_ai_text_segmentation()`
- `gui_segment_track_video(path, text_prompt, mode?, use_countgd?, model_name?, to_frame?)`
- `gui_label_behavior_segments(path?, behavior_labels?, segment_mode?, segment_frames?, max_segments?, subject?, overwrite_existing?, llm_profile?, llm_provider?, llm_model?)`
  - Saves timeline labels and exports `<video_stem>_timestamps.csv` automatically.
- `gui_start_realtime_stream(camera_source?, model_name?, target_behaviors?, confidence_threshold?, viewer_type?, classify_eye_blinks?, blink_ear_threshold?, blink_min_consecutive_frames?)`
- `gui_stop_realtime_stream()`

## Communication and Delegation

- `message(content, channel?, chat_id?)`
- `spawn(task, label?)`

## Scheduling

- `cron` actions:
  - `add`, `list`, `remove`, `enable`, `disable`, `run`, `status`

Use cron to schedule full agent workflows, automated web routines, or recurring actions (e.g. sending emails, checking updates).
