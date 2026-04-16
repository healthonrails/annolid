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
- `memory_set(key?, value?, note?, mode?)`
  - `mode`: `replace` (default, upsert for `key`+`value`) or `append` (always append a new line)

## Execution

- `exec(command, working_dir?)`
- `exec_start(command, working_dir?, background?, timeout_s?, pty?)`
- `exec_process(action, session_id?, wait_ms?, tail_lines?, text?, submit?)`
- `annolid_run(command?, argv?, working_dir?, allow_mutation?)`

## Box

Use the `box` tool for Cornell Box and Box cloud storage tasks. Do not fall back to `list_dir`
or common mount-point guessing when the user asks about Box folders.

- `box(action, folder_id?, file_id?, query?, destination_path?, file_path?, file_name?, limit?, offset?, item_type?, fields?, overwrite?)`
- Common actions:
  - `list_folder_items`
  - `search`
  - `get_file_info`
  - `download_file`
  - `upload_file`

Examples:

- "List the folders in Cornell Box for this project."
- "Search Cornell Box for `session_042_annotations.json`."
- "Show the latest files in the Box project folder."

Safety notes:

- Prefer non-destructive commands.
- Keep command scope limited to relevant directories.
- Treat external downloads and scripts as untrusted until verified.
- For multi-step shell workflows, prefer `exec_start` + `exec_process` instead of chaining large one-shot shell commands.
- Prefer `annolid_run` over shelling out when the task maps to `annolid-run` CLI functionality. Mutating subcommands require `allow_mutation=true`.

## Git and GitHub

- `git_status(repo_path?, short?)`
- `git_cli(args, repo_path?, allow_mutation?)`
- `git_diff(repo_path?, cached?, target?, name_only?)`
- `git_log(repo_path?, max_count?, oneline?)`
- `github_pr_status(repo_path?)`
- `gh_cli(args, repo_path?, allow_mutation?)`
- `github_pr_checks(repo_path?)`

## Web

- `web_search(query, count?)` (Scrapling-first backend for extraction-oriented web results; falls back to Brave Search API when configured)
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
- `sam3_agent_video_track(video_path, agent_prompt, window_size?, stride?, output_dir?, summary_filename?, checkpoint_path?, propagation_direction?, device?, agent_det_thresh?, score_threshold_detection?, new_det_thresh?, max_num_objects?, multiplex_count?, compile_model?, offload_video_to_cpu?, use_explicit_window_reseed?, boundary_mask_match_iou_threshold?, allow_private_state_mutation?, max_generations?, debug?, dry_run?)`
  - Runs SAM3 Agent on each window's key frame, then propagates masks through the window with overlap-aware session carry-over.
  - Writes a JSON run summary to `output_dir` when provided, or to a default `<video_stem>_sam3_agent/` workspace folder.
  - Reuses the current Annolid bot provider/model for the SAM3 VLM seed call when invoked from the GUI.

## GUI (Annolid Bot)

- `gui_context()`
- `gui_shared_image_path()`
- `gui_open_video(path)`
- `gui_open_url(url)`
- `gui_open_in_browser(url)`
- `gui_web_get_dom_text(max_chars?)`
- `gui_web_capture_screenshot(max_width?)`
- `gui_web_describe_view(max_width?)`
- `gui_web_extract_structured(fields?, regex_overrides?, selector_hints?, extraction_mode?, max_chars?, include_excerpt?)`
- `gui_web_click(selector)`
- `gui_web_type(selector, text, submit?)`
- `gui_web_scroll(delta_y?)`
- `gui_web_find_forms()`
- `gui_web_run_steps(steps, stop_on_error?, max_steps?)`
  Supported web actions include: `open_url`, `open_in_browser`, `get_text`, `click`, `type`, `scroll`, `find_forms`, `capture_screenshot`, `describe_view`, `wait`.
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
  - Uses the legacy GroundingDINO+SAM workflow for non-SAM3 models.
  - When `model_name` is `SAM3` and `mode="track"`, the GUI backend routes to `sam3_agent_video_track` instead.
- `gui_label_behavior_segments(path?, behavior_labels?, use_defined_behavior_list?, segment_mode?, segment_frames?, segment_seconds?, sample_frames_per_segment?, max_segments?, subject?, overwrite_existing?, llm_profile?, llm_provider?, llm_model?, video_description?, instance_count?, experiment_context?, behavior_definitions?, focus_points?)`
  - Saves timeline labels and exports `<video_stem>_timestamps.csv` automatically.
  - Writes a segment-level labeling log to `<video_stem>_behavior_segment_labels.json`.
  - Set `use_defined_behavior_list=true` to force model outputs to your schema/flags behavior list.
  - For fixed-duration windows, set `segment_mode="uniform"` and pass `segment_seconds` (for example `1.0` for 1-second segments).
  - `sample_frames_per_segment` controls how many frames are sampled per segment for VLM label voting.
  - Use `video_description`, `instance_count`, `experiment_context`, `behavior_definitions`, and `focus_points` to guide model labeling behavior for specific experimental protocols.
- `gui_start_realtime_stream(camera_source?, model_name?, target_behaviors?, confidence_threshold?, viewer_type?, classify_eye_blinks?, blink_ear_threshold?, blink_min_consecutive_frames?)`
- `gui_stop_realtime_stream()`

## Communication and Delegation

- `message(content, channel?, chat_id?)`
- `spawn(task, label?)`

## Scheduling

- `cron` actions:
  - `add`, `list`, `remove`, `enable`, `disable`, `run`, `status`

Use cron to schedule full agent workflows, automated web routines, or recurring actions (e.g. sending emails, checking updates).
