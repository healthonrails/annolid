---
name: Logs Manager
description: Inspect, open, and clean Annolid logs using dedicated GUI log tools.
---

# Logs Manager

Use these tools to manage logs without coupling to dataset dialogs.

## Tools

- `gui_list_logs`: List known log targets and paths.
- `gui_open_log_folder`: Open one target in the system file browser.
- `gui_remove_log_folder`: Remove a target folder recursively.
- `gui_list_log_files`: List files under a target with optional glob pattern, recursion, and sort controls.
- `gui_read_log_file`: Read tail-friendly content from a log file path.
- `gui_search_logs`: Search text across files in a target with optional glob pattern, regex, and case-sensitivity.

Supported targets:

- `logs`
- `realtime`
- `runs`
- `label_index`
- `app`

## Recommended Flow

1. Call `gui_list_logs` first and confirm the target path exists.
2. If user asks to inspect, call `gui_open_log_folder` with an explicit target.
3. For content inspection:
   - call `gui_list_log_files` to identify candidate files (`pattern`, `sort_by`, `descending`)
   - call `gui_read_log_file` for tail output
   - call `gui_search_logs` for keyword triage (`pattern`, `use_regex`, `case_sensitive`)
4. If user asks to clean/remove logs, call `gui_remove_log_folder` with an explicit target and report result.

## Safety

- Do not remove logs unless the user explicitly requested deletion.
- Prefer removing the most specific target (`realtime`, `runs`, `label_index`, or `app`) before deleting `logs`.
