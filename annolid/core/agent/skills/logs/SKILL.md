---
name: logs
description: Inspect, list, search, tail, open, and clean Annolid log folders using dedicated GUI log tools.
metadata: '{"annolid":{"always":false,"tags":["logs","log","debug","diagnostics","tail","traceback","errors","runs","realtime"]}}'
---

# Logs Manager

Use these tools to inspect, search, and clean Annolid logs safely without opening unrelated dialogs.

## Tools

- `gui_list_logs`: List known log targets and resolved paths.
- `gui_list_realtime_logs`: List active realtime detection and bot-event log file paths.
- `gui_list_log_files(target, pattern?, limit?, recursive?, sort_by?, descending?)`: Discover files under a target (`sort_by`: `"name"`, `"mtime"`, `"size"`).
- `gui_read_log_file(path, tail_lines?, max_chars?)`: Read tail-friendly content from an allowed log file.
- `gui_search_logs(query, target?, pattern?, case_sensitive?, use_regex?, max_matches?, max_files?)`: Search across log files for error signatures or keywords.
- `gui_open_log_folder(target)`: Open a target folder in the system file browser.
- `gui_remove_log_folder(target)`: Recursively delete a target folder (explicit user request only).

Supported targets: `logs`, `realtime`, `runs`, `label_index`, `app`.

## Recommended Flow

1. Call `gui_list_logs` first to confirm which target directories exist.
2. For debugging or triage:
   - Call `gui_list_log_files(target=..., sort_by="mtime", descending=true)` to locate the newest files.
   - Call `gui_search_logs(query="ERROR|Traceback", target=..., use_regex=true)` to isolate failures.
   - Call `gui_read_log_file(path=..., tail_lines=100)` to inspect context.
3. If the user asks to browse a folder, call `gui_open_log_folder(target=...)`.
4. If the user explicitly asks to clean logs, remove the most specific target (`realtime`, `runs`, `label_index`, or `app`) rather than all of `logs`, and report the result.

## Safety

- Never call `gui_remove_log_folder` unless the user explicitly requested deletion.
