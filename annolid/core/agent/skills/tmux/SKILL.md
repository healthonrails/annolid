---
name: tmux
description: Interact with tmux sessions for persistent multi-process terminal workflows and interactive CLIs.
metadata: '{"annolid":{"always":false,"tags":["tmux","terminal","multiplexer","pane","background process","interactive shell","long running"],"requires":{"bins":["tmux"]}}}'
---

# tmux Session Control

Use this skill when the task involves persistent interactive terminal sessions or multi-pane terminal workflows that must outlive a single command.

## Prefer Simpler Tools First

- For ordinary background commands (training, evaluation, scripts) where `exec_start` and `exec_process` are available, prefer `exec_start(..., background=true)` + `exec_process(action="poll"|"log"|"kill")` (or GUI `/session list|poll|logs`) before launching tmux.
- Use `tmux` when you need named persistent sessions, interactive TUIs, or multi-window multiplexing.

## Suggested Steps

1. Discover existing sessions and windows before creating or attaching (`scripts/find-sessions.sh`).
2. Keep session names explicit and task-specific (avoid colliding with user sessions).
3. Send non-destructive inspection commands first.
4. Capture pane output or poll for a completion token (`scripts/wait-for-text.sh`) for status updates and debugging.

## Bundled Scripts

- `scripts/find-sessions.sh`: list active tmux sessions across sockets
- `scripts/wait-for-text.sh`: poll a tmux pane until expected output appears or timeout expires
