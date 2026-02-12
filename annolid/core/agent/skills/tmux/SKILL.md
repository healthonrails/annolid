---
name: tmux
description: Interact with tmux sessions for multi-process workflows.
metadata: '{"annolid":{"requires":{"bins":["tmux"]}}}'
---

Use this skill when the task involves long-running terminal jobs or session multiplexing.

Suggested steps:

1. Discover sessions and windows before attaching.
2. Send non-destructive commands first.
3. Capture pane output for debugging/status updates.
4. Keep session names explicit and task-specific.

Scripts:

- `scripts/find-sessions.sh`: list sessions
- `scripts/wait-for-text.sh`: poll pane output for a token
