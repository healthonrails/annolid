---
name: tmux
description: "Create, manage, and interact with tmux sessions for long-running processes and terminal multiplexing. Use when the user asks to run background jobs, manage terminal sessions, monitor long-running commands, split panes, or check on running processes."
metadata: '{"annolid":{"requires":{"bins":["tmux"]}}}'
---

# tmux Session Management

Use this skill when the task involves long-running terminal jobs or session multiplexing.

## Workflow

1. **Discover** existing sessions before creating new ones:
   ```bash
   tmux list-sessions
   # Or use the bundled helper:
   # scripts/find-sessions.sh
   ```
2. **Create** a named session for the task:
   ```bash
   tmux new-session -d -s <task-name>
   ```
3. **Send** commands to a target session/window:
   ```bash
   tmux send-keys -t <session>:<window> '<command>' Enter
   ```
4. **Capture** pane output for status checks:
   ```bash
   tmux capture-pane -t <session> -p | tail -20
   ```
5. **Poll** for expected output using the bundled helper:
   ```bash
   # scripts/wait-for-text.sh <session> <expected-token> <timeout-seconds>
   ```
6. **Kill** a session only when the user explicitly requests it:
   ```bash
   tmux kill-session -t <session>
   ```

## Bundled Scripts

- `scripts/find-sessions.sh` — list all active tmux sessions with window counts
- `scripts/wait-for-text.sh` — poll a pane for a specific token (useful for waiting on build/test completion)

## Safety

- Always list sessions before sending commands to verify the target exists.
- Send non-destructive read commands first; confirm before running mutations.
- Keep session names explicit and task-specific to avoid collisions.
