---
name: multi-agent-orchestration
description: Delegate background tasks, spawn specialized behavior subagents, run multi-agent consensus swarms, or drive long-lived coding sessions cleanly.
metadata: '{"annolid":{"always":false,"tags":["subagent","spawn","swarm","multi-agent","delegate","orchestration","roundtable","consensus","coding session","codex","background task","behavior subagent"]}}'
---

# Multi-Agent Orchestration & Delegation

Use this skill when a task benefits from parallel delegation, specialized behavior subagent profiles, multi-perspective consensus swarms, or a dedicated persistent coding session.

## Choose the Right Orchestration Tool

### 1. Specialized Behavior Subagents (`spawn_behavior_subagent`)
Use `spawn_behavior_subagent(task, profile, label?, skill_names?, wait?)` for focused behavior-analysis roles (each loads a curated skill bundle automatically):

- `behavior_assay_inference` (`vision-language-analysis`, `behavior-assay-taxonomy`)
- `behavior_feature_planning` (`behavior-feature-selection`, `experimental-design`)
- `behavior_perception_routing` (`perception-routing`, `model-policy`)
- `behavior_segmentation` (`behavior-segmentation`, `timeline-reasoning`)
- `behavior_analysis_coding` (`sandboxed-analysis`, `metrics-derivation`)
- `behavior_polygon_classifier` (`polygon-classifier-training`, `behavior-evaluation`, `scientific-reporting`, `provenance`)
- `behavior_reporting` (`scientific-reporting`, `provenance`)

### 2. General Subagents & Task Control (`spawn`, `list_tasks`, `cancel_task`)
- `spawn(task, label?, runtime?, provider?, model?, workspace?, profile?, skill_names?, wait?)`: run an isolated subagent task (background by default unless `wait=true`).
- `list_tasks()`: inspect running or recently completed subagent tasks.
- `cancel_task(task_id)`: cancel a runaway or superseded task.

### 3. Multi-Perspective Swarm / Roundtable (`run_swarm`)
- `run_swarm(task, max_turns?, agents?)`: run a bounded collaborative discussion across multiple personas/perspectives (for brainstorming experimental designs, architecture tradeoffs, or peer critique).

### 4. Persistent Coding Sessions (`coding_session_*`)
Use when an external ACP coding harness (e.g., Codex CLI) is configured for multi-turn code editing and verification:
- `coding_session_start(task, label?, provider?, model?, workspace?, sandbox?)`
- `coding_session_send(session_id, message)`
- `coding_session_poll(session_id, tail_messages?)`
- `coding_session_list()`
- `coding_session_abort(session_id)` / `coding_session_close(session_id)`

## Delegation Best Practices

1. **Give Self-Contained Prompts**: Include exact file paths, workspace context, acceptance criteria, and expected artifact outputs — subagents do not automatically see implicit assumptions.
2. **Avoid Unnecessary Spawning**: Handle simple 1–2 tool lookups directly; delegate when parallelism, specialized skill isolation, or long-running background execution adds clear value.
3. **Clean Up & Verify**: Close completed coding sessions, cancel obsolete tasks, and always verify subagent-produced paths and metrics before reporting them to the user.
