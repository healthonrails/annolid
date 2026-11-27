---
name: cron
description: Schedule, inspect, pause, and manage recurring or one-shot agent tasks, reminders, scheduled emails, and periodic automations.
metadata: '{"annolid":{"always":false,"tags":["cron","schedule","reminder","email","recurring","timer","automation","job","periodic"]}}'
---

# Scheduling, Reminders & Cron Jobs

Use this skill for one-shot reminders, scheduled emails, recurring cron workloads, and lightweight periodic automations.

## Tool Selection

1. **`cron`** — general persistent scheduler supporting one-shot (`at`), fixed-interval (`every_seconds`), and cron expressions (`cron_expr` + optional IANA `tz`), plus direct scheduled emails (`email_to`, `email_subject`, `email_content`, `attachment_paths`).
2. **`automation_schedule`** — lightweight recurring bot task presets (`task_type`: `"camera_check"`, `"periodic_report"`, `"email_summary"` with `every_seconds`, `camera_source`, `email_to`, `notes`, `run_immediately`, `max_runs`).

## `cron` Actions & Modes

- `action="add"` with one schedule mode:
  - `at`: ISO-8601 datetime for one-shot runs (e.g. `"2026-06-15T09:00:00-04:00"` or UTC `"Z"`).
  - `every_seconds`: integer interval (`> 0`).
  - `cron_expr`: 5-field cron expression + optional IANA `tz` (e.g. `"America/New_York"`).
- Management actions: `list`, `status`, `check` (with `job_id`), `enable`, `disable`, `run`, `remove` (or `cancel`).
- Chat slash shortcuts: `/cron status`, `/cron list`, `/cron check <job_id>`, `/automation status`, `/automation list`.

## Direct Scheduled Email (Preferred Over Agent-Turn Email)

When scheduling an email whose body is already known, pass direct email fields instead of embedding a prompt in `message`:

- `cron(action="add", at="2026-06-15T09:00:00-04:00", message="Send lab report email", email_to="lab@example.org", email_subject="Weekly Report", email_content="Attached is the weekly summary.", attachment_paths=["/workspace/report.pdf"])`

Only rely on an agent-turn `message` (without `email_to`/`email_content`) when the email content must be freshly gathered at trigger time (e.g. checking a website or recent detections first).

## Examples

- **One-shot reminder:** `cron(action="add", message="Remind me to check incubator temperature", at="2026-06-15T14:30:00-04:00", deliver=true)`
- **Recurring autonomous workflow:** `cron(action="add", message="Summarize new tracking stats in the workspace and save summary.md", cron_expr="0 8 * * 1-5", tz="America/New_York")`
- **Periodic camera check:** `automation_schedule(action="add", task_type="camera_check", every_seconds=300, camera_source="0", email_to="lab@example.org")`

## Rules

- Resolve relative dates ("tomorrow at 9am") using the system prompt's current local datetime and timezone.
- Always confirm the resulting `job_id` (or `task_id`) and next schedule in the reply.
- Verify attachment paths stay within allowed workspace/read roots.
