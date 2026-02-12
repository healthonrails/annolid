---
name: cron
description: Schedule and manage recurring or one-shot agent tasks.
metadata: '{"annolid":{"requires":{"bins":[]}}}'
---

Use this skill when users want reminders or periodic automation.

Tool usage (Three Modes):

1. `cron` `add` with one of:
   - `every_seconds` for fixed intervals
   - `cron_expr` for cron-style expressions
   - `at` for a one-time run using ISO datetime (e.g. `2026-02-13T09:30:00Z`)
2. `cron` `list` to confirm schedule and identifiers.
3. `cron` `disable`/`enable` for temporary control.
4. `cron` `run` to trigger manually.
5. `cron` `remove` to delete.

Example one-time job:

- `cron` action=`add`, message=`"Review tracking report"`, at=`"2026-02-13T09:30:00Z"`

Always include the resulting `job_id` in user-facing confirmations.
