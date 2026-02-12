---
name: cron
description: Schedule and manage recurring or one-shot agent tasks.
metadata: '{"annolid":{"requires":{"bins":[]}}}'
---

Use this skill when users want reminders or periodic automation.

Tool usage:

1. `cron` `add` with one of:
   - `every_seconds`
   - `cron_expr`
   - `at_ms`
2. `cron` `list` to confirm schedule and identifiers.
3. `cron` `disable`/`enable` for temporary control.
4. `cron` `run` to trigger manually.
5. `cron` `remove` to delete.

Always include the resulting `job_id` in user-facing confirmations.
