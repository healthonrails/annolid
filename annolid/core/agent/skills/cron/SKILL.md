---
name: cron
description: Schedule and manage recurring or one-shot agent tasks.
metadata: '{"annolid":{"requires":{"bins":[]}}}'
---

Use this skill to schedule **Headless Agent Workloads**.

When a cron job triggers, it simulates the user typing the assigned `message` into the chat. *This means your scheduled jobs can perform complex automation pipelines, including browsing the web, checking email, querying APIs, or writing files.*

Tool usage (Three Modes):

1. `cron` `add` with one of:
   - `every_seconds` for fixed intervals
   - `cron_expr` for cron-style expressions
   - `at` for a one-time run using ISO datetime (e.g. `2026-02-13T09:30:00Z`)
2. `cron` `list` to confirm schedule and identifiers.
3. `cron` `disable`/`enable` for temporary control.
4. `cron` `run` to trigger manually.
5. `cron` `remove` to delete.

### Examples

**Simple Reminder:**
- `cron` action=`add`, message=`"Remind me to call John"`, at=`"2026-02-13T09:30:00Z"`

**Schedule an Email:**
- `cron` action=`add`, message=`"Send an email to annolid@gmail.com saying: Attached are the PDF files."`, at=`"2026-02-20T17:56:00Z"`

**Advanced Automation (Autonomous Background Agent):**
- `cron` action=`add`, message=`"Check HackerNews and summarize the top 3 posts. Save the Markdown to ~/Desktop/hn-summary.md"`, every_seconds=`10800`

Always include the resulting `job_id` in user-facing confirmations.
