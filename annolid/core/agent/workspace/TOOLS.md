# Available Tools

This workspace is intended for the Annolid agent tool stack.

## File Tools

- `read_file(path)`
- `write_file(path, content)`
- `edit_file(path, old_text, new_text)`
- `list_dir(path)`

## Execution

- `exec(command, working_dir?)`

Safety notes:

- Prefer non-destructive commands.
- Keep command scope limited to relevant directories.
- Treat external downloads and scripts as untrusted until verified.

## Web

- `web_search(query, count?)`
- `web_fetch(url, extractMode?, maxChars?)`

## Communication and Delegation

- `message(content, channel?, chat_id?)`
- `spawn(task, label?)`

## Scheduling

- `cron` actions:
  - `add`, `list`, `remove`, `enable`, `disable`, `run`, `status`

Use cron for periodic reminders and routine checks.
