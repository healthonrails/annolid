# Annolid Built-in Skills

This directory contains built-in skills for the Annolid agent runtime.

## Format

Each skill is a folder containing `SKILL.md` with:

- frontmatter (`name`, `description`, `metadata`)
- markdown instructions used by the agent

`metadata` is a JSON string and may include an `annolid` section:

- `always`: load this skill automatically in context
- `requires.bins`: required command-line tools
- `requires.env`: required environment variables

Example:

```yaml
---
name: weather
description: Query weather conditions
metadata: '{"annolid":{"requires":{"bins":["curl"]}}}'
---
```
