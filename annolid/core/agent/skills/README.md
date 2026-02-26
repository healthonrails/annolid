# Annolid Built-in Skills

This directory contains built-in skills for the Annolid agent runtime.

## Format

Each skill is a folder containing `SKILL.md` with:

- frontmatter (`name`, `description`, optional invocation/gating keys)
- markdown instructions used by the agent

`metadata` may be a JSON string/object and can include `annolid` or `openclaw`
namespaces. Supported fields include:

- `always`: load this skill automatically in context
- `user-invocable` (default `true`)
- `disable-model-invocation` (default `false`)
- `command-dispatch`, `command-tool`, `command-arg-mode`
- `os`: allowed operating systems (`darwin`, `linux`, `win32`)
- `requires.bins`: required command-line tools
- `requires.anyBins`: at least one CLI required
- `requires.env`: required environment variables
- `requires.config`: required config paths (dot notation)

## Discovery & precedence

Skills are discovered from:

1. `<workspace>/skills` (highest precedence)
2. `~/.annolid/skills` (managed/shared)
3. Built-in skills in `annolid/core/agent/skills`
4. Extra dirs from `ANNOLID_SKILLS_EXTRA_DIRS` or `config.json` `skills.load.extraDirs` (lowest)

When names conflict, higher precedence wins.

## Hot Reload

Skill discovery supports optional mtime-based refresh checks between turns.

- Enable with env: `ANNOLID_SKILLS_LOAD_WATCH=1` (or `ANNOLID_SKILLS_WATCH=1`)
- Configure in `~/.annolid/config.json`:
  - `skills.load.watch: true`
  - `skills.load.pollSeconds: 1.0`
- Optional env override for polling interval:
  - `ANNOLID_SKILLS_WATCH_POLL_SECONDS=0.5`

Example:

```yaml
---
name: weather
description: Query weather conditions
metadata: '{"annolid":{"requires":{"bins":["curl"]}}}'
---
```
