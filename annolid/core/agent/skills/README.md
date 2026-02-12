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

Example:

```yaml
---
name: weather
description: Query weather conditions
metadata: '{"annolid":{"requires":{"bins":["curl"]}}}'
---
```
