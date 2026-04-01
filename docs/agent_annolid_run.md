# Annolid Agent and `annolid-run`

Use this guide when you want Annolid Bot or other Annolid agent surfaces to drive
CLI workflows through `annolid-run`.

## What This Enables

Annolid agents can now invoke `annolid-run` through a typed `annolid_run` tool
instead of falling back to generic shell execution.

That gives you:

- structured argument handling,
- captured `stdout` and `stderr`,
- workspace-aware working directory resolution,
- safer defaults for mutating commands.

## Recommended Usage

Use read-only commands directly when you want status, discovery, or inspection.

Examples:

```text
annolid-run agent-status
annolid-run agent-onboard
annolid-run agent-onboard --dry-run
annolid-run agent-onboard --update
annolid-run agent-onboard --prune-bootstrap
annolid-run list-models
annolid-run help train
annolid-run help train dino_kpseg
annolid-run help predict dino_kpseg
annolid-run update check --channel stable
annolid-run agent memory inspect --workspace /path/to/workspace
```

These commands route through the dedicated tool path and return the captured CLI
output back into chat.

## Safety Model

The `annolid_run` tool blocks mutating commands unless `allow_mutation=true` is
set explicitly.

Examples of commands that may mutate state:

- `annolid-run update run`
- `annolid-run update rollback --previous-version 1.0.0`
- `annolid-run agent memory flush`
- `annolid-run agent skills refresh`
- `annolid-run train <model> ...`
- `annolid-run predict <model> ...`

Direct chat commands such as:

```text
annolid-run update run
```

do not auto-opt into mutation. They fail closed and instruct the caller to use
the typed tool path intentionally.

## Tool Contract

The `annolid_run` tool accepts:

- `command`: string form of the CLI command
- `argv`: optional explicit argument list
- `working_dir`: optional path resolved inside allowed roots
- `allow_mutation`: explicit opt-in for mutating commands

Typical typed invocation shape:

```json
{
  "command": "annolid-run update check --channel stable",
  "working_dir": "/Users/you/project",
  "allow_mutation": false
}
```

## Good Patterns

- Prefer `annolid-run` for Annolid-native CLI actions instead of raw shell.
- Use `annolid-run help <command>` for command-level guidance and `annolid-run help train <model>` or `annolid-run help predict <model>` for model-specific flags.
- Use explicit workspaces for agent memory, secrets, eval, and skills commands.
- Start with read-only inspection commands before any `train`, `predict`, `refresh`, or `apply` action.
- Keep GUI-driven direct commands for discovery and status; reserve typed mutation for deliberate automation.

## Workspace Onboarding and Template Updates

Use `agent-onboard` to initialize or synchronize workspace bootstrap files under
`~/.annolid/workspace` (or a custom `--workspace` path).

Recommended flow:

```text
annolid-run agent-onboard --dry-run
annolid-run agent-onboard
annolid-run agent-onboard --update
```

- `--dry-run`: preview file actions without writing.
- `--update`: overwrite changed template files with current versions.
- `--overwrite`: legacy alias for `--update`.
- `--no-backup`: when used with `--update`, skip backup copies.
- `--backup-dir <path>`: write overwritten-file backups to a custom directory.
- `--prune-bootstrap`: remove stale bootstrap-managed files no longer present in current template set.

By default, update mode creates backup copies in:

`<workspace>/.annolid/bootstrap-backups/<timestamp>/`

This keeps user-modified files recoverable while allowing template updates.

GUI workflow:

- Open `Settings -> Agent Workspace Onboarding…`
- Preview actions, apply onboarding/update, optionally prune stale bootstrap files, restore from the latest backup, and verify workspace template status.

## Help Patterns

The CLI and agent tool normalize these help forms to the same underlying
behavior:

```text
annolid-run help
annolid-run help train
annolid-run help predict
annolid-run help train dino_kpseg
annolid-run help predict dino_kpseg
```

Use the command-level help form first, then switch to the model-specific help
form when you need plugin arguments and examples.

Built-in model plugins now expose curated help groups such as `Required inputs`,
`Model and runtime`, and `Inference controls`, so `annolid-run help train <model>`
and `annolid-run help predict <model>` are easier to scan before the full flag
list.

## Related Docs

- [Workflows](workflows.md)
- [Reference](reference.md)
- [Agent Security](agent_security.md)
- [Google Workspace](agent_workspace.md)
- [Memory Subsystem](memory.md)
