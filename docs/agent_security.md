# Agent Security

Annolid includes agent security checks for local state, channel exposure, session routing, tool policy risk, and signature enforcement.

Use this page together with [Agent Secrets](agent_secrets.md) when hardening a local or shared Annolid agent deployment.

## Commands

Primary security commands:

```bash
annolid-run agent-security-check
annolid-run agent-security-audit
annolid-run agent-security-audit --fix
annolid-run agent-secrets-audit
```

Operator-style aliases are also supported:

```bash
annolid-run agent security audit
annolid-run agent security audit --fix
```

## What Each Command Does

### `annolid-run agent-security-check`

Runs a focused configuration hygiene check. It reports:

- plaintext secrets in persisted LLM settings,
- plaintext agent secrets in `~/.annolid/config.json`,
- unresolved secret refs,
- local file permission problems for security-sensitive files.

Use this when you want a quick health check for credentials and private file modes.

### `annolid-run agent-security-audit`

Runs a broader posture audit across the agent configuration. It inspects:

- secret hygiene,
- `config.json`, secret-store, and sessions directory permissions,
- DM session scope safety for shared messaging channels,
- missing channel allowlists,
- disabled runtime tool guard rails,
- risky tool-policy combinations,
- unsigned skill-install and auto-update exposure.

This command exits with a warning status when findings are present.

### `annolid-run agent-security-audit --fix`

Applies only safe local permission repairs. It can tighten modes for:

- the agent config directory,
- `~/.annolid/config.json`,
- `~/.annolid/agent_secrets.json`,
- the local sessions directory.

It does **not** automatically:

- change session-routing policy,
- rewrite tool allow/deny policy,
- disable channels,
- migrate secrets,
- change update/signature policy.

Those changes are intentionally left for explicit operator review.

## What the Audit Flags

### Plaintext Secrets

The audit reports plaintext secrets still stored in `~/.annolid/config.json`.

Recommended action:

```bash
annolid-run agent-secrets-migrate
annolid-run agent-secrets-migrate --apply
```

Or attach explicit refs:

```bash
annolid-run agent-secrets-set --path tools.zulip.api_key --env ZULIP_API_KEY
```

### Unresolved Secret Refs

If a secret ref exists but its environment variable or local-store value is missing, the audit reports it as unresolved.

Recommended action:

- populate the required environment variable, or
- write the secret to the local secret store and attach a `local` ref.

### Unsafe DM Session Scope

If external channels such as Zulip, WhatsApp, or email are enabled while DM session scope is `main`, the audit flags it.

Why this matters:

- `main` can collapse independent DM conversations into the same session state,
- shared inbox or multi-user messaging setups can leak conversation context across senders.

Recommended action:

- prefer `per-account-channel-peer` for shared messaging environments,
- otherwise use `per-peer` or `per-channel-peer` depending on the channel design.

### Empty Channel Allowlists

If an external channel is enabled and `allow_from` is empty, the audit flags it.

Why this matters:

- the channel may accept inbound messages from any sender the integration can see.

Recommended action:

- explicitly populate `allow_from` with trusted senders or accounts before using the integration in production.

### Disabled Runtime Guard

If `agents.defaults.strict_runtime_tool_guard` is disabled, the audit flags it.

Why this matters:

- Annolid has deny-by-default runtime protections for combinations such as shell execution plus messaging or automation primitives.
- disabling the guard removes a key safety backstop if policy is too broad.

Recommended action:

- re-enable `strict_runtime_tool_guard` unless the deployment has a narrow, reviewed exception.

### Risky Tool Policy Combinations

The audit looks for high-risk requested policy shapes, especially when runtime guard rails are disabled.

Examples:

- process execution plus email/message/automation tools,
- process execution plus broad web/browser tooling,
- skill installation without signed-skill enforcement.

Recommended action:

- split broad profiles into narrower task-specific profiles,
- keep runtime execution separate from messaging and scheduling where possible,
- require signed skills in production-like environments.

### Unsigned Auto Updates

If automatic updates are enabled without strict signature requirements, the audit flags it.

Recommended action:

- enable signature enforcement in both config and environment before using automatic updates in production.

## Files and Permissions

Security-sensitive local paths:

- agent config: `~/.annolid/config.json`
- local secret store: `~/.annolid/agent_secrets.json`
- sessions dir: `~/.annolid/sessions/`

Recommended modes:

- directories: `700`
- files: `600`

`annolid-run agent-security-audit --fix` can repair these modes when the files are writable by the current user.

## Recommended Hardening Flow

For a local but security-conscious setup:

```bash
annolid-run agent-secrets-audit
annolid-run agent-secrets-migrate --apply
annolid-run agent-security-check
annolid-run agent-security-audit
```

If the audit reports only permission issues:

```bash
annolid-run agent-security-audit --fix
```

Then re-run:

```bash
annolid-run agent-security-audit
```

## Example Operator Review

Typical findings worth fixing before enabling shared messaging:

- `plaintext-config-secrets`
- `dm-scope-main`
- `channel-allowlist-zulip`
- `strict-runtime-tool-guard-disabled`
- `unsigned-auto-update`

The audit output is JSON so it can be inspected manually or consumed by scripts.

## Notes

- The security audit is intentionally additive and does not break backward compatibility.
- `--fix` is deliberately conservative and only changes local file permissions.
- Secret hygiene is documented in more detail on [Agent Secrets](agent_secrets.md).
