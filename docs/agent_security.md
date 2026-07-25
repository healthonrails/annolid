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

For a WhatsApp Cloud API webhook, store the Meta app secret through a ref:

```bash
annolid-run agent-secrets-set \
  --path tools.whatsapp.app_secret \
  --env WHATSAPP_APP_SECRET
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

### Runtime Network and Workspace Guards

Annolid Bot blocks shell commands that target private or internal HTTP(S)
addresses, including local hosts and cloud metadata ranges. Workspace-scoped
shell execution also treats the configured workspace as the trusted root, so a
tool call cannot widen access by passing a different `working_dir`.

Agent web-fetch and download clients apply the same public-target validation to
every outgoing request, including each redirect hop. Redirect targets that
resolve to localhost, private networks, or cloud metadata services are rejected
before the redirected request is sent. The validated public DNS result is pinned
to the actual HTTP connection to prevent DNS rebinding between validation and
connect. Environment proxies are disabled for these requests because a proxy
would resolve the destination outside Annolid's pinned connection boundary.

Workspace restriction is enabled by default. Annolid does not register the host-backed
`exec_start` and `exec_process` tools. The Docker-backed `exec` tool also fails
closed when Docker is unavailable instead of silently falling back to the host.
Shell guards also inspect absolute paths after option separators such as
`--output=/outside/path`.

The default Docker image is pinned to a reviewed immutable SHA-256 digest.
Floating image tags are refused at execution time and reported by
`agent-security-audit`. Existing configurations can explicitly set
`tools.restrict_to_workspace=false`, but doing so re-enables host-backed managed
shell sessions and is reported as a high-severity finding.

Filesystem tools report workspace-boundary failures as hard policy boundaries.
If the same outside target is retried across equivalent tools, the tool registry
returns a refusal that tells the agent to ask the user how to proceed instead of
trying shell or path-workaround variants.

Text read/edit tools reject files larger than 100 MiB before loading them into
memory. Use a format-specific or streaming workflow for larger artifacts.

### WhatsApp Webhook Authentication

When `tools.whatsapp.app_secret` is configured, POST requests must carry a valid
Meta `X-Hub-Signature-256` signature over the exact request body. Invalid or
missing signatures are rejected before the payload reaches the message bus.

Annolid refuses to start an unsigned listener, including on loopback. This
prevents another local process or a browser-originated cross-site request from
injecting a message into the agent. Configure the app secret before enabling
the webhook or forwarding it through a public HTTPS tunnel.

Outbound WhatsApp Cloud API calls also require an HTTPS, publicly resolvable
`graph.facebook.com` endpoint before Annolid attaches the access token.

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
