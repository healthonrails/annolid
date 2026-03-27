# Agent Secrets

Annolid agent config can now reference secrets instead of storing raw credentials in `~/.annolid/config.json`.

For broader posture checks around session routing, tool policy, permissions, and channel exposure, see [Agent Security](agent_security.md).

This is intended for:

- provider API keys,
- Zulip API keys,
- email passwords,
- WhatsApp access and verify tokens,
- Box access tokens,
- other agent channel credentials.

## Files

- Main agent config: `~/.annolid/config.json`
- Local private secret store: `~/.annolid/agent_secrets.json`

`config.json` is safe to inspect and version locally because ref-backed secret fields are scrubbed on save.

## Secret Sources

Annolid supports two secret reference sources:

- `env`: read the secret from an environment variable.
- `local`: read the secret from `~/.annolid/agent_secrets.json`.

At runtime, Annolid resolves secret refs and passes normal string values to the existing agent/provider/channel code. This means current features keep working without changing how Zulip, WhatsApp, email, or provider clients consume credentials.

## Recommended Flow

For existing plaintext config:

```bash
annolid-run agent-secrets-audit
annolid-run agent-secrets-migrate
annolid-run agent-secrets-migrate --apply
```

For environment-managed secrets:

```bash
export ZULIP_API_KEY="..."
annolid-run agent-secrets-set --path tools.zulip.api_key --env ZULIP_API_KEY
```

For a local private secret:

```bash
annolid-run agent-secrets-set \
  --path tools.zulip.api_key \
  --local tools.zulip.api_key \
  --value "..."
```

To remove a ref:

```bash
annolid-run agent-secrets-remove --path tools.zulip.api_key
```

To remove a local ref and also delete the stored secret value:

```bash
annolid-run agent-secrets-remove \
  --path tools.zulip.api_key \
  --delete-local-value
```

## Commands

### `annolid-run agent-secrets-audit`

Audits the current agent config and reports:

- plaintext secret paths still present in `config.json`,
- configured secret refs,
- unresolved refs,
- whether the private local secret store exists.

This command exits with a warning status when plaintext secrets or unresolved refs are found.

### `annolid-run agent-secrets-migrate`

Finds plaintext secret fields in `config.json` and shows what would be moved into the local secret store.

Dry run:

```bash
annolid-run agent-secrets-migrate
```

Apply:

```bash
annolid-run agent-secrets-migrate --apply
```

When applied, Annolid:

1. copies plaintext secret values into `~/.annolid/agent_secrets.json`,
2. creates `local` secret refs in `config.json`,
3. clears the plaintext values from `config.json`.

### `annolid-run agent-secrets-set`

Attaches a secret ref to a config path.

Examples:

```bash
annolid-run agent-secrets-set --path providers.gemini.api_key --env GEMINI_API_KEY
annolid-run agent-secrets-set --path tools.email.password --local tools.email.password --value "app-password"
annolid-run agent-secrets-set --path tools.box.access_token --env BOX_ACCESS_TOKEN
annolid-run agent-secrets-set --path tools.box.client_secret --env BOX_CLIENT_SECRET
annolid-run agent-secrets-set --path tools.box.refresh_token --env BOX_REFRESH_TOKEN
```

### `annolid-run agent-secrets-remove`

Removes a secret ref from a config path. By default this removes only the ref metadata. Use `--delete-local-value` to also delete the local stored secret value.

## Security Check

`annolid-run agent-security-check` now also inspects the agent secret posture. It reports:

- plaintext secrets still present in `~/.annolid/config.json`,
- unresolved secret refs,
- private file permissions for the local secret store.

## Example Config

Environment-backed ref:

```json
{
  "tools": {
    "zulip": {
      "enabled": true,
      "serverUrl": "https://zulip.example.com",
      "user": "annolid-bot@example.com",
      "apiKey": ""
    }
  },
  "secrets": {
    "refs": {
      "tools.zulip.api_key": {
        "source": "env",
        "name": "ZULIP_API_KEY"
      }
    }
  }
}
```

Local-store-backed ref:

```json
{
  "secrets": {
    "refs": {
      "tools.zulip.api_key": {
        "source": "local",
        "name": "tools.zulip.api_key"
      }
    }
  }
}
```

## Notes

- Secret refs are additive and backward-compatible with existing config fields.
- If a ref is configured but cannot be resolved, audit and security-check commands will flag it.
- Ref-backed fields are intentionally persisted as empty strings in `config.json`.
