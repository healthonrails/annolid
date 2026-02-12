# Security Policy

## Supported Versions

Annolid is actively maintained on the latest release line.

| Version | Supported |
| --- | --- |
| Latest release (`main` / newest tag) | Yes |
| Older releases | Best effort only |

If a report applies to an older version, please also check whether it reproduces on the latest release.

## Reporting a Vulnerability

Please report security issues privately.

1. Do **not** open a public GitHub issue with exploit details.
2. Prefer GitHub Security Advisories for this repository.
3. If needed, contact maintainers directly (see repository contacts, including `healthonrails@gmail.com`).
4. Include:
   - affected version/commit
   - clear reproduction steps
   - expected vs actual behavior
   - impact and any proof-of-concept
   - suggested mitigation (optional)

Target response window: initial triage within 72 hours.

## Security Model and Trust Boundaries

Annolid includes:

- local GUI and CLI workflows
- optional remote model providers (for example OpenAI/Gemini/OpenRouter)
- local model runtimes (for example Ollama)
- agent capabilities with file, shell, web, cron, channel, and bus integrations

Treat any model/tool execution path as privileged automation. Run with least privilege and explicit operator oversight.

## Hardening Guidance

### 1) Secrets and Credentials

- Do not commit credentials, tokens, cookies, or private keys.
- Prefer environment variables or OS secret stores for API tokens.
- Annolid LLM settings scrub secret fields before writing `~/.annolid/llm_settings.json`.
- Annolid enforces restrictive permissions for LLM settings storage (`~/.annolid` 700, settings file 600).
- Rotate compromised keys immediately.
- Use separate credentials for development and production.

See `docs/source/llm_key_security.md` for setup instructions and platform-specific examples.

### 2) Agent Tooling Risk (`exec`, file tools, web tools)

- Run Annolid under a non-admin/non-root account.
- Restrict accessible directories whenever possible.
- Review tool usage and outputs in logs.
- Do not disable safety guards around shell/file operations.

### 3) Networked Integrations

- Only enable channels/providers you need.
- Apply allow-lists and identity checks for inbound channels.
- Keep bridge endpoints local/private unless protected by auth + TLS.
- Monitor outbound traffic for unexpected destinations.

### 4) Local Data Protection

- Protect `~/.annolid` and workspace directories with strict filesystem permissions.
- Remember that chat/session/memory artifacts may contain sensitive research data.
- Use encrypted disks or encrypted home directories on shared machines.

### 5) Dependency Security

- Keep Python dependencies current.
- Regularly run vulnerability scanners in CI or locally (for example `pip-audit`).
- Update promptly when security advisories affect transitive dependencies.

## Operational Checklist

Before using Annolid in a sensitive environment:

- [ ] Run as a dedicated non-privileged user
- [ ] Restrict filesystem permissions for data/config directories
- [ ] Configure provider/channel allow-lists and credentials
- [ ] Review enabled tools and disable unnecessary capabilities
- [ ] Keep dependencies and Annolid version updated
- [ ] Enable logging/monitoring and retain incident-relevant logs
- [ ] Define credential rotation and incident response steps

## Incident Response

If compromise is suspected:

1. Revoke/rotate all potentially exposed credentials.
2. Disable risky integrations (channels/tools/providers) temporarily.
3. Preserve logs and affected artifacts for investigation.
4. Upgrade to the latest patched version.
5. Report the incident privately with scope and timeline.
