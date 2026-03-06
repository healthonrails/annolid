# Annolid Docs

Annolid is a practical toolkit for annotation, segmentation, tracking, behavior analysis, and model-assisted workflows in real research settings.

The current codebase includes:

- the desktop GUI launched with `annolid`,
- the plugin-based CLI launched with `annolid-run`,
- Annolid Bot in the GUI, including multimodal chat and optional background integrations,
- deployment pipelines for the landing page, docs portal, and notebook-style tutorial book.

## Start Here

- [Installation](installation.md): choose the right install path for your platform.
- [uv Setup](install_with_uv.md): create a local development environment with `.venv`.
- [Workflows](workflows.md): follow the main GUI, CLI, and analysis flows.
- [Tutorials](tutorials.md): jump to practical notebooks, videos, and focused guides.
- [MCP](mcp.md): extend Annolid Bot with Model Context Protocol servers.
- [Agent Secrets](agent_secrets.md): keep agent/provider credentials out of plaintext `config.json`.
- [Agent Security](agent_security.md): audit session scope, tool policy, channel exposure, and local agent state.
- [SAM 3D](sam3d.md): configure the optional 3D reconstruction integration.
- [Reference](reference.md): commands, config files, and operational paths.
- [Deployment](deployment.md): understand how docs and the website are published.

## Current Documentation Surfaces

- `README.md`: repository overview and quick-start guidance.
- `docs/`: canonical MkDocs content published to the docs portal and mirrored root routes.
- `website/`: the landing page source for `annolid.com/`.
- `docs/tutorials/`: notebook tutorials tracked in the repository.

## Current Product Status

- Python support in the package metadata is `>=3.10`; current docs and CI focus on Python 3.10 to 3.13.
- The primary entry points are `annolid` for the GUI and `annolid-run` for model/plugin workflows.
- Annolid Bot is an active part of the GUI and now includes a Zulip draft/send workflow in the bot dock when Zulip is configured.
- Agent/provider credentials can now be managed through `annolid-run agent-secrets-audit`, `agent-secrets-set`, and `agent-secrets-migrate` so `config.json` can reference environment or local private secrets instead of storing plaintext.
- Agent security posture can now be reviewed through `annolid-run agent-security-check` and `annolid-run agent-security-audit`, with `--fix` available for safe local permission repairs.
- Docs are built with MkDocs in strict mode through the `docs-quality` and `docs-pages` GitHub Actions workflows.
