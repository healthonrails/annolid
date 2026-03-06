# Reference

This page collects the current operational reference points for Annolid.

## Commands

Primary entry points:

- `annolid`: launch the GUI
- `annolid-run`: run plugin-based training and inference workflows

Common CLI discovery commands:

```bash
annolid --help
annolid-run --help
annolid-run list-models
annolid-run train <model> --help-model
annolid-run predict <model> --help-model
```

Security-focused agent commands:

```bash
annolid-run agent-security-check
annolid-run agent-security-audit
annolid-run agent-security-audit --fix
annolid-run agent-secrets-audit
```

## Important Config Files

- LLM and Annolid Bot model settings:
  `~/.annolid/llm_settings.json`
- Annolid agent runtime and channel config:
  `~/.annolid/config.json`
- Annolid agent local private secret store:
  `~/.annolid/agent_secrets.json`
- Annolid agent sessions directory:
  `~/.annolid/sessions/`
- LabelMe/GUI-style user config:
  `~/.labelmerc`

## Current Docs Pages

- [Installation](installation.md)
- [uv Setup](install_with_uv.md)
- [One-Line Installer](one_line_install_choices.md)
- [Workflows](workflows.md)
- [Tutorials](tutorials.md)
- [MCP](mcp.md)
- [Agent Secrets](agent_secrets.md)
- [Agent Security](agent_security.md)
- [SAM 3D](sam3d.md)
- [Deployment](deployment.md)

## Current Bot/Agent Surfaces

Annolid Bot currently spans:

- GUI chat dock in the desktop app,
- optional background channels such as Zulip and WhatsApp,
- MCP integrations,
- browser/web tooling when configured,
- automation scheduling and auxiliary services defined in the agent config.

The current Zulip UI flow is part of the Annolid Bot dock and depends on the Zulip channel block in `~/.annolid/config.json`.

## Selected Repository Paths

- GUI code: `annolid/gui/`
- Bot widget code: `annolid/gui/widgets/ai_chat_widget.py`
- Agent/channel config schema: `annolid/core/agent/config/`
- CLI plugin entry point: `annolid/engine/cli.py`
- Tutorial notebooks: `docs/tutorials/`
- MkDocs config: `mkdocs.yml`

## Current Documentation Rule

User-facing guidance should be added to `docs/` first. Historical book content and legacy links are secondary references, not the primary source of truth.
