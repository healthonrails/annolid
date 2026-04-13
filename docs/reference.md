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
annolid-run help train
annolid-run help predict
annolid-run help train <model>
annolid-run help predict <model>
```

Model help details:

- Built-in model plugins now show curated quick-reference groups before the
  full flag list.
- Older forms such as `annolid-run train <model> --help-model` remain
  supported.
- Plugin-author contract and examples: [Model Plugin Help](model_plugin_help.md)

Agent-side CLI execution:

- [Annolid Agent and annolid-run](agent_annolid_run.md)

Security-focused agent commands:

```bash
annolid-run agent-security-check
annolid-run agent-security-audit
annolid-run agent-security-audit --fix
annolid-run agent-secrets-audit
annolid-run agent-box-auth-url --redirect-uri https://your-app.example.com/oauth/callback
annolid-run agent-box-auth-url --authorize-base-url https://my_org_xxx.account.box.com --redirect-uri https://your-app.example.com/oauth/callback
annolid-run agent-box-auth-url --open-browser --redirect-uri https://your-app.example.com/oauth/callback
annolid-run agent-box-auth-exchange --redirect-uri https://your-app.example.com/oauth/callback --code <auth_code>
annolid-run agent-box-auth-exchange --authorize-base-url https://my_org_xxx.account.box.com --redirect-uri https://your-app.example.com/oauth/callback --code <auth_code>
annolid-run agent-box-token-refresh
```

The Agent Runtime settings dialog includes Box auth host, client ID, client secret, and redirect URI fields so the GUI auth button can reuse the same values after Annolid restarts. The redirect URI defaults to `http://localhost:8765/oauth/callback` and includes a copy button for pasting into the Box app registration page. When that redirect points to localhost, Annolid starts a temporary local callback listener so Box can complete the OAuth round trip instead of showing a connection error page. Use `http://` for localhost callbacks; `https://localhost/...` is not supported by the local listener.

ACP bridge command:

```bash
annolid-run agent acp bridge --workspace /path/to/repo
```

## Important Config Files

- LLM and Annolid Bot model settings:
  `~/.annolid/llm_settings.json`
- Annolid agent runtime and channel config:
  `~/.annolid/config.json`
- Annolid agent local private secret store:
  `~/.annolid/agent_secrets.json`
- Annolid agent Google Calendar credentials:
  `~/.annolid/agent/google_calendar_credentials.json`
- Annolid agent Google Calendar cached token:
  `~/.annolid/agent/google_calendar_token.json`
- Annolid agent sessions directory:
  `~/.annolid/sessions/`
- LabelMe/GUI-style user config:
  `~/.labelmerc`

## Current Docs Pages

- [Installation](installation.md)
- [uv Setup](install_with_uv.md)
- [One-Line Installer](one_line_install_choices.md)
- [Workflows](workflows.md)
- [Simulation and FlyBody](simulation_flybody.md)
- [COCO Data Flow](coco_data_flow.md)
- [Tutorials](tutorials.md)
- [Zone Analysis](zone_analysis.md)
- [Identity Governor](identity_governor.md)
- [MCP](mcp.md)
- [Model Plugin Help](model_plugin_help.md)
- [Codex and ACP](codex_and_acp.md)
- [Agent Calendar](agent_calendar.md)
- [Agent Box](agent_box.md)
- [Agent Secrets](agent_secrets.md)
- [Agent Security](agent_security.md)
- [SAM 3D](sam3d.md)
- [Brain 3D Reconstruction](brain_3d_reconstruction.md)
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
