# Google Workspace Integration

Annolid Bot can interact with Google Workspace services (Drive, Gmail, Calendar, Sheets, Docs, Chat and more) through the [Google Workspace CLI (gws)](https://github.com/googleworkspace/cli).

## Prerequisites

- [Node.js](https://nodejs.org/) 18 or later
- npm (included with Node.js)

## Quick Start

### 1. Install the CLI

```bash
npm install -g @googleworkspace/cli
```

### 2. Authenticate

If you have `gcloud` installed:

```bash
gws auth setup
```

If you do not have `gcloud`, use the manual OAuth flow and then:

```bash
gws auth login
```

Verify authentication:

```bash
gws auth status
```

### 3. Enable in Annolid Config

Add the `gws` block to `~/.annolid/config.json`:

```json
{
  "tools": {
    "gws": {
      "enabled": true,
      "services": ["drive", "gmail", "calendar", "sheets", "slides"]
    }
  }
}
```

### 4. Link Skills

The agent can set up skills automatically. Ask the bot:

```text
Set up Google Workspace skills
```

This clones the [gws repo](https://github.com/googleworkspace/cli) to `~/.annolid/gws-cli/` and symlinks the `gws-*` skill directories into `~/.annolid/workspace/skills/`.

You can also do it manually:

```bash
git clone --depth 1 https://github.com/googleworkspace/cli.git ~/.annolid/gws-cli
mkdir -p ~/.annolid/workspace/skills
ln -s ~/.annolid/gws-cli/skills/gws-shared ~/.annolid/workspace/skills/
ln -s ~/.annolid/gws-cli/skills/gws-drive ~/.annolid/workspace/skills/
ln -s ~/.annolid/gws-cli/skills/gws-gmail ~/.annolid/workspace/skills/
ln -s ~/.annolid/gws-cli/skills/gws-calendar ~/.annolid/workspace/skills/
ln -s ~/.annolid/gws-cli/skills/gws-sheets ~/.annolid/workspace/skills/
ln -s ~/.annolid/gws-cli/skills/gws-slides ~/.annolid/workspace/skills/
```

## Config Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Register the Google Workspace tool |
| `auto_install` | bool | `false` | Register the setup tool even when `gws` is not on PATH |
| `services` | list | `["drive", "gmail", "calendar", "sheets"]` | Workspace services to expose |

Full example with all options:

```json
{
  "tools": {
    "gws": {
      "enabled": true,
      "autoInstall": false,
      "services": ["drive", "gmail", "calendar", "sheets", "docs", "chat"]
    }
  }
}
```

## How It Works

```
Agent decides it needs a Google Workspace action
  → Invokes a gws-* skill for guidance
  → Calls the google_workspace tool
  → Tool shells out to the gws CLI
  → gws uses stored OAuth credentials
  → Structured JSON result returned to agent
```

The `google_workspace` tool accepts:

- **service** — which Workspace API (drive, gmail, calendar, sheets, docs, chat, etc.)
- **resource** — API resource within the service (e.g. `files`, `users`, `events`)
- **method** — method to invoke (e.g. `list`, `create`, `get`, `send`)
- **params** — JSON-encoded query parameters
- **json_body** — JSON-encoded request body
- **extra_flags** — additional CLI flags like `--page-all` or `--dry-run`

## Available Skills

Recommended starter set:

| Skill | Description |
|-------|-------------|
| `gws-shared` | Shared auth patterns, global flags, auto-install block |
| `gws-drive` | List, upload, download, manage Drive files and folders |
| `gws-gmail` | Send, read, list, and search email |
| `gws-calendar` | List, create, and manage calendar events |
| `gws-sheets` | Read, write, and append to spreadsheets |

Additional skills available in the gws repo:

| Skill | Description |
|-------|-------------|
| `gws-docs` | Read and write Google Docs |
| `gws-slides` | Read and write presentations |
| `gws-chat` | Manage Chat spaces and messages |
| `gws-tasks` | Manage task lists and tasks |
| `gws-people` | Manage contacts and profiles |
| `gws-forms` | Read and write Google Forms |
| `gws-meet` | Manage Google Meet conferences |

To link all available skills at once:

```bash
ln -s ~/.annolid/gws-cli/skills/gws-* ~/.annolid/workspace/skills/
```

## GWS Setup Tool

The `gws_setup` tool provides bootstrap actions:

| Action | Description |
|--------|-------------|
| `check` | Verify gws is installed, npm is available, check linked skills |
| `install` | Install gws via `npm install -g @googleworkspace/cli` |
| `auth_status` | Run `gws auth status` to check authentication |
| `link_skills` | Clone the gws repo and symlink skills into `~/.annolid/workspace/skills/` |
| `update_skills` | Pull latest changes in the cloned repo |

## Calendar with GWS Backend

Instead of the OAuth-based `GoogleCalendarTool`, you can use `gws` as the calendar backend:

```json
{
  "tools": {
    "calendar": {
      "enabled": true,
      "provider": "gws"
    }
  }
}
```

This delegates calendar operations to the `gws` CLI, using its own stored credentials instead of separate Google API OAuth tokens.

## Example Prompts

Once set up, try these prompts with the agent:

- *"List my latest 5 Drive files"*
- *"Send an email to alice@example.com about the meeting tomorrow"*
- *"Show my calendar events for this week"*
- *"Read the first 10 rows from my Budget spreadsheet"*
- *"Create a new Google Doc called Meeting Notes"*
- *"Check my Google Workspace setup status"*

## Updating Skills

Skills are symlinked from the cloned repo, so updating is a single command:

```bash
git -C ~/.annolid/gws-cli pull
```

Or ask the agent:

```text
Update my Google Workspace skills
```

## Troubleshooting

### gws command not found

Install the CLI:

```bash
npm install -g @googleworkspace/cli
```

Confirm it is on your PATH:

```bash
which gws
```

### Authentication errors

Re-authenticate:

```bash
gws auth login
```

Or set up from scratch:

```bash
gws auth setup
```

### Skills not appearing

Verify symlinks exist:

```bash
ls -la ~/.annolid/workspace/skills/gws-*
```

If empty, re-link:

```bash
ln -s ~/.annolid/gws-cli/skills/gws-* ~/.annolid/workspace/skills/
```

### Tool not registered

Check that `tools.gws.enabled` is `true` in `~/.annolid/config.json` and restart Annolid.

## Related Docs

- [Agent Calendar](agent_calendar.md)
- [MCP](mcp.md)
- [Agent Secrets](agent_secrets.md)
- [Agent Security](agent_security.md)
- [Workflows](workflows.md)
