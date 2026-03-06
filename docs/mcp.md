# Model Context Protocol (MCP)

Annolid Bot can register tools and resources from Model Context Protocol servers.

## Current Config Path

Configure MCP in the Annolid agent config:

`~/.annolid/agent/config.json`

The schema currently stores MCP servers under the agent tools block as `mcpServers` / `mcp_servers`.

## Minimal Example

```json
{
  "tools": {
    "mcpServers": {
      "filesystem": {
        "command": "npx",
        "args": [
          "-y",
          "@modelcontextprotocol/server-filesystem",
          "/path/to/allowed/directory"
        ]
      }
    }
  }
}
```

## Supported Shape

Each server entry can define:

- `command`: executable to launch the server
- `args`: argument list
- `env`: optional environment variables
- `url`: optional remote server URL for HTTP/SSE-style MCP servers

## How It Is Used

Once configured, Annolid Bot can connect to those MCP servers during chat sessions and expose their tools/resources to the agent.

Typical uses:

- restricted filesystem access
- external search or data APIs
- research or lab-specific internal tooling

## Practical Notes

- Install the `annolid_bot` extra if you want the common Annolid Bot integration stack.
- Make sure the runtime needed by your MCP server exists locally, for example `node`, `npx`, or `python`.
- Prefer narrow filesystem/server scopes over broad access.

## Troubleshooting

- Check Annolid Bot logs and startup output for MCP registration failures.
- Verify the config path is correct: `~/.annolid/agent/config.json`.
- Confirm required environment variables are present in the server definition.
- If a server starts but tools do not appear useful, test the server independently outside Annolid first.
