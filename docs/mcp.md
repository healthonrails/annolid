# Model Context Protocol (MCP) Tutorial

Annolid supports the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), allowing you to extend the AI Chat's capabilities with external tools and data sources.

## Configuration

MCP servers are configured in your Annolid configuration file, typically located at `~/.annolid/config.json`.

If the file does not exist, you can create it with the following structure:

```json
{
  "tools": {
    "mcpServers": {
      "google-search": {
        "command": "npx",
        "args": [
          "-y",
          "@modelcontextprotocol/server-google-search"
        ],
        "env": {
          "GOOGLE_API_KEY": "your_api_key",
          "GOOGLE_SEARCH_ENGINE_ID": "your_engine_id"
        }
      }
    }
  }
}
```

### Server Configuration Options

- `command`: The command to run the MCP server (e.g., `npx`, `python`, `node`).
- `args`: A list of arguments to pass to the command.
- `env`: (Optional) A dictionary of environment variables required by the server.
- `url`: (Optional) The URL of a remote MCP server (using SSE/HTTP).

## Using MCP Tools

Once configured, Annolid will automatically connect to the MCP servers when you start a chat session. The tools provided by these servers will be registered and made available to the AI agent.

You can use them by simply asking the bot to perform a task that requires them, for example:

- "Search Google for the latest version of MediaPipe."
- "What is the weather in New York?" (if a weather MCP server is configured)

## Common MCP Servers

You can find a variety of MCP servers in the [MCP GitHub organization](https://github.com/modelcontextprotocol/servers) or other community repositories.

### Example: File System

To give the agent access to specific parts of your file system:

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

## Troubleshooting

- **Logs**: Check the Annolid console or logs for messages prefixed with `MCP:`.
- **Dependencies**: Ensure the required runtimes (Node.js, Python, etc.) are installed on your system if you are running servers via `npx` or `python`.
- **API Keys**: Verified that all necessary environment variables are correctly set in the `env` section.
