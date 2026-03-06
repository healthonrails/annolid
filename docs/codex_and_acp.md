# Codex and ACP

Annolid now supports Codex in more than one way. The easiest mental model is:

- `openai_codex`: Codex as a cloud model, authenticated through ChatGPT/Codex OAuth-style credentials.
- `codex_cli`: Codex as a local CLI runtime for conservative text-first execution.
- `ACP`: a long-lived external coding session that Annolid can keep open across turns.

These modes solve different problems. You do not need all of them at once.

## Which Mode Should I Use?

Use `openai_codex` when:

- you want Codex to be the main model in Annolid Bot,
- you want normal model-style chat behavior,
- you want the best default behavior for cloud Codex access.

Use `codex_cli` when:

- you already have the `codex` CLI installed locally,
- you want a simple local fallback,
- you want session continuity through the Codex CLI thread system.

Use ACP when:

- you want Annolid to keep a coding session alive across multiple prompts,
- you want an IDE or external client to talk to Annolid over stdio,
- you want to route follow-up coding instructions into the same long-lived session instead of starting over each turn.

## Codex Provider Modes

### 1. `openai_codex` Cloud Provider

This is the main Codex provider inside Annolid.

Current behavior:

- default model: `openai-codex/gpt-5.4`
- default transport: `auto`
- `auto` means WebSocket-first with SSE fallback
- explicit `transport` values supported: `auto`, `websocket`, `sse`

Annolid keeps the provider separate from regular `openai` configuration because the auth and transport behavior are different.

Practical note:

- Annolid auto-detects local Codex OAuth credentials in the runtime. You do not need to paste an API key into the Annolid UI for `openai_codex`.
- In AI Model Settings, use `Refresh` to check the local Codex login state immediately, or click `Save` and Annolid will detect and store non-secret auth status metadata automatically.
- If the required Python packages are missing, Annolid will return a direct setup error instead of a vague provider failure.

Required packages for this mode:

- `oauth_cli_kit`
- `httpx`
- `websockets`

Example install inside the project environment:

```bash
source .venv/bin/activate
pip install oauth-cli-kit httpx websockets
```

### 2. `codex_cli` Local Provider

This mode uses the local `codex` executable instead of Annolid’s cloud provider stack.

Current behavior:

- default model: `codex-cli/gpt-5.3-codex`
- text-first runtime
- Codex CLI thread IDs are persisted so later turns can resume the same Codex thread
- GUI chat sessions pass the Annolid session ID through to preserve continuity

This mode is useful when the local Codex CLI is already installed and you want a simpler runtime path.

Requirements:

- `codex` must be available on `PATH`

Check it with:

```bash
codex --help
```

## Transport Behavior for `openai_codex`

Annolid’s `openai_codex` provider now supports the same transport model users expect from OpenClaw-style Codex integration:

- `websocket`: force websocket transport
- `sse`: force SSE transport
- `auto`: try websocket first, then fall back to SSE automatically

Why this matters:

- websocket is the preferred fast path,
- SSE is the compatibility path,
- `auto` gives the best default for most users.

Advanced users can set this in `~/.annolid/llm_settings.json`:

```json
{
  "provider": "openai_codex",
  "openai_codex": {
    "base_url": "https://chatgpt.com/backend-api/codex/responses",
    "transport": "auto",
    "preferred_models": ["openai-codex/gpt-5.4"]
  }
}
```

If the websocket attempt fails in `auto` mode, Annolid falls back to SSE automatically.

## Long-Lived ACP Sessions

ACP stands for Agent Client Protocol. In Annolid, ACP is the session-oriented runtime for external coding harness workflows.

What that means in practice:

- Annolid can create a coding session once,
- keep it open across prompts,
- queue follow-up requests into the same session,
- cancel the active turn without destroying the whole session,
- expose the session over stdio for IDE integrations.

This is different from one-shot provider chat. ACP is for ongoing coding work.

### What Annolid Stores for ACP Sessions

Annolid keeps ACP session state under the local session store in:

- `~/.annolid/sessions/`

Annolid also tracks Codex CLI thread continuity for ACP-backed Codex CLI runs, so follow-up prompts can resume the same Codex context instead of starting a fresh local thread each time.

### Cancel Behavior

ACP `cancel` does not tear down the whole session anymore.

Current behavior:

- the active turn is aborted,
- the session stays alive,
- the next prompt can reuse the same ACP session.

This makes ACP suitable for IDE workflows where the user may stop a long turn and immediately retry with a narrower instruction.

## Stdio ACP Bridge for IDEs

Annolid now exposes a formal stdio ACP bridge.

Run it with:

```bash
annolid-run agent acp bridge --workspace /path/to/repo
```

This starts a stdio server that external tools can spawn and communicate with.

Current ACP-native methods supported:

- `initialize`
- `newSession`
- `loadSession`
- `prompt`
- `cancel`
- `listSessions`
- `shutdown`

Annolid also accepts compatibility aliases used by OpenClaw-style clients, including:

- `sessions_spawn`
- `sessions_send`
- `sessions_poll`
- `sessions_list`
- `sessions_close`

### What the Bridge Does

The stdio bridge:

- maps external ACP client sessions to Annolid ACP runtime sessions,
- keeps session IDs stable for the life of the bridge process,
- emits session update notifications,
- supports long-lived coding workflows over stdio.

Annolid also bounds idle ACP client sessions so a long-running IDE process does not grow an unbounded in-memory session map.

## Example IDE Launch Pattern

If your editor supports custom stdio agent servers, the command usually looks like:

```bash
annolid-run agent acp bridge --workspace /absolute/path/to/repo
```

The editor then sends ACP messages over stdin/stdout.

## Recommended Setup Paths

### Cloud Codex as Main Model

Use this if you want Annolid Bot itself to talk directly to Codex:

1. Ensure Codex OAuth-style credentials are available to Annolid.
2. Install required packages in `.venv`.
3. Set provider to `openai_codex`. Do not enter an API key in the UI for this provider.
4. Leave transport on `auto` unless you are debugging transport issues.

### Local Codex CLI Fallback

Use this if you prefer the local CLI path:

1. Install the `codex` CLI.
2. Set provider to `codex_cli`.
3. Keep using the same Annolid chat session to preserve Codex CLI thread continuity.

### IDE or Thread-Oriented Coding Work

Use this if you want persistent coding sessions:

1. Launch `annolid-run agent acp bridge --workspace /path/to/repo`.
2. Connect your IDE or ACP-compatible client over stdio.
3. Reuse the returned ACP session ID for follow-up prompts.

## Troubleshooting

If `openai_codex` fails immediately:

- verify `oauth_cli_kit`, `httpx`, and `websockets` are installed in `.venv`
- check that the Codex OAuth/token source is available to the current runtime

If `codex_cli` fails immediately:

- verify `codex` is installed and on `PATH`
- run `codex --help` manually

If websocket transport is unstable:

- leave `transport` set to `auto` so Annolid can fall back to SSE
- force `sse` only if you want to disable websocket attempts entirely

If an ACP session seems stuck:

- send `cancel`
- reuse the same session with a narrower prompt
- restart the bridge if the external client and bridge state are out of sync

## Summary

Annolid’s Codex support is now split into clear layers:

- `openai_codex` for cloud Codex with transport-aware runtime selection
- `codex_cli` for local Codex CLI execution with thread continuity
- ACP for long-lived external coding sessions
- stdio ACP bridge for IDE and tool integration

That separation makes Codex behavior easier to understand, easier to debug, and safer to extend.
