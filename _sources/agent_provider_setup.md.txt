# Annolid Bot Provider and Model Setup Tutorial

This tutorial shows how to configure Annolid Bot providers, agent models, local models, and search keys.

## 1. Where Annolid reads settings

Annolid resolves provider/model settings from:

1. Environment variables (recommended for secrets).
2. `~/.annolid/llm_settings.json` for non-secret config.
3. Profile overrides in `profiles`.

Annolid removes secret fields (such as `api_key`) before persisting settings. Keep keys in environment variables.

## 2. Core environment variables

Set only what you need:

- `OLLAMA_HOST` for local Ollama (default: `http://localhost:11434`)
- `OPENAI_API_KEY` for OpenAI-compatible providers
- `OPENAI_BASE_URL` for OpenAI-compatible gateways (for example OpenRouter)
- `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) for Gemini
- `BRAVE_API_KEY` for `web_search` tool calls

macOS/Linux example:

```bash
export OLLAMA_HOST="http://127.0.0.1:11434"
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
export GEMINI_API_KEY="..."
export BRAVE_API_KEY="..."
```

## 3. Configure local models (Ollama)

1. Start Ollama and pull model(s), for example:
   - `ollama pull qwen3-vl`
   - `ollama pull llama3.2-vision:latest`
2. In Annolid Bot, choose provider `Ollama (local)`.
3. Pick an available model from the model selector.

Optional `~/.annolid/llm_settings.json` block:

```json
{
  "provider": "ollama",
  "ollama": {
    "host": "http://127.0.0.1:11434",
    "preferred_models": ["qwen3-vl", "llama3.2-vision:latest"]
  }
}
```

## 4. Configure OpenAI models

1. Set `OPENAI_API_KEY`.
2. Keep `OPENAI_BASE_URL=https://api.openai.com/v1` (or unset it).
3. In Annolid Bot GUI, choose provider `OpenAI GPT` and choose model.

Optional settings block:

```json
{
  "openai": {
    "base_url": "https://api.openai.com/v1",
    "preferred_models": ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"]
  }
}
```

## 5. Configure OpenRouter (OpenAI-compatible path)

Use OpenRouter through the OpenAI-compatible channel:

```bash
export OPENAI_API_KEY="sk-or-..."
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
```

GUI path:

1. Open Annolid Bot.
2. Click `Configure…` in the chat panel.
3. In `OpenRouter` tab, set:
   - `API key` to your `sk-or-...` token
   - `Base URL` to `https://openrouter.ai/api/v1`

Then use provider `OpenRouter` in the GUI and pick/set an OpenRouter model id (for example `openai/gpt-4o-mini` or another OpenRouter-supported id).

Notes:

- This path is robust with current Annolid provider resolution.
- For agent loop usage, keep model ids exactly as your gateway expects.

## 6. Configure Gemini

1. Set one of:
   - `GEMINI_API_KEY`
   - `GOOGLE_API_KEY`
2. In GUI, choose provider `Google Gemini`.

Optional settings block:

```json
{
  "gemini": {
    "preferred_models": ["gemini-1.5-flash", "gemini-1.5-pro"]
  }
}
```

## 7. Configure agent profiles and model routing

Use `profiles` in `~/.annolid/llm_settings.json` to assign different providers/models per agent role:

```json
{
  "profiles": {
    "playground": {"provider": "openai", "model": "gpt-4o-mini"},
    "caption": {"provider": "ollama", "model": "qwen3-vl"},
    "research_agent": {"provider": "openai", "model": "gpt-4.1-mini"},
    "polygon_agent": {"provider": "gemini", "model": "gemini-1.5-pro"}
  },
  "agent": {
    "temperature": 0.7,
    "max_tool_iterations": 12,
    "max_history_messages": 24,
    "memory_window": 50
  }
}
```

Memory behavior:

- `memory/MEMORY.md` is injected as long-term facts.
- `memory/HISTORY.md` is an append-only archive for recall/search (not auto-injected).

Model name flexibility:

- In Annolid Bot model selector, you can type any model id (not limited to the predefined list).
- Typed model ids are remembered per provider and shown in future sessions.

## 8. Configure search key for tool calls

`web_search` tool calls require:

```bash
export BRAVE_API_KEY="..."
```

If unset, tool execution returns `Error: BRAVE_API_KEY not configured`.

## 9. Validate your setup

Run:

```bash
annolid agent-security-check
```

This reports:

- key/config file permission posture,
- whether secret-like fields were persisted,
- provider key presence for OpenAI/Gemini.

Exit code:

- `0`: OK
- `1`: Warning

## 10. Recommended secure workflow

1. Keep secrets only in env vars.
2. Keep `llm_settings.json` for model/provider preferences.
3. Use `agent-security-check` after changes.
4. Rotate keys immediately if exposed.
