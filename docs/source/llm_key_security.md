# LLM API Key Security

This page explains how to configure LLM providers in Annolid while keeping API keys safe.

## What Annolid Stores

Annolid stores provider/model preferences in:

- `~/.annolid/llm_settings.json`

Annolid **does not persist API keys** to this file. Secret fields such as `api_key`, `token`, and `access_token` are scrubbed before writing settings.

## Recommended Key Setup

Use environment variables for secrets:

- OpenAI: `OPENAI_API_KEY`
- OpenRouter: `OPENROUTER_API_KEY`
- Gemini: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`)
- Optional OpenAI-compatible endpoint: `OPENAI_BASE_URL`
- Optional local Ollama endpoint: `OLLAMA_HOST`

Annolid loads environment variables in this order:

1. Parent process environment (already exported in shell/session)
2. `./.env` in the current working directory (if present)
3. `~/.annolid/.env` as a global fallback

`.env` files do not override variables that are already set.

Annolid also supports inline env config in `~/.annolid/llm_settings.json`:

```json
{
  "env": {
    "OPENROUTER_API_KEY": "sk-or-...",
    "vars": {
      "GROQ_API_KEY": "gsk-..."
    }
  }
}
```

Inline env values are applied only when the variable is not already present in the process environment.

In the **AI Model Settings** dialog, you can also opt in to:

- `Persist entered credentials to ~/.annolid/.env (optional)`

This is disabled by default. When enabled, entered API keys and provider base-URL env values are written to `~/.annolid/.env`.

### macOS / Linux (bash/zsh)

```bash
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
```

To persist them for your shell, add to `~/.zshrc` or `~/.bashrc`.

### Windows PowerShell

```powershell
setx OPENAI_API_KEY "sk-..."
setx GEMINI_API_KEY "..."
```

Restart the terminal (or app) after `setx`.

## File Permission Hardening

Annolid enforces strict permissions for LLM settings storage:

- `~/.annolid` is set to `700`
- `~/.annolid/llm_settings.json` is set to `600`

This limits read access to your user account.

## Safe Operational Practices

- Never commit `.env`, API keys, or copied key values to git.
- Keep `~/.annolid/.env` and project `.env` out of source control.
- Avoid pasting keys into logs, screenshots, or issue trackers.
- Rotate keys immediately if accidentally exposed.
- Use separate keys per environment (dev, test, production).
- Prefer least-privilege tokens and provider-side usage limits.

## Verify Configuration

You can verify that keys are available without writing them to disk:

```bash
python -c "import os; print(bool(os.getenv('OPENAI_API_KEY')), bool(os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')))"
```

If either value prints `False`, configure the corresponding environment variable and restart Annolid.

You can also run Annolid's built-in security check:

```bash
annolid agent-security-check
```

The command reports:

- settings file/dir permission posture,
- whether any secret-like fields were persisted,
- OpenAI/Gemini key availability (including env var fallback).

Exit code is `0` for `ok`, `1` for `warning`.

## Adding New Providers in GUI

In **AI Model Settings**, click **Add Provider** to create a new OpenAI-compatible provider (for example NVIDIA NIM):

- provider id (e.g. `nvidia`)
- display name
- base URL (e.g. `https://integrate.api.nvidia.com/v1`)
- API key env var name (e.g. `NVIDIA_API_KEY`)

After saving, the provider appears in the provider selector without code changes.
