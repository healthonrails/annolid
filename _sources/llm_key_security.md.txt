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
