# Annolid Bot Background Services Setup

This tutorial helps you configure optional background services safely:

- Email polling/sending
- WhatsApp bridge/webhook channel
- Google Calendar tool

If optional packages are missing, Annolid now skips those features and keeps the app running.

## 1. Install optional extras you actually need

```bash
# Recommended bundle for most Annolid Bot users
pip install "annolid[annolid_bot]"
```

For WhatsApp QR bridge, also install Chromium runtime:

```bash
python -m playwright install chromium
```

## 2. Minimal safe config (`~/.annolid/agent/config.json`)

Start with disabled services, then enable one at a time:

```json
{
  "tools": {
    "email": {
      "enabled": false
    },
    "whatsapp": {
      "enabled": false,
      "autoStart": false
    },
    "calendar": {
      "enabled": false,
      "provider": "google"
    }
  }
}
```

## 3. Enable WhatsApp (optional)

See full guide: `docs/source/agent_whatsapp_tutorial.md`

- Set `tools.whatsapp.enabled=true`
- Set `tools.whatsapp.autoStart=true`
- Keep `bridgeMode="python"` for local QR bridge mode

If `websockets`/Playwright deps are missing, Annolid logs a warning and continues without WhatsApp startup.

## 4. Enable Google Calendar (optional)

See full guide: `docs/source/agent_google_calendar_tutorial.md`

- Set `tools.calendar.enabled=true`
- Install `annolid[annolid_bot]`
- Provide OAuth credentials file

If Google auth/client deps are missing, Annolid logs a warning and skips registering the calendar tool.

## 5. Common errors and quick fixes

- `No module named 'websockets'`
  - Run: `pip install "annolid[annolid_bot]"`
- `No module named 'google.auth'`
  - Run: `pip install "annolid[annolid_bot]"`
- Background services should not auto-start
  - Set:
    - `tools.whatsapp.autoStart=false`
    - `tools.email.enabled=false`
    - `tools.calendar.enabled=false`

## 6. Verify from your environment

```bash
python -c "import importlib.util as u; print('websockets', u.find_spec('websockets') is not None)"
python -c "import importlib.util as u; print('google.auth', u.find_spec('google.auth') is not None)"
```

If both print `False`, keep those services disabled in config until you install matching extras.
