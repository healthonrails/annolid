# Annolid Bot Google Calendar Tutorial

This guide shows how to let Annolid Bot read and manage Google Calendar events.

## 1. Install optional dependency

```bash
# Annolid Bot extras bundle (includes Google Calendar + WhatsApp + MCP)
pip install "annolid[annolid_bot]"
```

If this optional package is not installed, Annolid skips registering calendar tools and other features keep working normally.

## 2. Create Google OAuth desktop credentials

1. Open Google Cloud Console.
2. Enable **Google Calendar API** for your project.
3. Create OAuth client credentials of type **Desktop app**.
4. Download the JSON and save it to:
   `~/.annolid/agent/google_calendar_credentials.json`

## 3. Configure Annolid

Edit `~/.annolid/agent/config.json`:

```json
{
  "tools": {
    "calendar": {
      "enabled": true,
      "provider": "google",
      "credentialsFile": "~/.annolid/agent/google_calendar_credentials.json",
      "tokenFile": "~/.annolid/agent/google_calendar_token.json",
      "calendarId": "primary",
      "timezone": "America/Los_Angeles",
      "defaultEventDurationMinutes": 30
    }
  }
}
```

If the file does not exist yet, Annolid now creates a default template on first bot startup.

On first use, a browser login flow opens and token is cached in `tokenFile`.

## 4. Example prompts

- "List my next 5 calendar events."
- "Create a calendar event tomorrow at 10am called Lab meeting."
- "Create a weekly recurring event tomorrow at 10am called Lab meeting."
- "Update event `<event_id>` summary to Project sync."
- "Update event `<event_id>` to repeat weekly."
- "Delete event `<event_id>`."

## 5. Security notes

- Keep credentials/token files outside source control.
- Use a dedicated Google account for automation if needed.
- Revoke token in Google account security settings if compromised.
