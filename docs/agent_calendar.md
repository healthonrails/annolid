# Agent Calendar

This page documents how Annolid Bot uses Google Calendar credentials, cached OAuth tokens, and token renewal.

## Google Calendar Files

Annolid reads Google Calendar settings from `~/.annolid/config.json`.

Default paths:

- OAuth client credentials: `~/.annolid/agent/google_calendar_credentials.json`
- Cached OAuth token: `~/.annolid/agent/google_calendar_token.json`

The credentials file is the Google OAuth client JSON downloaded from Google Cloud. The token file is created locally after a successful interactive OAuth flow.

Annolid keeps these files private when possible:

- token file mode: `0600`
- token parent directory mode: `0700`

## Relevant Config

Example config block:

```json
{
  "tools": {
    "calendar": {
      "enabled": true,
      "provider": "google",
      "credentialsFile": "~/.annolid/agent/google_calendar_credentials.json",
      "tokenFile": "~/.annolid/agent/google_calendar_token.json",
      "calendarId": "primary",
      "timezone": "America/New_York",
      "defaultEventDurationMinutes": 30,
      "allowInteractiveAuth": false
    }
  }
}
```

Important fields:

- `enabled`: enables Google Calendar tool registration.
- `credentialsFile`: path to the Google OAuth client JSON.
- `tokenFile`: path to the locally cached OAuth token.
- `allowInteractiveAuth`: allows Annolid to open the browser-based OAuth flow when no valid cached token is available.

## Default Behavior

Annolid is cache-first by default:

1. If the token file exists, Annolid uses it.
2. If the token is expired but refreshable, Google auth refreshes it.
3. If no usable token exists, Annolid only starts the OAuth browser flow when `allowInteractiveAuth=true`.
4. If `allowInteractiveAuth=false` and no valid token exists, the Google Calendar tool is skipped or returns an actionable error instead of opening a browser unexpectedly.

This is intentional so background agents do not trigger OAuth prompts.

## First-Time Authentication

Use this flow when `google_calendar_token.json` does not exist yet.

1. Put your Google OAuth credentials JSON at the configured `credentialsFile` path.
2. Set `tools.calendar.allowInteractiveAuth` to `true` in `~/.annolid/config.json`.
3. Launch Annolid from an interactive terminal session:

```bash
annolid
```

4. Open a fresh Annolid Bot session.
5. Ask the bot to perform a calendar action, for example:

```text
List my next 3 Google Calendar events
```

6. Complete the browser-based Google OAuth flow.
7. Confirm that the token file now exists at the configured `tokenFile` path.
8. Set `allowInteractiveAuth` back to `false` if you want the safer background default.

## Token Renewal

If the cached token has expired and can no longer be refreshed, re-authenticate:

1. Confirm that `credentialsFile` still points to a valid Google OAuth client JSON.
2. Set `tools.calendar.allowInteractiveAuth=true`.
3. Remove the stale token file at the configured `tokenFile` path.
4. Launch Annolid from a terminal:

```bash
annolid
```

5. Trigger a calendar action again.
6. Complete the OAuth flow in the browser.
7. Verify that a new token file was written.
8. Optionally set `allowInteractiveAuth=false` again after renewal.

## Common Failure Cases

### Token file is never created

Check all of the following:

- `tools.calendar.enabled` is `true`
- `provider` is `google`
- `credentialsFile` exists
- `allowInteractiveAuth` is `true`
- Annolid was launched from an interactive terminal session
- the Google Calendar tool is present in the current toolset

If the bot can already list events, the tool is registered. In that case, a missing token file usually means the auth flow has not been triggered in the current process or the token path is not writable.

### The bot says the token expired and needs to be refreshed

That means the cached token is no longer usable. Use the token renewal flow above:

1. enable `allowInteractiveAuth`
2. remove the stale token file
3. relaunch Annolid from a terminal
4. trigger a calendar action
5. complete OAuth again

### Annolid says interactive auth is disabled

Set this in `~/.annolid/config.json`:

```json
{
  "tools": {
    "calendar": {
      "allowInteractiveAuth": true
    }
  }
}
```

Restart Annolid after changing the config.

### Annolid says credentials or token files are missing

Verify both paths in `~/.annolid/config.json` and confirm the files exist where configured.

### The bot can list events but refuses to create one

Use a current Annolid build and restart the app after calendar-related updates. A direct prompt such as this should trigger the calendar tool:

```text
Create a Google Calendar event tomorrow at 10 AM called Agent Coding Event
```

### Browser auth opens but renewal still fails

Typical causes:

- wrong Google OAuth client JSON
- token file path not writable
- Google OAuth app missing Calendar scope approval
- running from a non-interactive environment

## Recommended Operating Pattern

- Keep `allowInteractiveAuth=false` for normal background use.
- Turn it on only for first-time setup or token renewal.
- Launch Annolid from a terminal when you need OAuth.
- Switch it back off after a fresh token is created.

## Related Docs

- [Workflows](workflows.md)
- [Reference](reference.md)
- [Agent Secrets](agent_secrets.md)
- [Agent Security](agent_security.md)
