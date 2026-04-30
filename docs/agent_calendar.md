# Agent Calendar

This page documents how Annolid Bot uses Google Calendar with shared Google OAuth credentials.

## Google Calendar Files

Annolid reads settings from `~/.annolid/config.json`.

Default shared paths:

- OAuth client credentials: `~/.annolid/agent/google_oauth_credentials.json`
- Cached OAuth token: `~/.annolid/agent/google_oauth_token.json`

These files are shared by Google Calendar and Google Drive tools.

Annolid keeps these files private when possible:

- token file mode: `0600`
- token parent directory mode: `0700`

## Relevant Config

Example config block:

```json
{
  "tools": {
    "google_auth": {
      "credentialsFile": "~/.annolid/agent/google_oauth_credentials.json",
      "tokenFile": "~/.annolid/agent/google_oauth_token.json",
      "allowInteractiveAuth": false
    },
    "calendar": {
      "enabled": true,
      "calendarId": "primary",
      "timezone": "America/New_York",
      "defaultEventDurationMinutes": 30
    }
  }
}
```

Important fields:

- `tools.calendar.enabled`: enables calendar tool registration.
- `tools.google_auth.credentialsFile`: shared Google OAuth client JSON path.
- `tools.google_auth.tokenFile`: shared cached token path.
- `tools.google_auth.allowInteractiveAuth`: allows browser OAuth when no usable token exists.

## Default Behavior

Annolid is cache-first:

1. If token exists, Annolid uses it.
2. If token is expired but refreshable, Google auth refreshes it.
3. If no usable token exists, browser OAuth starts only when `allowInteractiveAuth=true`.
4. If `allowInteractiveAuth=false` and no valid token exists, the tool is skipped or returns an actionable error.

This prevents unexpected OAuth prompts in background sessions.

## First-Time Authentication

Use this when `google_oauth_token.json` does not exist.

1. Put your Google OAuth credentials JSON at configured `credentialsFile`.
2. Set `tools.google_auth.allowInteractiveAuth=true`.
3. Launch Annolid from an interactive terminal:

```bash
annolid
```

4. Ask the bot to perform a calendar action:

```text
List my next 3 Google Calendar events
```

5. Complete browser OAuth.
6. Confirm token file exists at configured `tokenFile`.
7. Optionally set `allowInteractiveAuth=false` again.

## Token Renewal

If token is expired and cannot refresh:

1. Confirm `credentialsFile` points to a valid OAuth client JSON.
2. Set `tools.google_auth.allowInteractiveAuth=true`.
3. Remove stale token file.
4. Relaunch Annolid from terminal.
5. Trigger a calendar action again.
6. Complete OAuth flow.
7. Verify new token file.
8. Optionally set `allowInteractiveAuth=false`.

## Common Failure Cases

### Token file is never created

Check:

- `tools.calendar.enabled` is `true`
- `tools.google_auth.credentialsFile` exists
- `tools.google_auth.allowInteractiveAuth` is `true`
- Annolid was launched from an interactive terminal
- Calendar tool is present in current toolset

### Annolid says interactive auth is disabled

Set:

```json
{
  "tools": {
    "google_auth": {
      "allowInteractiveAuth": true
    }
  }
}
```

Restart Annolid after config changes.

### Annolid says credentials/token files are missing

Verify paths in `~/.annolid/config.json` and confirm files exist where configured.

### Browser auth opens but renewal still fails

Typical causes:

- wrong OAuth client JSON
- token path not writable
- Google OAuth app missing Calendar scope approval
- running from non-interactive environment

### Access blocked (Error 403: access_denied)

If Google shows an "app not completed verification process" 403 error:

1. Open Google Cloud Console for your OAuth project.
2. Go to `APIs & Services` -> `OAuth consent screen`.
3. If `Publishing status` is `Testing`, add the Gmail account under `Test users`.
4. Retry OAuth from Annolid.

For non-test users, move the app to `Production` and complete Google verification as required by Google scopes and policy.

## Recommended Operating Pattern

- keep `allowInteractiveAuth=false` for normal background use
- turn it on only for first-time setup or renewal
- launch Annolid from terminal when OAuth is needed
- turn it off again after a fresh token is created

## Related Docs

- [Google Integrations](agent_workspace.md)
- [Workflows](workflows.md)
- [Reference](reference.md)
- [Agent Secrets](agent_secrets.md)
- [Agent Security](agent_security.md)
