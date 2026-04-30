# Google Integrations

Annolid Bot now supports native Google API integrations with shared OAuth credentials.

Current tools:

- `google_drive` (optional, enabled by config)
- `google_calendar` (optional, enabled by config)

The Google OAuth token/credentials are shared so Drive and Calendar use one local auth setup. This also prepares a single auth surface for future Gmail tooling.

## What Changed

The legacy Google Workspace CLI (`gws`) path and `gws_setup` tool are removed.

- no `gws` install step
- no `tools.gws` config block
- no `gws-*` skill symlinks

Use the native `google_drive` and `google_calendar` tools instead.

## Prerequisites

Install Annolid with the bot integrations bundle:

```bash
pip install "annolid[annolid_bot]"
```

(Equivalent editable install works too: `pip install -e ".[annolid_bot]"`.)

## Shared OAuth Files

Default paths:

- OAuth client credentials: `~/.annolid/agent/google_oauth_credentials.json`
- Cached OAuth token: `~/.annolid/agent/google_oauth_token.json`

`google_oauth_credentials.json` is your Google Cloud OAuth client JSON.
`google_oauth_token.json` is generated locally after successful OAuth auth.

Annolid tries to keep them private:

- file mode: `0600`
- parent dir mode: `0700`

## Config Reference

Configure in `~/.annolid/config.json`:

```json
{
  "tools": {
    "google_auth": {
      "credentialsFile": "~/.annolid/agent/google_oauth_credentials.json",
      "tokenFile": "~/.annolid/agent/google_oauth_token.json",
      "allowInteractiveAuth": false
    },
    "google_drive": {
      "enabled": true
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

Field summary:

- `tools.google_auth.credentialsFile`: shared Google OAuth client file.
- `tools.google_auth.tokenFile`: shared cached token file.
- `tools.google_auth.allowInteractiveAuth`: allow browser OAuth when token is missing/invalid.
- `tools.google_drive.enabled`: register `google_drive` tool.
- `tools.calendar.enabled`: register `google_calendar` tool.

## First-Time OAuth Setup

1. Put your Google OAuth client JSON at `credentialsFile`.
2. Set `tools.google_auth.allowInteractiveAuth=true`.
3. Start Annolid from an interactive terminal:

```bash
annolid
```

4. Trigger a Drive or Calendar action in Annolid Bot, for example:

```text
List my latest 5 Google Drive files
```

or

```text
List my next 3 Google Calendar events
```

5. Complete browser OAuth.
6. Confirm token is created at `tokenFile`.
7. Optionally set `allowInteractiveAuth=false` again for safer background operation.

## Example Prompts

- `List my latest 5 Google Drive files`
- `Show details for Google Drive file <file_id>`
- `Create a Google Drive folder named Lab Notes`
- `Delete Google Drive file <file_id>`
- `List my calendar events for this week`

## Video Upload Workflows (Google Drive)

Annolid Bot supports resumable, retry-safe video uploads to Google Drive.

### Upload one video file

Example prompt:

```text
Use google_drive upload_file for /path/to/video.mp4 into remote_folder_path annolid/saved_videos
```

Recommended fields:

- `local_path`
- `remote_folder_id` or `remote_folder_path`
- `chunk_size_mb` (default 8)
- `skip_if_exists` (default true, name+size dedupe)

### Upload saved videos in bulk

Example prompt:

```text
Upload saved videos from /path/to/project/output to Drive folder annolid/saved_videos
```

This uses `upload_saved_videos` and filters to recent non-realtime video files.

### Upload realtime-detect videos in bulk

Example prompt:

```text
Upload realtime detect videos from ~/.annolid/realtime to Drive folder annolid/realtime_detect
```

This uses `upload_realtime_videos` and focuses on realtime/detection-like paths and names.

## Realtime GUI Auto-Upload

In `Realtime Control` -> `Run & Output`:

1. Enable `Save MP4 segments on detections`.
2. Use `Check Auth` to verify Google OAuth readiness.
3. Use `Login` to run browser auth if needed.
4. Enable `Auto-upload saved segments to Google Drive`.
5. Set `Upload Delay` and `Drive Folder Path` (for example `annolid/realtime_detect`).
6. Start realtime inference.

When a detection segment is saved to disk, Annolid schedules an automatic Drive upload after the configured delay.

### Reliability behavior

- resumable uploads with chunked transfer
- retry on transient Drive errors (`429`, `500`, `502`, `503`, `504`)
- optional skip when same name and size already exists in destination
- batch result summary with uploaded/skipped/failed counts

## Troubleshooting

### Google tool not registered

Check:

- `tools.google_drive.enabled=true` for Drive
- `tools.calendar.enabled=true` for Calendar
- required Python deps installed (`annolid[annolid_bot]`)

Restart Annolid after config changes.

### OAuth token missing and no browser prompt

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

Then relaunch Annolid from an interactive terminal and retry a Google action.

### Access blocked (Error 403: access_denied)

If Google shows:

```text
Access blocked: <your app> has not completed the Google verification process
Error 403: access_denied
```

your OAuth app is usually in `Testing` mode and the signed-in Gmail account is not approved.

Fix:

1. Open [Google Cloud Console](https://console.cloud.google.com/) for the same project as your OAuth client.
2. Go to `APIs & Services` -> `OAuth consent screen`.
3. Check `Publishing status`.
4. If status is `Testing`, add your Gmail account under `Test users`.
5. Save changes and retry Annolid auth.
6. If you need broader user access, publish to `Production` and complete Google verification for sensitive/restricted scopes.

Notes:

- `Internal` app type only allows users in the same Google Workspace organization.
- Workspace org policies can block external OAuth clients until an admin approves them.

### Credentials/token path issues

Verify configured paths in `~/.annolid/config.json` and ensure files are readable/writable.

## Related Docs

- [Agent Calendar](agent_calendar.md)
- [MCP](mcp.md)
- [Agent Secrets](agent_secrets.md)
- [Agent Security](agent_security.md)
- [Workflows](workflows.md)
