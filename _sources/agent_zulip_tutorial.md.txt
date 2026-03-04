# Annolid Bot Zulip Setup Tutorial

This guide shows how to connect Annolid Bot to Zulip for inbound polling and outbound replies.

## 1. Create a Zulip bot

1. In Zulip, open **Settings → Your bots → Add a new bot**.
2. Choose a bot type suitable for your workflow (typically generic bot).
3. Copy:
   - Bot email (for example `annolid-bot@your-org.zulipchat.com`)
   - API key
   - Server URL (for example `https://your-org.zulipchat.com`)

## 2. Configure Annolid (`~/.annolid/agent/config.json`)

```json
{
  "tools": {
    "zulip": {
      "enabled": true,
      "server_url": "https://your-org.zulipchat.com",
      "user": "annolid-bot@your-org.zulipchat.com",
      "api_key": "YOUR_ZULIP_BOT_API_KEY",
      "stream": "annolid",
      "topic": "bot",
      "polling_interval": 30,
      "allow_from": ["alice@your-org.com"],
      "cursor_state_path": "~/.annolid/agent/channels/zulip_cursor_custom.json",
      "max_processed_ids": 4096,
      "log_skip_reasons": false,
      "bot_name": "Annolid Bot",
      "unread_backfill_enabled": true,
      "unread_backfill_on_empty_only": true,
      "unread_backfill_limit": 50,
      "unread_backfill_cooldown_s": 300
    }
  }
}
```

Notes:
- `stream`/`topic` are optional filters for inbound polling.
- `allow_from` is optional. If set, only matching senders are ingested.
- `polling_interval` has a minimum bound and defaults to 30 seconds.
- Annolid persists Zulip cursor state by default at
  `~/.annolid/agent/channels/zulip_cursor_<scope-hash>.json` to avoid replying to old
  messages after restart.
- `cursor_state_path` is optional and overrides the default state file location.
- `max_processed_ids` controls the in-memory/on-disk dedupe window (default `4096`,
  minimum `256`).
- `log_skip_reasons` enables debug logs for skipped messages (`duplicate`, `read`,
  `historical`, etc.) to diagnose ingestion behavior.
- `bot_name` controls mention matching when `topic` is configured. Messages from other
  topics are still ingested if they mention the bot.
- `unread_backfill_enabled` enables a second query for older unread messages.
- `unread_backfill_on_empty_only` runs unread backfill only when main polling returns no
  new messages.
- `unread_backfill_limit` sets how many unread messages are scanned per backfill query
  (default `50`, max `200`).
- `unread_backfill_cooldown_s` throttles unread backfill requests (default `300s`).

## 3. Start Annolid

1. Launch Annolid.
2. Open Annolid Bot.
3. Check logs for Zulip startup and polling messages.

## 3.1 Quick API connectivity check (optional)

Before starting Annolid, validate the bot credentials from terminal:

```bash
export ZULIP_SERVER_URL="https://your-org.zulipchat.com"
export ZULIP_USER="annolid-bot@your-org.zulipchat.com"
export ZULIP_API_KEY="YOUR_ZULIP_BOT_API_KEY"

curl -sS -u "${ZULIP_USER}:${ZULIP_API_KEY}" \
  "${ZULIP_SERVER_URL}/api/v1/users/me" | python -m json.tool
```

Expected:
- `"result": "success"`
- your bot identity in the response payload

If this fails, fix URL/user/key first, then continue with Annolid setup.

## 4. Message routing behavior

- Stream messages are mapped to `chat_id` format:
  - `stream:<stream_name>:<topic>`
- Private messages are mapped to:
  - `pm:<comma_separated_recipients>`
- Outbound replies use the same `chat_id` format and are sent back to stream/topic or PM recipients.

## 5. Troubleshooting

- If polling does not run:
  - verify `enabled=true`
  - verify `server_url`, `user`, and `api_key`
  - verify runtime logs include:
    - `Polling Zulip for ...`
    - `Zulip poll complete ...`
- If no inbound messages appear:
  - check `allow_from` filters
  - confirm the bot has permission to read the target stream/topic
  - if you expect older unread messages to be picked up quickly, lower
    `unread_backfill_cooldown_s` (for example to `30`).
  - inspect `skip_reasons=` in `Zulip poll complete ...` logs. Common values:
    - `allow_from_blocked`: sender not in allow list
    - `self_or_missing_sender`: bot's own message or missing sender
    - `read`: message already marked read
    - `historical`: pre-start history guard
- If outbound replies fail:
  - verify bot send permissions and API key validity
  - inspect Annolid logs for Zulip API error messages
