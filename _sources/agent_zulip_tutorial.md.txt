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
      "allow_from": ["alice@your-org.com"]
    }
  }
}
```

Notes:
- `stream`/`topic` are optional filters for inbound polling.
- `allow_from` is optional. If set, only matching senders are ingested.
- `polling_interval` has a minimum bound and defaults to 30 seconds.

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
- If no inbound messages appear:
  - check `allow_from` filters
  - confirm the bot has permission to read the target stream/topic
- If outbound replies fail:
  - verify bot send permissions and API key validity
  - inspect Annolid logs for Zulip API error messages
