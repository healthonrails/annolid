# Annolid Bot WhatsApp Setup Tutorial

This tutorial covers two ways to connect WhatsApp:

- **Default (safer local): Embedded QR bridge**
- **Optional (production/public webhook): WhatsApp Cloud API**

## 1. Default: Embedded QR bridge mode

QR bridge keeps traffic local and avoids exposing a public webhook:

- no public callback URL required
- no Meta developer token required
- local-only bridge on `127.0.0.1`

Install optional dependencies:

```bash
pip install "annolid[whatsapp]"
python -m playwright install chromium
```

Configure `~/.annolid/agent/config.json`:

```json
{
  "tools": {
    "whatsapp": {
      "enabled": true,
      "autoStart": true,
      "bridgeMode": "python",
      "bridgeHost": "127.0.0.1",
      "bridgePort": 3001,
      "bridgeSessionDir": "~/.annolid/whatsapp-web-session",
      "bridgeHeadless": false,
      "ingestOutgoingMessages": false,
      "allowFrom": []
    }
  }
}
```

Notes:

- `allowFrom` is optional. Keep it empty for first-run testing.
- After first inbound message is logged, set `allowFrom` to trusted senders only.
- Use full E.164 numbers in `allowFrom` (country code + number, no `+`).
- Keep `bridgeHeadless=false` so QR can be scanned.
- `ingestOutgoingMessages` defaults to `false` to prevent self-reply loops.
- Set `autoStart=false` if you want to keep WhatsApp config but not start bridge/webhook/channel at app launch.

## 2. QR flow

1. Start Annolid and open Annolid Bot.
2. Scan WhatsApp Web QR:
   WhatsApp -> Settings -> Linked Devices -> Link a Device
3. Send a test message.

## 3. No-reply troubleshooting (QR bridge)

If WhatsApp shows connected but the bot does not answer:

1. Confirm inbound detection in logs:
   - expect: `WhatsApp Playwright detected message ...`
   - expect: `WhatsApp bridge message detected ...`
2. If you only see `polling active; no new messages detected in the last 60s`:
   - reopen the target chat in WhatsApp Web
   - send a new plain text message
3. If you see `rejected by allow_from`:
   - clear `allowFrom` temporarily, retest, then add only trusted numbers
4. Self-chat (`Message yourself`) is supported, but make sure the self chat is the active WhatsApp Web chat.
5. Verify dependency/runtime:
   - `pip install "annolid[whatsapp]"`
   - `python -m playwright install chromium`

## 4. Optional: Cloud API mode

Use Cloud API when you need enterprise reliability and can manage public webhook exposure.

```json
{
  "tools": {
    "whatsapp": {
      "enabled": true,
      "bridgeMode": "external",
      "bridgeUrl": "",
      "accessToken": "EAAG....",
      "phoneNumberId": "123456789012345",
      "verifyToken": "your-verify-token",
      "apiVersion": "v22.0",
      "apiBase": "https://graph.facebook.com",
      "webhookEnabled": true,
      "webhookHost": "0.0.0.0",
      "webhookPort": 18081,
      "webhookPath": "/whatsapp/webhook",
      "allowFrom": ["15551234567"]
    }
  }
}
```

Business number setup (recommended for production):

1. In Meta Business Manager, create/select your business portfolio.
2. In WhatsApp Manager, add or migrate your business phone number.
3. In Meta App dashboard:
   - copy temporary/permanent `accessToken`
   - copy `phoneNumberId`
   - set your own `verifyToken`
4. Expose webhook URL publicly over HTTPS (for example via reverse proxy or tunnel).
5. Configure webhook in Meta:
   - callback URL: your public `https://.../whatsapp/webhook`
   - verify token: same as `verifyToken` in Annolid config
   - subscribe to `messages` field
6. In Annolid config:
   - set `bridgeMode` to `external`
   - leave `bridgeUrl` empty
   - set `webhookEnabled=true`
   - set `accessToken`, `phoneNumberId`, `verifyToken`
7. Send a real inbound message from another WhatsApp account (not self-chat).

## 5. Optional: External bridge mode

If you run your own compatible bridge service:

```json
{
  "tools": {
    "whatsapp": {
      "enabled": true,
      "bridgeMode": "external",
      "bridgeUrl": "ws://127.0.0.1:3001",
      "bridgeToken": ""
    }
  }
}
```

## 6. Security checklist

- Keep tokens out of source control.
- Use a strong `verifyToken`.
- Keep bridge host on localhost unless remote access is required.
- Restrict webhook endpoint to HTTPS.
- Rotate access tokens regularly.
