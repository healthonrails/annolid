from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from unittest.mock import patch

import pytest

from annolid.core.agent.bus import MessageBus
from annolid.core.agent.channels.whatsapp import WhatsAppChannel
from annolid.core.agent.channels.whatsapp_webhook_server import WhatsAppWebhookServer


def _webhook_payload() -> dict:
    return {
        "entry": [
            {
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "metadata": {"phone_number_id": "123"},
                            "messages": [
                                {
                                    "id": "wamid.TEST",
                                    "from": "15551234567",
                                    "type": "text",
                                    "text": {"body": "hello"},
                                }
                            ],
                        },
                    }
                ]
            }
        ]
    }


def test_whatsapp_webhook_server_lifecycle_and_ingest() -> None:
    class _FakeHttpServer:
        def __init__(self, addr, handler_cls):
            del addr, handler_cls
            self.server_port = 18081

        def serve_forever(self) -> None:
            return

        def shutdown(self) -> None:
            return

        def server_close(self) -> None:
            return

    bus = MessageBus()
    channel = WhatsAppChannel(
        {
            "verify_token": "verify-123",
            "app_secret": "meta-app-secret",
        },
        bus,
    )
    server = WhatsAppWebhookServer(channel=channel, port=0)

    with patch(
        "annolid.core.agent.channels.whatsapp_webhook_server.ThreadingHTTPServer",
        _FakeHttpServer,
    ):
        url = server.start()
        assert url == "http://127.0.0.1:18081/whatsapp/webhook"

    ingested = server._run_ingest(_webhook_payload())
    assert ingested == 1

    async def _consume():
        return await bus.consume_inbound(timeout_s=0.5)

    inbound = asyncio.run(_consume())
    assert inbound.channel == "whatsapp"
    assert inbound.content == "hello"
    assert inbound.sender_id == "15551234567"
    server.stop()


def test_whatsapp_webhook_rejects_start_without_app_secret() -> None:
    bus = MessageBus()
    channel = WhatsAppChannel({"verify_token": "verify-123"}, bus)
    server = WhatsAppWebhookServer(channel=channel, port=0)

    with pytest.raises(RuntimeError, match="unsigned WhatsApp webhook"):
        server.start()


def test_whatsapp_webhook_requires_valid_post_signature() -> None:
    app_secret = "meta-app-secret"
    bus = MessageBus()
    channel = WhatsAppChannel(
        {"verify_token": "verify-123", "app_secret": app_secret},
        bus,
    )
    server = WhatsAppWebhookServer(channel=channel, port=0)
    body = json.dumps(_webhook_payload()).encode("utf-8")

    assert server._is_valid_post_signature(body, "sha256=invalid") is False

    signature = (
        "sha256="
        + hmac.new(
            app_secret.encode("utf-8"),
            body,
            hashlib.sha256,
        ).hexdigest()
    )
    assert server._is_valid_post_signature(body, signature) is True
    assert server._is_valid_post_signature(body + b" ", signature) is False
