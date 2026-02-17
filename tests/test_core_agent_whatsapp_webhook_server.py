from __future__ import annotations

import asyncio
from unittest.mock import patch

from annolid.core.agent.bus import MessageBus
from annolid.core.agent.channels.whatsapp import WhatsAppChannel
from annolid.core.agent.channels.whatsapp_webhook_server import WhatsAppWebhookServer


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
    channel = WhatsAppChannel({"verify_token": "verify-123"}, bus)
    server = WhatsAppWebhookServer(channel=channel, port=0)

    with patch(
        "annolid.core.agent.channels.whatsapp_webhook_server.ThreadingHTTPServer",
        _FakeHttpServer,
    ):
        url = server.start()
        assert url == "http://127.0.0.1:18081/whatsapp/webhook"

    payload = {
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
    ingested = server._run_ingest(payload)
    assert ingested == 1

    async def _consume():
        return await bus.consume_inbound(timeout_s=0.5)

    inbound = asyncio.run(_consume())
    assert inbound.channel == "whatsapp"
    assert inbound.content == "hello"
    assert inbound.sender_id == "15551234567"
    server.stop()
