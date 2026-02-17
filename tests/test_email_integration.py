from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch


from annolid.core.agent.bus import MessageBus, OutboundMessage
from annolid.core.agent.channels.email import EmailChannel
from annolid.core.agent.tools.email import EmailTool


def test_email_channel_poll_interval_resolution_uses_config_and_env(monkeypatch):
    bus = MessageBus()
    channel = EmailChannel({"polling_interval": 300}, bus)

    monkeypatch.delenv("ANNOLID_EMAIL_POLL_INTERVAL_SECONDS", raising=False)
    monkeypatch.delenv("NANOBOT_EMAIL_POLL_INTERVAL_SECONDS", raising=False)
    assert channel._resolve_poll_interval_seconds() == 300

    monkeypatch.setenv("ANNOLID_EMAIL_POLL_INTERVAL_SECONDS", "45")
    assert channel._resolve_poll_interval_seconds() == 45

    monkeypatch.delenv("ANNOLID_EMAIL_POLL_INTERVAL_SECONDS", raising=False)
    monkeypatch.setenv("NANOBOT_EMAIL_POLL_INTERVAL_SECONDS", "75")
    assert channel._resolve_poll_interval_seconds() == 75


def test_email_channel_poll_interval_resolution_clamps_invalid_values(monkeypatch):
    bus = MessageBus()
    channel = EmailChannel({"polling_interval": "bad-value"}, bus)

    monkeypatch.delenv("ANNOLID_EMAIL_POLL_INTERVAL_SECONDS", raising=False)
    monkeypatch.delenv("NANOBOT_EMAIL_POLL_INTERVAL_SECONDS", raising=False)
    assert channel._resolve_poll_interval_seconds() == 300

    monkeypatch.setenv("ANNOLID_EMAIL_POLL_INTERVAL_SECONDS", "1")
    assert channel._resolve_poll_interval_seconds() == 10


def test_email_tool_send_mocked():
    """Test EmailTool using mocked smtplib."""

    async def _run():
        tool = EmailTool(
            smtp_host="localhost",
            smtp_port=587,
            user="bot@example.com",
            password="password",
        )

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            result = await tool.execute(
                to="user@example.com", content="Hello world", subject="Test Subject"
            )

            assert "successfully sent" in result
            mock_server.starttls.assert_called_once()
            mock_server.login.assert_called_once_with("bot@example.com", "password")
            mock_server.send_message.assert_called_once()

    asyncio.run(_run())


def test_email_channel_send_mocked():
    """Test EmailChannel outbound sending using mocked smtplib."""

    async def _run():
        bus = MessageBus()
        config = {
            "smtp_host": "localhost",
            "smtp_port": 587,
            "user": "bot@example.com",
            "password": "password",
        }
        channel = EmailChannel(config, bus)

        with patch("smtplib.SMTP") as mock_smtp:
            mock_server = MagicMock()
            mock_smtp.return_value.__enter__.return_value = mock_server

            msg = OutboundMessage(
                channel="email",
                chat_id="user@example.com",
                content="Outbound message",
                metadata={"subject": "Outbound Test"},
            )

            await channel.send(msg)

            # Wait a bit for thread to finish
            await asyncio.sleep(0.1)

            mock_server.send_message.assert_called_once()

    asyncio.run(_run())


def test_email_channel_send_skips_empty_content():
    async def _run():
        bus = MessageBus()
        channel = EmailChannel({}, bus)

        msg = OutboundMessage(
            channel="email",
            chat_id="user@example.com",
            content="   ",
            metadata={"subject": "Empty"},
        )

        with patch("smtplib.SMTP") as mock_smtp:
            await channel.send(msg)
            mock_smtp.assert_not_called()

    asyncio.run(_run())


def test_email_channel_imap_polling_mocked():
    """Test EmailChannel IMAP polling and ingestion."""

    async def _run():
        bus = MessageBus()
        config = {
            "imap_host": "localhost",
            "imap_port": 993,
            "user": "bot@example.com",
            "password": "password",
            "polling_interval": 0.1,
        }
        channel = EmailChannel(config, bus)

        # Create a mock email message
        mock_msg = MagicMock()
        mock_msg.is_multipart.return_value = False
        # mock_msg.get_payload.return_value = b"Incoming email body"
        # We need to simulate the return of get_payload(decode=True)
        # In the code: body = msg.get_payload(decode=True).decode()
        mock_msg.get_payload.return_value = b"Incoming email body"

        # Mock get for From and Subject
        def mock_get(key, default=None):
            mapping = {"From": "sender@example.com", "Subject": "Hello Bot"}
            return mapping.get(key, default)

        mock_msg.get.side_effect = mock_get

        with patch("imaplib.IMAP4_SSL") as mock_imap:
            mock_server = MagicMock()
            mock_imap.return_value = mock_server
            mock_server.login.return_value = ("OK", [b"Logged in"])
            mock_server.select.return_value = ("OK", [b"1"])
            mock_server.search.return_value = ("OK", [b"1"])
            mock_server.fetch.return_value = ("OK", [(b"1", b"RFC822 content")])

            with patch("email.message_from_bytes") as mock_from_bytes:
                mock_from_bytes.return_value = mock_msg

                await channel.start()

                # Wait for poll loop to run at least once
                inbound = await bus.consume_inbound(timeout_s=2.0)

                assert inbound is not None
                assert "Incoming email body" in inbound.content
                assert inbound.sender_id == "sender@example.com"

                await channel.stop()

    asyncio.run(_run())
