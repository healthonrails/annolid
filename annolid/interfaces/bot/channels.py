"""Bot-channel interface adapters."""

from annolid.core.agent.channels import (
    BaseChannel,
    ChannelManager,
    DiscordChannel,
    EmailChannel,
    SlackChannel,
    TelegramChannel,
    WhatsAppChannel,
    WhatsAppPythonBridge,
    WhatsAppWebhookServer,
    ZulipChannel,
    markdown_to_telegram_html,
)

__all__ = [
    "BaseChannel",
    "ChannelManager",
    "DiscordChannel",
    "EmailChannel",
    "SlackChannel",
    "TelegramChannel",
    "WhatsAppChannel",
    "WhatsAppPythonBridge",
    "WhatsAppWebhookServer",
    "ZulipChannel",
    "markdown_to_telegram_html",
]
