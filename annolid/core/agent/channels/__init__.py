"""Channel adapters for bus-based agent messaging."""

from .base import BaseChannel
from .discord import DiscordChannel
from .email import EmailChannel
from .manager import ChannelManager
from .slack import SlackChannel
from .telegram import TelegramChannel, markdown_to_telegram_html
from .whatsapp import WhatsAppChannel
from .whatsapp_python_bridge import WhatsAppPythonBridge
from .whatsapp_webhook_server import WhatsAppWebhookServer

__all__ = [
    "BaseChannel",
    "ChannelManager",
    "TelegramChannel",
    "DiscordChannel",
    "SlackChannel",
    "EmailChannel",
    "WhatsAppChannel",
    "WhatsAppPythonBridge",
    "WhatsAppWebhookServer",
    "markdown_to_telegram_html",
]
