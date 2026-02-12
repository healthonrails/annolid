"""Channel adapters for bus-based agent messaging."""

from .base import BaseChannel
from .discord import DiscordChannel
from .email import EmailChannel
from .manager import ChannelManager
from .slack import SlackChannel
from .telegram import TelegramChannel, markdown_to_telegram_html
from .whatsapp import WhatsAppChannel

__all__ = [
    "BaseChannel",
    "ChannelManager",
    "TelegramChannel",
    "DiscordChannel",
    "SlackChannel",
    "EmailChannel",
    "WhatsAppChannel",
    "markdown_to_telegram_html",
]
