"""Bot-channel interface adapters."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "BaseChannel": "annolid.interfaces.bot.channels",
    "ChannelManager": "annolid.interfaces.bot.channels",
    "DiscordChannel": "annolid.interfaces.bot.channels",
    "EmailChannel": "annolid.interfaces.bot.channels",
    "SlackChannel": "annolid.interfaces.bot.channels",
    "TelegramChannel": "annolid.interfaces.bot.channels",
    "WhatsAppChannel": "annolid.interfaces.bot.channels",
    "WhatsAppPythonBridge": "annolid.interfaces.bot.channels",
    "WhatsAppWebhookServer": "annolid.interfaces.bot.channels",
    "ZulipChannel": "annolid.interfaces.bot.channels",
    "markdown_to_telegram_html": "annolid.interfaces.bot.channels",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name)
    return getattr(module, name)
