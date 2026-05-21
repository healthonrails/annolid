import asyncio

from annolid.core.agent.config.schema import CalendarToolConfig
from annolid.core.agent.tools.function_registry import FunctionToolRegistry
from annolid.core.agent.tools import nanobot as nanobot_mod


class _DummyCalendarTool:
    name = "google_calendar"

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def preflight(**kwargs):
        return True, "ok"


def test_register_nanobot_style_tools_logs_calendar_registration_once(monkeypatch):
    info_messages = []

    def _capture_info(message, *args, **kwargs):
        del kwargs
        if args:
            message = str(message) % args
        info_messages.append(str(message))

    monkeypatch.setattr(nanobot_mod, "GoogleCalendarTool", _DummyCalendarTool)
    monkeypatch.setattr(nanobot_mod.logger, "info", _capture_info)
    nanobot_mod._INFO_LOG_ONCE_KEYS.clear()

    registry = FunctionToolRegistry()
    cfg = CalendarToolConfig(enabled=True)

    asyncio.run(register_once(registry, cfg))
    asyncio.run(register_once(registry, cfg))

    matching = [
        msg
        for msg in info_messages
        if "Google Calendar tool registered with Google OAuth backend." in msg
    ]
    assert len(matching) == 1


async def register_once(registry, cfg):
    await nanobot_mod.register_nanobot_style_tools(
        registry,
        calendar_cfg=cfg,
    )
