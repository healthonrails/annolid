from __future__ import annotations

import asyncio
import importlib.util
import json

from annolid.core.agent.tools.calendar import GoogleCalendarTool


class _FakeEventsAPI:
    def __init__(self) -> None:
        self.last_op = ""
        self._payload = {}

    def list(self, **kwargs):
        self.last_op = "list"
        self._payload = kwargs
        return self

    def insert(self, **kwargs):
        self.last_op = "insert"
        self._payload = kwargs
        return self

    def get(self, **kwargs):
        self.last_op = "get"
        self._payload = kwargs
        return self

    def update(self, **kwargs):
        self.last_op = "update"
        self._payload = kwargs
        return self

    def delete(self, **kwargs):
        self.last_op = "delete"
        self._payload = kwargs
        return self

    def execute(self):
        if self.last_op == "list":
            return {"items": [{"id": "evt-1", "summary": "demo"}]}
        if self.last_op == "insert":
            body = dict(self._payload.get("body") or {})
            body.setdefault("id", "evt-created")
            return body
        if self.last_op == "get":
            return {"id": "evt-1", "summary": "old"}
        if self.last_op == "update":
            body = dict(self._payload.get("body") or {})
            body.setdefault("id", self._payload.get("eventId", "evt-1"))
            return body
        return {}


class _FakeCalendarService:
    def __init__(self) -> None:
        self.api = _FakeEventsAPI()

    def events(self):
        return self.api


def test_google_calendar_tool_missing_optional_packages() -> None:
    async def _run() -> None:
        tool = GoogleCalendarTool()
        tool._get_service = lambda: (_ for _ in ()).throw(ImportError("missing deps"))
        text = await tool.execute(action="list_events")
        assert "dependencies are not installed" in text.lower()

    asyncio.run(_run())


def test_google_calendar_is_available_handles_missing_google_namespace(
    monkeypatch,
) -> None:
    real_find_spec = importlib.util.find_spec

    def _fake_find_spec(name: str):
        if name == "google.auth.transport.requests":
            raise ModuleNotFoundError("No module named 'google.auth'")
        return real_find_spec(name)

    monkeypatch.setattr(importlib.util, "find_spec", _fake_find_spec)
    assert GoogleCalendarTool.is_available() is False


def test_google_calendar_tool_list_and_create() -> None:
    async def _run() -> None:
        tool = GoogleCalendarTool()
        tool._get_service = lambda: _FakeCalendarService()
        listed = await tool.execute(action="list_events", max_results=5)
        listed_payload = json.loads(listed)
        assert listed_payload["count"] == 1

        created = await tool.execute(
            action="create_event",
            summary="Team sync",
            start_time="2026-03-01T10:00:00+00:00",
            end_time="2026-03-01T10:30:00+00:00",
            recurrence_rule="weekly",
        )
        created_payload = json.loads(created)
        assert created_payload["created"]["summary"] == "Team sync"
        assert created_payload["created"]["recurrence"] == ["RRULE:FREQ=WEEKLY"]

    asyncio.run(_run())


def test_google_calendar_tool_update_delete_validation() -> None:
    async def _run() -> None:
        tool = GoogleCalendarTool()
        tool._get_service = lambda: _FakeCalendarService()
        missing = await tool.execute(action="update_event")
        assert "requires `event_id`" in missing

        updated = await tool.execute(
            action="update_event",
            event_id="evt-1",
            summary="Updated",
            recurrence_rule="RRULE:FREQ=WEEKLY;BYDAY=MO",
        )
        updated_payload = json.loads(updated)
        assert updated_payload["updated"]["summary"] == "Updated"
        assert updated_payload["updated"]["recurrence"] == [
            "RRULE:FREQ=WEEKLY;BYDAY=MO"
        ]

        deleted = await tool.execute(action="delete_event", event_id="evt-1")
        deleted_payload = json.loads(deleted)
        assert deleted_payload["deleted_event_id"] == "evt-1"

    asyncio.run(_run())


def test_google_calendar_tool_invalid_action_does_not_initialize_service() -> None:
    async def _run() -> None:
        tool = GoogleCalendarTool()
        called = {"value": False}

        def _fail_service():
            called["value"] = True
            raise AssertionError("service should not be initialized")

        tool._get_service = _fail_service
        result = await tool.execute(action="unknown_action")
        assert "unsupported action" in result.lower()
        assert called["value"] is False

    asyncio.run(_run())


def test_google_calendar_tool_update_requires_fields() -> None:
    async def _run() -> None:
        tool = GoogleCalendarTool()
        tool._get_service = lambda: _FakeCalendarService()
        result = await tool.execute(action="update_event", event_id="evt-1")
        assert "at least one field to update" in result.lower()

    asyncio.run(_run())
