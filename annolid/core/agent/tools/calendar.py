from __future__ import annotations

import asyncio
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from .function_base import FunctionTool


class GoogleCalendarTool(FunctionTool):
    """Google Calendar tool with OAuth-based local token caching."""

    _SCOPES = ["https://www.googleapis.com/auth/calendar"]
    _ACTIONS = {
        "list_events",
        "create_event",
        "update_event",
        "delete_event",
    }

    def __init__(
        self,
        *,
        credentials_file: str = "~/.annolid/agent/google_calendar_credentials.json",
        token_file: str = "~/.annolid/agent/google_calendar_token.json",
        calendar_id: str = "primary",
        timezone_name: str = "",
        default_event_duration_minutes: int = 30,
    ) -> None:
        self._credentials_file = str(credentials_file or "").strip()
        self._token_file = str(token_file or "").strip()
        self._calendar_id = str(calendar_id or "primary").strip() or "primary"
        self._timezone_name = str(timezone_name or "").strip()
        self._default_event_duration_minutes = max(
            1, int(default_event_duration_minutes or 30)
        )
        self._service_cache: Any = None

    @property
    def name(self) -> str:
        return "google_calendar"

    @property
    def description(self) -> str:
        return (
            "Manage Google Calendar events (list/create/update/delete) for the "
            "configured calendar."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": sorted(self._ACTIONS),
                },
                "event_id": {"type": "string"},
                "summary": {"type": "string"},
                "description": {"type": "string"},
                "location": {"type": "string"},
                "start_time": {
                    "type": "string",
                    "description": "ISO-8601 datetime (e.g. 2026-03-01T10:00:00-08:00)",
                },
                "end_time": {
                    "type": "string",
                    "description": "ISO-8601 datetime; optional for create/update",
                },
                "max_results": {"type": "integer", "minimum": 1, "maximum": 50},
                "recurrence_rule": {
                    "type": "string",
                    "description": (
                        "Recurrence rule, e.g. RRULE:FREQ=WEEKLY, or shortcut: "
                        "daily|weekly|monthly|yearly"
                    ),
                },
                "recurrence": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        'Google recurrence lines, e.g. ["RRULE:FREQ=WEEKLY;BYDAY=MO"]'
                    ),
                },
                "clear_recurrence": {
                    "type": "boolean",
                    "description": "Set true in update_event to remove recurrence.",
                },
            },
            "required": ["action"],
        }

    async def execute(self, action: str, **kwargs: Any) -> str:
        action_name = str(action or "").strip().lower()
        if action_name not in self._ACTIONS:
            return (
                "Error: Unsupported action. Use one of: "
                "list_events, create_event, update_event, delete_event."
            )

        # Validate required inputs before OAuth/service initialization.
        if action_name == "create_event":
            summary = str(kwargs.get("summary", "") or "").strip()
            start_time = str(kwargs.get("start_time", "") or "").strip()
            if not summary or not start_time:
                return "Error: create_event requires `summary` and `start_time`."
        if action_name in {"update_event", "delete_event"}:
            event_id = str(kwargs.get("event_id", "") or "").strip()
            if not event_id:
                return f"Error: {action_name} requires `event_id`."

        try:
            service = await asyncio.to_thread(self._get_service)
        except ImportError as exc:
            return (
                "Error: Google Calendar dependencies are not installed. "
                'Install optional extras with `pip install "annolid[google_calendar]"`. '
                f"Details: {exc}"
            )
        except Exception as exc:
            return f"Error: Failed to initialize Google Calendar service: {exc}"

        if action_name == "list_events":
            max_results = int(kwargs.get("max_results", 10) or 10)
            return await asyncio.to_thread(self._list_events, service, max_results)
        if action_name == "create_event":
            return await asyncio.to_thread(
                self._create_event,
                service,
                str(kwargs.get("summary", "") or "").strip(),
                str(kwargs.get("start_time", "") or "").strip(),
                str(kwargs.get("end_time", "") or "").strip(),
                str(kwargs.get("description", "") or "").strip(),
                str(kwargs.get("location", "") or "").strip(),
                str(kwargs.get("recurrence_rule", "") or "").strip(),
                kwargs.get("recurrence"),
            )
        if action_name == "update_event":
            return await asyncio.to_thread(
                self._update_event,
                service,
                str(kwargs.get("event_id", "") or "").strip(),
                kwargs,
            )
        if action_name == "delete_event":
            return await asyncio.to_thread(
                self._delete_event,
                service,
                str(kwargs.get("event_id", "") or "").strip(),
            )
        return "Error: Unsupported action."

    @classmethod
    def is_available(cls) -> bool:
        required_modules = (
            "google.auth.transport.requests",
            "google.oauth2.credentials",
            "google_auth_oauthlib.flow",
            "googleapiclient.discovery",
        )
        for name in required_modules:
            try:
                if importlib.util.find_spec(name) is None:
                    return False
            except (ImportError, ModuleNotFoundError):
                return False
        return True

    def _import_google_modules(self) -> tuple[Any, Any, Any, Any]:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build

        return Request, Credentials, InstalledAppFlow, build

    def _get_service(self) -> Any:
        if self._service_cache is not None:
            return self._service_cache
        Request, Credentials, InstalledAppFlow, build = self._import_google_modules()
        creds = None
        token_path = Path(self._token_file).expanduser()
        credentials_path = Path(self._credentials_file).expanduser()
        if token_path.exists():
            creds = Credentials.from_authorized_user_file(str(token_path), self._SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not credentials_path.exists():
                    raise FileNotFoundError(
                        "Google Calendar credentials file not found at "
                        f"{credentials_path}"
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(credentials_path), self._SCOPES
                )
                creds = flow.run_local_server(port=0)
            token_path.parent.mkdir(parents=True, exist_ok=True)
            token_path.write_text(creds.to_json(), encoding="utf-8")
            try:
                token_path.chmod(0o600)
            except OSError:
                pass
        self._service_cache = build("calendar", "v3", credentials=creds)
        return self._service_cache

    def _list_events(self, service: Any, max_results: int) -> str:
        now = datetime.now(timezone.utc).isoformat()
        result = (
            service.events()
            .list(
                calendarId=self._calendar_id,
                timeMin=now,
                maxResults=max(1, min(50, int(max_results))),
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        events = result.get("items", [])
        return json.dumps({"count": len(events), "events": events}, ensure_ascii=True)

    def _create_event(
        self,
        service: Any,
        summary: str,
        start_time: str,
        end_time: str,
        description: str,
        location: str,
        recurrence_rule: str,
        recurrence_items: Any,
    ) -> str:
        start_dt = self._parse_iso_datetime(start_time)
        end_dt = (
            self._parse_iso_datetime(end_time)
            if end_time
            else start_dt + timedelta(minutes=self._default_event_duration_minutes)
        )
        event_body: dict[str, Any] = {
            "summary": summary,
            "start": self._format_event_time(start_dt),
            "end": self._format_event_time(end_dt),
        }
        if description:
            event_body["description"] = description
        if location:
            event_body["location"] = location
        recurrence = self._normalize_recurrence(
            recurrence_rule=recurrence_rule,
            recurrence_items=recurrence_items,
        )
        if recurrence:
            event_body["recurrence"] = recurrence
        created = (
            service.events()
            .insert(calendarId=self._calendar_id, body=event_body)
            .execute()
        )
        return json.dumps({"created": created}, ensure_ascii=True)

    def _update_event(
        self, service: Any, event_id: str, updates: dict[str, Any]
    ) -> str:
        event = (
            service.events()
            .get(calendarId=self._calendar_id, eventId=event_id)
            .execute()
        )
        summary = str(updates.get("summary", "") or "").strip()
        description = str(updates.get("description", "") or "").strip()
        location = str(updates.get("location", "") or "").strip()
        start_time = str(updates.get("start_time", "") or "").strip()
        end_time = str(updates.get("end_time", "") or "").strip()
        recurrence_rule = str(updates.get("recurrence_rule", "") or "").strip()
        recurrence_items = updates.get("recurrence")
        clear_recurrence = bool(updates.get("clear_recurrence", False))
        if summary:
            event["summary"] = summary
        if description:
            event["description"] = description
        if location:
            event["location"] = location
        if start_time:
            event["start"] = self._format_event_time(
                self._parse_iso_datetime(start_time)
            )
        if end_time:
            event["end"] = self._format_event_time(self._parse_iso_datetime(end_time))
        recurrence = self._normalize_recurrence(
            recurrence_rule=recurrence_rule,
            recurrence_items=recurrence_items,
        )
        if clear_recurrence:
            event.pop("recurrence", None)
        elif recurrence:
            event["recurrence"] = recurrence
        if not any([summary, description, location, start_time, end_time]):
            if not clear_recurrence and not recurrence:
                return "Error: update_event requires at least one field to update."
        # if recurrence requested but normalization failed, surface error clearly
        if (recurrence_rule or recurrence_items) and not recurrence:
            return (
                "Error: recurrence format is invalid. Use RRULE:FREQ=... or "
                "shortcut daily|weekly|monthly|yearly."
            )
        updated = (
            service.events()
            .update(calendarId=self._calendar_id, eventId=event_id, body=event)
            .execute()
        )
        return json.dumps({"updated": updated}, ensure_ascii=True)

    def _delete_event(self, service: Any, event_id: str) -> str:
        service.events().delete(
            calendarId=self._calendar_id, eventId=event_id
        ).execute()
        return json.dumps({"deleted_event_id": event_id}, ensure_ascii=True)

    def _parse_iso_datetime(self, value: str) -> datetime:
        text = str(value or "").strip()
        if not text:
            raise ValueError("datetime value is required")
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    def _format_event_time(self, dt: datetime) -> dict[str, str]:
        payload = {"dateTime": dt.isoformat()}
        if self._timezone_name:
            try:
                ZoneInfo(self._timezone_name)
                payload["timeZone"] = self._timezone_name
            except Exception:
                pass
        return payload

    def _normalize_recurrence(
        self, *, recurrence_rule: str, recurrence_items: Any
    ) -> list[str]:
        entries: list[str] = []
        rule = str(recurrence_rule or "").strip()
        if rule:
            mapped = self._normalize_recurrence_line(rule)
            if not mapped:
                return []
            entries.append(mapped)
        if isinstance(recurrence_items, (list, tuple)):
            for item in recurrence_items:
                line = str(item or "").strip()
                if not line:
                    continue
                mapped = self._normalize_recurrence_line(line)
                if not mapped:
                    return []
                entries.append(mapped)
        deduped: list[str] = []
        for line in entries:
            if line not in deduped:
                deduped.append(line)
        return deduped

    def _normalize_recurrence_line(self, line: str) -> str:
        value = str(line or "").strip()
        if not value:
            return ""
        shortcuts = {
            "daily": "RRULE:FREQ=DAILY",
            "weekly": "RRULE:FREQ=WEEKLY",
            "monthly": "RRULE:FREQ=MONTHLY",
            "yearly": "RRULE:FREQ=YEARLY",
        }
        lowered = value.lower()
        if lowered in shortcuts:
            return shortcuts[lowered]
        upper = value.upper()
        if upper.startswith(("RRULE:", "EXRULE:", "RDATE:", "EXDATE:")):
            return upper
        return ""
