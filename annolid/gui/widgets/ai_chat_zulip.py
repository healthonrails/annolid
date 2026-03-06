from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


DEFAULT_ZULIP_TOPIC = "annolid"


@dataclass(frozen=True)
class ZulipDraftTarget:
    chat_id: str
    summary: str
    target_type: str


def _unique_recipients(raw_values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    recipients: list[str] = []
    for raw in raw_values:
        value = str(raw or "").strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        recipients.append(value)
    return recipients


def parse_zulip_recipients(value: str) -> list[str]:
    text = str(value or "")
    return _unique_recipients(text.replace(";", ",").split(","))


def build_zulip_draft_target(
    target_type: str,
    *,
    stream: str = "",
    topic: str = "",
    recipients: str = "",
    default_stream: str = "",
    default_topic: str = DEFAULT_ZULIP_TOPIC,
) -> ZulipDraftTarget:
    kind = str(target_type or "stream").strip().lower()
    if kind == "dm":
        normalized = parse_zulip_recipients(recipients)
        if not normalized:
            raise ValueError("Add at least one Zulip recipient email.")
        label = ", ".join(normalized)
        return ZulipDraftTarget(
            chat_id=f"pm:{','.join(normalized)}",
            summary=f"DM {label}",
            target_type="dm",
        )

    stream_name = str(stream or default_stream or "").strip()
    if not stream_name:
        raise ValueError("Choose a Zulip stream.")
    topic_name = str(topic or default_topic or DEFAULT_ZULIP_TOPIC).strip()
    if not topic_name:
        topic_name = DEFAULT_ZULIP_TOPIC
    return ZulipDraftTarget(
        chat_id=f"stream:{stream_name}:{topic_name}",
        summary=f"Stream #{stream_name} > {topic_name}",
        target_type="stream",
    )


def missing_zulip_config_fields(config: Mapping[str, Any] | None) -> list[str]:
    payload = dict(config or {})
    missing: list[str] = []
    if not bool(payload.get("enabled", False)):
        missing.append("enabled")
    for key in ("server_url", "user", "api_key"):
        if not str(payload.get(key) or "").strip():
            missing.append(key)
    return missing
