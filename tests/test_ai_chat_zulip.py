from __future__ import annotations

import pytest

from annolid.gui.widgets.ai_chat_zulip import (
    build_zulip_draft_target,
    missing_zulip_config_fields,
    parse_zulip_recipients,
)


def test_parse_zulip_recipients_normalizes_and_deduplicates() -> None:
    recipients = parse_zulip_recipients(
        " Alice@example.com ;bob@example.com, alice@example.com ,,"
    )
    assert recipients == ["alice@example.com", "bob@example.com"]


def test_build_zulip_stream_target_uses_defaults() -> None:
    target = build_zulip_draft_target(
        "stream",
        default_stream="annolid",
        default_topic="bot-ui",
    )
    assert target.chat_id == "stream:annolid:bot-ui"
    assert target.summary == "Stream #annolid > bot-ui"
    assert target.target_type == "stream"


def test_build_zulip_dm_target_requires_recipient() -> None:
    with pytest.raises(ValueError, match="recipient"):
        build_zulip_draft_target("dm", recipients="")


def test_build_zulip_dm_target_builds_chat_id() -> None:
    target = build_zulip_draft_target(
        "dm",
        recipients="alice@example.com, Bob@example.com",
    )
    assert target.chat_id == "pm:alice@example.com,bob@example.com"
    assert target.summary == "DM alice@example.com, bob@example.com"
    assert target.target_type == "dm"


def test_missing_zulip_config_fields_reports_required_values() -> None:
    missing = missing_zulip_config_fields(
        {
            "enabled": False,
            "server_url": "",
            "user": "annolid-bot@example.com",
            "api_key": "",
        }
    )
    assert missing == ["enabled", "server_url", "api_key"]
