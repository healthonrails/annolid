from __future__ import annotations

import json

from annolid.core.agent.gui_backend import session_io


def test_outbound_chat_event_roundtrip() -> None:
    event = session_io.OutboundChatEvent(
        direction="outbound",
        kind="final",
        text="done",
        is_error=False,
    )
    payload = session_io.encode_outbound_chat_event(event)
    decoded = session_io.decode_outbound_chat_event(payload)
    assert decoded is not None
    assert decoded.direction == "outbound"
    assert decoded.kind == "final"
    assert decoded.text == "done"
    assert decoded.is_error is False


def test_emit_chunk_prefers_unified_consumer(monkeypatch) -> None:
    calls: list[str] = []

    def _fake_invoke(_obj, member, *_args):
        calls.append(str(member))
        return True

    monkeypatch.setattr(session_io.QMetaObject, "invokeMethod", _fake_invoke)

    class _Widget:
        def consume_outbound_chat_event(self, _payload: str) -> None:
            return None

    session_io.emit_chunk(widget=_Widget(), chunk="hello")
    assert calls == ["consume_outbound_chat_event"]


def test_emit_chunk_prefers_bus_enqueue_slot(monkeypatch) -> None:
    calls: list[str] = []

    def _fake_invoke(_obj, member, *_args):
        calls.append(str(member))
        return True

    monkeypatch.setattr(session_io.QMetaObject, "invokeMethod", _fake_invoke)

    class _Widget:
        def enqueue_outbound_bus_message(self, _payload: str) -> None:
            return None

        def consume_outbound_chat_event(self, _payload: str) -> None:
            return None

    session_io.emit_chunk(widget=_Widget(), chunk="hello")
    assert calls == ["enqueue_outbound_bus_message"]


def test_emit_chunk_falls_back_when_bus_enqueue_invoke_fails(monkeypatch) -> None:
    calls: list[str] = []
    count = {"n": 0}

    def _fake_invoke(_obj, member, *_args):
        calls.append(str(member))
        count["n"] += 1
        if count["n"] == 1:
            return False
        return True

    monkeypatch.setattr(session_io.QMetaObject, "invokeMethod", _fake_invoke)

    class _Widget:
        def enqueue_outbound_bus_message(self, _payload: str) -> None:
            return None

        def consume_outbound_chat_event(self, _payload: str) -> None:
            return None

    session_io.emit_chunk(widget=_Widget(), chunk="hello")
    assert calls == ["enqueue_outbound_bus_message", "consume_outbound_chat_event"]


def test_emit_chunk_falls_back_to_legacy_slot(monkeypatch) -> None:
    calls: list[str] = []

    def _fake_invoke(_obj, member, *_args):
        calls.append(str(member))
        return True

    monkeypatch.setattr(session_io.QMetaObject, "invokeMethod", _fake_invoke)

    class _Widget:
        pass

    session_io.emit_chunk(widget=_Widget(), chunk="hello")
    assert calls == ["stream_chat_chunk"]


def test_decode_outbound_event_rejects_unknown_kind() -> None:
    payload = json.dumps(
        {
            "direction": "outbound",
            "kind": "unknown",
            "text": "x",
            "is_error": False,
        }
    )
    assert session_io.decode_outbound_chat_event(payload) is None
