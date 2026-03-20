from __future__ import annotations

from annolid.services.chat_session import (
    clear_chat_session,
    delete_chat_history_message,
    emit_chat_chunk,
    emit_chat_final,
    emit_chat_progress,
    get_chat_session_store,
    load_chat_history_messages,
    persist_chat_turn,
)


def test_chat_session_store_singleton(monkeypatch) -> None:
    import annolid.services.chat_session as session_mod

    class _Manager:
        pass

    class _Store:
        def __init__(self, manager):
            self.manager = manager

    monkeypatch.setattr(session_mod, "_SESSION_STORE", None)
    monkeypatch.setattr(session_mod, "AgentSessionManager", _Manager)
    monkeypatch.setattr(session_mod, "PersistentSessionStore", _Store)

    first = get_chat_session_store()
    second = get_chat_session_store()

    assert isinstance(first, _Store)
    assert first is second


def test_chat_session_wrappers(monkeypatch) -> None:
    import annolid.services.chat_session as session_mod

    captured = {}

    class _Store:
        def clear_session(self, session_id):
            captured["cleared"] = session_id

        def delete_history_message(
            self,
            session_id,
            *,
            message_id="",
            history_index,
            expected_role="",
            expected_content="",
        ):
            captured["deleted"] = {
                "session_id": session_id,
                "message_id": message_id,
                "history_index": history_index,
                "expected_role": expected_role,
                "expected_content": expected_content,
            }
            return True

    monkeypatch.setattr(
        session_mod,
        "gui_emit_chunk",
        lambda **kwargs: captured.setdefault("chunk", kwargs),
    )
    monkeypatch.setattr(
        session_mod,
        "gui_emit_progress",
        lambda **kwargs: captured.setdefault("progress", kwargs) or "next",
    )
    monkeypatch.setattr(
        session_mod,
        "gui_emit_final",
        lambda **kwargs: captured.setdefault("final", kwargs),
    )
    monkeypatch.setattr(
        session_mod,
        "gui_load_history_messages",
        lambda **kwargs: [{"role": "user", "content": "hello"}],
    )
    monkeypatch.setattr(
        session_mod,
        "gui_persist_turn",
        lambda **kwargs: captured.setdefault("persist", kwargs),
    )

    clear_chat_session("gui:test", session_store=_Store())
    deleted = delete_chat_history_message(
        "gui:test",
        message_id="msg-3",
        history_index=3,
        expected_role="assistant",
        expected_content="done",
        session_store=_Store(),
    )
    emit_chat_chunk(widget="widget", chunk="hi", turn_id="turn-1")
    progress = emit_chat_progress(
        widget="widget",
        update="loading",
        last_progress_update=None,
        turn_id="turn-1",
        turn_status="running",
    )
    emit_chat_final(
        widget="widget",
        message="done",
        is_error=False,
        emit_progress_cb=lambda _text: None,
        turn_id="turn-1",
        turn_status="completed",
        error_type="",
    )
    history = load_chat_history_messages(
        session_store="store",
        session_id="gui:test",
        max_history_messages=4,
    )
    persist_chat_turn(
        user_text="u",
        assistant_text="a",
        session_id="gui:test",
        session_store="store",
        max_history_messages=4,
        workspace_memory="memory",
        persist_session_history=False,
    )

    assert captured["cleared"] == "gui:test"
    assert deleted is True
    assert captured["deleted"]["session_id"] == "gui:test"
    assert captured["deleted"]["message_id"] == "msg-3"
    assert captured["deleted"]["history_index"] == 3
    assert captured["chunk"]["chunk"] == "hi"
    assert captured["progress"]["update"] == "loading"
    assert progress == captured["progress"]
    assert captured["final"]["message"] == "done"
    assert history == [{"role": "user", "content": "hello"}]
    assert captured["persist"]["assistant_text"] == "a"
