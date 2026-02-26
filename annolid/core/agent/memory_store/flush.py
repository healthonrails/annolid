from __future__ import annotations

from datetime import datetime

from .store import WorkspaceMemoryStore


def build_pre_compaction_flush_entry(
    *,
    session_id: str,
    transcript: str,
    archive_len: int,
    max_chars: int = 6000,
) -> str:
    text = str(transcript or "").strip()
    if not text:
        return ""
    limited = text[: max(256, int(max_chars))].rstrip()
    if len(text) > len(limited):
        limited += "\n...[archive transcript truncated]..."
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        f"## [{stamp}] Pre-compaction Memory Flush\n\n"
        f"- session_id: {str(session_id).strip()}\n"
        f"- archived_messages: {int(max(0, archive_len))}\n\n"
        "```text\n"
        f"{limited}\n"
        "```"
    )


def append_pre_compaction_flush(
    *,
    store: WorkspaceMemoryStore,
    session_id: str,
    transcript: str,
    archive_len: int,
    max_chars: int = 6000,
) -> bool:
    entry = build_pre_compaction_flush_entry(
        session_id=session_id,
        transcript=transcript,
        archive_len=archive_len,
        max_chars=max_chars,
    )
    if not entry:
        return False
    try:
        store.append_today(entry)
        return True
    except Exception:
        return False
