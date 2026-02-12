from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from annolid.core.agent.memory import AgentMemoryStore


def test_memory_context_includes_long_term_today_and_yesterday(tmp_path: Path) -> None:
    store = AgentMemoryStore(tmp_path)
    store.write_long_term("Use deterministic exports.")

    today = date.today()
    yesterday = today - timedelta(days=1)
    older = today - timedelta(days=2)
    (store.memory_dir / f"{today.strftime('%Y-%m-%d')}.md").write_text(
        "today note",
        encoding="utf-8",
    )
    (store.memory_dir / f"{yesterday.strftime('%Y-%m-%d')}.md").write_text(
        "yesterday note",
        encoding="utf-8",
    )
    (store.memory_dir / f"{older.strftime('%Y-%m-%d')}.md").write_text(
        "older note",
        encoding="utf-8",
    )

    context = store.get_memory_context(recent_days=2)
    assert "Long-term Memory" in context
    assert "Use deterministic exports." in context
    assert "today note" in context
    assert "yesterday note" in context
    assert "older note" not in context


def test_memory_get_guards_path_scope(tmp_path: Path) -> None:
    store = AgentMemoryStore(tmp_path)
    (store.memory_dir / "MEMORY.md").write_text(
        "line1\nline2\nline3\n", encoding="utf-8"
    )

    payload = store.memory_get("memory/MEMORY.md", start_line=2, end_line=3)
    assert payload["line_start"] == 2
    assert payload["line_end"] == 3
    assert "line1" not in payload["content"]
    assert "line2" in payload["content"]

    try:
        store.memory_get("../x.md")
        assert False, "expected ValueError for escaping memory directory"
    except ValueError:
        pass
