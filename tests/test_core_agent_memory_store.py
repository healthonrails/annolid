from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from annolid.core.agent.memory import AgentMemoryStore


def test_memory_context_includes_only_long_term_by_default(tmp_path: Path) -> None:
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
    assert "today note" not in context
    assert "yesterday note" not in context
    assert "older note" not in context


def test_memory_append_history_writes_history_file(tmp_path: Path) -> None:
    store = AgentMemoryStore(tmp_path)
    store.append_history("[2026-01-01 10:00] User asked for export defaults.")
    content = (store.memory_dir / "HISTORY.md").read_text(encoding="utf-8")
    assert "# History" in content
    assert "export defaults" in content


def test_memory_get_guards_path_scope(tmp_path: Path) -> None:
    store = AgentMemoryStore(tmp_path)
    (store.memory_dir / "MEMORY.md").write_text(
        "line1\nline2\nline3\n", encoding="utf-8"
    )
    store.append_history("[2026-01-01 10:00] Added line range test.")

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

    history_payload = store.memory_get("memory/HISTORY.md")
    assert history_payload["path"] == "memory/HISTORY.md"
    assert "line range test" in history_payload["content"]


def test_memory_store_supports_custom_retrieval_plugin(tmp_path: Path) -> None:
    class _FakePlugin:
        name = "fake_retrieval_v1"

        def search(
            self, store, query: str, *, top_k: int = 5, max_snippet_chars: int = 700
        ):
            del store, query, top_k, max_snippet_chars
            return [{"path": "memory/MEMORY.md", "score": 1.0, "snippet": "fake"}]

    store = AgentMemoryStore(tmp_path, retrieval_plugin=_FakePlugin())
    results = store.memory_search("anything")
    assert store.retrieval_plugin_name == "fake_retrieval_v1"
    assert len(results) == 1
    assert results[0]["snippet"] == "fake"


def test_memory_store_defaults_to_semantic_keyword_plugin(tmp_path: Path) -> None:
    store = AgentMemoryStore(tmp_path)
    assert store.retrieval_plugin_name == "workspace_semantic_keyword_v1"


def test_memory_search_falls_back_to_keyword_when_semantic_empty(
    tmp_path: Path,
) -> None:
    store = AgentMemoryStore(tmp_path)
    store.write_long_term("Annolid install guide and setup steps.")
    store.memory_search_semantic = lambda *a, **k: []  # type: ignore[method-assign]
    results = store.memory_search("install setup", top_k=3)
    assert len(results) >= 1
    assert "install" in results[0]["snippet"].lower()
