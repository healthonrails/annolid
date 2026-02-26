from __future__ import annotations

import os
from typing import Any, Dict, List, Protocol


class MemoryRetrievalPlugin(Protocol):
    name: str

    def search(
        self,
        store: Any,
        query: str,
        *,
        top_k: int = 5,
        max_snippet_chars: int = 700,
    ) -> List[Dict[str, Any]]: ...


class WorkspaceLexicalRetrievalPlugin:
    name = "workspace_lexical_v1"

    def search(
        self,
        store: Any,
        query: str,
        *,
        top_k: int = 5,
        max_snippet_chars: int = 700,
    ) -> List[Dict[str, Any]]:
        return store.memory_search_lexical(
            query,
            top_k=top_k,
            max_snippet_chars=max_snippet_chars,
        )


class WorkspaceSemanticKeywordRetrievalPlugin:
    """Default local retrieval: semantic ranking + keyword fallback."""

    name = "workspace_semantic_keyword_v1"

    def search(
        self,
        store: Any,
        query: str,
        *,
        top_k: int = 5,
        max_snippet_chars: int = 700,
    ) -> List[Dict[str, Any]]:
        limit = max(1, int(top_k))
        primary = store.memory_search_semantic(
            query,
            top_k=limit,
            max_snippet_chars=max_snippet_chars,
        )
        if len(primary) >= limit:
            return primary[:limit]

        fallback = store.memory_search_lexical(
            query,
            top_k=limit,
            max_snippet_chars=max_snippet_chars,
        )
        merged: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for row in list(primary) + list(fallback):
            path = str(row.get("path") or "")
            line_start = int(row.get("line_start") or 0)
            line_end = int(row.get("line_end") or 0)
            key = f"{path}:{line_start}:{line_end}"
            if key in seen:
                continue
            seen.add(key)
            merged.append(dict(row))
            if len(merged) >= limit:
                break
        return merged


def _resolve_default_plugin() -> MemoryRetrievalPlugin:
    name = str(os.getenv("ANNOLID_MEMORY_RETRIEVAL_PLUGIN") or "").strip().lower()
    if name in {"lexical", "workspace_lexical_v1"}:
        return WorkspaceLexicalRetrievalPlugin()
    return WorkspaceSemanticKeywordRetrievalPlugin()


_DEFAULT_MEMORY_RETRIEVAL_PLUGIN: MemoryRetrievalPlugin = _resolve_default_plugin()


def set_memory_retrieval_plugin(plugin: MemoryRetrievalPlugin) -> None:
    global _DEFAULT_MEMORY_RETRIEVAL_PLUGIN
    _DEFAULT_MEMORY_RETRIEVAL_PLUGIN = plugin


def get_memory_retrieval_plugin() -> MemoryRetrievalPlugin:
    return _DEFAULT_MEMORY_RETRIEVAL_PLUGIN
