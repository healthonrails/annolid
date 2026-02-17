from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from annolid.utils.citations import (
    BibEntry,
    entry_to_dict,
    load_bibtex,
    remove_entry,
    save_bibtex,
    search_entries,
    upsert_entry,
)

from .common import _resolve_read_path, _resolve_write_path
from .function_base import FunctionTool


class BibtexListEntriesTool(FunctionTool):
    def __init__(
        self,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ):
        self._allowed_dir = allowed_dir
        self._allowed_read_roots = tuple(allowed_read_roots or ())

    @property
    def name(self) -> str:
        return "bibtex_list_entries"

    @property
    def description(self) -> str:
        return "List or search entries in a BibTeX (.bib) file."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "query": {"type": "string"},
                "field": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
            },
            "required": ["path"],
        }

    async def execute(
        self,
        path: str,
        query: str = "",
        field: str = "",
        limit: int = 50,
        **kwargs: Any,
    ) -> str:
        del kwargs
        try:
            bib_path = _resolve_read_path(
                path,
                allowed_dir=self._allowed_dir,
                allowed_read_roots=self._allowed_read_roots,
            )
        except PermissionError as exc:
            return json.dumps({"error": str(exc), "path": path})
        if not bib_path.exists():
            return json.dumps({"error": "File not found", "path": path})
        entries = load_bibtex(bib_path)
        normalized_limit = max(1, min(int(limit), 1000))
        if str(query or "").strip():
            rows = search_entries(
                entries,
                str(query),
                field=(str(field).strip().lower() or None),
                limit=normalized_limit,
            )
        else:
            rows = list(entries[:normalized_limit])
        return json.dumps(
            {
                "path": str(bib_path),
                "total_entries": len(entries),
                "returned": len(rows),
                "entries": [entry_to_dict(entry) for entry in rows],
            }
        )


class BibtexUpsertEntryTool(FunctionTool):
    def __init__(self, allowed_dir: Path | None = None):
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "bibtex_upsert_entry"

    @property
    def description(self) -> str:
        return "Create or update one BibTeX entry in a .bib file."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "key": {"type": "string", "minLength": 1},
                "entry_type": {"type": "string", "minLength": 1},
                "fields": {"type": "object"},
                "sort_keys": {"type": "boolean"},
            },
            "required": ["path", "key", "fields"],
        }

    async def execute(
        self,
        path: str,
        key: str,
        fields: dict[str, Any],
        entry_type: str = "article",
        sort_keys: bool = True,
        **kwargs: Any,
    ) -> str:
        del kwargs
        try:
            bib_path = _resolve_write_path(path, allowed_dir=self._allowed_dir)
        except PermissionError as exc:
            return json.dumps({"error": str(exc), "path": path})

        normalized_key = str(key or "").strip()
        normalized_type = str(entry_type or "").strip().lower()
        if not normalized_key:
            return json.dumps({"error": "key must be non-empty"})
        if not normalized_type:
            return json.dumps({"error": "entry_type must be non-empty"})
        if not isinstance(fields, dict) or not fields:
            return json.dumps({"error": "fields must be a non-empty object"})

        normalized_fields: dict[str, str] = {}
        for field_name, value in fields.items():
            clean_name = str(field_name or "").strip().lower()
            if not clean_name:
                continue
            text_value = str(value).strip()
            if text_value:
                normalized_fields[clean_name] = text_value
        if not normalized_fields:
            return json.dumps({"error": "fields must include at least one value"})

        entries = load_bibtex(bib_path) if bib_path.exists() else []
        updated, created = upsert_entry(
            entries,
            BibEntry(
                entry_type=normalized_type,
                key=normalized_key,
                fields=normalized_fields,
            ),
        )
        save_bibtex(bib_path, updated, sort_keys=bool(sort_keys))
        return json.dumps(
            {
                "path": str(bib_path),
                "key": normalized_key,
                "created": bool(created),
                "total_entries": len(updated),
            }
        )


class BibtexRemoveEntryTool(FunctionTool):
    def __init__(self, allowed_dir: Path | None = None):
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "bibtex_remove_entry"

    @property
    def description(self) -> str:
        return "Remove one BibTeX entry by key from a .bib file."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "key": {"type": "string", "minLength": 1},
                "sort_keys": {"type": "boolean"},
            },
            "required": ["path", "key"],
        }

    async def execute(
        self,
        path: str,
        key: str,
        sort_keys: bool = True,
        **kwargs: Any,
    ) -> str:
        del kwargs
        try:
            bib_path = _resolve_write_path(path, allowed_dir=self._allowed_dir)
        except PermissionError as exc:
            return json.dumps({"error": str(exc), "path": path})
        if not bib_path.exists():
            return json.dumps({"error": "File not found", "path": str(bib_path)})
        entries = load_bibtex(bib_path)
        updated, removed = remove_entry(entries, str(key))
        if removed:
            save_bibtex(bib_path, updated, sort_keys=bool(sort_keys))
        return json.dumps(
            {
                "path": str(bib_path),
                "key": str(key),
                "removed": bool(removed),
                "total_entries": len(updated if removed else entries),
            }
        )
