from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from annolid.core.agent.memory import AgentMemoryStore
from annolid.core.agent.utils import get_agent_data_path

from .function_base import FunctionTool


class MemorySearchTool(FunctionTool):
    def __init__(self, workspace: Path | None = None):
        root = Path(workspace).expanduser().resolve() if workspace is not None else None
        self._memory = AgentMemoryStore(root or (get_agent_data_path() / "workspace"))

    @property
    def name(self) -> str:
        return "memory_search"

    @property
    def description(self) -> str:
        return (
            "Search markdown memory files (including memory/HISTORY.md) and return "
            "top matching snippets with path and line range."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                "max_snippet_chars": {
                    "type": "integer",
                    "minimum": 80,
                    "maximum": 4000,
                },
            },
            "required": ["query"],
        }

    async def execute(
        self,
        query: str,
        top_k: int = 5,
        max_snippet_chars: int = 700,
        **kwargs: Any,
    ) -> str:
        del kwargs
        results = self._memory.memory_search(
            query,
            top_k=max(1, int(top_k)),
            max_snippet_chars=max(80, int(max_snippet_chars)),
        )
        return json.dumps(
            {
                "query": str(query or ""),
                "count": len(results),
                "results": results,
            }
        )


class MemoryGetTool(FunctionTool):
    def __init__(self, workspace: Path | None = None):
        root = Path(workspace).expanduser().resolve() if workspace is not None else None
        self._memory = AgentMemoryStore(root or (get_agent_data_path() / "workspace"))

    @property
    def name(self) -> str:
        return "memory_get"

    @property
    def description(self) -> str:
        return (
            "Read MEMORY.md, HISTORY.md, or a daily memory file under memory/ "
            "with an optional line range."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
                "max_chars": {"type": "integer", "minimum": 64, "maximum": 50000},
            },
            "required": ["path"],
        }

    async def execute(
        self,
        path: str,
        start_line: int = 1,
        end_line: int | None = None,
        max_chars: int = 8000,
        **kwargs: Any,
    ) -> str:
        del kwargs
        try:
            payload = self._memory.memory_get(
                path,
                start_line=max(1, int(start_line)),
                end_line=None if end_line is None else max(1, int(end_line)),
                max_chars=max(64, int(max_chars)),
            )
            return json.dumps(payload)
        except ValueError as exc:
            return json.dumps({"error": str(exc), "path": str(path or "")})
        except Exception as exc:
            return json.dumps({"error": str(exc), "path": str(path or "")})


class MemorySetTool(FunctionTool):
    def __init__(self, workspace: Path | None = None):
        root = Path(workspace).expanduser().resolve() if workspace is not None else None
        self._memory = AgentMemoryStore(root or (get_agent_data_path() / "workspace"))

    @property
    def name(self) -> str:
        return "memory_set"

    @property
    def description(self) -> str:
        return "Remember a durable long-term fact by appending it to memory/MEMORY.md."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": [],
        }

    async def execute(
        self,
        key: str | None = None,
        value: str | None = None,
        note: str | None = None,
        **kwargs: Any,
    ) -> str:
        del kwargs
        key_text = str(key or "").strip()
        value_text = str(value or "").strip()
        note_text = str(note or "").strip()

        if key_text and value_text:
            line = f"- {key_text}: {value_text}"
        elif note_text:
            line = f"- {note_text}"
        else:
            return json.dumps(
                {
                    "error": "Provide either key+value or note.",
                    "path": "memory/MEMORY.md",
                }
            )

        existing = self._memory.read_long_term().rstrip()
        updated = (existing + "\n" + line).strip() + "\n" if existing else line + "\n"
        self._memory.write_long_term(updated)
        return json.dumps(
            {
                "ok": True,
                "path": "memory/MEMORY.md",
                "line": line,
            }
        )


__all__ = ["MemorySearchTool", "MemoryGetTool", "MemorySetTool"]
