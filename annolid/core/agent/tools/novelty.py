from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from annolid.services.novelty import novelty_preflight_check

from .common import (
    _normalize_allowed_read_roots,
    _resolve_read_path,
)
from .function_base import FunctionTool


class AnnolidNoveltyCheckTool(FunctionTool):
    def __init__(
        self,
        *,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ) -> None:
        self._allowed_dir = (
            Path(allowed_dir).expanduser().resolve()
            if allowed_dir is not None
            else None
        )
        self._allowed_read_roots = _normalize_allowed_read_roots(
            self._allowed_dir,
            allowed_read_roots,
        )

    @property
    def name(self) -> str:
        return "annolid_novelty_check"

    @property
    def description(self) -> str:
        return (
            "Run novelty preflight for research drafting by scoring overlap "
            "between proposed idea text and related-work entries."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "idea_title": {"type": "string"},
                "idea_summary": {"type": "string", "minLength": 1},
                "related_work": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {"type": "string"},
                            {
                                "type": "object",
                                "properties": {
                                    "title": {"type": "string"},
                                    "abstract": {"type": "string"},
                                    "summary": {"type": "string"},
                                    "keywords": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                                "required": [],
                            },
                        ]
                    },
                },
                "related_work_json_path": {"type": "string"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
                "abort_overlap_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "differentiate_overlap_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
            "required": ["idea_summary"],
        }

    async def execute(
        self,
        idea_summary: str,
        idea_title: str = "",
        related_work: Sequence[object] | None = None,
        related_work_json_path: str = "",
        top_k: int = 5,
        abort_overlap_threshold: float = 0.72,
        differentiate_overlap_threshold: float = 0.45,
        **kwargs: Any,
    ) -> str:
        del kwargs
        try:
            rows: list[object] = list(related_work or [])
            if str(related_work_json_path or "").strip():
                path = _resolve_read_path(
                    related_work_json_path,
                    allowed_dir=self._allowed_dir,
                    allowed_read_roots=self._allowed_read_roots,
                )
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    rows = list(payload)
                elif isinstance(payload, dict) and isinstance(
                    payload.get("related_work"), list
                ):
                    rows = list(payload.get("related_work") or [])
                else:
                    raise ValueError(
                        "related_work_json_path must point to a JSON array or object with 'related_work' array."
                    )
            result = novelty_preflight_check(
                idea_title=idea_title,
                idea_summary=idea_summary,
                related_work=rows,
                top_k=top_k,
                abort_overlap_threshold=abort_overlap_threshold,
                differentiate_overlap_threshold=differentiate_overlap_threshold,
            )
            return json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                },
                ensure_ascii=False,
            )


__all__ = ["AnnolidNoveltyCheckTool"]
