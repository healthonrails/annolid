from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timedelta
import math
from pathlib import Path
import re
from typing import Any, Dict, Iterable, List, Sequence

from annolid.core.agent.observability import emit_governance_event

from .memory_core import MemoryRetrievalPlugin, get_memory_retrieval_plugin


def _today_str() -> str:
    return date.today().strftime("%Y-%m-%d")


class WorkspaceMemoryStore:
    """Workspace markdown memory store with pluggable retrieval."""

    _DAILY_FILE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\.md$")

    def __init__(
        self,
        workspace: Path,
        *,
        retrieval_plugin: MemoryRetrievalPlugin | None = None,
    ):
        self.workspace = Path(workspace)
        self.memory_dir = self.workspace / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self._retrieval_plugin: MemoryRetrievalPlugin = (
            retrieval_plugin or get_memory_retrieval_plugin()
        )

    def get_today_file(self) -> Path:
        return self.memory_dir / f"{_today_str()}.md"

    def read_today(self) -> str:
        path = self.get_today_file()
        if path.exists():
            return path.read_text(encoding="utf-8")
        return ""

    def append_today(self, content: str) -> None:
        path = self.get_today_file()
        text = str(content or "").strip()
        if not text:
            return
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            path.write_text(
                (existing.rstrip() + "\n\n" + text).strip() + "\n", encoding="utf-8"
            )
        else:
            path.write_text(f"# {_today_str()}\n\n{text}\n", encoding="utf-8")
        emit_governance_event(
            event_type="memory",
            action="append_today",
            outcome="ok",
            actor="system",
            details={
                "workspace": str(self.workspace),
                "path": str(path),
                "chars": len(text),
            },
        )

    def append_history(self, content: str) -> None:
        path = self.history_file
        text = str(content or "").strip()
        if not text:
            return
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            path.write_text(
                (existing.rstrip() + "\n\n" + text).strip() + "\n", encoding="utf-8"
            )
        else:
            path.write_text("# History\n\n" + text + "\n", encoding="utf-8")
        emit_governance_event(
            event_type="memory",
            action="append_history",
            outcome="ok",
            actor="system",
            details={
                "workspace": str(self.workspace),
                "path": str(path),
                "chars": len(text),
            },
        )

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        text = str(content or "")
        self.memory_file.write_text(text, encoding="utf-8")
        emit_governance_event(
            event_type="memory",
            action="write_long_term",
            outcome="ok",
            actor="system",
            details={
                "workspace": str(self.workspace),
                "path": str(self.memory_file),
                "chars": len(text),
            },
        )

    def get_recent_memories(self, days: int = 7) -> str:
        parts: List[str] = []
        total_days = max(1, int(days))
        today = datetime.now().date()
        for i in range(total_days):
            d = today - timedelta(days=i)
            p = self.memory_dir / f"{d.strftime('%Y-%m-%d')}.md"
            if p.exists():
                parts.append(p.read_text(encoding="utf-8"))
        return "\n\n---\n\n".join(parts)

    def list_memory_files(self) -> List[Path]:
        files = list(self.memory_dir.glob("????-??-??.md"))
        return sorted(files, reverse=True)

    def get_recent_daily_files(self, days: int = 2) -> List[Path]:
        total_days = max(1, int(days))
        today = datetime.now().date()
        files: List[Path] = []
        for i in range(total_days):
            d = today - timedelta(days=i)
            p = self.memory_dir / f"{d.strftime('%Y-%m-%d')}.md"
            if p.exists():
                files.append(p)
        return files

    def get_memory_context(
        self, *, recent_days: int = 2, max_chars: int = 12000
    ) -> str:
        del recent_days
        parts: List[str] = []
        long_term = self.read_long_term().strip()
        if long_term:
            parts.append("## Long-term Memory\n" + long_term)

        merged = "\n\n".join(parts).strip()
        if len(merged) <= max(0, int(max_chars)):
            return merged
        if max_chars <= 0:
            return ""
        suffix = "\n\n[Memory context truncated]"
        cutoff = max(0, int(max_chars) - len(suffix))
        return merged[:cutoff].rstrip() + suffix

    def _iter_searchable_files(self) -> Sequence[Path]:
        files: List[Path] = []
        if self.memory_file.exists():
            files.append(self.memory_file)
        if self.history_file.exists():
            files.append(self.history_file)
        files.extend(self.list_memory_files())
        return files

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        raw = re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower())
        return [tok for tok in raw if len(tok) > 1]

    def _iter_chunks(
        self, lines: Sequence[str], *, chunk_size: int = 36, overlap: int = 8
    ) -> Iterable[tuple[int, int, str]]:
        if not lines:
            return []
        size = max(1, int(chunk_size))
        step = max(1, size - max(0, int(overlap)))
        chunks: List[tuple[int, int, str]] = []
        start = 0
        total = len(lines)
        while start < total:
            end = min(total, start + size)
            snippet = "\n".join(lines[start:end]).strip()
            if snippet:
                chunks.append((start + 1, end, snippet))
            if end >= total:
                break
            start += step
        return chunks

    @staticmethod
    def _daily_age_boost(path: Path) -> float:
        name = path.name
        if not WorkspaceMemoryStore._DAILY_FILE_RE.match(name):
            if name == "MEMORY.md":
                return 0.12
            if name == "HISTORY.md":
                return 0.08
            return 0.0
        try:
            day = datetime.strptime(path.stem, "%Y-%m-%d").date()
            age = max(0, (date.today() - day).days)
            return max(0.0, 0.25 * (1.0 - min(age, 14) / 14.0))
        except ValueError:
            return 0.0

    def memory_search(
        self,
        query: str,
        *,
        top_k: int = 5,
        max_snippet_chars: int = 700,
    ) -> List[Dict[str, Any]]:
        return self._retrieval_plugin.search(
            self,
            query,
            top_k=max(1, int(top_k)),
            max_snippet_chars=max(80, int(max_snippet_chars)),
        )

    def memory_search_lexical(
        self,
        query: str,
        *,
        top_k: int = 5,
        max_snippet_chars: int = 700,
    ) -> List[Dict[str, Any]]:
        query_text = str(query or "").strip()
        if not query_text:
            return []
        query_lc = query_text.lower()
        tokens = self._tokenize(query_text)
        if not tokens:
            return []

        candidates: List[Dict[str, Any]] = []
        for path in self._iter_searchable_files():
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            lines = text.splitlines()
            for line_start, line_end, snippet in self._iter_chunks(lines):
                lower = snippet.lower()
                token_hits = sum(1 for tok in tokens if tok in lower)
                if token_hits == 0 and query_lc not in lower:
                    continue
                lexical = token_hits / max(1, len(tokens))
                phrase = 0.35 if query_lc in lower else 0.0
                score = lexical + phrase + self._daily_age_boost(path)
                capped = snippet[: max(80, int(max_snippet_chars))].strip()
                candidates.append(
                    {
                        "path": f"memory/{path.name}",
                        "line_start": line_start,
                        "line_end": line_end,
                        "score": round(float(score), 4),
                        "snippet": capped,
                    }
                )

        candidates.sort(key=lambda item: float(item["score"]), reverse=True)
        limit = max(1, int(top_k))
        return candidates[:limit]

    @staticmethod
    def _char_trigram_vector(text: str) -> Counter[str]:
        value = re.sub(r"\s+", " ", str(text or "").lower()).strip()
        if not value:
            return Counter()
        padded = f"  {value}  "
        return Counter(padded[i : i + 3] for i in range(max(0, len(padded) - 2)))

    @staticmethod
    def _cosine_similarity(a: Counter[str], b: Counter[str]) -> float:
        if not a or not b:
            return 0.0
        dot = 0.0
        for key, val in a.items():
            dot += float(val) * float(b.get(key, 0))
        norm_a = math.sqrt(sum(float(v) * float(v) for v in a.values()))
        norm_b = math.sqrt(sum(float(v) * float(v) for v in b.values()))
        denom = norm_a * norm_b
        if denom <= 0.0:
            return 0.0
        return max(0.0, min(1.0, dot / denom))

    def memory_search_semantic(
        self,
        query: str,
        *,
        top_k: int = 5,
        max_snippet_chars: int = 700,
    ) -> List[Dict[str, Any]]:
        query_text = str(query or "").strip()
        if not query_text:
            return []
        query_vec = self._char_trigram_vector(query_text)
        query_terms = set(self._tokenize(query_text))
        if not query_vec:
            return []

        candidates: List[Dict[str, Any]] = []
        for path in self._iter_searchable_files():
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                continue
            lines = text.splitlines()
            for line_start, line_end, snippet in self._iter_chunks(lines):
                snippet_text = str(snippet or "").strip()
                if not snippet_text:
                    continue
                sem = self._cosine_similarity(
                    query_vec, self._char_trigram_vector(snippet_text)
                )
                if sem <= 0.08:
                    continue
                snippet_terms = set(self._tokenize(snippet_text))
                overlap = 0.0
                if query_terms:
                    overlap = len(query_terms & snippet_terms) / float(len(query_terms))
                score = (0.75 * sem) + (0.25 * overlap) + self._daily_age_boost(path)
                capped = snippet_text[: max(80, int(max_snippet_chars))].strip()
                candidates.append(
                    {
                        "path": f"memory/{path.name}",
                        "line_start": line_start,
                        "line_end": line_end,
                        "score": round(float(score), 4),
                        "snippet": capped,
                    }
                )

        candidates.sort(key=lambda item: float(item["score"]), reverse=True)
        return candidates[: max(1, int(top_k))]

    @property
    def retrieval_plugin_name(self) -> str:
        return str(getattr(self._retrieval_plugin, "name", "unknown") or "unknown")

    def set_retrieval_plugin(self, plugin: MemoryRetrievalPlugin) -> None:
        self._retrieval_plugin = plugin

    def _resolve_memory_path(self, path: str) -> Path:
        raw = str(path or "").strip().replace("\\", "/")
        if raw.startswith("memory/"):
            raw = raw.split("/", 1)[1]
        if "/" in raw:
            raise ValueError(
                "Only MEMORY.md, HISTORY.md, or memory/YYYY-MM-DD.md are allowed."
            )
        if raw not in {"MEMORY.md", "HISTORY.md"} and not self._DAILY_FILE_RE.match(
            raw
        ):
            raise ValueError(
                "Only MEMORY.md, HISTORY.md, or memory/YYYY-MM-DD.md are allowed."
            )
        resolved = (self.memory_dir / raw).resolve()
        memory_root = self.memory_dir.resolve()
        if not str(resolved).startswith(str(memory_root)):
            raise ValueError("Path escapes memory directory.")
        return resolved

    def memory_get(
        self,
        path: str,
        *,
        start_line: int = 1,
        end_line: int | None = None,
        max_chars: int = 8000,
    ) -> Dict[str, Any]:
        resolved = self._resolve_memory_path(path)
        if not resolved.exists():
            return {
                "error": "memory file not found",
                "path": f"memory/{resolved.name}",
            }
        lines = resolved.read_text(encoding="utf-8").splitlines()
        total_lines = len(lines)
        if total_lines == 0:
            return {
                "path": f"memory/{resolved.name}",
                "line_start": 0,
                "line_end": 0,
                "total_lines": 0,
                "truncated": False,
                "content": "",
            }
        start = min(max(1, int(start_line)), max(1, total_lines))
        end = (
            total_lines
            if end_line is None
            else min(max(start, int(end_line)), total_lines)
        )
        content = "\n".join(lines[start - 1 : end]).strip()
        chars_limit = max(64, int(max_chars))
        truncated = len(content) > chars_limit
        if truncated:
            content = content[:chars_limit].rstrip()
        return {
            "path": f"memory/{resolved.name}",
            "line_start": start,
            "line_end": end,
            "total_lines": total_lines,
            "truncated": truncated,
            "content": content,
        }
