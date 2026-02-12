from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List


def _today_str() -> str:
    return date.today().strftime("%Y-%m-%d")


class AgentMemoryStore:
    """Persistent markdown memory store (daily notes + long-term memory)."""

    def __init__(self, workspace: Path):
        self.workspace = Path(workspace)
        self.memory_dir = self.workspace / "memory"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.memory_file = self.memory_dir / "MEMORY.md"

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

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(str(content or ""), encoding="utf-8")

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

    def get_memory_context(self) -> str:
        parts: List[str] = []
        long_term = self.read_long_term().strip()
        if long_term:
            parts.append("## Long-term Memory\n" + long_term)
        today = self.read_today().strip()
        if today:
            parts.append("## Today's Notes\n" + today)
        return "\n\n".join(parts)
