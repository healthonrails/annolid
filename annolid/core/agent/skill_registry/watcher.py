from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable, List


class SkillRegistryWatcher:
    """Lightweight polling watcher for skill folder snapshots."""

    def __init__(self, *, poll_seconds: float = 1.0) -> None:
        self.poll_seconds = max(0.0, float(poll_seconds))
        self._last_check_at = 0.0
        self._fingerprint: Dict[str, int] = {}

    def reset(self, roots: Iterable[Path]) -> None:
        self._fingerprint = self.compute_fingerprint(roots)
        self._last_check_at = time.monotonic()

    def changed(self, roots: Iterable[Path]) -> bool:
        now = time.monotonic()
        if (
            self._last_check_at > 0.0
            and self.poll_seconds > 0.0
            and (now - self._last_check_at) < self.poll_seconds
        ):
            return False
        current = self.compute_fingerprint(roots)
        self._last_check_at = now
        changed = current != self._fingerprint
        if changed:
            self._fingerprint = current
        return changed

    @staticmethod
    def compute_fingerprint(roots: Iterable[Path]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for root in roots:
            if not root.exists() or not root.is_dir():
                continue
            entries: List[Path] = sorted(root.iterdir(), key=lambda p: p.name)
            for candidate in entries:
                if not candidate.is_dir():
                    continue
                skill_file = candidate / "SKILL.md"
                if not skill_file.exists():
                    continue
                try:
                    out[str(skill_file)] = int(
                        getattr(skill_file.stat(), "st_mtime_ns", 0)
                    )
                except OSError:
                    continue
        return out
