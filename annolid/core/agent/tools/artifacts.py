from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


def content_hash(payload: Any) -> str:
    """Return a stable content hash for a JSON-serializable payload."""
    try:
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    except TypeError:
        raw = repr(payload).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ArtifactPaths:
    base_dir: Path
    run_id: str
    cache_dir: Optional[Path] = None

    @property
    def run_dir(self) -> Path:
        return self.base_dir / ".agent_runs" / self.run_id

    def cache_root(self) -> Optional[Path]:
        if self.cache_dir is not None:
            return self.cache_dir
        return self.base_dir / ".cache"


class FileArtifactStore:
    """Filesystem-backed artifact store (Phase 3)."""

    def __init__(
        self,
        *,
        base_dir: Path,
        run_id: str,
        cache_dir: Optional[Path] = None,
    ) -> None:
        self._paths = ArtifactPaths(
            base_dir=Path(base_dir),
            run_id=str(run_id),
            cache_dir=cache_dir,
        )

    @property
    def run_dir(self) -> Path:
        return self._paths.run_dir

    def resolve(self, *parts: str, kind: str = "run") -> Path:
        root = self.run_dir
        if kind == "cache":
            cache_root = self._paths.cache_root()
            root = cache_root if cache_root is not None else root
        return root.joinpath(*parts)

    def ensure_dir(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(
        self, path: Path, payload: Dict[str, Any], *, indent: int = 2
    ) -> Path:
        self.ensure_dir(path)
        path.write_text(
            json.dumps(payload, indent=indent, ensure_ascii=False), encoding="utf-8"
        )
        return path

    def write_ndjson(self, path: Path, records: Iterable[Dict[str, Any]]) -> Path:
        self.ensure_dir(path)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")
        return path

    def write_text(self, path: Path, text: str) -> Path:
        self.ensure_dir(path)
        path.write_text(text, encoding="utf-8")
        return path

    def write_bytes(self, path: Path, blob: bytes) -> Path:
        self.ensure_dir(path)
        path.write_bytes(blob)
        return path

    def write_meta(self, path: Path, payload: Dict[str, Any]) -> Path:
        return self.write_json(path, payload, indent=2)

    def should_reuse_cache(self, meta_path: Path, hash_value: str) -> bool:
        if not meta_path.exists():
            return False
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        return payload.get("content_hash") == hash_value
