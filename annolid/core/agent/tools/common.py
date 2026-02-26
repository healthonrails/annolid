from __future__ import annotations

import contextlib
import html
import os
import re
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse


def _normalize_workspace_alias_path(path: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return raw
    normalized = raw.replace("\\", "/")
    home_workspace_prefix = str((Path.home() / ".annolid" / "workspace")).replace(
        "\\", "/"
    )

    if normalized.startswith(".annolid/workspace/"):
        return str(Path.home() / normalized)
    if normalized.startswith("./.annolid/workspace/"):
        return str(Path.home() / normalized[2:])
    if "/.annolid/workspace/" in normalized and not normalized.startswith(
        home_workspace_prefix
    ):
        suffix = normalized.split("/.annolid/workspace/", 1)[1]
        return str(Path.home() / ".annolid" / "workspace" / suffix)
    return raw


def _resolve_path(path: str, allowed_dir: Path | None = None) -> Path:
    resolved = Path(path).expanduser().resolve()
    if allowed_dir and not str(resolved).startswith(str(allowed_dir.resolve())):
        raise PermissionError(f"Path {path} is outside allowed directory {allowed_dir}")
    return resolved


def _normalize_allowed_read_roots(
    allowed_dir: Path | None, allowed_read_roots: Sequence[str | Path] | None
) -> tuple[Path, ...]:
    roots: list[Path] = []
    if allowed_dir is not None:
        roots.append(Path(allowed_dir).expanduser().resolve())
    if allowed_read_roots:
        for raw in allowed_read_roots:
            text = str(raw).strip()
            if not text:
                continue
            with contextlib.suppress(Exception):
                candidate = Path(text).expanduser().resolve()
                if candidate not in roots:
                    roots.append(candidate)
    return tuple(roots)


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_read_path(
    path: str,
    *,
    allowed_dir: Path | None = None,
    allowed_read_roots: Sequence[str | Path] | None = None,
) -> Path:
    normalized_path = _normalize_workspace_alias_path(path)
    resolved = Path(normalized_path).expanduser().resolve()
    roots = _normalize_allowed_read_roots(allowed_dir, allowed_read_roots)
    if roots and not any(_is_within_root(resolved, root) for root in roots):
        allowed = ", ".join(str(root) for root in roots)
        raise PermissionError(f"Path {path} is outside allowed read roots: [{allowed}]")
    return resolved


def _resolve_write_path(path: str, *, allowed_dir: Path | None = None) -> Path:
    resolved = Path(path).expanduser().resolve()
    if allowed_dir is not None and not _is_within_root(
        resolved, Path(allowed_dir).expanduser().resolve()
    ):
        raise PermissionError(f"Path {path} is outside allowed directory {allowed_dir}")
    return resolved


def _iter_text_files(root: Path, *, include_hidden: bool = False) -> Sequence[Path]:
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for filename in filenames:
            if not include_hidden and filename.startswith("."):
                continue
            files.append(Path(dirpath) / filename)
    return files


def _is_probably_text_file(path: Path, *, probe_bytes: int = 2048) -> bool:
    try:
        data = path.read_bytes()[:probe_bytes]
    except Exception:
        return False
    if not data:
        return True
    if b"\x00" in data:
        return False
    return True


def _strip_tags(text: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", "", text, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.I)
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)
    except ImportError:
        text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def _normalize(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _validate_url(url: str) -> tuple[bool, str]:
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False, f"Only http/https allowed, got '{p.scheme or 'none'}'"
        if not p.netloc:
            return False, "Missing domain"
        return True, ""
    except Exception as exc:
        return False, str(exc)
