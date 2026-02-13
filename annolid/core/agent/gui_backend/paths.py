from __future__ import annotations

import os
from pathlib import Path
import re
from typing import List, Optional, Set


def extract_pdf_path_candidates(raw: str) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []

    candidates: List[str] = []
    if re.search(r"\.pdf\b", text, flags=re.IGNORECASE):
        candidates.append(text)
    for line in (ln.strip() for ln in text.splitlines() if ln.strip()):
        if re.search(r"\.pdf\b", line, flags=re.IGNORECASE):
            candidates.append(line)

    for quoted in re.findall(
        r'["\']([^"\']+\.pdf)["\']',
        text,
        flags=re.IGNORECASE,
    ):
        candidates.append(quoted.strip())

    for token in re.findall(
        r"https?://[^\s`\"']+?\.pdf(?:\?[^\s`\"']*)?",
        text,
        flags=re.IGNORECASE,
    ):
        candidates.append(token.strip())

    for token in re.findall(
        r'(?:~?/|/)[^\s`"\']+\.pdf',
        text,
        flags=re.IGNORECASE,
    ):
        candidates.append(token.strip())

    for token in re.findall(
        r"\b[\w.\-]+\.pdf\b",
        text,
        flags=re.IGNORECASE,
    ):
        candidates.append(token.strip())

    if text.startswith("gui_open_pdf(") and "path=" in text:
        match = re.search(r'path\s*=\s*["\']([^"\']+)["\']', text)
        if match:
            candidates.append(match.group(1).strip())

    return _dedupe_path_text(candidates)


def extract_video_path_candidates(raw: str) -> List[str]:
    text = str(raw or "").strip()
    if not text:
        return []

    candidates: List[str] = [text]
    candidates.extend(ln.strip() for ln in text.splitlines() if ln.strip())

    for quoted in re.findall(
        r'["\']([^"\']+\.(?:mp4|avi|mov|mkv|m4v|wmv|flv))["\']',
        text,
        flags=re.IGNORECASE,
    ):
        candidates.append(quoted.strip())

    for token in re.findall(
        r'(?:~?/|/)[^\s`"\']+\.(?:mp4|avi|mov|mkv|m4v|wmv|flv)',
        text,
        flags=re.IGNORECASE,
    ):
        candidates.append(token.strip())
    for token in re.findall(
        r"\b[\w.\-]+\.(?:mp4|avi|mov|mkv|m4v|wmv|flv)\b",
        text,
        flags=re.IGNORECASE,
    ):
        candidates.append(token.strip())

    if text.startswith("gui_open_video(") and "path=" in text:
        match = re.search(r'path\s*=\s*["\']([^"\']+)["\']', text)
        if match:
            candidates.append(match.group(1).strip())

    return _dedupe_path_text(candidates)


def find_video_by_basename_in_roots(
    *, basenames: Set[str], roots: List[Path], max_scan: int = 30000
) -> Optional[Path]:
    targets = {name.lower() for name in basenames if str(name).strip()}
    if not targets:
        return None
    scanned = 0
    for root in roots:
        try:
            root_resolved = root.expanduser()
        except Exception:
            continue
        if not root_resolved.exists() or not root_resolved.is_dir():
            continue
        try:
            for dirpath, _dirnames, filenames in os.walk(root_resolved):
                for filename in filenames:
                    scanned += 1
                    if scanned > max_scan:
                        return None
                    if filename.lower() in targets:
                        return Path(dirpath) / filename
        except Exception:
            continue
    return None


def build_workspace_roots(workspace: Path, read_roots_cfg: List[str]) -> List[Path]:
    roots: List[Path] = [workspace]
    roots.extend(
        Path(str(root)).expanduser() for root in read_roots_cfg if str(root).strip()
    )
    return _dedupe_roots(roots)


def build_pdf_search_roots(workspace: Path, read_roots_cfg: List[str]) -> List[Path]:
    roots: List[Path] = [workspace / "downloads", workspace]
    roots.extend(
        Path(str(root)).expanduser() for root in read_roots_cfg if str(root).strip()
    )
    return _dedupe_roots(roots)


def resolve_pdf_path_for_roots(raw_path: str, roots: List[Path]) -> Optional[Path]:
    candidates = extract_pdf_path_candidates(raw_path)
    if not candidates:
        return None

    for candidate in candidates:
        path_obj = Path(candidate).expanduser()
        if path_obj.exists() and path_obj.is_file():
            return path_obj
        if not path_obj.is_absolute():
            for root in roots:
                joined = (root / path_obj).expanduser()
                if joined.exists() and joined.is_file():
                    return joined

    for candidate in candidates:
        basename = Path(candidate).name
        if not basename:
            continue
        for root in roots:
            joined = (root / basename).expanduser()
            if joined.exists() and joined.is_file():
                return joined
    return None


def list_available_pdfs_in_roots(
    roots: List[Path], *, limit: int = 8, max_scan: int = 40000
) -> List[Path]:
    matches: List[Path] = []
    scanned = 0
    for root in roots:
        try:
            root_resolved = root.expanduser()
        except Exception:
            continue
        if not root_resolved.exists() or not root_resolved.is_dir():
            continue
        try:
            for dirpath, _dirnames, filenames in os.walk(root_resolved):
                for filename in filenames:
                    scanned += 1
                    if scanned > int(max_scan):
                        break
                    if not str(filename).lower().endswith(".pdf"):
                        continue
                    matches.append((Path(dirpath) / filename).resolve())
                if scanned > int(max_scan):
                    break
        except Exception:
            continue
        if scanned > int(max_scan):
            break
    unique: dict[str, Path] = {}
    for path in matches:
        unique[str(path)] = path
    ranked = sorted(
        unique.values(),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    return ranked[: int(limit)]


def resolve_video_path_for_roots(
    raw_path: str,
    roots: List[Path],
    *,
    active_video_raw: str = "",
    max_scan: int = 30000,
) -> Optional[Path]:
    candidates = extract_video_path_candidates(raw_path)
    if not candidates:
        return None

    active_name = Path(active_video_raw).name.lower() if active_video_raw else ""
    active_stem = Path(active_video_raw).stem.lower() if active_video_raw else ""
    if active_video_raw:
        for candidate in candidates:
            cleaned = str(candidate).strip().strip("\"'`")
            if not cleaned:
                continue
            lowered = cleaned.lower()
            if lowered in {"current video", "this video", "active video"}:
                return Path(active_video_raw).expanduser()
            candidate_name = Path(cleaned).name.lower()
            candidate_stem = Path(cleaned).stem.lower()
            if candidate_name and candidate_name == active_name:
                return Path(active_video_raw).expanduser()
            if candidate_stem and candidate_stem == active_stem:
                return Path(active_video_raw).expanduser()

    for candidate in candidates:
        path_obj = Path(candidate).expanduser()
        if path_obj.exists():
            return path_obj
        if not path_obj.is_absolute():
            for root in roots:
                joined = (root / path_obj).expanduser()
                if joined.exists():
                    return joined

    for candidate in candidates:
        basename = Path(candidate).name
        if not basename:
            continue
        for root in roots:
            joined = (root / basename).expanduser()
            if joined.exists():
                return joined

    candidate_basenames = {
        Path(str(candidate)).name
        for candidate in candidates
        if Path(str(candidate)).name.strip()
    }
    if candidate_basenames:
        found = find_video_by_basename_in_roots(
            basenames=candidate_basenames,
            roots=roots,
            max_scan=max_scan,
        )
        if found is not None:
            return found
    return None


def _dedupe_path_text(candidates: List[str]) -> List[str]:
    cleaned: List[str] = []
    seen: set[str] = set()
    for item in candidates:
        value = str(item or "").strip().strip("`").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        cleaned.append(value)
    return cleaned


def _dedupe_roots(roots: List[Path]) -> List[Path]:
    deduped: List[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root)
    return deduped
