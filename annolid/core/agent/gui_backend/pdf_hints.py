from __future__ import annotations

from pathlib import Path
import re


def parse_source_page_hint(text: str) -> tuple[str, int] | None:
    match = re.search(
        r"source\s*:\s*([^)\n]*?\.pdf)\s+page\s*(\d+)",
        str(text or ""),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    source_name = str(match.group(1) or "").strip().strip("\"' ")
    page = max(1, int(match.group(2) or 1))
    return (source_name, page)


def source_matches_active_pdf(
    source_name: str,
    active_path: str,
    active_title: str,
) -> bool:
    source_base = Path(str(source_name or "").strip()).name.lower()
    active_base = Path(str(active_path or "").strip()).name.lower()
    active_title_base = Path(str(active_title or "").strip()).name.lower()
    return bool(
        source_base
        and (
            source_base == active_base
            or source_base == active_title_base
            or source_base in active_base
        )
    )
