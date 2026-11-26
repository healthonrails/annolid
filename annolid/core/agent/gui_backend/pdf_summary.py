from __future__ import annotations

import hashlib
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict

from annolid.services.pdf_parsing import (
    dehyphenate_pdf_text,
    extract_pdf_pages,
    normalize_pages_for_display,
)
from annolid.utils.atomic_io import atomic_write_text

_ABSTRACT_HEADING_RE = re.compile(
    r"(?im)^\s*(?:(?:\d+(?:\.\d+)*|[IVXLCDM]+)\.?\s+)?abstract\b(?:\s*[:.\-—]|\s*$)"
)
_AFFILIATION_LINE_RE = re.compile(
    r"\b(?:university|institute|department|school|laboratory|corresponding\s+author)\b",
    flags=re.IGNORECASE,
)


def looks_like_raw_pdf_extract_response(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return True
    lowered = value.lower()
    if value.count("@") >= 2:
        return True
    if "using the currently open pdf" in lowered:
        return value.count("@") >= 1
    noisy_affiliations = (
        "university",
        "institute",
        "department",
        "school",
        "laboratory",
        "corresponding author",
    )
    if sum(1 for token in noisy_affiliations if token in lowered) >= 3:
        return True
    if len(value) > 900 and ("\n-" not in value and "\n1." not in value):
        return True
    return False


def clean_pdf_text_for_summary(text: str) -> str:
    value = dehyphenate_pdf_text(str(text or ""))
    if not value.strip():
        return ""
    # Only jump to an actual Abstract heading near the front of the paper so
    # body-text mentions like "an abstract representation" never drop earlier pages.
    leading_window = value[: max(8000, int(len(value) * 0.35))]
    abstract_match = _ABSTRACT_HEADING_RE.search(leading_window)
    if abstract_match:
        value = value[abstract_match.start() :]
    lines: list[str] = []
    for raw in value.splitlines():
        line = str(raw or "").strip()
        if not line:
            continue
        if "@" in line and len(line) < 160:
            continue
        if _AFFILIATION_LINE_RE.search(line) and len(line) < 140:
            continue
        if re.match(r"^(?:##\s*)?page\s+\d+\b", line, flags=re.IGNORECASE):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _slice_text(value: str, *, max_chars: int) -> str:
    compact = str(value or "").strip()
    if not compact:
        return ""
    return compact[: int(max_chars)].strip()


def _extract_section_slice(
    text: str,
    *,
    heading_pattern: str,
    stop_patterns: tuple[str, ...] = (),
    max_chars: int,
) -> str:
    value = str(text or "")
    if not value:
        return ""
    match = re.search(heading_pattern, value, flags=re.IGNORECASE | re.MULTILINE)
    if not match:
        return ""
    start = match.start()
    search_offset = match.end()
    end = len(value)
    for stop_pattern in stop_patterns:
        stop_match = re.search(
            stop_pattern,
            value[search_offset:],
            flags=re.IGNORECASE | re.MULTILINE,
        )
        if not stop_match:
            continue
        end = min(end, search_offset + stop_match.start())
    return _slice_text(value[start:end], max_chars=max_chars)


def build_llm_summary_source_candidates(text: str) -> list[dict[str, str]]:
    cleaned = clean_pdf_text_for_summary(text)
    if not cleaned:
        return []

    candidates: list[dict[str, str]] = []

    section_prefix = r"^\s*(?:(?:\d+(?:\.\d+)*|[IVXLCDM]+)\.?\s+)?"
    abstract_slice = _extract_section_slice(
        cleaned,
        heading_pattern=rf"{section_prefix}abstract\b(?:\s*[:.\-—]|\s*$)",
        stop_patterns=(
            rf"{section_prefix}(?:introduction|background|methods?|methodology|materials?(?:\s+and\s+methods?)?|results?)\b(?:\s*[:.\-—]?\s*$|\s*[:\-—])",
        ),
        max_chars=3200,
    )
    if abstract_slice:
        candidates.append({"label": "abstract", "text": abstract_slice})

    intro_slice = _extract_section_slice(
        cleaned,
        heading_pattern=rf"{section_prefix}(?:introduction|background)\b(?:\s*[:.\-—]|\s*$)",
        stop_patterns=(
            rf"{section_prefix}(?:methods?|methodology|materials?(?:\s+and\s+methods?)?|results?|discussion|conclusion[s]?)\b(?:\s*[:.\-—]?\s*$|\s*[:\-—])",
        ),
        max_chars=2400,
    )
    ending_slice = _extract_section_slice(
        cleaned,
        heading_pattern=rf"{section_prefix}(?:discussion|conclusion[s]?|limitations?)\b(?:\s*[:.\-—]|\s*$)",
        stop_patterns=(
            rf"{section_prefix}(?:references|bibliography|acknowledgm?ents?)\b(?:\s*[:.\-—]?\s*$|\s*[:\-—])",
        ),
        max_chars=1800,
    )
    combined_sections = "\n\n".join(
        part for part in [abstract_slice, intro_slice, ending_slice] if part
    ).strip()
    if combined_sections:
        candidates.append(
            {
                "label": "abstract_intro_conclusion",
                "text": _slice_text(combined_sections, max_chars=5200),
            }
        )

    lead_slice = _slice_text(cleaned, max_chars=5200)
    if lead_slice:
        candidates.append({"label": "lead_pages", "text": lead_slice})

    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in candidates:
        candidate_text = str(item.get("text") or "").strip()
        if not candidate_text or candidate_text in seen:
            continue
        seen.add(candidate_text)
        deduped.append(
            {"label": str(item.get("label") or "candidate"), "text": candidate_text}
        )
    return deduped


def summarize_active_pdf_with_cache(
    *,
    workspace: Path,
    get_pdf_state: Callable[[], Dict[str, Any]],
    build_summary: Callable[..., str],
    build_llm_summary: Callable[..., str] | None = None,
    require_llm_summary: bool = False,
    max_pages: int = 80,
    max_extract_chars: int = 350000,
    on_telemetry: Callable[[Dict[str, Any]], None] | None = None,
) -> str:
    started = time.perf_counter()
    telemetry: Dict[str, Any] = {
        "ok": False,
        "reason": "",
        "cache_hit": False,
        "pdf_path": "",
        "cache_path": "",
        "pages_scanned": 0,
        "text_chars": 0,
        "extract_ms": 0.0,
        "summary_ms": 0.0,
        "llm_ms": 0.0,
        "summary_mode": "extractive",
        "total_ms": 0.0,
    }

    def _emit_telemetry() -> None:
        telemetry["total_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        if callable(on_telemetry):
            try:
                on_telemetry(dict(telemetry))
            except Exception:
                pass

    state = get_pdf_state()
    if not isinstance(state, dict) or not bool(state.get("ok")):
        telemetry["reason"] = "state_unavailable"
        _emit_telemetry()
        return ""
    if not bool(state.get("has_pdf")):
        telemetry["reason"] = "no_pdf_open"
        _emit_telemetry()
        return ""
    path_text = str(state.get("path") or "").strip()
    if not path_text:
        telemetry["reason"] = "missing_pdf_path"
        _emit_telemetry()
        return ""
    pdf_path = Path(path_text).expanduser()
    telemetry["pdf_path"] = str(pdf_path)
    if not pdf_path.exists() or not pdf_path.is_file():
        telemetry["reason"] = "pdf_not_found"
        _emit_telemetry()
        return ""

    try:
        stat = pdf_path.stat()
        cache_key = hashlib.sha1(
            f"{pdf_path.resolve()}:{int(stat.st_mtime_ns)}:{int(stat.st_size)}".encode(
                "utf-8"
            ),
            usedforsecurity=False,
        ).hexdigest()[:12]
    except Exception:
        cache_key = hashlib.sha1(
            str(pdf_path).encode("utf-8"), usedforsecurity=False
        ).hexdigest()[:12]
    safe_stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", pdf_path.stem).strip("._-")
    if not safe_stem:
        safe_stem = "paper"
    cache_dir = workspace / "pdf_text_cache"
    cache_path = cache_dir / f"{safe_stem[:80]}_{cache_key}.md"
    telemetry["cache_path"] = str(cache_path)

    extracted_text = ""
    extract_started = time.perf_counter()
    if cache_path.exists():
        try:
            extracted_text = str(cache_path.read_text(encoding="utf-8") or "").strip()
            telemetry["cache_hit"] = bool(extracted_text)
        except Exception:
            extracted_text = ""
    if not extracted_text:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            telemetry["reason"] = "cache_dir_unavailable"
            _emit_telemetry()
            return ""
        try:
            pages, pages_scanned, _backend = extract_pdf_pages(
                pdf_path,
                max_pages=max_pages,
                max_chars=max_extract_chars,
                dehyphenate=True,
            )
            telemetry["pages_scanned"] = int(pages_scanned)
        except RuntimeError:
            telemetry["reason"] = "fitz_unavailable"
            _emit_telemetry()
            return ""
        except Exception:
            telemetry["reason"] = "pdf_extract_failed"
            _emit_telemetry()
            return ""
        body = normalize_pages_for_display(pages)
        if not body:
            telemetry["reason"] = "pdf_text_empty"
            telemetry["extract_ms"] = round(
                (time.perf_counter() - extract_started) * 1000.0, 1
            )
            _emit_telemetry()
            return ""
        extracted_text = f"# Extracted PDF Text\n\nSource: {pdf_path}\n\n{body}\n"
        try:
            atomic_write_text(cache_path, extracted_text, encoding="utf-8")
        except Exception:
            pass
    telemetry["extract_ms"] = round((time.perf_counter() - extract_started) * 1000.0, 1)
    telemetry["text_chars"] = len(extracted_text)

    summary_started = time.perf_counter()
    summary_source = clean_pdf_text_for_summary(extracted_text)
    summary = ""
    llm_started = time.perf_counter()
    if callable(build_llm_summary):
        try:
            summary = str(
                build_llm_summary(
                    summary_source,
                    max_chars=1800,
                    pdf_path=str(pdf_path),
                )
                or ""
            ).strip()
        except TypeError:
            try:
                summary = str(build_llm_summary(summary_source) or "").strip()
            except Exception:
                summary = ""
        except Exception:
            summary = ""
    telemetry["llm_ms"] = round((time.perf_counter() - llm_started) * 1000.0, 1)
    if summary:
        telemetry["summary_mode"] = "llm"
    else:
        if bool(require_llm_summary):
            telemetry["reason"] = "llm_summary_unavailable"
            telemetry["summary_ms"] = round(
                (time.perf_counter() - summary_started) * 1000.0, 1
            )
            _emit_telemetry()
            return ""
        summary = build_summary(summary_source, max_sentences=10, max_chars=1800)
    if not summary:
        compact = " ".join(summary_source.split())
        if not compact:
            telemetry["reason"] = "summary_empty"
            telemetry["summary_ms"] = round(
                (time.perf_counter() - summary_started) * 1000.0, 1
            )
            _emit_telemetry()
            return ""
        summary = compact[:1800].rstrip()
    telemetry["summary_ms"] = round((time.perf_counter() - summary_started) * 1000.0, 1)
    telemetry["ok"] = True
    _emit_telemetry()
    return (
        f"Summary of the open PDF ({pdf_path.name}):\n"
        f"{summary}\n\n"
        f"Cached extraction: {cache_path}"
    )
