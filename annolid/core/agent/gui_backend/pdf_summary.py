from __future__ import annotations

import hashlib
from pathlib import Path
import re
import time
from typing import Any, Callable, Dict


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
        "atlanta",
        "corresponding author",
    )
    if sum(1 for token in noisy_affiliations if token in lowered) >= 3:
        return True
    if len(value) > 900 and ("\n-" not in value and "\n1." not in value):
        return True
    return False


def clean_pdf_text_for_summary(text: str) -> str:
    value = str(text or "").replace("\u2029", "\n")
    if not value.strip():
        return ""
    abstract_match = re.search(r"\babstract\b[:\s]", value, flags=re.IGNORECASE)
    if abstract_match:
        value = value[abstract_match.start() :]
    lines: list[str] = []
    for raw in value.splitlines():
        line = str(raw or "").strip()
        if not line:
            continue
        if "@" in line and len(line) < 160:
            continue
        if (
            re.search(
                r"\b(university|institute|department|school|atlanta)\b",
                line,
                flags=re.IGNORECASE,
            )
            and len(line) < 140
        ):
            continue
        if re.match(r"^page\s+\d+\b", line, flags=re.IGNORECASE):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def summarize_active_pdf_with_cache(
    *,
    workspace: Path,
    get_pdf_state: Callable[[], Dict[str, Any]],
    build_summary: Callable[..., str],
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
        import fitz  # type: ignore
    except Exception:
        telemetry["reason"] = "fitz_unavailable"
        _emit_telemetry()
        return ""

    try:
        stat = pdf_path.stat()
        cache_key = hashlib.sha1(
            f"{pdf_path.resolve()}:{int(stat.st_mtime_ns)}:{int(stat.st_size)}".encode(
                "utf-8"
            )
        ).hexdigest()[:12]
    except Exception:
        cache_key = hashlib.sha1(str(pdf_path).encode("utf-8")).hexdigest()[:12]
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
        parts: list[str] = []
        try:
            with fitz.open(str(pdf_path)) as doc:
                total_pages = int(getattr(doc, "page_count", 0) or 0)
                for idx in range(min(total_pages, max_pages)):
                    page = doc.load_page(idx)
                    page_text = str(page.get_text("text") or "").strip()
                    telemetry["pages_scanned"] = int(idx + 1)
                    if not page_text:
                        continue
                    parts.append(f"\n\n## Page {idx + 1}\n{page_text}")
                    if sum(len(item) for item in parts) >= max_extract_chars:
                        break
        except Exception:
            telemetry["reason"] = "pdf_extract_failed"
            _emit_telemetry()
            return ""
        body = "".join(parts).strip()
        if not body:
            telemetry["reason"] = "pdf_text_empty"
            telemetry["extract_ms"] = round(
                (time.perf_counter() - extract_started) * 1000.0, 1
            )
            _emit_telemetry()
            return ""
        extracted_text = f"# Extracted PDF Text\n\nSource: {pdf_path}\n\n{body}\n"
        try:
            cache_path.write_text(extracted_text, encoding="utf-8")
        except Exception:
            pass
    telemetry["extract_ms"] = round((time.perf_counter() - extract_started) * 1000.0, 1)
    telemetry["text_chars"] = len(extracted_text)

    summary_started = time.perf_counter()
    summary_source = clean_pdf_text_for_summary(extracted_text)
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
