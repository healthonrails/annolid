"""Framework-agnostic PDF parsing, text extraction, repair, and section utilities.

All optional PDF backends (PyMuPDF / ``fitz`` and ``pypdf``) are imported lazily
inside functions so lean ``annolid[gui]`` installs remain unaffected.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
from typing import Any, Optional, Sequence

from annolid.utils.logger import logger

_PDF_HEADER_SCAN_BYTES = 1024
_PDF_TAIL_SCAN_BYTES = 4096

_COMMON_SECTION_TITLES = frozenset(
    {
        "abstract",
        "introduction",
        "background",
        "related work",
        "methods",
        "methodology",
        "materials and methods",
        "experimental setup",
        "experiments",
        "results",
        "results and discussion",
        "discussion",
        "limitations",
        "conclusion",
        "conclusions",
        "future work",
        "references",
        "bibliography",
        "acknowledgments",
        "acknowledgements",
        "appendix",
        "supplementary materials",
        "supporting information",
    }
)

_NUMBERED_SECTION_RE = re.compile(
    r"^(?:(?:\d+(?:\.\d+){0,4})\.?|(?:[IVXLCDM]{1,6})\.)\s+([A-Za-z][^\n]*)$"
)

_SENTENCE_CONTINUATION_WORDS = frozenset(
    {
        "a",
        "an",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "did",
        "do",
        "for",
        "from",
        "had",
        "has",
        "have",
        "in",
        "is",
        "it",
        "may",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "these",
        "this",
        "those",
        "to",
        "was",
        "we",
        "were",
        "which",
        "while",
        "with",
    }
)

_HYPHEN_WRAP_RE = re.compile(
    r"([A-Za-z]{2,})[-\u2010\u2011\u2012\u2013]\s*(?:\r?\n)+\s*([a-z]{2,})"
)

_INLINE_HYPHEN_WRAP_RE = re.compile(
    r"([A-Za-z]{2,})[-\u2010\u2011\u2012\u2013][ \t]+([a-z]{2,})"
)

_CJK_CHAR_CLASS = r"\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af"
_CJK_NEWLINE_GAP_RE = re.compile(
    rf"([{_CJK_CHAR_CLASS}])[ \t]*\r?\n[ \t]*([{_CJK_CHAR_CLASS}])"
)


def _import_fitz() -> Any | None:
    try:
        import fitz  # type: ignore[import-not-found]
    except Exception:
        return None
    _silence_mupdf_diagnostics(fitz)
    return fitz


def _silence_mupdf_diagnostics(fitz_mod: Any) -> None:
    try:
        tools = getattr(fitz_mod, "TOOLS", None)
        if tools is None:
            return
        disable_errors = getattr(tools, "mupdf_display_errors", None)
        disable_warnings = getattr(tools, "mupdf_display_warnings", None)
        if callable(disable_errors):
            disable_errors(False)
        if callable(disable_warnings):
            disable_warnings(False)
    except Exception:
        return


def _fitz_page_count(doc: Any) -> int:
    raw = getattr(doc, "page_count", None)
    if raw is not None:
        try:
            return max(0, int(raw))
        except Exception:
            pass
    try:
        return max(0, int(len(doc)))
    except Exception:
        return 0


def _fitz_get_page(doc: Any, index: int) -> Any:
    loader = getattr(doc, "load_page", None)
    if callable(loader):
        return loader(index)
    return doc[index]


def _unlock_fitz_doc(doc: Any) -> bool:
    """Attempt empty-password unlock for restricted PDFs opened via PyMuPDF."""
    try:
        if not bool(getattr(doc, "needs_pass", False)):
            return True
        auth = getattr(doc, "authenticate", None)
        if callable(auth):
            return bool(auth(""))
    except Exception:
        return False
    return False


def _open_pypdf_reader(path: Path | str, *, strict: bool = False) -> Any:
    """Open a ``pypdf.PdfReader`` tolerantly and unlock empty-password PDFs."""
    from pypdf import PdfReader  # type: ignore[import-not-found]

    try:
        reader = PdfReader(str(path), strict=strict)
    except TypeError:
        reader = PdfReader(str(path))
    try:
        if bool(getattr(reader, "is_encrypted", False)):
            decrypt = getattr(reader, "decrypt", None)
            if callable(decrypt):
                decrypt("")
    except Exception:
        logger.debug("pypdf empty-password decrypt attempt failed", exc_info=True)
    return reader


def _extract_pypdf_page_text(page: Any) -> str:
    """Safely extract text from a single ``pypdf`` page without aborting whole-doc scans."""
    try:
        raw = page.extract_text()
    except Exception:
        logger.debug("pypdf page.extract_text() failed; skipping page", exc_info=True)
        return ""
    return str(raw or "")


def _extract_pypdf_outline(reader: Any) -> list[dict[str, object]]:
    """Flatten nested ``pypdf`` outline entries into ``[{level, title, page}]``."""
    raw_outline = getattr(reader, "outline", None)
    if not raw_outline:
        return []
    get_page_num = getattr(reader, "get_destination_page_number", None)
    out: list[dict[str, object]] = []

    def _walk(items: Any, level: int) -> None:
        if not isinstance(items, (list, tuple)):
            return
        for entry in items:
            if isinstance(entry, (list, tuple)):
                _walk(entry, level + 1)
                continue
            title_raw: object = getattr(entry, "title", None)
            if title_raw is None and isinstance(entry, dict):
                title_raw = entry.get("/Title") or entry.get("title")
            clean_title = " ".join(str(title_raw or "").split()).strip()
            if not clean_title:
                continue
            page_idx: Optional[int] = None
            if callable(get_page_num):
                try:
                    resolved = get_page_num(entry)
                    if resolved is not None:
                        page_idx = max(0, int(resolved))
                except Exception:
                    page_idx = None
            if page_idx is None:
                continue
            out.append({"level": int(level), "title": clean_title, "page": page_idx})

    try:
        _walk(raw_outline, 1)
    except Exception:
        return []
    return out


def dehyphenate_pdf_text(text: str, *, merge_inline_space_hyphens: bool = False) -> str:
    """Clean discretionary hyphens and line-wrap hyphenation artifacts in PDF text."""
    value = str(text or "")
    if not value:
        return ""
    value = value.replace("\u00ad", "").replace("\u2029", "\n").replace("\u00a0", " ")
    value = _HYPHEN_WRAP_RE.sub(r"\1\2", value)
    if merge_inline_space_hyphens:
        value = _INLINE_HYPHEN_WRAP_RE.sub(r"\1\2", value)
    return value


def normalize_pdf_text(
    text: str,
    *,
    preserve_paragraphs: bool = True,
    merge_inline_space_hyphens: bool = False,
) -> str:
    """Normalize extracted PDF text while preserving paragraph structure when requested."""
    cleaned = dehyphenate_pdf_text(
        text, merge_inline_space_hyphens=merge_inline_space_hyphens
    )
    if not cleaned:
        return ""
    cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
    if not preserve_paragraphs:
        cleaned = _CJK_NEWLINE_GAP_RE.sub(r"\1\2", cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.splitlines()]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def pdf_needs_repair_hint(path: Path) -> bool:
    """Return True if a PDF file lacks a valid header or EOF marker."""
    try:
        with Path(path).open("rb") as fh:
            head = fh.read(_PDF_HEADER_SCAN_BYTES)
            if b"%PDF-" not in head:
                return True
            try:
                fh.seek(-_PDF_TAIL_SCAN_BYTES, os.SEEK_END)
            except OSError:
                fh.seek(0)
            tail = fh.read()
        return b"%%EOF" not in tail
    except Exception:
        return False


def repair_pdf(
    path: Path,
    *,
    reason: str = "load",
    out_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Attempt to rewrite a structurally damaged PDF into a clean temporary copy.

    Tries ``pypdf`` first (tolerant non-strict parser) and falls back to PyMuPDF
    ``save(..., garbage=3, deflate=True)`` when ``pypdf`` is unavailable or fails.
    """
    source = Path(path)
    if not source.exists() or not source.is_file():
        return None
    target_dir = (
        Path(out_dir)
        if out_dir is not None
        else Path(tempfile.gettempdir()) / "annolid_pdf_repair"
    )
    try:
        mtime_ns = int(source.stat().st_mtime_ns)
    except OSError:
        mtime_ns = 0
    repaired_path = target_dir / f"{source.stem}.{mtime_ns}.repaired.pdf"

    # Strategy 1: pypdf tolerant rewrite
    try:
        from pypdf import PdfWriter  # type: ignore[import-not-found]

        reader = _open_pypdf_reader(source, strict=False)
        if len(reader.pages) > 0:
            writer = PdfWriter()
            for page in reader.pages:
                writer.add_page(page)
            target_dir.mkdir(parents=True, exist_ok=True)
            tmp_path: Optional[Path] = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb",
                    dir=str(target_dir),
                    prefix=f".{repaired_path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as out:
                    tmp_path = Path(out.name)
                    writer.write(out)
                    out.flush()
                os.replace(tmp_path, repaired_path)
                tmp_path = None
            finally:
                if tmp_path is not None:
                    tmp_path.unlink(missing_ok=True)
            logger.warning(
                "Repaired PDF via pypdf: %s (reason=%s)", repaired_path, reason
            )
            return repaired_path
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("pypdf repair failed for %s (%s): %s", source, reason, exc)

    # Strategy 2: PyMuPDF clean rewrite
    fitz = _import_fitz()
    if fitz is not None:
        tmp_path = None
        try:
            with fitz.open(str(source)) as doc:
                _unlock_fitz_doc(doc)
                if _fitz_page_count(doc) > 0 and callable(getattr(doc, "save", None)):
                    target_dir.mkdir(parents=True, exist_ok=True)
                    fd, raw_tmp = tempfile.mkstemp(
                        dir=str(target_dir),
                        prefix=f".{repaired_path.name}.",
                        suffix=".tmp",
                    )
                    os.close(fd)
                    tmp_path = Path(raw_tmp)
                    doc.save(str(tmp_path), garbage=3, deflate=True)
                    os.replace(tmp_path, repaired_path)
                    tmp_path = None
                    logger.warning(
                        "Repaired PDF via PyMuPDF: %s (reason=%s)",
                        repaired_path,
                        reason,
                    )
                    return repaired_path
        except Exception as exc:
            logger.warning("PyMuPDF repair failed for %s (%s): %s", source, reason, exc)
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
    return None


def _extract_fitz_page_text(
    page: Any,
    *,
    fallback_blocks_words: bool = False,
    min_fallback_chars: int = 60,
) -> str:
    chunk = ""
    try:
        chunk = str(page.get_text("text") or "")
    except Exception:
        chunk = ""
    chunk = chunk.replace("\u2029", "\n").strip()

    if not fallback_blocks_words or len(chunk) >= min_fallback_chars:
        return chunk

    # Fallback to blocks/words only when standard "text" extraction yielded little
    # or nothing (common on fragmented PDFs). Silently ignored for minimal mocks.
    try:
        blocks = page.get_text("blocks") or []
        block_texts: list[str] = []
        for b in blocks:
            if not isinstance(b, (list, tuple)) or len(b) < 5:
                continue
            t = str(b[4] or "").strip()
            if t:
                block_texts.append(t)
        if block_texts:
            chunk = (chunk + "\n" + "\n".join(block_texts)).strip()
    except Exception:
        pass

    if len(chunk) < min_fallback_chars:
        try:
            words = page.get_text("words") or []
            word_texts: list[str] = []
            for w in words:
                if not isinstance(w, (list, tuple)) or len(w) < 5:
                    continue
                t = str(w[4] or "").strip()
                if t:
                    word_texts.append(t)
            if word_texts:
                chunk = (chunk + "\n" + " ".join(word_texts)).strip()
        except Exception:
            pass
    return chunk


def extract_pdf_text(
    pdf_path: Path | str,
    *,
    start_index: int = 0,
    pages_limit: int = 20,
    dehyphenate: bool = False,
    fallback_blocks_words: bool = False,
    join_separator: str = "\n\n",
) -> tuple[str, int, int, str]:
    """Extract text from a page range of ``pdf_path``.

    Returns ``(text, pages_read, total_pages, backend)`` where ``backend`` is
    ``"pymupdf"`` or ``"pypdf"``. Raises ``RuntimeError`` if neither backend is
    installed.
    """
    path = Path(pdf_path)
    start_idx = max(0, int(start_index))
    limit = max(1, int(pages_limit))

    fitz = _import_fitz()
    if fitz is not None:
        with fitz.open(str(path)) as doc:
            _unlock_fitz_doc(doc)
            total_pages = _fitz_page_count(doc)
            if start_idx >= total_pages:
                return "", 0, total_pages, "pymupdf"
            end_idx = min(total_pages, start_idx + limit)
            parts: list[str] = []
            for idx in range(start_idx, end_idx):
                page = _fitz_get_page(doc, idx)
                chunk = _extract_fitz_page_text(
                    page, fallback_blocks_words=fallback_blocks_words
                )
                if dehyphenate:
                    chunk = dehyphenate_pdf_text(chunk)
                parts.append(chunk)
            return (
                join_separator.join(parts),
                end_idx - start_idx,
                total_pages,
                "pymupdf",
            )

    try:
        reader = _open_pypdf_reader(path, strict=False)
    except ImportError as exc:
        raise RuntimeError(
            "PDF extraction backend is not available. Install pymupdf or pypdf."
        ) from exc

    total_pages = len(reader.pages)
    if start_idx >= total_pages:
        return "", 0, total_pages, "pypdf"
    end_idx = min(total_pages, start_idx + limit)
    parts = []
    for idx in range(start_idx, end_idx):
        chunk = _extract_pypdf_page_text(reader.pages[idx])
        if dehyphenate:
            chunk = dehyphenate_pdf_text(chunk)
        parts.append(chunk)
    return (
        join_separator.join(parts),
        end_idx - start_idx,
        total_pages,
        "pypdf",
    )


def extract_pdf_pages(
    pdf_path: Path | str,
    *,
    max_pages: int = 80,
    max_chars: int = 350000,
    dehyphenate: bool = True,
) -> tuple[list[tuple[int, str]], int, str]:
    """Extract ``(1_based_page_number, page_text)`` tuples up to limits.

    Returns ``(pages, pages_scanned, backend)``.
    """
    path = Path(pdf_path)
    page_limit = max(1, int(max_pages))
    char_limit = max(1, int(max_chars))

    fitz = _import_fitz()
    if fitz is not None:
        pages_out: list[tuple[int, str]] = []
        pages_scanned = 0
        total_chars = 0
        with fitz.open(str(path)) as doc:
            _unlock_fitz_doc(doc)
            total_pages = _fitz_page_count(doc)
            for idx in range(min(total_pages, page_limit)):
                page = _fitz_get_page(doc, idx)
                chunk = _extract_fitz_page_text(page, fallback_blocks_words=False)
                if dehyphenate:
                    chunk = dehyphenate_pdf_text(chunk)
                pages_scanned = idx + 1
                if not chunk:
                    continue
                pages_out.append((idx + 1, chunk))
                total_chars += len(chunk)
                if total_chars >= char_limit:
                    break
        return pages_out, pages_scanned, "pymupdf"

    try:
        reader = _open_pypdf_reader(path, strict=False)
    except ImportError as exc:
        raise RuntimeError(
            "PDF extraction backend is not available. Install pymupdf or pypdf."
        ) from exc

    pages_out = []
    pages_scanned = 0
    total_chars = 0
    total_pages = len(reader.pages)
    for idx in range(min(total_pages, page_limit)):
        chunk = _extract_pypdf_page_text(reader.pages[idx]).strip()
        if dehyphenate:
            chunk = dehyphenate_pdf_text(chunk)
        pages_scanned = idx + 1
        if not chunk:
            continue
        pages_out.append((idx + 1, chunk))
        total_chars += len(chunk)
        if total_chars >= char_limit:
            break
    return pages_out, pages_scanned, "pypdf"


def extract_pdf_text_sample(
    pdf_path: Path | str,
    *,
    max_pages: int = 3,
    max_chars: int = 12000,
) -> str:
    """Extract a bounded text sample (stem, metadata, first few pages) for language detection."""
    parts: list[str] = []
    path = Path(pdf_path)
    try:
        if path.stem:
            parts.append(path.stem)
    except Exception:
        pass

    fitz = _import_fitz()
    if fitz is not None:
        try:
            with fitz.open(str(path)) as doc:
                _unlock_fitz_doc(doc)
                meta = getattr(doc, "metadata", None) or {}
                if isinstance(meta, dict):
                    for key in ("title", "subject", "keywords"):
                        val = str(meta.get(key) or "").strip()
                        if val:
                            parts.append(val)
                page_limit = min(max(1, int(max_pages)), _fitz_page_count(doc))
                for idx in range(page_limit):
                    try:
                        page = _fitz_get_page(doc, idx)
                    except Exception:
                        continue
                    chunk = _extract_fitz_page_text(page, fallback_blocks_words=True)
                    if chunk:
                        parts.append(chunk)
                    if sum(len(item) for item in parts) >= max_chars:
                        break
            return "\n".join(p for p in parts if p).strip()[:max_chars]
        except Exception:
            pass

    try:
        reader = _open_pypdf_reader(path, strict=False)
        meta = getattr(reader, "metadata", None)
        if meta is not None:
            for attr in ("title", "subject"):
                val = str(getattr(meta, attr, "") or "").strip()
                if val:
                    parts.append(val)
        page_limit = min(max(1, int(max_pages)), len(reader.pages))
        for idx in range(page_limit):
            chunk = _extract_pypdf_page_text(reader.pages[idx]).strip()
            if chunk:
                parts.append(chunk)
            if sum(len(item) for item in parts) >= max_chars:
                break
    except Exception:
        pass
    return "\n".join(p for p in parts if p).strip()[:max_chars]


def extract_pdf_title(pdf_path: Path | str) -> Optional[str]:
    """Infer a PDF document title from metadata or the first substantive text line."""
    path = Path(pdf_path)

    def _first_meaningful_line(raw_text: str) -> Optional[str]:
        for line in str(raw_text or "").splitlines():
            candidate = " ".join(line.split()).strip()
            if len(candidate) >= 8:
                return candidate
        return None

    fitz = _import_fitz()
    if fitz is not None:
        try:
            with fitz.open(str(path)) as doc:
                _unlock_fitz_doc(doc)
                metadata = getattr(doc, "metadata", None) or {}
                if isinstance(metadata, dict):
                    title = str(metadata.get("title") or "").strip()
                    if title:
                        return title
                if _fitz_page_count(doc) > 0:
                    first_line = _first_meaningful_line(
                        _extract_fitz_page_text(_fitz_get_page(doc, 0))
                    )
                    if first_line:
                        return first_line
        except Exception:
            logger.debug("PyMuPDF could not extract a PDF title", exc_info=True)

    try:
        reader = _open_pypdf_reader(path, strict=False)
        title = str(
            getattr(getattr(reader, "metadata", None), "title", "") or ""
        ).strip()
        if title:
            return title
        if reader.pages:
            first_line = _first_meaningful_line(
                _extract_pypdf_page_text(reader.pages[0])
            )
            if first_line:
                return first_line
    except Exception:
        logger.debug("pypdf could not extract a PDF title", exc_info=True)
    return None


def extract_pdf_outline(pdf_path: Path | str) -> list[dict[str, object]]:
    """Extract embedded PDF outline/TOC bookmarks via PyMuPDF or ``pypdf``."""
    path = Path(pdf_path)
    bookmarks: list[dict[str, object]] = []
    fitz = _import_fitz()
    if fitz is not None:
        try:
            with fitz.open(str(path)) as doc:
                _unlock_fitz_doc(doc)
                get_toc = getattr(doc, "get_toc", None)
                toc = get_toc(simple=True) if callable(get_toc) else []
                for entry in toc or []:
                    if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                        continue
                    level, title, page = entry[0], entry[1], entry[2]
                    clean_title = " ".join(str(title or "").split()).strip()
                    if not clean_title:
                        continue
                    try:
                        bookmarks.append(
                            {
                                "level": int(level),
                                "title": clean_title,
                                "page": max(0, int(page) - 1),
                            }
                        )
                    except Exception:
                        continue
            return bookmarks
        except Exception:
            return []

    try:
        reader = _open_pypdf_reader(path, strict=False)
        return _extract_pypdf_outline(reader)
    except Exception:
        return []


def is_section_heading_candidate(line: str) -> bool:
    """Return True if a single line looks like a real document section heading."""
    text = " ".join(str(line or "").split()).strip()
    if len(text) < 3 or len(text) > 140:
        return False
    if text.endswith(".") and len(text.split()) > 5:
        return False

    lowered = text.lower().rstrip(":")
    if lowered in _COMMON_SECTION_TITLES:
        return True

    numbered = _NUMBERED_SECTION_RE.match(text)
    if numbered:
        heading_body = numbered.group(1).strip()
        body_words = heading_body.split()
        if 1 <= len(body_words) <= 12 and not heading_body.endswith("."):
            first_word = body_words[0].lower()
            # Reject prose sentences starting with a count or pronoun like
            # "10 mice were tested..." or "I. have a...".
            if len(body_words) <= 6 or first_word not in _SENTENCE_CONTINUATION_WORDS:
                if body_words[0][:1].isupper() or heading_body.lower().rstrip(":") in (
                    _COMMON_SECTION_TITLES
                ):
                    return True

    words = text.split()
    if len(words) > 12:
        return False
    if text.endswith((",", ";")):
        return False
    alpha = [c for c in text if c.isalpha()]
    if len(alpha) < 3:
        return False
    upper_ratio = sum(1 for c in alpha if c.isupper()) / max(len(alpha), 1)
    title_ratio = sum(1 for w in words if w[:1].isupper()) / max(len(words), 1)
    starts_cap = bool(text[:1].isupper())
    return starts_cap and (
        upper_ratio >= 0.65 or (len(words) <= 8 and title_ratio >= 0.75)
    )


def extract_pdf_sections(
    pdf_path: Path | str,
    *,
    max_sections: int = 20,
    max_pages: int = 12,
) -> dict[str, object]:
    """Detect section headings from PDF outline (preferred) and page text heuristics."""
    path = Path(pdf_path)
    section_limit = max(1, min(int(max_sections or 20), 200))
    page_limit = max(1, min(int(max_pages or 12), 100))

    found: list[dict[str, object]] = []
    seen: set[str] = set()
    total_pages = 0
    scanned_pages = 0

    def _add_section(title: str, page_num: int, line_num: int, source: str) -> None:
        clean = " ".join(str(title or "").split()).strip()
        key = clean.lower()
        if not key or key in seen or len(found) >= section_limit:
            return
        seen.add(key)
        found.append(
            {
                "title": clean,
                "page": int(page_num),
                "line": int(line_num),
                "source": source,
            }
        )

    fitz = _import_fitz()
    if fitz is not None:
        with fitz.open(str(path)) as doc:
            _unlock_fitz_doc(doc)
            total_pages = _fitz_page_count(doc)
            if total_pages <= 0:
                return {"ok": False, "error": "PDF has no pages."}

            # 1) Prefer embedded PDF outline/TOC when available.
            get_toc = getattr(doc, "get_toc", None)
            toc = get_toc(simple=True) if callable(get_toc) else []
            for entry in toc or []:
                if not isinstance(entry, (list, tuple)) or len(entry) < 3:
                    continue
                _, title, page_num = entry[0], entry[1], entry[2]
                try:
                    p_num = int(page_num)
                except Exception:
                    continue
                if p_num < 1 or p_num > min(page_limit, total_pages):
                    continue
                _add_section(str(title), p_num, 0, "outline")
                if len(found) >= section_limit:
                    break

            # 2) Supplement with page text heading scan when outline is sparse/absent.
            end_page = min(page_limit, total_pages)
            if len(found) < section_limit:
                for page_index in range(end_page):
                    scanned_pages += 1
                    page = _fitz_get_page(doc, page_index)
                    lines = _extract_fitz_page_text(page).splitlines()
                    for line_idx, line in enumerate(lines):
                        title = " ".join(str(line or "").split()).strip()
                        if not is_section_heading_candidate(title):
                            continue
                        _add_section(title, page_index + 1, line_idx + 1, "text")
                        if len(found) >= section_limit:
                            break
                    if len(found) >= section_limit:
                        break
            else:
                scanned_pages = end_page
    else:
        try:
            reader = _open_pypdf_reader(path, strict=False)
        except ImportError:
            return {"ok": False, "error": "PyMuPDF (pymupdf) is required."}
        total_pages = len(reader.pages)
        if total_pages <= 0:
            return {"ok": False, "error": "PDF has no pages."}
        end_page = min(page_limit, total_pages)
        for outline_entry in _extract_pypdf_outline(reader):
            p_num = int(outline_entry.get("page", 0)) + 1
            if 1 <= p_num <= end_page:
                _add_section(str(outline_entry.get("title") or ""), p_num, 0, "outline")
            if len(found) >= section_limit:
                break
        if len(found) < section_limit:
            for page_index in range(end_page):
                scanned_pages += 1
                lines = _extract_pypdf_page_text(reader.pages[page_index]).splitlines()
                for line_idx, line in enumerate(lines):
                    title = " ".join(str(line or "").split()).strip()
                    if not is_section_heading_candidate(title):
                        continue
                    _add_section(title, page_index + 1, line_idx + 1, "text")
                    if len(found) >= section_limit:
                        break
                if len(found) >= section_limit:
                    break
        else:
            scanned_pages = end_page

    return {
        "ok": True,
        "path": str(path),
        "title": path.name,
        "sections": found,
        "count": len(found),
        "max_sections": section_limit,
        "scanned_pages": scanned_pages,
        "total_pages": total_pages,
    }


def detect_document_language(text: str) -> str:
    """Best-effort Kokoro-compatible language code detection from a document text sample."""
    sample = (text or "").strip()[:8000]
    if not sample:
        return ""
    counts = {
        "hiragana": 0,
        "katakana": 0,
        "han": 0,
        "hangul": 0,
        "cyrillic": 0,
        "devanagari": 0,
        "latin": 0,
    }
    for ch in sample:
        o = ord(ch)
        if 0x3040 <= o <= 0x309F:
            counts["hiragana"] += 1
        elif 0x30A0 <= o <= 0x30FF:
            counts["katakana"] += 1
        elif 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:
            counts["han"] += 1
        elif 0xAC00 <= o <= 0xD7AF:
            counts["hangul"] += 1
        elif 0x0400 <= o <= 0x04FF:
            counts["cyrillic"] += 1
        elif 0x0900 <= o <= 0x097F:
            counts["devanagari"] += 1
        elif (0x0041 <= o <= 0x007A) or (0x00C0 <= o <= 0x024F):
            counts["latin"] += 1

    kana = counts["hiragana"] + counts["katakana"]
    han = counts["han"]
    total = max(
        1,
        kana
        + han
        + counts["hangul"]
        + counts["cyrillic"]
        + counts["devanagari"]
        + counts["latin"],
    )

    if kana >= 8 and kana / total >= 0.06:
        return "ja"
    if counts["hangul"] >= 8 and counts["hangul"] / total >= 0.06:
        return "ko"
    if han >= 5 and kana < 6 and (han / total >= 0.12 or han >= 30):
        return "zh"
    if counts["cyrillic"] >= 10 and counts["cyrillic"] / total >= 0.08:
        return "ru"
    if counts["devanagari"] >= 10 and counts["devanagari"] / total >= 0.08:
        return "hi"
    if counts["latin"] >= 24 and counts["latin"] / total >= 0.20:
        return "en-us"
    return ""


def normalize_pages_for_display(pages: Sequence[tuple[int, str]]) -> str:
    """Format extracted ``(page_num, text)`` entries into a markdown cache body."""
    parts = [
        f"\n\n## Page {int(page_num)}\n{str(text).strip()}"
        for page_num, text in pages
        if str(text or "").strip()
    ]
    return "".join(parts).strip()


__all__ = [
    "dehyphenate_pdf_text",
    "detect_document_language",
    "extract_pdf_outline",
    "extract_pdf_pages",
    "extract_pdf_sections",
    "extract_pdf_text",
    "extract_pdf_text_sample",
    "extract_pdf_title",
    "is_section_heading_candidate",
    "normalize_pages_for_display",
    "normalize_pdf_text",
    "pdf_needs_repair_hint",
    "repair_pdf",
]
