from __future__ import annotations

import re
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Optional


def extract_first_web_url(
    text: str,
    *,
    extract_web_urls: Callable[[str], List[str]],
) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""
    candidates = extract_web_urls(raw)
    if raw.lower().startswith(("http://", "https://")) and raw not in candidates:
        candidates.insert(0, raw.rstrip(").,;!?"))
    if not candidates:
        domain_match = re.search(
            r"\b(?:www\.)?[a-z0-9][a-z0-9\-]{0,62}"
            r"(?:\.[a-z0-9][a-z0-9\-]{0,62})+(?::\d+)?(?:/[^\s<>\"]*)?",
            raw,
            flags=re.IGNORECASE,
        )
        if domain_match:
            domain_url = str(domain_match.group(0) or "").strip().rstrip(").,;!?")
            if domain_url:
                return f"https://{domain_url}"
        return ""
    return str(candidates[0] or "").strip()


async def open_url_tool(
    url: str,
    *,
    extract_first_web_url_fn: Callable[[str], str],
    emit_progress: Callable[[str], None],
    run_arxiv_search: Callable[[str], Awaitable[None]],
    invoke_open_url: Callable[[str], bool],
) -> Dict[str, object]:
    raw_text = str(url or "").strip()
    target_url = extract_first_web_url_fn(raw_text)

    candidate_url = target_url or raw_text
    arxiv_match = re.search(
        r"arxiv\.org/(?:pdf|abs)/(\d{4}\.\d{4,5}(?:v\d+)?)", candidate_url
    )
    if arxiv_match:
        arxiv_id = arxiv_match.group(1)
        emit_progress(f"arXiv: {arxiv_id}")
        await run_arxiv_search(f"id:{arxiv_id}")
        return {
            "ok": True,
            "url": f"https://arxiv.org/abs/{arxiv_id}",
            "id": arxiv_id,
        }

    if not target_url:
        candidate = raw_text
        lowered = candidate.lower()
        for prefix in ("open ", "load ", "show ", "open url ", "open file "):
            if lowered.startswith(prefix):
                candidate = candidate[len(prefix) :].strip()
                break
        candidate_path = Path(candidate).expanduser()
        if candidate_path.exists() and candidate_path.is_file():
            target_url = str(candidate_path)

    if not target_url:
        return {
            "ok": False,
            "error": "URL or local file path not found in provided text.",
            "input": raw_text,
            "hint": (
                "Provide a URL (e.g. google.com) or an existing local file path "
                "(e.g. /path/to/file.html)."
            ),
        }

    lower_target = target_url.lower()
    if not (
        lower_target.startswith(("http://", "https://", "file://"))
        or Path(target_url).expanduser().is_file()
    ):
        return {
            "ok": False,
            "error": "Only http(s) URLs or existing local files are supported.",
            "url": target_url,
        }
    if not invoke_open_url(target_url):
        return {"ok": False, "error": "Failed to queue GUI URL open action"}
    return {"ok": True, "queued": True, "url": target_url}


def open_in_browser_tool(
    url: str,
    *,
    extract_first_web_url_fn: Callable[[str], str],
    invoke_open_in_browser: Callable[[str], bool],
) -> Dict[str, object]:
    target_url = extract_first_web_url_fn(url)
    if not target_url:
        return {
            "ok": False,
            "error": "URL not found in provided text.",
            "input": str(url or "").strip(),
            "hint": "Provide a URL, for example google.com or https://example.org.",
        }
    if not invoke_open_in_browser(target_url):
        return {"ok": False, "error": "Failed to queue browser open action"}
    return {"ok": True, "queued": True, "url": target_url}


async def open_pdf_tool(
    path: str,
    *,
    extract_pdf_path_candidates: Callable[[str], List[str]],
    extract_web_urls: Callable[[str], List[str]],
    download_pdf: Callable[[str], Awaitable[Optional[Path]]],
    resolve_pdf_path: Callable[[str], Optional[Path]],
    list_available_pdfs: Callable[[int], List[Path]],
    invoke_open_pdf: Callable[[Path], bool],
) -> Dict[str, object]:
    path_text = str(path or "").strip()
    path_candidates = extract_pdf_path_candidates(path_text) if path_text else []
    generic_url_candidates = extract_web_urls(path_text) if path_text else []
    has_explicit_pdf_path = bool(path_candidates)
    resolved_path: Optional[Path] = None

    if has_explicit_pdf_path or generic_url_candidates:
        url_candidate = next(
            (
                candidate
                for candidate in path_candidates
                if str(candidate).lower().startswith(("http://", "https://"))
            ),
            "",
        )
        if not url_candidate and generic_url_candidates:
            url_candidate = str(generic_url_candidates[0] or "").strip()
        if url_candidate:
            resolved_path = await download_pdf(url_candidate)
        if resolved_path is None:
            resolved_path = resolve_pdf_path(path_text)

    if (has_explicit_pdf_path or generic_url_candidates) and resolved_path is None:
        return {
            "ok": False,
            "error": "PDF not found or URL did not resolve to a PDF.",
            "input": path_text,
            "hint": (
                "Provide an absolute/local PDF path, or a URL that serves application/pdf."
            ),
        }

    if resolved_path is None:
        available = list_available_pdfs(8)
        if not available:
            return {
                "ok": False,
                "error": (
                    "No PDF files found in workspace/read-roots. "
                    "Download a PDF first or provide a path."
                ),
            }
        if len(available) > 1:
            choices = [str(item) for item in available]
            return {
                "ok": False,
                "error": "Multiple PDFs are available. Please specify which PDF to open.",
                "choices": choices,
            }
        resolved_path = available[0]

    if not invoke_open_pdf(resolved_path):
        return {"ok": False, "error": "Failed to queue GUI PDF open action"}
    return {"ok": True, "queued": True, "path": str(resolved_path)}
