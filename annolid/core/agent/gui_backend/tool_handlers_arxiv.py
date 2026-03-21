from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional
import urllib.error
import urllib.parse
import urllib.request

from annolid.core.agent.tools.pdf import DownloadPdfTool
from annolid.services.literature_search import search_literature

_LOGGER = logging.getLogger(__name__)

_ARXIV_API_URL = "https://export.arxiv.org/api/query"
_ARXIV_CB_CLOSED = "closed"
_ARXIV_CB_OPEN = "open"
_ARXIV_CB_HALF_OPEN = "half_open"
_ARXIV_CB_FAIL_THRESHOLD = 3
_ARXIV_CB_COOLDOWN_SEC = 180.0

_arxiv_cb_state = _ARXIV_CB_CLOSED
_arxiv_cb_failures = 0
_arxiv_cb_opened_at = 0.0
_arxiv_cb_lock = threading.Lock()


def _reset_arxiv_circuit_breaker() -> None:
    global _arxiv_cb_state, _arxiv_cb_failures, _arxiv_cb_opened_at  # noqa: PLW0603
    with _arxiv_cb_lock:
        _arxiv_cb_state = _ARXIV_CB_CLOSED
        _arxiv_cb_failures = 0
        _arxiv_cb_opened_at = 0.0


def _arxiv_circuit_allow_request() -> bool:
    global _arxiv_cb_state  # noqa: PLW0603
    with _arxiv_cb_lock:
        if _arxiv_cb_state == _ARXIV_CB_CLOSED:
            return True
        if _arxiv_cb_state == _ARXIV_CB_OPEN:
            elapsed = time.monotonic() - float(_arxiv_cb_opened_at)
            if elapsed >= _ARXIV_CB_COOLDOWN_SEC:
                _arxiv_cb_state = _ARXIV_CB_HALF_OPEN
                return True
            return False
        return True


def _arxiv_circuit_on_success() -> None:
    global _arxiv_cb_state, _arxiv_cb_failures  # noqa: PLW0603
    with _arxiv_cb_lock:
        _arxiv_cb_state = _ARXIV_CB_CLOSED
        _arxiv_cb_failures = 0


def _arxiv_circuit_on_rate_limit() -> None:
    global _arxiv_cb_state, _arxiv_cb_failures, _arxiv_cb_opened_at  # noqa: PLW0603
    with _arxiv_cb_lock:
        _arxiv_cb_failures += 1
        if (
            _arxiv_cb_state == _ARXIV_CB_HALF_OPEN
            or _arxiv_cb_failures >= _ARXIV_CB_FAIL_THRESHOLD
        ):
            _arxiv_cb_state = _ARXIV_CB_OPEN
            _arxiv_cb_opened_at = time.monotonic()


def _parse_retry_after_seconds(raw_value: object) -> float:
    text = str(raw_value or "").strip()
    if not text:
        return 0.0
    try:
        return max(0.0, float(int(text)))
    except Exception:
        pass
    try:
        from email.utils import parsedate_to_datetime
    except Exception:
        return 0.0
    try:
        dt = parsedate_to_datetime(text)
        if dt is None:
            return 0.0
        if dt.tzinfo is None:
            from datetime import timezone

            dt = dt.replace(tzinfo=timezone.utc)
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        delta = (dt - now).total_seconds()
        return max(0.0, float(delta))
    except Exception:
        return 0.0


def _fetch_arxiv_feed_bytes(
    *,
    url: str,
    timeout_s: float = 10.0,
    max_attempts: int = 4,
    emit_progress: Callable[[str], None] | None = None,
) -> bytes:
    if not _arxiv_circuit_allow_request():
        raise RuntimeError(
            "arXiv temporarily unavailable due to repeated rate limiting. Retry later."
        )
    attempts = max(1, int(max_attempts))
    delay = 1.0
    for attempt in range(1, attempts + 1):
        if not _arxiv_circuit_allow_request():
            raise RuntimeError(
                "arXiv temporarily unavailable due to repeated rate limiting. Retry later."
            )
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "annolid-arxiv-client/1.0"},
            )
            with urllib.request.urlopen(
                req, timeout=max(1.0, float(timeout_s))
            ) as response:
                payload = response.read()
            _arxiv_circuit_on_success()
            return payload
        except urllib.error.HTTPError as exc:
            if int(getattr(exc, "code", 0)) == 429:
                _arxiv_circuit_on_rate_limit()
                retry_after = _parse_retry_after_seconds(
                    getattr(exc, "headers", {}).get("Retry-After")
                )
                wait_s = max(retry_after, delay)
                if emit_progress is not None:
                    emit_progress(
                        f"[rate-limit] arXiv responded with 429; retrying in {wait_s:.1f}s "
                        f"(attempt {attempt}/{attempts})."
                    )
                if attempt >= attempts:
                    break
                time.sleep(wait_s)
                delay = min(30.0, delay * 2.0)
                continue
            if attempt >= attempts:
                raise
            if emit_progress is not None:
                emit_progress(
                    f"arXiv request failed (HTTP {exc.code}); retrying in {delay:.1f}s "
                    f"(attempt {attempt}/{attempts})."
                )
            time.sleep(delay)
            delay = min(30.0, delay * 2.0)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt >= attempts:
                raise RuntimeError(
                    f"arXiv request failed after {attempts} attempts: {exc}"
                ) from exc
            if emit_progress is not None:
                emit_progress(
                    f"arXiv network error; retrying in {delay:.1f}s (attempt {attempt}/{attempts})."
                )
            time.sleep(delay)
            delay = min(30.0, delay * 2.0)
    raise RuntimeError("arXiv request failed after repeated rate-limit responses.")


async def safe_run_arxiv_search(
    *,
    query: str,
    run_arxiv_search: Callable[..., Awaitable[Dict[str, Any]]],
    emit_progress: Callable[[str], None],
    log_error: Callable[[str], None],
) -> None:
    try:
        result = await run_arxiv_search(query=query)
        if not result.get("ok"):
            error_msg = result.get("error", "Unknown error")
            emit_progress(f"arXiv download failed: {error_msg}")
        else:
            open_result = result.get("open_result", {})
            if not open_result.get("ok"):
                emit_progress(
                    f"Downloaded but failed to open: {open_result.get('error')}"
                )
            else:
                emit_progress("Opened PDF successfully.")
    except Exception as exc:
        log_error(str(exc))
        emit_progress(f"Error during arXiv operation: {str(exc)}")


def list_local_pdfs(
    *,
    workspace: Path,
    query: Optional[str] = None,
    max_results: int = 20,
) -> Dict[str, Any]:
    search_dirs = [workspace, workspace / "downloads"]
    found_files: List[Path] = []
    for item in search_dirs:
        if item.exists() and item.is_dir():
            found_files.extend(list(item.glob("*.pdf")))

    unique_files = {str(f.absolute()): f for f in found_files}
    pdf_files = list(unique_files.values())
    if query:
        q = str(query).lower().strip()
        pdf_files = [f for f in pdf_files if q in f.name.lower()]
    pdf_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    max_items = max(1, int(max_results or 20))
    truncated = len(pdf_files) > max_items
    pdf_files = pdf_files[:max_items]

    rel_paths = []
    for item in pdf_files:
        try:
            rel_paths.append(str(item.relative_to(workspace)))
        except ValueError:
            rel_paths.append(str(item))

    return {
        "ok": True,
        "files": rel_paths,
        "count": len(found_files),
        "showing": len(rel_paths),
        "truncated": truncated,
    }


async def arxiv_search_tool(
    *,
    query: str,
    max_results: int,
    workspace: Path,
    emit_progress: Callable[[str], None],
    open_pdf: Callable[[str], Awaitable[Dict[str, Any]]],
) -> Dict[str, Any]:
    try:
        emit_progress(f"Searching arXiv for '{query}'...")
        emit_progress(f"arXiv: {query}")
        loop = asyncio.get_running_loop()
        sources: tuple[str, ...] = (
            ("arxiv", "openalex", "crossref")
            if str(query or "").strip().startswith("id:")
            else ("openalex", "crossref", "arxiv")
        )
        emit_progress(f"[literature] searching sources: {', '.join(sources)}")

        search_payload = await loop.run_in_executor(
            None,
            lambda: search_literature(
                str(query or ""),
                max_results=max(1, int(max_results or 1)),
                sources=sources,
                cache_dir=workspace / ".annolid_cache" / "literature",
            ),
        )
        candidates = search_payload.get("results")
        rows = candidates if isinstance(candidates, list) else []
        if not rows:
            return {
                "ok": False,
                "error": f"No papers found for query: {query}",
                "query": query,
                "search": {
                    "source_counts": search_payload.get("source_counts", {}),
                    "degraded_sources": search_payload.get("degraded_sources", []),
                    "cache_hit": bool(search_payload.get("cache_hit")),
                },
            }
        emit_progress("Metadata received. Selecting best downloadable result...")

        selected = None
        for row in rows:
            if isinstance(row, dict) and str(row.get("pdf_url") or "").strip():
                selected = row
                break
        if selected is None:
            for row in rows:
                if isinstance(row, dict) and str(row.get("arxiv_id") or "").strip():
                    selected = row
                    break
        if selected is None and rows and isinstance(rows[0], dict):
            selected = rows[0]

        if not isinstance(selected, dict):
            return {
                "ok": False,
                "error": f"No usable paper metadata found for query: {query}",
                "query": query,
            }

        title = str(selected.get("title") or "Untitled").strip()
        pdf_link = str(selected.get("pdf_url") or "").strip()
        arxiv_id = str(selected.get("arxiv_id") or "").strip()
        if not pdf_link and arxiv_id:
            pdf_link = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        if not pdf_link:
            return {
                "ok": False,
                "error": "Could not find PDF link for paper.",
                "title": title,
                "source": str(selected.get("source") or ""),
            }

        emit_progress(f"Found: {title}")
        emit_progress("Downloading PDF...")
        downloads_dir = workspace / "downloads"
        downloads_dir.mkdir(parents=True, exist_ok=True)
        safe_title = "".join(c for c in title if c.isalnum() or c in " ._-").strip()
        if len(safe_title) > 100:
            safe_title = safe_title[:100].rstrip(" ._-")
        filename = f"{safe_title}.pdf"
        output_path = str(downloads_dir / filename)

        downloader = DownloadPdfTool(allowed_dir=workspace)
        dl_result_json = await downloader.execute(
            url=pdf_link, output_path=output_path, overwrite=True
        )
        try:
            dl_result = json.loads(dl_result_json)
        except Exception:
            dl_result = {"error": "Invalid download response"}
        if dl_result.get("error"):
            return {
                "ok": False,
                "error": f"Download failed: {dl_result['error']}",
                "url": pdf_link,
            }

        final_path = dl_result.get("output_path", output_path)
        emit_progress("Opening PDF...")
        open_res = await open_pdf(final_path)
        return {
            "ok": True,
            "title": title,
            "path": final_path,
            "open_result": open_res,
            "source": str(selected.get("source") or ""),
            "search": {
                "source_counts": search_payload.get("source_counts", {}),
                "degraded_sources": search_payload.get("degraded_sources", []),
                "cache_hit": bool(search_payload.get("cache_hit")),
            },
        }
    except Exception as exc:
        _LOGGER.warning("arXiv search workflow failed: %s", exc)
        return {"ok": False, "error": str(exc)}
