"""Reusable literature search service with source fallback and cache."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
import threading
import time
from typing import Iterable, Sequence
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import json

_LOGGER = logging.getLogger(__name__)

_ARXIV_API_URL = "https://export.arxiv.org/api/query"
_OPENALEX_WORKS_URL = "https://api.openalex.org/works"
_CROSSREF_WORKS_URL = "https://api.crossref.org/works"
_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
_DEFAULT_TIMEOUT_S = 10.0
_CACHE_TTL_S = 24.0 * 60.0 * 60.0
_CACHE_TTL_NON_ARXIV_S = 72.0 * 60.0 * 60.0
_DEFAULT_SOURCES = ("openalex", "crossref", "arxiv")
_SOURCE_MIN_GAP_S = {
    "openalex": 0.2,
    "crossref": 0.2,
    "arxiv": 1.0,
}
_SOURCE_MAX_ATTEMPTS = {
    "openalex": 3,
    "crossref": 3,
    "arxiv": 3,
}
_SOURCE_RETRYABLE_EXC = (
    urllib.error.HTTPError,
    urllib.error.URLError,
    TimeoutError,
    OSError,
    RuntimeError,
)


@dataclass(frozen=True)
class LiteratureResult:
    source: str
    title: str
    summary: str = ""
    id_url: str = ""
    abs_url: str = ""
    pdf_url: str = ""
    arxiv_id: str = ""
    doi: str = ""
    year: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "title": self.title,
            "summary": self.summary,
            "id_url": self.id_url,
            "abs_url": self.abs_url,
            "pdf_url": self.pdf_url,
            "arxiv_id": self.arxiv_id,
            "doi": self.doi,
            "year": self.year,
        }


_CACHE_LOCK = threading.Lock()
_CACHE: dict[tuple[str, int, tuple[str, ...]], tuple[float, dict[str, object]]] = {}
_SOURCE_STATE_LOCK = threading.Lock()
_SOURCE_LAST_CALL: dict[str, float] = {}


def _cache_ttl_seconds(sources: Sequence[str]) -> float:
    return (
        _CACHE_TTL_S
        if "arxiv" in {str(src).lower() for src in sources}
        else _CACHE_TTL_NON_ARXIV_S
    )


def _default_cache_dir() -> Path:
    return Path.home() / ".annolid" / "cache" / "literature"


def _cache_file_path(*, cache_dir: Path, key: tuple[str, int, tuple[str, ...]]) -> Path:
    digest = hashlib.sha256(
        f"{key[0]}|{key[1]}|{','.join(key[2])}".encode("utf-8", errors="replace")
    ).hexdigest()
    return cache_dir / f"{digest}.json"


def _read_disk_cache(
    *,
    cache_dir: Path,
    key: tuple[str, int, tuple[str, ...]],
    ttl_s: float,
) -> tuple[dict[str, object], float] | None:
    path = _cache_file_path(cache_dir=cache_dir, key=key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    created = payload.get("created_at_epoch")
    created_epoch = float(created) if isinstance(created, (int, float)) else 0.0
    if created_epoch <= 0.0:
        return None
    age_s = max(0.0, time.time() - created_epoch)
    if age_s > float(ttl_s):
        return None
    cached_payload = payload.get("payload")
    if not isinstance(cached_payload, dict):
        return None
    return cached_payload, age_s


def _write_disk_cache(
    *,
    cache_dir: Path,
    key: tuple[str, int, tuple[str, ...]],
    payload: dict[str, object],
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_file_path(cache_dir=cache_dir, key=key)
    wrapped = {
        "created_at_epoch": time.time(),
        "payload": payload,
    }
    path.write_text(json.dumps(wrapped, ensure_ascii=False), encoding="utf-8")


def _pace_source(source: str) -> None:
    src = str(source or "").lower().strip()
    min_gap = float(_SOURCE_MIN_GAP_S.get(src, 0.0))
    if min_gap <= 0.0:
        return
    now = time.monotonic()
    with _SOURCE_STATE_LOCK:
        prev = float(_SOURCE_LAST_CALL.get(src, 0.0))
        wait_s = max(0.0, min_gap - max(0.0, now - prev))
        _SOURCE_LAST_CALL[src] = now + wait_s
    if wait_s > 0.0:
        time.sleep(wait_s)


def _run_source_with_retry(
    *,
    source: str,
    run_once,
) -> tuple[list[LiteratureResult], dict[str, object]]:
    src = str(source or "").lower().strip()
    attempts = max(1, int(_SOURCE_MAX_ATTEMPTS.get(src, 2)))
    backoff = 0.2
    start = time.perf_counter()
    last_error = ""
    tries = 0
    for attempt in range(1, attempts + 1):
        tries = attempt
        try:
            _pace_source(src)
            rows = run_once()
            latency_ms = (time.perf_counter() - start) * 1000.0
            return list(rows), {
                "attempts": tries,
                "latency_ms": round(latency_ms, 2),
                "success": True,
                "error": "",
            }
        except _SOURCE_RETRYABLE_EXC as exc:
            last_error = str(exc)
            if attempt >= attempts:
                break
            time.sleep(backoff)
            backoff = min(5.0, backoff * 2.0)
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            break
    latency_ms = (time.perf_counter() - start) * 1000.0
    return [], {
        "attempts": tries,
        "latency_ms": round(latency_ms, 2),
        "success": False,
        "error": last_error,
    }


def _http_json(url: str, *, timeout_s: float = _DEFAULT_TIMEOUT_S) -> dict[str, object]:
    req = urllib.request.Request(
        url=url,
        headers={
            "User-Agent": "annolid-literature-search/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=max(1.0, float(timeout_s))) as response:
        raw = response.read()
    payload = json.loads(raw.decode("utf-8", errors="replace"))
    return payload if isinstance(payload, dict) else {}


def _http_bytes(url: str, *, timeout_s: float = _DEFAULT_TIMEOUT_S) -> bytes:
    req = urllib.request.Request(
        url=url,
        headers={"User-Agent": "annolid-literature-search/1.0"},
    )
    with urllib.request.urlopen(req, timeout=max(1.0, float(timeout_s))) as response:
        return response.read()


def _extract_arxiv_id(text: str) -> str:
    value = str(text or "").strip().rstrip("/")
    if not value:
        return ""
    token = value.rsplit("/", 1)[-1]
    token = token.replace("abs/", "").replace("pdf/", "")
    if token.endswith(".pdf"):
        token = token[:-4]
    return token


def _search_arxiv(
    query: str, *, max_results: int, timeout_s: float
) -> list[LiteratureResult]:
    final_query = query if query.startswith("id:") else f"all:{query}"
    safe_query = urllib.parse.quote(final_query)
    url = f"{_ARXIV_API_URL}?search_query={safe_query}&start=0&max_results={max(1, int(max_results))}"
    xml_bytes = _http_bytes(url, timeout_s=timeout_s)
    root = ET.fromstring(xml_bytes)
    out: list[LiteratureResult] = []
    for entry in root.findall("atom:entry", _ARXIV_NS):
        title = (
            entry.findtext("atom:title", default="", namespaces=_ARXIV_NS) or ""
        ).strip()
        if not title:
            continue
        abs_url = (
            entry.findtext("atom:id", default="", namespaces=_ARXIV_NS) or ""
        ).strip()
        summary = (
            entry.findtext("atom:summary", default="", namespaces=_ARXIV_NS) or ""
        ).strip()
        pdf_url = ""
        for link in entry.findall("atom:link", _ARXIV_NS):
            if str(link.attrib.get("title") or "").strip().lower() == "pdf":
                pdf_url = str(link.attrib.get("href") or "").strip()
                break
        arxiv_id = _extract_arxiv_id(abs_url)
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        if pdf_url.startswith("http://"):
            pdf_url = "https://" + pdf_url[len("http://") :]
        if abs_url.startswith("http://"):
            abs_url = "https://" + abs_url[len("http://") :]
        out.append(
            LiteratureResult(
                source="arxiv",
                title=title.replace("\n", " "),
                summary=summary.replace("\n", " "),
                id_url=abs_url,
                abs_url=abs_url,
                pdf_url=pdf_url,
                arxiv_id=arxiv_id,
            )
        )
    return out


def _search_openalex(
    query: str, *, max_results: int, timeout_s: float
) -> list[LiteratureResult]:
    url = (
        f"{_OPENALEX_WORKS_URL}?search={urllib.parse.quote(query)}"
        f"&per-page={max(1, int(max_results))}"
    )
    payload = _http_json(url, timeout_s=timeout_s)
    rows = payload.get("results")
    if not isinstance(rows, list):
        return []
    out: list[LiteratureResult] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title = str(row.get("title") or "").strip()
        if not title:
            continue
        ids = row.get("ids")
        ids_map = ids if isinstance(ids, dict) else {}
        arxiv_abs = str(ids_map.get("arxiv") or "").strip()
        arxiv_id = _extract_arxiv_id(arxiv_abs)
        doi_url = str(row.get("doi") or "").strip()
        doi = doi_url.split("doi.org/")[-1] if "doi.org/" in doi_url else doi_url
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf" if arxiv_id else ""
        out.append(
            LiteratureResult(
                source="openalex",
                title=title,
                id_url=str(row.get("id") or "").strip(),
                abs_url=(f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""),
                pdf_url=pdf_url,
                arxiv_id=arxiv_id,
                doi=doi,
                year=(
                    int(row.get("publication_year"))
                    if isinstance(row.get("publication_year"), int)
                    else None
                ),
            )
        )
    return out


def _search_crossref(
    query: str, *, max_results: int, timeout_s: float
) -> list[LiteratureResult]:
    url = (
        f"{_CROSSREF_WORKS_URL}?query.bibliographic={urllib.parse.quote(query)}"
        f"&rows={max(1, int(max_results))}"
    )
    payload = _http_json(url, timeout_s=timeout_s)
    message = payload.get("message")
    message_map = message if isinstance(message, dict) else {}
    items = message_map.get("items")
    rows = items if isinstance(items, list) else []
    out: list[LiteratureResult] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        title_list = row.get("title")
        title_items = title_list if isinstance(title_list, list) else []
        title = str(title_items[0] if title_items else "").strip()
        if not title:
            continue
        doi = str(row.get("DOI") or "").strip()
        out.append(
            LiteratureResult(
                source="crossref",
                title=title,
                id_url=str(row.get("URL") or "").strip(),
                doi=doi,
                year=_crossref_year(row),
            )
        )
    return out


def _crossref_year(row: dict[str, object]) -> int | None:
    issued = row.get("issued")
    issued_map = issued if isinstance(issued, dict) else {}
    date_parts = issued_map.get("date-parts")
    if not isinstance(date_parts, list) or not date_parts:
        return None
    head = date_parts[0]
    if not isinstance(head, list) or not head:
        return None
    value = head[0]
    return int(value) if isinstance(value, int) else None


def _deduplicate(results: Iterable[LiteratureResult]) -> list[LiteratureResult]:
    out: list[LiteratureResult] = []
    seen: set[str] = set()
    for row in results:
        title_key = " ".join(str(row.title).lower().split())
        if row.doi:
            key = f"doi:{row.doi.lower()}"
        elif row.arxiv_id:
            key = f"arxiv:{row.arxiv_id.lower()}"
        else:
            key = f"title:{title_key}"
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def search_literature(
    query: str,
    *,
    max_results: int = 5,
    sources: Sequence[str] = _DEFAULT_SOURCES,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    use_cache: bool = True,
    cache_dir: str | Path | None = None,
) -> dict[str, object]:
    clean_query = str(query or "").strip()
    if not clean_query:
        return {
            "results": [],
            "source_counts": {},
            "degraded_sources": [],
            "cache_hit": False,
            "source_metrics": {},
        }
    source_tuple = tuple(
        str(src).strip().lower() for src in sources if str(src).strip()
    )
    if not source_tuple:
        source_tuple = _DEFAULT_SOURCES
    key = (clean_query, int(max_results), source_tuple)
    now = time.time()
    ttl_s = _cache_ttl_seconds(source_tuple)
    resolved_cache_dir = (
        Path(cache_dir).expanduser().resolve()
        if cache_dir is not None
        else _default_cache_dir()
    )
    if use_cache:
        disk_cached = _read_disk_cache(
            cache_dir=resolved_cache_dir,
            key=key,
            ttl_s=ttl_s,
        )
        if disk_cached is not None:
            cached_payload, age_s = disk_cached
            out = dict(cached_payload)
            out["cache_hit"] = True
            out["cache_age_seconds"] = age_s
            _LOGGER.info(
                "[cache] HIT literature query=%r sources=%s age=%.1fs",
                clean_query,
                ",".join(source_tuple),
                age_s,
            )
            return out
        with _CACHE_LOCK:
            cached = _CACHE.get(key)
        if cached is not None:
            ts, payload = cached
            if (now - float(ts)) <= ttl_s:
                out = dict(payload)
                out["cache_hit"] = True
                out["cache_age_seconds"] = max(0.0, now - float(ts))
                return out
    source_counts: dict[str, int] = {}
    degraded: list[str] = []
    source_metrics: dict[str, dict[str, object]] = {}
    merged: list[LiteratureResult] = []
    for source in source_tuple:
        try:
            if source == "openalex":
                rows, metrics = _run_source_with_retry(
                    source=source,
                    run_once=lambda: _search_openalex(
                        clean_query, max_results=max_results, timeout_s=timeout_s
                    ),
                )
            elif source == "crossref":
                rows, metrics = _run_source_with_retry(
                    source=source,
                    run_once=lambda: _search_crossref(
                        clean_query, max_results=max_results, timeout_s=timeout_s
                    ),
                )
            elif source == "arxiv":
                rows, metrics = _run_source_with_retry(
                    source=source,
                    run_once=lambda: _search_arxiv(
                        clean_query, max_results=max_results, timeout_s=timeout_s
                    ),
                )
            else:
                continue
            source_metrics[source] = metrics
            if not bool(metrics.get("success")):
                degraded.append(source)
                continue
            source_counts[source] = len(rows)
            merged.extend(rows)
        except Exception as exc:  # noqa: BLE001
            degraded.append(source)
            _LOGGER.warning(
                "Literature source %s failed for %r: %s", source, clean_query, exc
            )
    deduped = _deduplicate(merged)
    payload = {
        "results": [item.to_dict() for item in deduped[: max(1, int(max_results))]],
        "source_counts": source_counts,
        "degraded_sources": degraded,
        "cache_hit": False,
        "cache_age_seconds": 0.0,
        "source_metrics": source_metrics,
    }
    if use_cache:
        with _CACHE_LOCK:
            _CACHE[key] = (now, payload)
        try:
            _write_disk_cache(
                cache_dir=resolved_cache_dir,
                key=key,
                payload=payload,
            )
        except Exception as exc:  # noqa: BLE001
            _LOGGER.debug("Failed to write literature cache: %s", exc)
    return payload


__all__ = [
    "LiteratureResult",
    "search_literature",
]
