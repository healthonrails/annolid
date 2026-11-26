"""Reusable literature search service with source fallback and cache."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import html
import logging
from pathlib import Path
import re
import threading
import time
from typing import Iterable, Mapping, Sequence
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import json

from annolid.utils.atomic_io import atomic_write_text

_LOGGER = logging.getLogger(__name__)

_ARXIV_API_URL = "https://export.arxiv.org/api/query"
_OPENALEX_WORKS_URL = "https://api.openalex.org/works"
_CROSSREF_WORKS_URL = "https://api.crossref.org/works"
_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}
_DEFAULT_TIMEOUT_S = 10.0
_MAX_HTTP_RESPONSE_BYTES = 5 * 1024 * 1024
_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_ARXIV_VERSION_SUFFIX_RE = re.compile(r"v\d+$", re.IGNORECASE)
_XML_UNSAFE_DECL_RE = re.compile(rb"<!(?:DOCTYPE|ENTITY)\b", re.IGNORECASE)
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


def _coerce_authors(raw: object) -> tuple[str, ...]:
    if not raw:
        return ()
    if isinstance(raw, str):
        parts = re.split(r"\s+and\s+|;", raw)
        return tuple(p.strip() for p in parts if p.strip())
    if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        out: list[str] = []
        for item in raw:
            if isinstance(item, Mapping):
                name = str(item.get("display_name") or item.get("name") or "").strip()
                if not name:
                    given = str(item.get("given") or "").strip()
                    family = str(item.get("family") or "").strip()
                    name = f"{given} {family}".strip()
            else:
                name = str(item or "").strip()
            if name:
                out.append(name)
        return tuple(out)
    return ()


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
    authors: tuple[str, ...] = ()
    venue: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "LiteratureResult":
        def _text(key: str) -> str:
            return str(payload.get(key) or "").strip()

        raw_year = payload.get("year")
        year: int | None
        try:
            year = int(raw_year) if raw_year not in (None, "") else None
        except Exception:
            year = None

        summary = _text("summary") or _text("abstract")
        venue = _text("venue") or _text("journal") or _text("booktitle")
        authors = _coerce_authors(payload.get("authors") or payload.get("author"))

        return cls(
            source=_text("source"),
            title=_text("title"),
            summary=summary,
            id_url=_text("id_url"),
            abs_url=_text("abs_url"),
            pdf_url=_text("pdf_url"),
            arxiv_id=_text("arxiv_id"),
            doi=_text("doi"),
            year=year,
            authors=authors,
            venue=venue,
        )

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
            "authors": list(self.authors),
            "venue": self.venue,
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
    atomic_write_text(path, json.dumps(wrapped, ensure_ascii=False))


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
            if isinstance(exc, urllib.error.HTTPError):
                code = int(getattr(exc, "code", 0) or 0)
                if 400 <= code < 500 and code not in {408, 429}:
                    break
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


def _validate_http_url(url: str) -> str:
    parsed = urllib.parse.urlparse(str(url or "").strip())
    if parsed.scheme.lower() not in _ALLOWED_URL_SCHEMES or not parsed.netloc:
        raise ValueError(f"Unsupported URL scheme or host: {url!r}")
    return str(url).strip()


def _read_bounded_response(
    response: object, *, max_bytes: int = _MAX_HTTP_RESPONSE_BYTES
) -> bytes:
    reader = getattr(response, "read")
    chunk = reader(max_bytes + 1)
    if len(chunk) > max_bytes:
        raise ValueError(f"HTTP response exceeded {max_bytes} byte limit.")
    return chunk


def _http_json(url: str, *, timeout_s: float = _DEFAULT_TIMEOUT_S) -> dict[str, object]:
    safe_url = _validate_http_url(url)
    req = urllib.request.Request(
        url=safe_url,
        headers={
            "User-Agent": "annolid-literature-search/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=max(1.0, float(timeout_s))) as response:
        raw = _read_bounded_response(response)
    payload = json.loads(raw.decode("utf-8", errors="replace"))
    return payload if isinstance(payload, dict) else {}


def _http_bytes(url: str, *, timeout_s: float = _DEFAULT_TIMEOUT_S) -> bytes:
    safe_url = _validate_http_url(url)
    req = urllib.request.Request(
        url=safe_url,
        headers={"User-Agent": "annolid-literature-search/1.0"},
    )
    with urllib.request.urlopen(req, timeout=max(1.0, float(timeout_s))) as response:
        return _read_bounded_response(response)


def _extract_arxiv_id(text: str) -> str:
    value = str(text or "").strip().rstrip("/")
    if not value:
        return ""
    match = re.search(r"/(?:abs|pdf)/(.+?)(?:\.pdf)?$", value, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip("/")
    token = value.rsplit("/", 1)[-1]
    if token.endswith(".pdf"):
        token = token[:-4]
    return token


def _reconstruct_openalex_abstract(inverted_index: object) -> str:
    """Reconstruct plain abstract text from OpenAlex ``abstract_inverted_index``."""
    if not isinstance(inverted_index, Mapping):
        return ""
    positions: dict[int, str] = {}
    for word, idx_list in inverted_index.items():
        token = str(word or "").strip()
        if (
            not token
            or not isinstance(idx_list, Sequence)
            or isinstance(idx_list, (str, bytes))
        ):
            continue
        for pos in idx_list:
            try:
                idx = int(pos)
            except Exception:
                continue
            if idx >= 0:
                positions[idx] = token
    if not positions:
        return ""
    raw = " ".join(positions[idx] for idx in sorted(positions)).strip()
    return _strip_xml_tags(raw)


def _strip_xml_tags(text: str) -> str:
    stripped = _HTML_TAG_RE.sub(" ", str(text or ""))
    cleaned = html.unescape(stripped)
    return " ".join(cleaned.split()).strip()


def _parse_iso_year(value: str) -> int | None:
    match = re.match(r"^\s*(\d{4})\b", str(value or ""))
    if not match:
        return None
    try:
        year = int(match.group(1))
    except Exception:
        return None
    return year if 1800 <= year <= 2100 else None


def _search_arxiv(
    query: str, *, max_results: int, timeout_s: float
) -> list[LiteratureResult]:
    final_query = query if query.startswith("id:") else f"all:{query}"
    safe_query = urllib.parse.quote(final_query)
    url = f"{_ARXIV_API_URL}?search_query={safe_query}&start=0&max_results={max(1, int(max_results))}"
    xml_bytes = _http_bytes(url, timeout_s=timeout_s)
    if _XML_UNSAFE_DECL_RE.search(xml_bytes):
        raise ValueError(
            "Refusing to parse arXiv XML containing DOCTYPE/ENTITY declarations."
        )
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
        published = (
            entry.findtext("atom:published", default="", namespaces=_ARXIV_NS) or ""
        ).strip()
        year = _parse_iso_year(published)
        authors: list[str] = []
        for author_el in entry.findall("atom:author", _ARXIV_NS):
            name = (
                author_el.findtext("atom:name", default="", namespaces=_ARXIV_NS) or ""
            ).strip()
            if name:
                authors.append(name)
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
                title=" ".join(title.split()),
                summary=" ".join(summary.split()),
                id_url=abs_url,
                abs_url=abs_url,
                pdf_url=pdf_url,
                arxiv_id=arxiv_id,
                year=year,
                authors=tuple(authors),
                venue="arXiv",
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
        primary_loc = row.get("primary_location")
        loc_map = primary_loc if isinstance(primary_loc, dict) else {}
        source_obj = loc_map.get("source")
        venue = (
            str(source_obj.get("display_name") or "").strip()
            if isinstance(source_obj, dict)
            else ""
        )
        pdf_url = str(loc_map.get("pdf_url") or "").strip()
        if not pdf_url:
            best_oa = row.get("best_oa_location")
            if isinstance(best_oa, dict):
                pdf_url = str(best_oa.get("pdf_url") or "").strip()
        if not pdf_url:
            oa_block = row.get("open_access")
            if isinstance(oa_block, dict):
                oa_url = str(oa_block.get("oa_url") or "").strip()
                if oa_url.lower().endswith(".pdf"):
                    pdf_url = oa_url
        if not pdf_url and arxiv_id:
            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        authors: list[str] = []
        for authorship in row.get("authorships") or []:
            if not isinstance(authorship, dict):
                continue
            author_obj = authorship.get("author")
            if isinstance(author_obj, dict):
                display = str(author_obj.get("display_name") or "").strip()
                if display:
                    authors.append(display)
        summary = _reconstruct_openalex_abstract(row.get("abstract_inverted_index"))
        out.append(
            LiteratureResult(
                source="openalex",
                title=title,
                summary=summary,
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
                authors=tuple(authors),
                venue=venue,
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
        authors: list[str] = []
        for person in row.get("author") or []:
            if not isinstance(person, dict):
                continue
            given = str(person.get("given") or "").strip()
            family = str(person.get("family") or "").strip()
            literal = str(person.get("name") or "").strip()
            full_name = f"{given} {family}".strip() if (given or family) else literal
            if full_name:
                authors.append(full_name)
        container = row.get("container-title")
        venue = (
            str(container[0] or "").strip()
            if isinstance(container, list) and container
            else ""
        )
        summary = _strip_xml_tags(str(row.get("abstract") or ""))
        out.append(
            LiteratureResult(
                source="crossref",
                title=title,
                summary=summary,
                id_url=str(row.get("URL") or "").strip(),
                doi=doi,
                year=_crossref_year(row),
                authors=tuple(authors),
                venue=venue,
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


def _merge_literature_results(
    base: LiteratureResult, incoming: LiteratureResult
) -> LiteratureResult:
    """Combine complementary metadata when two sources describe the same paper."""
    return LiteratureResult(
        source=base.source or incoming.source,
        title=base.title or incoming.title,
        summary=base.summary
        if len(base.summary) >= len(incoming.summary)
        else incoming.summary,
        id_url=base.id_url or incoming.id_url,
        abs_url=base.abs_url or incoming.abs_url,
        pdf_url=base.pdf_url or incoming.pdf_url,
        arxiv_id=base.arxiv_id or incoming.arxiv_id,
        doi=base.doi or incoming.doi,
        year=base.year if base.year is not None else incoming.year,
        authors=base.authors
        if len(base.authors) >= len(incoming.authors)
        else incoming.authors,
        venue=base.venue
        if (base.venue and base.venue.lower() != "arxiv")
        else (incoming.venue or base.venue),
    )


def _deduplicate(results: Iterable[LiteratureResult]) -> list[LiteratureResult]:
    out: list[LiteratureResult] = []
    key_to_idx: dict[str, int] = {}
    for row in results:
        title_norm = " ".join(re.findall(r"[a-z0-9]+", str(row.title).lower()))
        keys: list[str] = []
        if row.doi:
            keys.append(f"doi:{row.doi.strip().lower()}")
        if row.arxiv_id:
            base_arxiv = _ARXIV_VERSION_SUFFIX_RE.sub("", row.arxiv_id.strip().lower())
            keys.append(f"arxiv:{base_arxiv}")
        if title_norm:
            keys.append(f"title:{title_norm}")
        existing_idx: int | None = None
        for k in keys:
            if k in key_to_idx:
                existing_idx = key_to_idx[k]
                break
        if existing_idx is not None:
            merged = _merge_literature_results(out[existing_idx], row)
            out[existing_idx] = merged
            for k in keys:
                key_to_idx[k] = existing_idx
            continue
        new_idx = len(out)
        out.append(row)
        for k in keys:
            key_to_idx[k] = new_idx
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
