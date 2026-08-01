from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import re
import time
import urllib.parse
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import Any

from annolid.core.agent.security_network import (
    public_httpx_client_kwargs,
    validate_http_url_shape,
    validate_public_url_target,
)

from .function_base import FunctionTool
from .common import _normalize, _resolve_write_path, _strip_tags
from .web_runtime import (
    DEFAULT_FETCH_MAX_BYTES,
    DEFAULT_SEARCH_MAX_BYTES,
    UNTRUSTED_WEB_CONTENT_BANNER,
    decode_response_bytes,
    is_textual_content_type,
    read_response_bytes,
    sanitize_web_error,
    stream_response_to_atomic_file,
    validate_download_request_headers,
)

logger = logging.getLogger(__name__)
_DDGS_SEARCH_LOCK = Lock()


def _clean_tool_url(url: object) -> str:
    return str(url or "").strip().strip("`'\"").strip()


def _validate_public_web_url(url: str) -> tuple[bool, str]:
    return validate_public_url_target(url)


class WebSearchTool(FunctionTool):
    _MAX_QUERY_CHARS = 1000
    _MAX_TITLE_CHARS = 500
    _MAX_DESCRIPTION_CHARS = 2000
    _USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    _DUCK_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
    _DUCK_CHALLENGE_MARKERS = (
        "unfortunately, bots use duckduckgo too",
        "select all squares containing a duck",
    )
    _DUCK_RESULT_LINK_RE = re.compile(
        r"<a[^>]*class=(?P<q1>[\"'])[^\"']*\bresult__a\b[^\"']*(?P=q1)"
        r"[^>]*href=(?P<q2>[\"'])(?P<url>.*?)(?P=q2)[^>]*>(?P<title>.*?)</a>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    _DUCK_SNIPPET_RE = re.compile(
        r"<(?:a|div|span)[^>]*class=(?P<q1>[\"'])[^\"']*\bresult__snippet\b[^\"']*(?P=q1)"
        r"[^>]*>(?P<snippet>.*?)</(?:a|div|span)>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    _VALID_BACKENDS = {"auto", "duckduckgo", "ddgs", "scrapling", "brave"}

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 5,
        backend: str = "auto",
        cache_ttl_seconds: float = 900.0,
        cache_max_entries: int = 128,
    ):
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY", "")
        self.max_results = self._bounded_int(
            max_results, default=5, minimum=1, maximum=10
        )
        self.backend = str(backend or "auto").strip().lower()
        self.cache_ttl_seconds = max(0.0, float(cache_ttl_seconds or 0.0))
        self.cache_max_entries = max(1, int(cache_max_entries or 1))
        self._cache: OrderedDict[tuple[str, str, int], tuple[float, str]] = (
            OrderedDict()
        )

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web (DDGS first, hardened DuckDuckGo HTML and Brave "
            "API fallbacks). "
            "Returns titles, URLs, and snippets."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "maxLength": self._MAX_QUERY_CHARS},
                "count": {"type": "integer", "minimum": 1, "maximum": 10},
                "backend": {
                    "type": "string",
                    "enum": [
                        "auto",
                        "duckduckgo",
                        "ddgs",
                        "scrapling",
                        "brave",
                    ],
                },
            },
            "required": ["query"],
        }

    async def execute(
        self,
        query: str,
        count: int | None = None,
        backend: str | None = None,
        **kwargs: Any,
    ) -> str:
        del kwargs
        query_text = str(query or "").strip()
        if not query_text:
            return "Error: query is required"
        if len(query_text) > self._MAX_QUERY_CHARS:
            return f"Error: query exceeds maximum length ({self._MAX_QUERY_CHARS})"
        n = self._bounded_int(count, default=self.max_results, minimum=1, maximum=10)
        requested_backend = str(backend or self.backend or "auto").strip().lower()
        preferred = (
            requested_backend if requested_backend in self._VALID_BACKENDS else "auto"
        )
        cache_key = self._cache_key(query_text, preferred, n)
        cached = self._get_cached(cache_key)
        if cached:
            return cached
        duckduckgo_available = False

        if preferred in {"auto", "duckduckgo", "ddgs", "scrapling"}:
            if preferred == "ddgs":
                duckduckgo_result = await self._search_with_ddgs(
                    query=query_text, count=n
                )
            elif preferred == "scrapling":
                duckduckgo_result = await self._search_with_duckduckgo_html(
                    query=query_text, count=n
                )
            else:
                duckduckgo_result = await self._search_with_scrapling(
                    query=query_text, count=n
                )
            duckduckgo_available = duckduckgo_result is not None
            if duckduckgo_result:
                text = self._format_results(query_text, duckduckgo_result)
                self._store_cached(cache_key, text)
                return text
            if preferred in {"duckduckgo", "ddgs", "scrapling"}:
                if duckduckgo_available:
                    return f"No results for: {query_text}"
                return (
                    f"Error: {preferred} search backend unavailable. "
                    "Install the Annolid Bot dependencies or configure "
                    "BRAVE_API_KEY and use backend='brave'."
                )

        brave_result = await self._search_with_brave(query=query_text, count=n)
        if brave_result is not None:
            text = self._format_results(query_text, brave_result)
            self._store_cached(cache_key, text)
            return text

        if duckduckgo_available:
            return f"No results for: {query_text}"
        if preferred == "brave":
            if self.api_key:
                return "Error: Brave search backend unavailable."
            return "Error: BRAVE_API_KEY not configured"
        return (
            "Error: DuckDuckGo search backends unavailable and "
            "BRAVE_API_KEY is not configured."
        )

    @staticmethod
    def _cache_key(query: str, backend: str, count: int) -> tuple[str, str, int]:
        normalized_query = " ".join(str(query or "").split()).strip().lower()
        normalized_backend = str(backend or "auto").strip().lower()
        return (normalized_backend, normalized_query, int(count))

    def _get_cached(self, key: tuple[str, str, int]) -> str:
        if self.cache_ttl_seconds <= 0:
            return ""
        cached = self._cache.get(key)
        if cached is None:
            return ""
        expires_at, text = cached
        if time.monotonic() >= expires_at:
            self._cache.pop(key, None)
            return ""
        self._cache.move_to_end(key)
        return text

    def _store_cached(self, key: tuple[str, str, int], text: str) -> None:
        if self.cache_ttl_seconds <= 0 or not str(text or "").strip():
            return
        self._cache.pop(key, None)
        self._cache[key] = (time.monotonic() + self.cache_ttl_seconds, text)
        while len(self._cache) > self.cache_max_entries:
            self._cache.popitem(last=False)

    @staticmethod
    def _bounded_int(
        value: object,
        *,
        default: int,
        minimum: int,
        maximum: int,
    ) -> int:
        try:
            resolved = int(value) if value is not None else int(default)
        except Exception:
            resolved = int(default)
        return min(max(resolved, int(minimum)), int(maximum))

    @staticmethod
    def _format_results(query: str, results: list[dict[str, str]]) -> str:
        if not results:
            return f"No results for: {query}"
        lines = [
            UNTRUSTED_WEB_CONTENT_BANNER,
            f"Results for: {query}\n",
        ]
        for i, item in enumerate(results, 1):
            title = WebSearchTool._safe_result_text(
                item.get("title", ""),
                max_chars=WebSearchTool._MAX_TITLE_CHARS,
            )
            description = WebSearchTool._safe_result_text(
                item.get("description", ""),
                max_chars=WebSearchTool._MAX_DESCRIPTION_CHARS,
            )
            lines.append(f"{i}. {title}\n   {item.get('url', '')}")
            if description:
                lines.append(f"   {description}")
        return "\n".join(lines)

    @staticmethod
    def _safe_result_text(value: object, *, max_chars: int) -> str:
        text = _normalize(_strip_tags(html.unescape(str(value or ""))))
        return re.sub(r"\s+", " ", text).strip()[: max(1, int(max_chars))]

    async def _search_with_brave(
        self, *, query: str, count: int
    ) -> list[dict[str, str]] | None:
        if not self.api_key:
            return None
        try:
            import httpx

            async with httpx.AsyncClient(
                timeout=10.0,
                **public_httpx_client_kwargs(),
            ) as client:
                response_body = b""
                for attempt in range(2):
                    retry = False
                    async with client.stream(
                        "GET",
                        "https://api.search.brave.com/res/v1/web/search",
                        params={"q": query, "count": count},
                        headers={
                            "Accept": "application/json",
                            "X-Subscription-Token": self.api_key,
                        },
                    ) as response:
                        retry = response.status_code == 429 and attempt == 0
                        if not retry:
                            response.raise_for_status()
                            response_body = await read_response_bytes(
                                response, max_bytes=DEFAULT_SEARCH_MAX_BYTES
                            )
                    if not retry:
                        break
                    await asyncio.sleep(0.25)
            response_payload = json.loads(response_body.decode("utf-8"))
            results = response_payload.get("web", {}).get("results", [])
            out: list[dict[str, str]] = []
            for item in results[:count]:
                url = str(item.get("url", "")).strip()
                ok, _ = validate_http_url_shape(url)
                if url and not ok:
                    continue
                out.append(
                    {
                        "title": self._safe_result_text(
                            item.get("title", ""),
                            max_chars=self._MAX_TITLE_CHARS,
                        ),
                        "url": url,
                        "description": self._safe_result_text(
                            item.get("description", ""),
                            max_chars=self._MAX_DESCRIPTION_CHARS,
                        ),
                    }
                )
            return [row for row in out if row.get("title") or row.get("url")]
        except Exception as exc:
            logger.debug("Brave search failed: %s", sanitize_web_error(exc))
            return None

    async def _search_with_scrapling(
        self, *, query: str, count: int
    ) -> list[dict[str, str]] | None:
        """Run the resilient keyless DuckDuckGo path.

        The method name is retained for compatibility with callers that
        monkeypatch the former Scrapling-backed implementation.
        """
        ddgs_result = await self._search_with_ddgs(query=query, count=count)
        if ddgs_result is not None:
            return ddgs_result
        return await self._search_with_duckduckgo_html(query=query, count=count)

    async def _search_with_ddgs(
        self, *, query: str, count: int
    ) -> list[dict[str, str]] | None:
        try:
            from ddgs import DDGS
        except ImportError:
            logger.debug(
                "DDGS search dependency is unavailable; using the HTML fallback"
            )
            return None

        def _run_search() -> list[dict[str, Any]]:
            with _DDGS_SEARCH_LOCK:
                raw = DDGS(timeout=10).text(query, max_results=count)
                return [dict(item) for item in (raw or []) if isinstance(item, dict)]

        try:
            raw_results = await asyncio.wait_for(
                asyncio.to_thread(_run_search),
                timeout=20.0,
            )
        except Exception as exc:
            logger.debug("DDGS search failed: %s", sanitize_web_error(exc))
            return None

        results: list[dict[str, str]] = []
        for item in raw_results:
            url = str(item.get("href") or item.get("url") or "").strip()
            ok, _ = validate_http_url_shape(url)
            if not ok:
                continue
            title = self._safe_result_text(
                item.get("title", ""),
                max_chars=self._MAX_TITLE_CHARS,
            )
            description = self._safe_result_text(
                item.get("body") or item.get("description") or "",
                max_chars=self._MAX_DESCRIPTION_CHARS,
            )
            if title or url:
                results.append(
                    {
                        "title": title,
                        "url": url,
                        "description": description,
                    }
                )
            if len(results) >= count:
                break
        return results

    async def _search_with_duckduckgo_html(
        self, *, query: str, count: int
    ) -> list[dict[str, str]] | None:
        target = self._DUCK_HTML_ENDPOINT + "?" + urllib.parse.urlencode({"q": query})
        try:
            html_text = await self._fetch_html_with_httpx(target)
            lowered = html_text.lower()
            if any(marker in lowered for marker in self._DUCK_CHALLENGE_MARKERS):
                raise RuntimeError(
                    "DuckDuckGo HTML returned an interactive bot challenge"
                )
            return self._parse_duckduckgo_results(html_text, count=count)
        except Exception as exc:
            logger.debug(
                "DuckDuckGo HTML search failed: %s",
                sanitize_web_error(exc),
            )
            return None

    async def _fetch_html_with_scrapling(self, url: str) -> str:
        """Compatibility wrapper for the former Scrapling-backed search path."""
        return await self._fetch_html_with_httpx(url)

    async def _fetch_html_with_httpx(self, url: str) -> str:
        import httpx

        async with httpx.AsyncClient(
            follow_redirects=True,
            max_redirects=5,
            timeout=15.0,
            **public_httpx_client_kwargs(),
        ) as client:
            async with client.stream(
                "GET",
                url,
                headers={"User-Agent": self._USER_AGENT},
            ) as response:
                response.raise_for_status()
                body = await read_response_bytes(
                    response, max_bytes=DEFAULT_SEARCH_MAX_BYTES
                )
                content_type = str(response.headers.get("content-type", ""))
        return decode_response_bytes(body, content_type).strip()

    @staticmethod
    def _parse_duckduckgo_results(
        source_html: str, *, count: int
    ) -> list[dict[str, str]]:
        text = str(source_html or "")
        if not text:
            return []
        rows: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        matches = list(WebSearchTool._DUCK_RESULT_LINK_RE.finditer(text))
        for idx, match in enumerate(matches):
            raw_url = html.unescape(match.group("url")).strip()
            title = _normalize(_strip_tags(html.unescape(match.group("title"))))
            if not raw_url or not title:
                continue
            if raw_url.startswith("//"):
                raw_url = f"https:{raw_url}"
            parsed = urllib.parse.urlparse(raw_url)
            q = urllib.parse.parse_qs(parsed.query)
            if "uddg" in q and q["uddg"]:
                url = html.unescape(q["uddg"][0]).strip()
            else:
                url = raw_url
            if not url:
                continue
            parsed_url = urllib.parse.urlparse(url)
            ok, _ = validate_http_url_shape(url)
            if not ok:
                continue
            if (parsed_url.netloc or "").lower().endswith(
                "duckduckgo.com"
            ) and "uddg" not in q:
                continue
            normalized_url = urllib.parse.urlunparse(parsed_url)
            if normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)
            next_start = (
                matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            )
            result_fragment = text[match.end() : next_start]
            snippet_match = WebSearchTool._DUCK_SNIPPET_RE.search(result_fragment)
            description = ""
            if snippet_match:
                description = _normalize(
                    _strip_tags(html.unescape(snippet_match.group("snippet")))
                )
                description = re.sub(r"\s+", " ", description).strip()
            rows.append({"title": title, "url": url, "description": description})
            if len(rows) >= int(count):
                break
        return rows


class WebFetchTool(FunctionTool):
    USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"

    def __init__(
        self,
        max_chars: int = 50000,
        max_response_bytes: int = DEFAULT_FETCH_MAX_BYTES,
    ):
        self.max_chars = max(100, int(max_chars))
        self.max_response_bytes = max(1, int(max_response_bytes))

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetch URL and extract readable content."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "extractMode": {"type": "string", "enum": ["markdown", "text"]},
                "maxChars": {"type": "integer", "minimum": 100},
            },
            "required": ["url"],
        }

    async def execute(
        self,
        url: str,
        extractMode: str = "markdown",
        maxChars: int | None = None,
        **kwargs: Any,
    ) -> str:
        del kwargs
        clean_url = _clean_tool_url(url)
        ok, err = _validate_public_web_url(clean_url)
        if not ok:
            return json.dumps({"error": f"URL validation failed: {err}"})

        max_chars = WebSearchTool._bounded_int(
            maxChars,
            default=self.max_chars,
            minimum=100,
            maximum=max(100, int(self.max_chars)),
        )
        extract_mode = str(extractMode or "markdown").strip().lower()
        if extract_mode not in {"markdown", "text"}:
            return json.dumps(
                {
                    "error": "extractMode must be 'markdown' or 'text'",
                    "url": clean_url,
                }
            )
        try:
            import httpx

            async with httpx.AsyncClient(
                follow_redirects=True,
                max_redirects=5,
                timeout=30.0,
                **public_httpx_client_kwargs(),
            ) as client:
                async with client.stream(
                    "GET",
                    clean_url,
                    headers={"User-Agent": self.USER_AGENT},
                ) as response:
                    response.raise_for_status()
                    final_url = str(response.url)
                    ok, err = _validate_public_web_url(final_url)
                    if not ok:
                        return json.dumps(
                            {
                                "error": f"Final URL validation failed: {err}",
                                "url": clean_url,
                                "finalUrl": final_url,
                                "status": response.status_code,
                            }
                        )
                    ctype = str(response.headers.get("content-type", ""))
                    if not is_textual_content_type(ctype):
                        return json.dumps(
                            {
                                "error": (
                                    f"Unsupported content-type "
                                    f"'{ctype or 'unknown'}'; use download_url "
                                    "for binary content"
                                ),
                                "url": clean_url,
                                "finalUrl": final_url,
                                "status": response.status_code,
                            }
                        )
                    body_bytes = await read_response_bytes(
                        response, max_bytes=self.max_response_bytes
                    )
                    status_code = int(response.status_code)

            response_text = decode_response_bytes(body_bytes, ctype)
            page_title = ""
            media_type = ctype.split(";", 1)[0].strip().lower()
            if media_type == "application/json" or media_type.endswith("+json"):
                try:
                    text = json.dumps(json.loads(response_text), indent=2)
                    extractor = "json"
                except json.JSONDecodeError:
                    text = response_text
                    extractor = "invalid-json-text"
            elif "text/html" in media_type or response_text[:256].lower().startswith(
                ("<!doctype", "<html")
            ):
                title_match = re.search(
                    r"<title[^>]*>(.*?)</title>",
                    response_text,
                    flags=re.IGNORECASE | re.DOTALL,
                )
                if title_match:
                    page_title = _normalize(
                        _strip_tags(html.unescape(title_match.group(1)))
                    )
                body = _strip_tags(response_text)
                text = _normalize(body)
                extractor = "html-strip"
            else:
                text = response_text
                extractor = "raw"

            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars]
            model_text = f"{UNTRUSTED_WEB_CONTENT_BANNER}\n\n{text}"
            return json.dumps(
                {
                    "url": clean_url,
                    "finalUrl": final_url,
                    "status": status_code,
                    "extractor": extractor,
                    "title": page_title,
                    "contentType": ctype,
                    "contentTrust": "untrusted_external_content",
                    "untrusted": True,
                    "truncated": truncated,
                    "bytesRead": len(body_bytes),
                    "contentLength": len(text),
                    "length": len(model_text),
                    "text": model_text,
                }
            )
        except Exception as exc:
            return json.dumps({"error": sanitize_web_error(exc), "url": clean_url})


class DownloadUrlTool(FunctionTool):
    USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"

    def __init__(
        self,
        allowed_dir: Path | None = None,
        max_bytes: int = 25 * 1024 * 1024,
        *,
        allowed_request_headers: frozenset[str] = frozenset(),
    ):
        self._allowed_dir = allowed_dir
        self._max_bytes = max(1, int(max_bytes))
        self._allowed_request_headers = frozenset(
            str(item).strip().lower()
            for item in allowed_request_headers
            if str(item).strip()
        )

    @property
    def name(self) -> str:
        return "download_url"

    @property
    def description(self) -> str:
        return "Download a URL to a local file with size/type safety checks."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "output_path": {"type": "string"},
                "max_bytes": {"type": "integer", "minimum": 1},
                "overwrite": {"type": "boolean"},
                "content_type_prefixes": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "request_headers": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["url", "output_path"],
        }

    async def execute(
        self,
        url: str,
        output_path: str,
        max_bytes: int | None = None,
        overwrite: bool = False,
        content_type_prefixes: list[str] | None = None,
        request_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> str:
        del kwargs
        clean_url = _clean_tool_url(url)
        ok, err = _validate_public_web_url(clean_url)
        if not ok:
            return json.dumps({"error": f"URL validation failed: {err}"})

        try:
            dst = _resolve_write_path(output_path, allowed_dir=self._allowed_dir)
        except PermissionError as exc:
            return json.dumps(
                {"error": str(exc), "url": clean_url, "output_path": output_path}
            )

        if dst.exists() and not overwrite:
            return json.dumps(
                {
                    "error": "Destination file exists; set overwrite=true to replace.",
                    "url": clean_url,
                    "output_path": output_path,
                }
            )

        dst.parent.mkdir(parents=True, exist_ok=True)
        effective_max = WebSearchTool._bounded_int(
            max_bytes,
            default=self._max_bytes,
            minimum=1,
            maximum=self._max_bytes,
        )
        allowed_types = [
            str(item).strip().lower()
            for item in (content_type_prefixes or [])
            if str(item).strip()
        ]
        headers: dict[str, str] = {
            "User-Agent": self.USER_AGENT,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
        }
        custom_headers, header_error = validate_download_request_headers(
            request_headers,
            extra_allowed_names=self._allowed_request_headers,
        )
        if header_error:
            return json.dumps(
                {
                    "error": header_error,
                    "url": clean_url,
                    "output_path": str(dst),
                }
            )
        headers.update(custom_headers)

        try:
            import httpx

            bytes_written = 0
            status_code = 0
            final_url = clean_url
            content_type = ""
            async with httpx.AsyncClient(
                follow_redirects=True,
                max_redirects=5,
                timeout=60.0,
                **public_httpx_client_kwargs(),
            ) as client:
                async with client.stream(
                    "GET",
                    clean_url,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    status_code = int(response.status_code)
                    final_url = str(response.url)
                    ok, err = _validate_public_web_url(final_url)
                    if not ok:
                        return json.dumps(
                            {
                                "error": f"Final URL validation failed: {err}",
                                "url": clean_url,
                                "status": status_code,
                            }
                        )
                    content_type = str(response.headers.get("content-type", "")).lower()
                    if allowed_types and not any(
                        content_type.startswith(prefix) for prefix in allowed_types
                    ):
                        return json.dumps(
                            {
                                "error": (
                                    f"content-type '{content_type or 'unknown'}' "
                                    "not allowed"
                                ),
                                "url": clean_url,
                                "finalUrl": final_url,
                                "status": status_code,
                            }
                        )
                    bytes_written = await stream_response_to_atomic_file(
                        response,
                        dst,
                        max_bytes=effective_max,
                        overwrite=overwrite,
                    )
            return json.dumps(
                {
                    "url": clean_url,
                    "finalUrl": final_url,
                    "status": status_code,
                    "output_path": str(dst),
                    "bytes": bytes_written,
                    "content_type": content_type,
                }
            )
        except Exception as exc:
            return json.dumps(
                {
                    "error": sanitize_web_error(exc),
                    "url": clean_url,
                    "output_path": str(dst),
                }
            )


__all__ = ["WebSearchTool", "WebFetchTool", "DownloadUrlTool"]
