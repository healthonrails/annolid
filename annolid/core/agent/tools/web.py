from __future__ import annotations

import contextlib
import html
import json
import os
import re
import urllib.parse
from pathlib import Path
from typing import Any

from .function_base import FunctionTool
from .common import _normalize, _resolve_write_path, _strip_tags, _validate_url


class WebSearchTool(FunctionTool):
    _USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    _DUCK_RESULT_LINK_RE = re.compile(
        r"<a[^>]*class=(?P<q1>[\"'])[^\"']*\bresult__a\b[^\"']*(?P=q1)"
        r"[^>]*href=(?P<q2>[\"'])(?P<url>.*?)(?P=q2)[^>]*>(?P<title>.*?)</a>",
        flags=re.IGNORECASE | re.DOTALL,
    )
    _VALID_BACKENDS = {"auto", "scrapling", "brave"}

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 5,
        backend: str = "auto",
    ):
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY", "")
        self.max_results = max_results
        self.backend = str(backend or "auto").strip().lower()

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web (Scrapling-first, Brave API fallback). "
            "Returns titles, URLs, and snippets."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "count": {"type": "integer", "minimum": 1, "maximum": 10},
                "backend": {"type": "string", "enum": ["auto", "scrapling", "brave"]},
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
        n = min(max(count or self.max_results, 1), 10)
        requested_backend = str(backend or self.backend or "auto").strip().lower()
        preferred = (
            requested_backend if requested_backend in self._VALID_BACKENDS else "auto"
        )
        scrapling_available = False

        if preferred in {"auto", "scrapling"}:
            scrapling_result = await self._search_with_scrapling(
                query=query_text, count=n
            )
            scrapling_available = scrapling_result is not None
            if scrapling_result:
                return self._format_results(query_text, scrapling_result)
            if preferred == "scrapling":
                if scrapling_available:
                    return f"No results for: {query_text}"
                return (
                    "Error: Scrapling search backend unavailable. "
                    "Configure BRAVE_API_KEY or use backend='brave'."
                )

        brave_result = await self._search_with_brave(query=query_text, count=n)
        if brave_result is not None:
            return self._format_results(query_text, brave_result)

        if scrapling_available:
            return f"No results for: {query_text}"
        return "Error: BRAVE_API_KEY not configured"

    @staticmethod
    def _format_results(query: str, results: list[dict[str, str]]) -> str:
        if not results:
            return f"No results for: {query}"
        lines = [f"Results for: {query}\n"]
        for i, item in enumerate(results, 1):
            lines.append(f"{i}. {item.get('title', '')}\n   {item.get('url', '')}")
            if item.get("description"):
                lines.append(f"   {item['description']}")
        return "\n".join(lines)

    async def _search_with_brave(
        self, *, query: str, count: int
    ) -> list[dict[str, str]] | None:
        if not self.api_key:
            return None
        try:
            import httpx

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": count},
                    headers={
                        "Accept": "application/json",
                        "X-Subscription-Token": self.api_key,
                    },
                    timeout=10.0,
                )
                response.raise_for_status()
            results = response.json().get("web", {}).get("results", [])
            out: list[dict[str, str]] = []
            for item in results[:count]:
                out.append(
                    {
                        "title": str(item.get("title", "")).strip(),
                        "url": str(item.get("url", "")).strip(),
                        "description": str(item.get("description", "")).strip(),
                    }
                )
            return [row for row in out if row.get("title") or row.get("url")]
        except Exception:
            return None

    async def _search_with_scrapling(
        self, *, query: str, count: int
    ) -> list[dict[str, str]] | None:
        target = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
        try:
            html_text = await self._fetch_html_with_scrapling(target)
            if not html_text:
                html_text = await self._fetch_html_with_httpx(target)
            return self._parse_duckduckgo_results(html_text, count=count)
        except Exception:
            return None

    @staticmethod
    def _extract_html_text(page: Any) -> str:
        for attr in (
            "html",
            "html_content",
            "text",
            "content",
            "body_html",
            "raw_html",
        ):
            value = getattr(page, attr, None)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    async def _fetch_html_with_scrapling(self, url: str) -> str:
        try:
            from scrapling.fetchers import AsyncFetcher  # type: ignore
        except Exception:
            return ""

        # Prefer class-based API (per Scrapling docs) and keep kwargs conservative.
        kwargs = {
            "headers": {"User-Agent": self._USER_AGENT},
            "follow_redirects": True,
            "timeout": 15,
        }

        fetch_method = getattr(AsyncFetcher, "fetch", None) or getattr(
            AsyncFetcher, "get", None
        )
        if callable(fetch_method):
            with contextlib.suppress(Exception):
                page = await fetch_method(url, **kwargs)
                html_text = self._extract_html_text(page)
                if html_text:
                    return html_text

        # Backward-compat fallback for versions exposing instance methods only.
        with contextlib.suppress(Exception):
            instance = AsyncFetcher()
            fetch_inst = getattr(instance, "fetch", None) or getattr(
                instance, "get", None
            )
            if callable(fetch_inst):
                page = await fetch_inst(url, **kwargs)
                html_text = self._extract_html_text(page)
                if html_text:
                    return html_text
        return ""

    async def _fetch_html_with_httpx(self, url: str) -> str:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                headers={"User-Agent": self._USER_AGENT},
                timeout=15.0,
                follow_redirects=True,
            )
            response.raise_for_status()
        return str(response.text or "").strip()

    @staticmethod
    def _parse_duckduckgo_results(
        source_html: str, *, count: int
    ) -> list[dict[str, str]]:
        text = str(source_html or "")
        if not text:
            return []
        rows: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for match in WebSearchTool._DUCK_RESULT_LINK_RE.finditer(text):
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
            if not parsed_url.scheme:
                continue
            if (parsed_url.netloc or "").lower().endswith(
                "duckduckgo.com"
            ) and "uddg" not in q:
                continue
            normalized_url = urllib.parse.urlunparse(parsed_url)
            if normalized_url in seen_urls:
                continue
            seen_urls.add(normalized_url)
            rows.append({"title": title, "url": url, "description": ""})
            if len(rows) >= int(count):
                break
        return rows


class WebFetchTool(FunctionTool):
    USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"

    def __init__(self, max_chars: int = 50000):
        self.max_chars = max_chars

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
        ok, err = _validate_url(url)
        if not ok:
            return json.dumps({"error": f"URL validation failed: {err}", "url": url})

        max_chars = maxChars or self.max_chars
        try:
            import httpx

            async with httpx.AsyncClient(
                follow_redirects=True,
                max_redirects=5,
                timeout=30.0,
            ) as client:
                response = await client.get(
                    url, headers={"User-Agent": self.USER_AGENT}
                )
                response.raise_for_status()

            ctype = response.headers.get("content-type", "")
            if "application/json" in ctype:
                text = json.dumps(response.json(), indent=2)
                extractor = "json"
            elif "text/html" in ctype or response.text[:256].lower().startswith(
                ("<!doctype", "<html")
            ):
                body = _strip_tags(response.text)
                text = _normalize(body)
                if extractMode == "markdown":
                    text = text
                extractor = "html-strip"
            else:
                text = response.text
                extractor = "raw"

            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars]
            return json.dumps(
                {
                    "url": url,
                    "finalUrl": str(response.url),
                    "status": response.status_code,
                    "extractor": extractor,
                    "truncated": truncated,
                    "length": len(text),
                    "text": text,
                }
            )
        except Exception as exc:
            return json.dumps({"error": str(exc), "url": url})


class DownloadUrlTool(FunctionTool):
    USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"

    def __init__(
        self, allowed_dir: Path | None = None, max_bytes: int = 25 * 1024 * 1024
    ):
        self._allowed_dir = allowed_dir
        self._max_bytes = max_bytes

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
        ok, err = _validate_url(url)
        if not ok:
            return json.dumps({"error": f"URL validation failed: {err}", "url": url})

        try:
            dst = _resolve_write_path(output_path, allowed_dir=self._allowed_dir)
        except PermissionError as exc:
            return json.dumps(
                {"error": str(exc), "url": url, "output_path": output_path}
            )

        if dst.exists() and not overwrite:
            return json.dumps(
                {
                    "error": "Destination file exists; set overwrite=true to replace.",
                    "url": url,
                    "output_path": output_path,
                }
            )

        dst.parent.mkdir(parents=True, exist_ok=True)
        effective_max = max(1, int(max_bytes or self._max_bytes))
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
        if isinstance(request_headers, dict):
            for key, value in request_headers.items():
                name = str(key or "").strip()
                val = str(value or "").strip()
                if name and val:
                    headers[name] = val

        try:
            import httpx

            bytes_written = 0
            status_code = 0
            final_url = url
            content_type = ""
            async with httpx.AsyncClient(
                follow_redirects=True,
                max_redirects=5,
                timeout=60.0,
            ) as client:
                async with client.stream(
                    "GET",
                    url,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    status_code = int(response.status_code)
                    final_url = str(response.url)
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
                                "url": url,
                                "finalUrl": final_url,
                                "status": status_code,
                            }
                        )
                    with dst.open("wb") as handle:
                        async for chunk in response.aiter_bytes():
                            if not chunk:
                                continue
                            bytes_written += len(chunk)
                            if bytes_written > effective_max:
                                raise ValueError(
                                    f"Download exceeds max_bytes ({effective_max})"
                                )
                            handle.write(chunk)

            return json.dumps(
                {
                    "url": url,
                    "finalUrl": final_url,
                    "status": status_code,
                    "output_path": str(dst),
                    "bytes": bytes_written,
                    "content_type": content_type,
                }
            )
        except Exception as exc:
            with contextlib.suppress(OSError):
                if dst.exists():
                    dst.unlink()
            return json.dumps({"error": str(exc), "url": url, "output_path": str(dst)})


__all__ = ["WebSearchTool", "WebFetchTool", "DownloadUrlTool"]
