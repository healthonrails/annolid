from __future__ import annotations

import asyncio
import contextlib
import html
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.parse import urlparse

from annolid.core.agent.cron import CronPayload, CronSchedule, CronService
from annolid.core.agent.utils import get_agent_data_path

from .function_base import FunctionTool
from .function_registry import FunctionToolRegistry


def _resolve_path(path: str, allowed_dir: Path | None = None) -> Path:
    resolved = Path(path).expanduser().resolve()
    if allowed_dir and not str(resolved).startswith(str(allowed_dir.resolve())):
        raise PermissionError(f"Path {path} is outside allowed directory {allowed_dir}")
    return resolved


def _strip_tags(text: str) -> str:
    text = re.sub(r"<script[\s\S]*?</script>", "", text, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def _normalize(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _validate_url(url: str) -> tuple[bool, str]:
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False, f"Only http/https allowed, got '{p.scheme or 'none'}'"
        if not p.netloc:
            return False, "Missing domain"
        return True, ""
    except Exception as exc:
        return False, str(exc)


class ReadFileTool(FunctionTool):
    def __init__(self, allowed_dir: Path | None = None):
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a UTF-8 text file at the given path."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }

    async def execute(self, path: str, **kwargs: Any) -> str:
        del kwargs
        try:
            file_path = _resolve_path(path, self._allowed_dir)
            if not file_path.exists():
                return f"Error: File not found: {path}"
            if not file_path.is_file():
                return f"Error: Not a file: {path}"
            if file_path.suffix.lower() == ".pdf":
                return (
                    "Error: PDF is a binary file. Use extract_pdf_text(path=...) "
                    "or extract_pdf_images(path=...) to read PDF content."
                )
            return file_path.read_text(encoding="utf-8")
        except PermissionError as exc:
            return f"Error: {exc}"
        except UnicodeDecodeError:
            return (
                "Error: File is not UTF-8 text. Use a format-specific tool "
                "(for PDF use extract_pdf_text or extract_pdf_images)."
            )
        except Exception as exc:
            return f"Error reading file: {exc}"


class ExtractPdfTextTool(FunctionTool):
    def __init__(
        self, allowed_dir: Path | None = None, default_max_chars: int = 120000
    ):
        self._allowed_dir = allowed_dir
        self._default_max_chars = default_max_chars

    @property
    def name(self) -> str:
        return "extract_pdf_text"

    @property
    def description(self) -> str:
        return "Extract text from a local PDF file for summarization and analysis."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_page": {"type": "integer", "minimum": 1},
                "max_pages": {"type": "integer", "minimum": 1},
                "max_chars": {"type": "integer", "minimum": 100},
            },
            "required": ["path"],
        }

    async def execute(
        self,
        path: str,
        start_page: int = 1,
        max_pages: int = 20,
        max_chars: int | None = None,
        **kwargs: Any,
    ) -> str:
        del kwargs
        try:
            pdf_path = _resolve_path(path, self._allowed_dir)
            if not pdf_path.exists():
                return json.dumps({"error": f"File not found: {path}", "path": path})
            if not pdf_path.is_file():
                return json.dumps({"error": f"Not a file: {path}", "path": path})
            if pdf_path.suffix.lower() != ".pdf":
                return json.dumps(
                    {
                        "error": "Path is not a PDF file (.pdf).",
                        "path": path,
                    }
                )

            start_index = max(0, int(start_page) - 1)
            pages_limit = max(1, int(max_pages))
            chars_limit = max(100, int(max_chars or self._default_max_chars))

            text, pages_read, total_pages, backend = self._read_pdf_text(
                pdf_path=pdf_path,
                start_index=start_index,
                pages_limit=pages_limit,
            )
            normalized = _normalize(text)
            truncated = len(normalized) > chars_limit
            if truncated:
                normalized = normalized[:chars_limit]
            return json.dumps(
                {
                    "path": str(pdf_path),
                    "backend": backend,
                    "start_page": start_index + 1,
                    "pages_read": pages_read,
                    "total_pages": total_pages,
                    "truncated": truncated,
                    "length": len(normalized),
                    "text": normalized,
                }
            )
        except PermissionError as exc:
            return json.dumps({"error": str(exc), "path": path})
        except Exception as exc:
            return json.dumps({"error": str(exc), "path": path})

    def _read_pdf_text(
        self, *, pdf_path: Path, start_index: int, pages_limit: int
    ) -> tuple[str, int, int, str]:
        try:
            import fitz  # type: ignore

            with fitz.open(str(pdf_path)) as doc:
                total_pages = len(doc)
                if start_index >= total_pages:
                    return "", 0, total_pages, "pymupdf"
                end_index = min(total_pages, start_index + pages_limit)
                parts: list[str] = []
                for index in range(start_index, end_index):
                    page = doc[index]
                    parts.append(page.get_text("text") or "")
                return (
                    "\n\n".join(parts),
                    end_index - start_index,
                    total_pages,
                    "pymupdf",
                )
        except ImportError:
            pass

        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(str(pdf_path))
            total_pages = len(reader.pages)
            if start_index >= total_pages:
                return "", 0, total_pages, "pypdf"
            end_index = min(total_pages, start_index + pages_limit)
            parts = []
            for index in range(start_index, end_index):
                parts.append(reader.pages[index].extract_text() or "")
            return "\n\n".join(parts), end_index - start_index, total_pages, "pypdf"
        except ImportError as exc:
            raise RuntimeError(
                "PDF extraction backend is not available. Install pymupdf or pypdf."
            ) from exc


class ExtractPdfImagesTool(FunctionTool):
    def __init__(self, allowed_dir: Path | None = None, default_dpi: int = 144):
        self._allowed_dir = allowed_dir
        self._default_dpi = default_dpi

    @property
    def name(self) -> str:
        return "extract_pdf_images"

    @property
    def description(self) -> str:
        return "Render PDF pages to images so vision models can read the document."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "output_dir": {"type": "string"},
                "start_page": {"type": "integer", "minimum": 1},
                "max_pages": {"type": "integer", "minimum": 1},
                "dpi": {"type": "integer", "minimum": 72, "maximum": 600},
                "overwrite": {"type": "boolean"},
            },
            "required": ["path"],
        }

    async def execute(
        self,
        path: str,
        output_dir: str | None = None,
        start_page: int = 1,
        max_pages: int = 10,
        dpi: int | None = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> str:
        del kwargs
        try:
            pdf_path = _resolve_path(path, self._allowed_dir)
            if not pdf_path.exists():
                return json.dumps({"error": f"File not found: {path}", "path": path})
            if not pdf_path.is_file():
                return json.dumps({"error": f"Not a file: {path}", "path": path})
            if pdf_path.suffix.lower() != ".pdf":
                return json.dumps(
                    {"error": "Path is not a PDF file (.pdf).", "path": path}
                )

            if output_dir:
                out_dir = _resolve_path(output_dir, self._allowed_dir)
            else:
                out_dir = pdf_path.parent / f"{pdf_path.stem}_pages"
                if self._allowed_dir is not None:
                    out_dir = _resolve_path(str(out_dir), self._allowed_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            start_index = max(0, int(start_page) - 1)
            pages_limit = max(1, int(max_pages))
            render_dpi = max(72, int(dpi or self._default_dpi))
            image_paths, total_pages = self._render_pdf_images(
                pdf_path=pdf_path,
                output_dir=out_dir,
                start_index=start_index,
                pages_limit=pages_limit,
                dpi=render_dpi,
                overwrite=bool(overwrite),
            )
            return json.dumps(
                {
                    "path": str(pdf_path),
                    "output_dir": str(out_dir),
                    "start_page": start_index + 1,
                    "total_pages": total_pages,
                    "pages_rendered": len(image_paths),
                    "dpi": render_dpi,
                    "images": image_paths,
                }
            )
        except PermissionError as exc:
            return json.dumps({"error": str(exc), "path": path})
        except Exception as exc:
            return json.dumps({"error": str(exc), "path": path})

    def _render_pdf_images(
        self,
        *,
        pdf_path: Path,
        output_dir: Path,
        start_index: int,
        pages_limit: int,
        dpi: int,
        overwrite: bool,
    ) -> tuple[list[str], int]:
        try:
            import fitz  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PDF image rendering requires pymupdf (fitz).") from exc

        written: list[str] = []
        scale = float(dpi) / 72.0
        matrix = fitz.Matrix(scale, scale)
        with fitz.open(str(pdf_path)) as doc:
            total_pages = len(doc)
            if start_index >= total_pages:
                return [], total_pages
            end_index = min(total_pages, start_index + pages_limit)
            for index in range(start_index, end_index):
                page = doc[index]
                page_number = index + 1
                out_path = output_dir / f"{pdf_path.stem}_p{page_number:04d}.png"
                if out_path.exists() and not overwrite:
                    written.append(str(out_path))
                    continue
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                pix.save(str(out_path))
                written.append(str(out_path))
        return written, total_pages


class WriteFileTool(FunctionTool):
    def __init__(self, allowed_dir: Path | None = None):
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file path. Creates parent directories if needed."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str, content: str, **kwargs: Any) -> str:
        del kwargs
        try:
            file_path = _resolve_path(path, self._allowed_dir)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return f"Successfully wrote {len(content)} bytes to {path}"
        except PermissionError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error writing file: {exc}"


class EditFileTool(FunctionTool):
    def __init__(self, allowed_dir: Path | None = None):
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return (
            "Edit a file by replacing old_text with new_text. "
            "The old_text must match exactly."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        }

    async def execute(
        self, path: str, old_text: str, new_text: str, **kwargs: Any
    ) -> str:
        del kwargs
        try:
            file_path = _resolve_path(path, self._allowed_dir)
            if not file_path.exists():
                return f"Error: File not found: {path}"

            content = file_path.read_text(encoding="utf-8")
            if old_text not in content:
                return (
                    "Error: old_text not found in file. Make sure it matches exactly."
                )
            count = content.count(old_text)
            if count > 1:
                return (
                    f"Warning: old_text appears {count} times. Please make it unique."
                )
            file_path.write_text(
                content.replace(old_text, new_text, 1), encoding="utf-8"
            )
            return f"Successfully edited {path}"
        except PermissionError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error editing file: {exc}"


class ListDirTool(FunctionTool):
    def __init__(self, allowed_dir: Path | None = None):
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "List the contents of a directory."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }

    async def execute(self, path: str, **kwargs: Any) -> str:
        del kwargs
        try:
            dir_path = _resolve_path(path, self._allowed_dir)
            if not dir_path.exists():
                return f"Error: Directory not found: {path}"
            if not dir_path.is_dir():
                return f"Error: Not a directory: {path}"
            items = []
            for item in sorted(dir_path.iterdir()):
                prefix = "DIR " if item.is_dir() else "FILE"
                items.append(f"{prefix}\t{item.name}")
            return "\n".join(items) if items else f"Directory {path} is empty"
        except PermissionError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error listing directory: {exc}"


class ExecTool(FunctionTool):
    def __init__(
        self,
        timeout: int = 60,
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
        allow_patterns: list[str] | None = None,
        restrict_to_workspace: bool = False,
    ):
        self.timeout = timeout
        self.working_dir = working_dir
        self.deny_patterns = deny_patterns or [
            r"\brm\s+-[rf]{1,2}\b",
            r"\bdel\s+/[fq]\b",
            r"\brmdir\s+/s\b",
            r"\b(format|mkfs|diskpart)\b",
            r"\bdd\s+if=",
            r">\s*/dev/sd",
            r"\b(shutdown|reboot|poweroff)\b",
            r":\(\)\s*\{.*\};\s*:",
        ]
        self.allow_patterns = allow_patterns or []
        self.restrict_to_workspace = restrict_to_workspace

    @property
    def name(self) -> str:
        return "exec"

    @property
    def description(self) -> str:
        return "Execute a shell command and return stdout/stderr."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "working_dir": {"type": "string"},
            },
            "required": ["command"],
        }

    async def execute(
        self, command: str, working_dir: str | None = None, **kwargs: Any
    ) -> str:
        del kwargs
        cwd = working_dir or self.working_dir or os.getcwd()
        guard_error = self._guard_command(command, cwd)
        if guard_error:
            return guard_error

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                return f"Error: Command timed out after {self.timeout} seconds"

            parts: list[str] = []
            if stdout:
                parts.append(stdout.decode("utf-8", errors="replace"))
            if stderr:
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    parts.append(f"STDERR:\n{stderr_text}")
            if proc.returncode != 0:
                parts.append(f"\nExit code: {proc.returncode}")
            result = "\n".join(parts) if parts else "(no output)"
            if len(result) > 10000:
                result = (
                    result[:10000]
                    + f"\n... (truncated, {len(result) - 10000} more chars)"
                )
            return result
        except Exception as exc:
            return f"Error executing command: {exc}"

    def _guard_command(self, command: str, cwd: str) -> str | None:
        cmd = command.strip()
        lower = cmd.lower()
        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                return "Error: Command blocked by safety guard (dangerous pattern detected)"
        if self.allow_patterns and not any(
            re.search(p, lower) for p in self.allow_patterns
        ):
            return "Error: Command blocked by safety guard (not in allowlist)"
        if self.restrict_to_workspace:
            if "..\\" in cmd or "../" in cmd:
                return (
                    "Error: Command blocked by safety guard (path traversal detected)"
                )
            cwd_path = Path(cwd).resolve()
            abs_paths = re.findall(r"(?:^|[\s|>])(/[^\s\"'>]+)", cmd)
            for raw in abs_paths:
                try:
                    p = Path(raw.strip()).resolve()
                except Exception:
                    continue
                if p.is_absolute() and cwd_path not in p.parents and p != cwd_path:
                    return (
                        "Error: Command blocked by safety guard "
                        "(path outside working dir)"
                    )
        return None


class WebSearchTool(FunctionTool):
    def __init__(self, api_key: str | None = None, max_results: int = 5):
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY", "")
        self.max_results = max_results

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web. Returns titles, URLs, and snippets."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "count": {"type": "integer", "minimum": 1, "maximum": 10},
            },
            "required": ["query"],
        }

    async def execute(self, query: str, count: int | None = None, **kwargs: Any) -> str:
        del kwargs
        if not self.api_key:
            return "Error: BRAVE_API_KEY not configured"
        try:
            import httpx

            n = min(max(count or self.max_results, 1), 10)
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": n},
                    headers={
                        "Accept": "application/json",
                        "X-Subscription-Token": self.api_key,
                    },
                    timeout=10.0,
                )
                response.raise_for_status()
            results = response.json().get("web", {}).get("results", [])
            if not results:
                return f"No results for: {query}"
            lines = [f"Results for: {query}\n"]
            for i, item in enumerate(results[:n], 1):
                lines.append(f"{i}. {item.get('title', '')}\n   {item.get('url', '')}")
                if item.get("description"):
                    lines.append(f"   {item['description']}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {exc}"


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
        **kwargs: Any,
    ) -> str:
        del kwargs
        ok, err = _validate_url(url)
        if not ok:
            return json.dumps({"error": f"URL validation failed: {err}", "url": url})

        try:
            dst = _resolve_path(output_path, self._allowed_dir)
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
                    headers={"User-Agent": self.USER_AGENT},
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


class MessageTool(FunctionTool):
    def __init__(
        self,
        send_callback: Callable[[str, str, str], Awaitable[None] | None] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
    ):
        self._send_callback = send_callback
        self._default_channel = default_channel
        self._default_chat_id = default_chat_id

    def set_context(self, channel: str, chat_id: str) -> None:
        self._default_channel = channel
        self._default_chat_id = chat_id

    @property
    def name(self) -> str:
        return "message"

    @property
    def description(self) -> str:
        return "Send a message to the user."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "channel": {"type": "string"},
                "chat_id": {"type": "string"},
            },
            "required": ["content"],
        }

    async def execute(
        self,
        content: str,
        channel: str | None = None,
        chat_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        del kwargs
        if self._send_callback is None:
            return "Error: Message sending not configured"
        resolved_channel = channel or self._default_channel
        resolved_chat_id = chat_id or self._default_chat_id
        if not resolved_channel or not resolved_chat_id:
            return "Error: No target channel/chat specified"
        ret = self._send_callback(resolved_channel, resolved_chat_id, content)
        if asyncio.iscoroutine(ret):
            await ret
        return f"Message sent to {resolved_channel}:{resolved_chat_id}"


class SpawnTool(FunctionTool):
    def __init__(
        self,
        spawn_callback: Callable[..., Awaitable[str] | str] | None = None,
    ):
        self._spawn_callback = spawn_callback
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        self._origin_channel = channel
        self._origin_chat_id = chat_id

    def set_spawn_callback(
        self, callback: Callable[..., Awaitable[str] | str] | None
    ) -> None:
        self._spawn_callback = callback

    @property
    def name(self) -> str:
        return "spawn"

    @property
    def description(self) -> str:
        return "Spawn a subagent/background task."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "label": {"type": "string"},
            },
            "required": ["task"],
        }

    async def execute(self, task: str, label: str | None = None, **kwargs: Any) -> str:
        del kwargs
        if self._spawn_callback is None:
            return "Error: spawn callback not configured"
        try:
            ret = self._spawn_callback(
                task=task,
                label=label,
                origin_channel=self._origin_channel,
                origin_chat_id=self._origin_chat_id,
            )
        except TypeError:
            ret = self._spawn_callback(task, label)
        if asyncio.iscoroutine(ret):
            return str(await ret)
        return str(ret)


class CronTool(FunctionTool):
    def __init__(
        self,
        *,
        store_path: Path | None = None,
        send_callback: Callable[[str, str, str], Awaitable[None] | None] | None = None,
    ):
        self._channel = ""
        self._chat_id = ""
        self._send_callback = send_callback
        if store_path is None:
            store_path = self._resolve_default_store_path()
        self._service = CronService(store_path=store_path, on_job=self._on_job)

    @staticmethod
    def _resolve_default_store_path() -> Path:
        data_path = get_agent_data_path()
        candidates = [
            data_path / "cron" / "jobs.json",
            Path.cwd() / ".annolid" / "cron" / "jobs.json",
            Path("/tmp") / "annolid" / "cron" / "jobs.json",
        ]
        for path in candidates:
            if CronTool._is_store_path_writable(path):
                return path
        return candidates[-1]

    @staticmethod
    def _is_store_path_writable(path: Path) -> bool:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            return False
        probe = path.parent / f".cron-write-probe-{os.getpid()}-{uuid.uuid4().hex}"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return True
        except OSError:
            return False

    def set_context(self, channel: str, chat_id: str) -> None:
        self._channel = channel
        self._chat_id = chat_id

    @property
    def name(self) -> str:
        return "cron"

    @property
    def description(self) -> str:
        return (
            "Schedule reminders and recurring tasks. Actions: "
            "add, list, remove, enable, disable, run, status."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "add",
                        "list",
                        "remove",
                        "enable",
                        "disable",
                        "run",
                        "status",
                    ],
                },
                "message": {"type": "string"},
                "every_seconds": {"type": "integer"},
                "cron_expr": {"type": "string"},
                "at_ms": {"type": "integer"},
                "deliver": {"type": "boolean"},
                "job_id": {"type": "string"},
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        message: str = "",
        every_seconds: int | None = None,
        cron_expr: str | None = None,
        at_ms: int | None = None,
        deliver: bool = False,
        job_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        del kwargs
        if action == "add":
            return self._add_job(
                message=message,
                every_seconds=every_seconds,
                cron_expr=cron_expr,
                at_ms=at_ms,
                deliver=bool(deliver),
            )
        if action == "list":
            return self._list_jobs()
        if action == "remove":
            return self._remove_job(job_id)
        if action == "enable":
            return self._enable_job(job_id, True)
        if action == "disable":
            return self._enable_job(job_id, False)
        if action == "run":
            return await self._run_job(job_id)
        if action == "status":
            return self._status()
        return f"Unknown action: {action}"

    def _add_job(
        self,
        *,
        message: str,
        every_seconds: int | None,
        cron_expr: str | None,
        at_ms: int | None,
        deliver: bool,
    ) -> str:
        if not message:
            return "Error: message is required for add"
        if not self._channel or not self._chat_id:
            return "Error: no session context (channel/chat_id)"
        if not every_seconds and not cron_expr and not at_ms:
            return "Error: one of every_seconds, cron_expr, or at_ms is required"
        if every_seconds and int(every_seconds) <= 0:
            return "Error: every_seconds must be > 0"

        if at_ms:
            schedule = CronSchedule(kind="at", at_ms=int(at_ms))
        elif every_seconds:
            schedule = CronSchedule(kind="every", every_ms=int(every_seconds) * 1000)
        else:
            schedule = CronSchedule(kind="cron", expr=str(cron_expr or "").strip())

        payload = CronPayload(
            kind="agent_turn",
            message=message,
            deliver=bool(deliver),
            channel=self._channel,
            to=self._chat_id,
        )
        job = self._service.add_job(
            name=message[:40],
            schedule=schedule,
            payload=payload,
            delete_after_run=(schedule.kind == "at"),
        )
        return f"Created job '{message[:30]}' (id: {job.id})"

    def _list_jobs(self) -> str:
        jobs = self._service.list_jobs(include_disabled=True)
        if not jobs:
            return "No scheduled jobs."
        lines = []
        for job in jobs:
            if job.schedule.kind == "every":
                mode = f"every={int((job.schedule.every_ms or 0) / 1000)}s"
            elif job.schedule.kind == "cron":
                mode = f"cron={job.schedule.expr}"
            else:
                mode = f"at={job.schedule.at_ms}"
            marker = "enabled" if job.enabled else "disabled"
            lines.append(
                f"- {job.payload.message[:30]} (id: {job.id}, {mode}, {marker})"
            )
        return "Scheduled jobs:\n" + "\n".join(lines)

    def _remove_job(self, job_id: str | None) -> str:
        if not job_id:
            return "Error: job_id is required for remove"
        if self._service.remove_job(job_id):
            return f"Removed job {job_id}"
        return f"Job {job_id} not found"

    def _enable_job(self, job_id: str | None, enabled: bool) -> str:
        if not job_id:
            return "Error: job_id is required"
        updated = self._service.enable_job(job_id, enabled=enabled)
        if updated is None:
            return f"Job {job_id} not found"
        return f"{'Enabled' if enabled else 'Disabled'} job {job_id}"

    async def _run_job(self, job_id: str | None) -> str:
        if not job_id:
            return "Error: job_id is required"
        ok = await self._service.run_job(job_id, force=True)
        if not ok:
            return f"Job {job_id} not found"
        return f"Ran job {job_id}"

    def _status(self) -> str:
        status = self._service.status()
        text = (
            f"Cron status: enabled={status.get('enabled')} "
            f"jobs={status.get('jobs')} next_wake_at_ms={status.get('next_wake_at_ms')}"
        )
        persistence_error = str(status.get("persistence_error") or "").strip()
        if persistence_error:
            text += f" persistence_error={persistence_error}"
        return text

    async def _on_job(self, job) -> str | None:
        message = str(job.payload.message or "")
        if job.payload.deliver and self._send_callback is not None:
            channel = str(job.payload.channel or self._channel or "")
            chat_id = str(job.payload.to or self._chat_id or "")
            if channel and chat_id and message:
                result = self._send_callback(channel, chat_id, message)
                if asyncio.iscoroutine(result):
                    await result
        return message


def register_nanobot_style_tools(
    registry: FunctionToolRegistry,
    *,
    allowed_dir: Path | None = None,
    send_callback: Callable[[str, str, str], Awaitable[None] | None] | None = None,
    spawn_callback: Callable[[str, str | None], Awaitable[str] | str] | None = None,
) -> None:
    """Register a Nanobot-like default tool set."""

    registry.register(ReadFileTool(allowed_dir=allowed_dir))
    registry.register(ExtractPdfTextTool(allowed_dir=allowed_dir))
    registry.register(ExtractPdfImagesTool(allowed_dir=allowed_dir))
    registry.register(WriteFileTool(allowed_dir=allowed_dir))
    registry.register(EditFileTool(allowed_dir=allowed_dir))
    registry.register(ListDirTool(allowed_dir=allowed_dir))
    registry.register(ExecTool())
    registry.register(WebSearchTool())
    registry.register(WebFetchTool())
    registry.register(DownloadUrlTool(allowed_dir=allowed_dir))
    registry.register(MessageTool(send_callback=send_callback))
    registry.register(SpawnTool(spawn_callback=spawn_callback))
    registry.register(CronTool(send_callback=send_callback))
