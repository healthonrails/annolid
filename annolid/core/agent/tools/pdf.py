from __future__ import annotations

import contextlib
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import parse_qsl, unquote, urlencode, urlparse, urlunparse

from .common import _normalize, _resolve_read_path, _resolve_write_path
from .function_base import FunctionTool
from .web import DownloadUrlTool


class ExtractPdfTextTool(FunctionTool):
    def __init__(
        self,
        allowed_dir: Path | None = None,
        default_max_chars: int = 120000,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ):
        self._allowed_dir = allowed_dir
        self._default_max_chars = default_max_chars
        self._allowed_read_roots = tuple(allowed_read_roots or ())

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
            pdf_path = _resolve_read_path(
                path,
                allowed_dir=self._allowed_dir,
                allowed_read_roots=self._allowed_read_roots,
            )
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


class OpenPdfTool(ExtractPdfTextTool):
    @property
    def name(self) -> str:
        return "open_pdf"

    @property
    def description(self) -> str:
        return (
            "Open a local PDF and return readable text plus page metadata. "
            "Alias of extract_pdf_text for PDF-first workflows."
        )


class ExtractPdfImagesTool(FunctionTool):
    def __init__(
        self,
        allowed_dir: Path | None = None,
        default_dpi: int = 144,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ):
        self._allowed_dir = allowed_dir
        self._default_dpi = default_dpi
        self._allowed_read_roots = tuple(allowed_read_roots or ())

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
            pdf_path = _resolve_read_path(
                path,
                allowed_dir=self._allowed_dir,
                allowed_read_roots=self._allowed_read_roots,
            )
            if not pdf_path.exists():
                return json.dumps({"error": f"File not found: {path}", "path": path})
            if not pdf_path.is_file():
                return json.dumps({"error": f"Not a file: {path}", "path": path})
            if pdf_path.suffix.lower() != ".pdf":
                return json.dumps(
                    {"error": "Path is not a PDF file (.pdf).", "path": path}
                )

            if output_dir:
                out_dir = _resolve_write_path(output_dir, allowed_dir=self._allowed_dir)
            else:
                if self._allowed_dir is not None:
                    out_dir = _resolve_write_path(
                        str(Path(self._allowed_dir) / f"{pdf_path.stem}_pages"),
                        allowed_dir=self._allowed_dir,
                    )
                else:
                    out_dir = pdf_path.parent / f"{pdf_path.stem}_pages"
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


class DownloadPdfTool(FunctionTool):
    def __init__(
        self, allowed_dir: Path | None = None, max_bytes: int = 100 * 1024 * 1024
    ):
        self._allowed_dir = allowed_dir
        self._max_bytes = max_bytes
        self._downloader = DownloadUrlTool(allowed_dir=allowed_dir, max_bytes=max_bytes)

    @property
    def name(self) -> str:
        return "download_pdf"

    @property
    def description(self) -> str:
        return (
            "Download a PDF URL to a local file with PDF content-type checks. "
            "If output_path is omitted, saves under downloads/<url-file-name>.pdf and "
            "auto-renames generic names (for example pdf.pdf) using PDF title metadata."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "output_path": {"type": "string"},
                "max_bytes": {"type": "integer", "minimum": 1},
                "overwrite": {"type": "boolean"},
            },
            "required": ["url"],
        }

    def _default_output_path(self, url: str) -> str:
        parsed = urlparse(url)
        raw_name = Path(unquote(parsed.path or "")).name
        file_name = raw_name or "download.pdf"
        if not file_name.lower().endswith(".pdf"):
            file_name = f"{file_name}.pdf"
        base_dir = (
            Path(self._allowed_dir) if self._allowed_dir is not None else Path(".")
        )
        return str(base_dir / "downloads" / file_name)

    @staticmethod
    def _is_generic_pdf_name(path: Path) -> bool:
        stem = str(path.stem or "").strip().lower()
        return stem in {"pdf", "download", "file", "document", "paper"}

    def _should_rename_downloaded_pdf(self, saved_path: Path, safe_title: str) -> bool:
        title_value = str(safe_title or "").strip()
        if not title_value:
            return False
        current_value = self._sanitize_filename(str(saved_path.stem or ""))
        if not current_value:
            return True
        return current_value.lower() != title_value.lower()

    @staticmethod
    def _sanitize_filename(text: str, *, max_len: int = 100) -> str:
        cleaned = "".join(
            ch if (ch.isalnum() or ch in {" ", "-", "_"}) else " "
            for ch in str(text or "")
        )
        cleaned = " ".join(cleaned.split()).strip(" ._-")
        if not cleaned:
            return ""
        if len(cleaned) > int(max_len):
            cleaned = cleaned[: int(max_len)].rstrip(" ._-")
        return cleaned.replace(" ", "_")

    @staticmethod
    def _extract_pmcid(url: str) -> str:
        text = str(url or "").strip()
        match = re.search(r"\b(PMC\d+)\b", text, flags=re.IGNORECASE)
        if not match:
            return ""
        return str(match.group(1) or "").upper()

    @staticmethod
    def _normalize_http_url(url: str) -> str:
        text = str(url or "").strip()
        if not text:
            return ""
        if text.startswith("ftp://ftp.ncbi.nlm.nih.gov/"):
            return (
                "https://ftp.ncbi.nlm.nih.gov/"
                + text[len("ftp://ftp.ncbi.nlm.nih.gov/") :]
            )
        if text.startswith("//"):
            return f"https:{text}"
        return text

    async def _fetch_pmc_oa_pdf_urls(self, pmcid: str) -> list[str]:
        identifier = str(pmcid or "").strip().upper()
        if not identifier.startswith("PMC"):
            return []
        endpoint = f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={identifier}"
        try:
            import httpx

            async with httpx.AsyncClient(
                follow_redirects=True,
                max_redirects=5,
                timeout=20.0,
            ) as client:
                response = await client.get(
                    endpoint,
                    headers={
                        "User-Agent": DownloadUrlTool.USER_AGENT,
                        "Accept": "application/xml,text/xml,*/*;q=0.8",
                    },
                )
                response.raise_for_status()
                body = str(response.text or "").strip()
        except Exception:
            return []

        if not body:
            return []
        try:
            root = ET.fromstring(body)
        except Exception:
            return []

        candidates: list[str] = []
        for link in root.findall(".//link"):
            fmt = str(link.attrib.get("format") or "").strip().lower()
            href = str(link.attrib.get("href") or "").strip()
            if fmt != "pdf" or not href:
                continue
            normalized = self._normalize_http_url(href)
            if normalized and normalized not in candidates:
                candidates.append(normalized)
        return candidates

    @staticmethod
    def _candidate_pdf_urls(url: str) -> list[str]:
        primary = str(url or "").strip()
        if not primary:
            return []
        candidates = [primary]
        parsed = urlparse(primary)
        host = str(parsed.netloc or "").lower()
        path = str(parsed.path or "")

        if "pmc.ncbi.nlm.nih.gov" in host and "/pdf/" in path.lower():
            pmcid = DownloadPdfTool._extract_pmcid(primary)
            if pmcid:
                epmc_url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
                if epmc_url not in candidates:
                    candidates.insert(0, epmc_url)

            query_items = dict(parse_qsl(parsed.query, keep_blank_values=True))
            if "download" not in {str(k).lower() for k in query_items.keys()}:
                query_items["download"] = "1"
                fallback = urlunparse(
                    parsed._replace(query=urlencode(query_items, doseq=True))
                )
                if fallback not in candidates:
                    candidates.append(fallback)

        if "arxiv.org" in host:
            abs_match = re.search(
                r"^/abs/(\d{4}\.\d{4,5}(?:v\d+)?)$",
                path,
                flags=re.IGNORECASE,
            )
            if abs_match:
                arxiv_id = str(abs_match.group(1) or "").strip()
                pdf_url = urlunparse(
                    parsed._replace(path=f"/pdf/{arxiv_id}.pdf", query="", fragment="")
                )
                if pdf_url not in candidates:
                    candidates.append(pdf_url)

            pdf_match = re.search(
                r"^/pdf/(\d{4}\.\d{4,5}(?:v\d+)?)(?:\.pdf)?$",
                path,
                flags=re.IGNORECASE,
            )
            if pdf_match:
                arxiv_id = str(pdf_match.group(1) or "").strip()
                normalized = urlunparse(
                    parsed._replace(path=f"/pdf/{arxiv_id}.pdf", query="", fragment="")
                )
                if normalized not in candidates:
                    candidates.append(normalized)
        return candidates

    def _extract_pdf_title(self, pdf_path: Path) -> str | None:
        try:
            import fitz  # type: ignore

            with fitz.open(str(pdf_path)) as doc:
                metadata = getattr(doc, "metadata", None) or {}
                if isinstance(metadata, dict):
                    title = str(metadata.get("title") or "").strip()
                    if title:
                        return title
                if len(doc) > 0:
                    text = str(doc[0].get_text("text") or "")
                    for line in text.splitlines():
                        candidate = str(line or "").strip()
                        if len(candidate) >= 8:
                            return candidate
        except Exception:
            pass

        try:
            from pypdf import PdfReader  # type: ignore

            reader = PdfReader(str(pdf_path))
            title = str(
                getattr(getattr(reader, "metadata", None), "title", "") or ""
            ).strip()
            if title:
                return title
            if reader.pages:
                text = str(reader.pages[0].extract_text() or "")
                for line in text.splitlines():
                    candidate = str(line or "").strip()
                    if len(candidate) >= 8:
                        return candidate
        except Exception:
            pass
        return None

    @staticmethod
    def _unique_path(path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        parent = path.parent
        for idx in range(2, 1000):
            candidate = parent / f"{stem}_{idx}{suffix}"
            if not candidate.exists():
                return candidate
        return path

    async def execute(
        self,
        url: str,
        output_path: str | None = None,
        max_bytes: int | None = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> str:
        del kwargs
        user_provided_output = output_path is not None
        destination = str(output_path or self._default_output_path(url))
        last_error_payload: dict[str, Any] | None = None
        result = ""
        pmcid = self._extract_pmcid(url)
        source_url = str(url or "").strip()
        lower_source = source_url.lower()
        candidates = self._candidate_pdf_urls(url) or [source_url]
        # Prefer canonical OA-hosted PDF links first for PMC URLs to avoid
        # predictable 403s on direct article PDF endpoints.
        if pmcid and "pmc.ncbi.nlm.nih.gov" in lower_source:
            oa_first = await self._fetch_pmc_oa_pdf_urls(pmcid)
            if oa_first:
                ordered: list[str] = []
                for item in oa_first + candidates:
                    value = str(item or "").strip()
                    if value and value not in ordered:
                        ordered.append(value)
                candidates = ordered
        oa_resolved = False
        idx = 0
        while idx < len(candidates):
            candidate_url = str(candidates[idx] or "").strip()
            idx += 1
            if not candidate_url:
                continue
            result = await self._downloader.execute(
                url=candidate_url,
                output_path=destination,
                max_bytes=max_bytes or self._max_bytes,
                overwrite=overwrite,
                content_type_prefixes=["application/pdf"],
                request_headers={
                    "Accept": "application/pdf,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": str(url or "").strip(),
                },
            )
            try:
                payload_try = json.loads(result)
            except Exception:
                return result
            if not isinstance(payload_try, dict):
                return result
            if not payload_try.get("error"):
                payload_try["requested_url"] = str(url or "").strip()
                result = json.dumps(payload_try)
                break
            last_error_payload = payload_try
            if (
                str(payload_try.get("error", ""))
                .strip()
                .startswith("Destination file exists")
            ):
                break
            if (
                idx >= len(candidates)
                and not oa_resolved
                and pmcid
                and "pmc.ncbi.nlm.nih.gov" in lower_source
            ):
                oa_resolved = True
                oa_urls = await self._fetch_pmc_oa_pdf_urls(pmcid)
                for oa_url in oa_urls:
                    if oa_url not in candidates:
                        candidates.append(oa_url)
        if not result and last_error_payload is not None:
            result = json.dumps(last_error_payload)
        try:
            payload = json.loads(result)
        except Exception:
            return result
        if payload.get("error"):
            return result
        content_type = str(payload.get("content_type", "")).lower()
        saved_path = Path(str(payload.get("output_path", destination)))
        if saved_path.suffix.lower() != ".pdf":
            with contextlib.suppress(OSError):
                if saved_path.exists():
                    saved_path.unlink()
            payload["error"] = (
                "Downloaded file is not a .pdf. Provide output_path ending with .pdf."
            )
            return json.dumps(payload)

        if not user_provided_output:
            title = self._extract_pdf_title(saved_path)
            safe_title = self._sanitize_filename(str(title or ""))
            if self._should_rename_downloaded_pdf(saved_path, safe_title):
                target_path = self._unique_path(
                    saved_path.with_name(f"{safe_title}.pdf")
                )
                if target_path != saved_path:
                    with contextlib.suppress(OSError):
                        saved_path.rename(target_path)
                        saved_path = target_path
                        payload["output_path"] = str(saved_path)
                        payload["renamed"] = True
                        payload["inferred_title"] = safe_title

        payload["tool"] = self.name
        payload["is_pdf"] = content_type.startswith("application/pdf")
        return json.dumps(payload)


__all__ = [
    "ExtractPdfTextTool",
    "OpenPdfTool",
    "ExtractPdfImagesTool",
    "DownloadPdfTool",
]
