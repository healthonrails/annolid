from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import unquote, urlparse

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
        result = await self._downloader.execute(
            url=url,
            output_path=destination,
            max_bytes=max_bytes or self._max_bytes,
            overwrite=overwrite,
            content_type_prefixes=["application/pdf"],
        )
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

        if not user_provided_output and self._is_generic_pdf_name(saved_path):
            title = self._extract_pdf_title(saved_path)
            safe_title = self._sanitize_filename(str(title or ""))
            if safe_title:
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
