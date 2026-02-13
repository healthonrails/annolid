from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .function_base import FunctionTool
from .common import _normalize, _resolve_read_path, _resolve_write_path


class ReadFileTool(FunctionTool):
    def __init__(
        self,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ):
        self._allowed_dir = allowed_dir
        self._allowed_read_roots = tuple(allowed_read_roots or ())

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
            file_path = _resolve_read_path(
                path,
                allowed_dir=self._allowed_dir,
                allowed_read_roots=self._allowed_read_roots,
            )
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
            file_path = _resolve_write_path(path, allowed_dir=self._allowed_dir)
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
            file_path = _resolve_write_path(path, allowed_dir=self._allowed_dir)
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
    def __init__(
        self,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ):
        self._allowed_dir = allowed_dir
        self._allowed_read_roots = tuple(allowed_read_roots or ())

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
            dir_path = _resolve_read_path(
                path,
                allowed_dir=self._allowed_dir,
                allowed_read_roots=self._allowed_read_roots,
            )
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


__all__ = [
    "ReadFileTool",
    "ExtractPdfTextTool",
    "ExtractPdfImagesTool",
    "WriteFileTool",
    "EditFileTool",
    "ListDirTool",
]
