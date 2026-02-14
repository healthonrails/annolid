from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .function_base import FunctionTool
from .common import _resolve_read_path, _resolve_write_path
from .pdf import ExtractPdfImagesTool, ExtractPdfTextTool, OpenPdfTool


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
                    "or open_pdf(path=...) to read PDF content."
                )
            return file_path.read_text(encoding="utf-8")
        except PermissionError as exc:
            return f"Error: {exc}"
        except UnicodeDecodeError:
            return (
                "Error: File is not UTF-8 text. Use a format-specific tool "
                "(for PDF use open_pdf, extract_pdf_text, or extract_pdf_images)."
            )
        except Exception as exc:
            return f"Error reading file: {exc}"


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


class RenameFileTool(FunctionTool):
    def __init__(self, allowed_dir: Path | None = None):
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "rename_file"

    @property
    def description(self) -> str:
        return (
            "Rename or move a file/directory within the writable workspace. "
            "Use new_name to rename in-place, or new_path for an explicit destination."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "new_name": {"type": "string"},
                "new_path": {"type": "string"},
                "overwrite": {"type": "boolean"},
            },
            "required": ["path"],
        }

    async def execute(
        self,
        path: str,
        new_name: str = "",
        new_path: str = "",
        overwrite: bool = False,
        **kwargs: Any,
    ) -> str:
        del kwargs
        try:
            source = _resolve_write_path(path, allowed_dir=self._allowed_dir)
            if not source.exists():
                return f"Error: File not found: {path}"

            target: Path
            explicit_target = str(new_path or "").strip()
            rename_name = str(new_name or "").strip()
            if explicit_target:
                target = _resolve_write_path(
                    explicit_target, allowed_dir=self._allowed_dir
                )
            elif rename_name:
                if Path(rename_name).name != rename_name:
                    return (
                        "Error: new_name must be a base name without path separators."
                    )
                target = source.with_name(rename_name)
                target = _resolve_write_path(str(target), allowed_dir=self._allowed_dir)
            else:
                return "Error: Provide either new_name or new_path."

            if source == target:
                return f"No-op: Source and destination are the same ({source})."

            if target.exists():
                if not overwrite:
                    return (
                        f"Error: Target already exists: {target}. "
                        "Set overwrite=true to replace it."
                    )
                if target.is_dir():
                    return (
                        "Error: overwrite=true does not replace existing directories."
                    )
                target.unlink()

            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
            return f"Successfully renamed {source} -> {target}"
        except PermissionError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error renaming file: {exc}"


__all__ = [
    "ReadFileTool",
    "ExtractPdfTextTool",
    "OpenPdfTool",
    "ExtractPdfImagesTool",
    "WriteFileTool",
    "EditFileTool",
    "ListDirTool",
    "RenameFileTool",
]
