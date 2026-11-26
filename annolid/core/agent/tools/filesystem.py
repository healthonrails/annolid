from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from annolid.utils.atomic_io import atomic_write_text

from .function_base import FunctionTool
from .common import _resolve_read_path, _resolve_write_path
from .pdf import ExtractPdfImagesTool, ExtractPdfTextTool, OpenPdfTool

_MAX_TEXT_FILE_SIZE_BYTES = 100 * 1024 * 1024


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
    def read_only(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Read the contents of a UTF-8 text file at the given path, "
            "optionally restricted to an inclusive 1-based line range."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "minimum": 1},
                "end_line": {"type": "integer", "minimum": 1},
            },
            "required": ["path"],
        }

    async def execute(
        self,
        path: str,
        start_line: int | None = None,
        end_line: int | None = None,
        **kwargs: Any,
    ) -> str:
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
            file_size = file_path.stat().st_size
            if file_size > _MAX_TEXT_FILE_SIZE_BYTES:
                return (
                    f"Error: File too large to read ({file_size / (1024 * 1024):.1f} "
                    "MiB). Maximum is 100 MiB."
                )
            if file_path.suffix.lower() == ".pdf":
                return (
                    "Error: PDF is a binary file. Use extract_pdf_text(path=...) "
                    "or open_pdf(path=...) to read PDF content."
                )
            text = file_path.read_text(encoding="utf-8")
            if start_line is None and end_line is None:
                return text
            start = max(1, int(start_line or 1))
            if end_line is not None and int(end_line) < start:
                return f"Error: end_line ({end_line}) must be >= start_line ({start})."
            lines = text.splitlines()
            end = len(lines) if end_line is None else min(len(lines), int(end_line))
            return "\n".join(lines[start - 1 : end])
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
            atomic_write_text(file_path, content, encoding="utf-8")
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
            "The old_text must match exactly unless replace_all=true."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["path", "old_text", "new_text"],
        }

    async def execute(
        self,
        path: str,
        old_text: str,
        new_text: str,
        replace_all: bool = False,
        **kwargs: Any,
    ) -> str:
        del kwargs
        try:
            file_path = _resolve_write_path(path, allowed_dir=self._allowed_dir)
            if not file_path.exists():
                return f"Error: File not found: {path}"
            if not file_path.is_file():
                return f"Error: Not a file: {path}"
            file_size = file_path.stat().st_size
            if file_size > _MAX_TEXT_FILE_SIZE_BYTES:
                return (
                    f"Error: File too large to edit ({file_size / (1024 * 1024):.1f} "
                    "MiB). Maximum is 100 MiB."
                )

            content = file_path.read_text(encoding="utf-8")
            if old_text not in content:
                return (
                    "Error: old_text not found in file. Make sure it matches exactly."
                )
            count = content.count(old_text)
            if count > 1 and not bool(replace_all):
                return (
                    f"Warning: old_text appears {count} times. "
                    "Please make it unique or set replace_all=true."
                )
            updated = (
                content.replace(old_text, new_text)
                if bool(replace_all)
                else content.replace(old_text, new_text, 1)
            )
            atomic_write_text(file_path, updated, encoding="utf-8")
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
    def read_only(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return "List the contents of a directory."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "max_entries": {"type": "integer", "minimum": 1, "maximum": 5000},
            },
            "required": ["path"],
        }

    async def execute(self, path: str, max_entries: int = 500, **kwargs: Any) -> str:
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
            cap = max(1, min(int(max_entries or 500), 5000))
            all_entries = sorted(dir_path.iterdir())
            items = []
            for item in all_entries[:cap]:
                prefix = "DIR " if item.is_dir() else "FILE"
                items.append(f"{prefix}\t{item.name}")
            if len(all_entries) > cap:
                items.append(f"... ({len(all_entries) - cap} more entries omitted)")
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

            target.parent.mkdir(parents=True, exist_ok=True)
            if overwrite:
                source.replace(target)
            else:
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
