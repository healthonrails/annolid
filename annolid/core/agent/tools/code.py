from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any, Sequence

from .function_base import FunctionTool
from .common import _is_probably_text_file, _iter_text_files, _resolve_read_path


class CodeSearchTool(FunctionTool):
    def __init__(
        self,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ):
        self._allowed_dir = allowed_dir
        self._allowed_read_roots = tuple(allowed_read_roots or ())

    @property
    def name(self) -> str:
        return "code_search"

    @property
    def description(self) -> str:
        return (
            "Search text across source files and return file/line matches with "
            "optional context."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "path": {"type": "string"},
                "glob": {"type": "string"},
                "regex": {"type": "boolean"},
                "case_sensitive": {"type": "boolean"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 2000},
                "context_lines": {"type": "integer", "minimum": 0, "maximum": 10},
            },
            "required": ["query"],
        }

    async def execute(
        self,
        query: str,
        path: str = ".",
        glob: str = "*",
        regex: bool = False,
        case_sensitive: bool = False,
        max_results: int = 100,
        context_lines: int = 0,
        **kwargs: Any,
    ) -> str:
        del kwargs
        query_text = str(query or "")
        if not query_text:
            return json.dumps({"error": "query must be non-empty"})

        try:
            search_root = _resolve_read_path(
                path,
                allowed_dir=self._allowed_dir,
                allowed_read_roots=self._allowed_read_roots,
            )
        except PermissionError as exc:
            return json.dumps({"error": str(exc), "path": path})

        if not search_root.exists():
            return json.dumps({"error": f"Path not found: {path}", "path": path})

        max_hits = max(1, min(int(max_results), 2000))
        ctx = max(0, min(int(context_lines), 10))
        file_glob = str(glob or "*").strip() or "*"
        use_regex = bool(regex)
        flags = 0 if case_sensitive else re.IGNORECASE

        try:
            pattern = re.compile(query_text, flags) if use_regex else None
        except re.error as exc:
            return json.dumps({"error": f"Invalid regex: {exc}", "query": query_text})

        candidates = []
        if search_root.is_dir():
            candidates = list(_iter_text_files(search_root))
        elif search_root.is_file():
            candidates = [search_root]
        else:
            return json.dumps({"error": f"Unsupported path: {path}", "path": path})

        results: list[dict[str, Any]] = []
        scanned_files = 0
        for file_path in candidates:
            rel_name = str(file_path.name)
            rel_path = str(file_path)
            try:
                if search_root.is_dir():
                    rel_name = str(file_path.relative_to(search_root))
                    rel_path = rel_name
            except Exception:
                pass
            if not Path(rel_name).match(file_glob):
                continue
            if not file_path.is_file():
                continue
            if not _is_probably_text_file(file_path):
                continue
            scanned_files += 1
            try:
                text = file_path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            except Exception:
                continue
            lines = text.splitlines()
            for idx, line in enumerate(lines, start=1):
                matched = (
                    bool(pattern.search(line))
                    if pattern is not None
                    else (
                        query_text in line
                        if case_sensitive
                        else query_text.lower() in line.lower()
                    )
                )
                if not matched:
                    continue
                item: dict[str, Any] = {
                    "path": rel_path,
                    "line": idx,
                    "text": line,
                }
                if ctx > 0:
                    start = max(1, idx - ctx)
                    end = min(len(lines), idx + ctx)
                    item["context"] = [
                        {"line": no, "text": lines[no - 1]}
                        for no in range(start, end + 1)
                    ]
                results.append(item)
                if len(results) >= max_hits:
                    return json.dumps(
                        {
                            "query": query_text,
                            "path": str(search_root),
                            "glob": file_glob,
                            "regex": use_regex,
                            "case_sensitive": bool(case_sensitive),
                            "scanned_files": scanned_files,
                            "count": len(results),
                            "truncated": True,
                            "results": results,
                        }
                    )
        return json.dumps(
            {
                "query": query_text,
                "path": str(search_root),
                "glob": file_glob,
                "regex": use_regex,
                "case_sensitive": bool(case_sensitive),
                "scanned_files": scanned_files,
                "count": len(results),
                "truncated": False,
                "results": results,
            }
        )


class CodeExplainTool(FunctionTool):
    def __init__(
        self,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ):
        self._allowed_dir = allowed_dir
        self._allowed_read_roots = tuple(allowed_read_roots or ())

    @property
    def name(self) -> str:
        return "code_explain"

    @property
    def description(self) -> str:
        return (
            "Explain a Python file or symbol using static AST analysis "
            "(docstrings, signatures, and call graph hints)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "symbol": {"type": "string"},
                "include_source": {"type": "boolean"},
                "max_source_lines": {
                    "type": "integer",
                    "minimum": 5,
                    "maximum": 500,
                },
            },
            "required": ["path"],
        }

    async def execute(
        self,
        path: str,
        symbol: str | None = None,
        include_source: bool = False,
        max_source_lines: int = 120,
        **kwargs: Any,
    ) -> str:
        del kwargs
        try:
            file_path = _resolve_read_path(
                path,
                allowed_dir=self._allowed_dir,
                allowed_read_roots=self._allowed_read_roots,
            )
        except PermissionError as exc:
            return json.dumps({"error": str(exc), "path": path})

        if not file_path.exists():
            return json.dumps({"error": f"File not found: {path}", "path": path})
        if not file_path.is_file():
            return json.dumps({"error": f"Not a file: {path}", "path": path})
        if file_path.suffix.lower() != ".py":
            return json.dumps(
                {
                    "error": "code_explain currently supports only Python files (.py).",
                    "path": str(file_path),
                }
            )
        try:
            source = file_path.read_text(encoding="utf-8")
        except Exception as exc:
            return json.dumps({"error": str(exc), "path": str(file_path)})
        try:
            module = ast.parse(source, filename=str(file_path))
        except SyntaxError as exc:
            return json.dumps(
                {
                    "error": f"Python parse error: {exc.msg}",
                    "line": int(exc.lineno or 0),
                    "path": str(file_path),
                }
            )

        lines = source.splitlines()
        module_doc = (ast.get_docstring(module) or "").strip()
        summary: dict[str, Any] = {
            "path": str(file_path),
            "module_docstring": module_doc,
            "imports": self._collect_imports(module),
            "classes": [],
            "functions": [],
        }

        target = str(symbol or "").strip()
        if target:
            node = self._find_symbol_node(module, target)
            if node is None:
                return json.dumps(
                    {
                        "path": str(file_path),
                        "symbol": target,
                        "error": f"Symbol '{target}' not found",
                    }
                )
            payload = self._describe_node(node, lines)
            payload["path"] = str(file_path)
            payload["symbol"] = target
            if include_source:
                payload["source"] = self._node_source(
                    node, lines, max_lines=max(5, int(max_source_lines))
                )
            return json.dumps(payload)

        for node in module.body:
            if isinstance(node, ast.ClassDef):
                summary["classes"].append(self._describe_node(node, lines))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                summary["functions"].append(self._describe_node(node, lines))
        return json.dumps(summary)

    @staticmethod
    def _collect_imports(module: ast.Module) -> list[str]:
        imports: list[str] = []
        for node in module.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                base = "." * int(node.level or 0) + str(node.module or "")
                names = ", ".join(alias.name for alias in node.names)
                imports.append(f"{base}:{names}")
        return imports

    def _find_symbol_node(self, module: ast.Module, symbol: str) -> ast.AST | None:
        for node in module.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name == symbol:
                    return node
                if isinstance(node, ast.ClassDef):
                    for child in node.body:
                        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            dotted = f"{node.name}.{child.name}"
                            if dotted == symbol:
                                return child
        return None

    @staticmethod
    def _node_source(node: ast.AST, lines: Sequence[str], *, max_lines: int) -> str:
        start = max(1, int(getattr(node, "lineno", 1) or 1))
        end = int(getattr(node, "end_lineno", start) or start)
        end = min(end, start + max_lines - 1)
        snippet = lines[start - 1 : end]
        return "\n".join(snippet)

    def _describe_node(self, node: ast.AST, lines: Sequence[str]) -> dict[str, Any]:
        start = int(getattr(node, "lineno", 0) or 0)
        end = int(getattr(node, "end_lineno", start) or start)
        doc = ""
        name = ""
        kind = "node"
        signature = ""
        methods: list[str] = []
        calls: list[str] = []
        if isinstance(node, ast.ClassDef):
            kind = "class"
            name = node.name
            doc = (ast.get_docstring(node) or "").strip()
            methods = [
                m.name
                for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            calls = self._collect_calls(node)
        elif isinstance(node, ast.AsyncFunctionDef):
            kind = "async_function"
            name = node.name
            doc = (ast.get_docstring(node) or "").strip()
            signature = self._signature_for_function(node, async_def=True)
            calls = self._collect_calls(node)
        elif isinstance(node, ast.FunctionDef):
            kind = "function"
            name = node.name
            doc = (ast.get_docstring(node) or "").strip()
            signature = self._signature_for_function(node, async_def=False)
            calls = self._collect_calls(node)
        header = ""
        if 1 <= start <= len(lines):
            header = lines[start - 1].strip()
        payload: dict[str, Any] = {
            "kind": kind,
            "name": name,
            "line_start": start,
            "line_end": end,
            "header": header,
            "docstring": doc,
            "calls": calls,
        }
        if signature:
            payload["signature"] = signature
        if methods:
            payload["methods"] = methods
        return payload

    @staticmethod
    def _signature_for_function(
        node: ast.FunctionDef | ast.AsyncFunctionDef, *, async_def: bool
    ) -> str:
        args: list[str] = []
        for arg in node.args.posonlyargs:
            args.append(arg.arg)
        for arg in node.args.args:
            args.append(arg.arg)
        if node.args.vararg is not None:
            args.append("*" + node.args.vararg.arg)
        for arg in node.args.kwonlyargs:
            args.append(arg.arg)
        if node.args.kwarg is not None:
            args.append("**" + node.args.kwarg.arg)
        prefix = "async def" if async_def else "def"
        return f"{prefix} {node.name}({', '.join(args)})"

    @staticmethod
    def _collect_calls(node: ast.AST) -> list[str]:
        calls: list[str] = []
        for child in ast.walk(node):
            if not isinstance(child, ast.Call):
                continue
            func = child.func
            call_name = ""
            if isinstance(func, ast.Name):
                call_name = func.id
            elif isinstance(func, ast.Attribute):
                parts: list[str] = [func.attr]
                cursor = func.value
                while isinstance(cursor, ast.Attribute):
                    parts.append(cursor.attr)
                    cursor = cursor.value
                if isinstance(cursor, ast.Name):
                    parts.append(cursor.id)
                parts.reverse()
                call_name = ".".join(parts)
            if call_name:
                calls.append(call_name)
        unique = sorted(set(calls))
        return unique[:100]


__all__ = ["CodeSearchTool", "CodeExplainTool"]
