"""Deterministic sandbox execution helper for generated analysis code."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from typing import Any


class SandboxExecutionError(RuntimeError):
    pass


_FORBIDDEN_ATTRS = frozenset(
    {
        "gi_frame",
        "gi_code",
        "gi_yieldfrom",
        "cr_frame",
        "cr_code",
        "cr_await",
        "ag_frame",
        "ag_code",
        "f_globals",
        "f_locals",
        "f_builtins",
        "f_back",
        "f_code",
        "tb_frame",
        "tb_next",
        "func_globals",
        "func_code",
        "func_closure",
        "format",
        "format_map",
    }
)
_MAX_SANDBOX_STDOUT_BYTES = 4 * 1024 * 1024

_CHILD_RUNNER = r"""
import ast, json, sys
try:
    import resource
    cpu_limit = max(1, int(float(sys.argv[1]))) if len(sys.argv) > 1 else 5
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit, cpu_limit))
except Exception:
    pass

payload = json.loads(sys.stdin.read())
code = payload["code"]
inputs = payload["inputs"]
tree = ast.parse(code, mode="exec")
safe_builtins = {
    "len": len,
    "min": min,
    "max": max,
    "sum": sum,
    "sorted": sorted,
    "range": range,
    "float": float,
    "int": int,
    "str": str,
    "bool": bool,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "abs": abs,
    "round": round,
    "enumerate": enumerate,
    "zip": zip,
    "any": any,
    "all": all,
}
ns = {"__builtins__": safe_builtins}
exec(compile(tree, "<generated_analysis>", "exec"), ns, ns)
run_callable = ns.get("run")
if not callable(run_callable):
    raise RuntimeError("Generated analysis must define run(inputs).")
result = run_callable(dict(inputs))
if not isinstance(result, dict):
    result = {"result": result}
sys.stdout.write(json.dumps(result))
"""


def _is_forbidden_identifier(name: str) -> bool:
    return name.startswith("__") or name.endswith("__")


def _validate_sandbox_ast(tree: ast.AST) -> None:
    match_class_cls = getattr(ast, "MatchClass", None)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SandboxExecutionError(
                "Imports are not allowed in sandboxed analysis."
            )
        if isinstance(node, ast.Attribute):
            attr = str(node.attr or "")
            if _is_forbidden_identifier(attr) or attr in _FORBIDDEN_ATTRS:
                raise SandboxExecutionError(
                    f"Disallowed attribute access in sandboxed analysis: {attr!r}"
                )
        if isinstance(node, ast.Name):
            name = str(node.id or "")
            if _is_forbidden_identifier(name):
                raise SandboxExecutionError(
                    f"Disallowed identifier in sandboxed analysis: {name!r}"
                )
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            def_name = str(node.name or "")
            if _is_forbidden_identifier(def_name):
                raise SandboxExecutionError(
                    f"Disallowed definition name in sandboxed analysis: {def_name!r}"
                )
        if isinstance(node, ast.arg):
            arg_name = str(node.arg or "")
            if _is_forbidden_identifier(arg_name):
                raise SandboxExecutionError(
                    f"Disallowed parameter name in sandboxed analysis: {arg_name!r}"
                )
        if isinstance(node, ast.keyword) and node.arg is not None:
            kw_name = str(node.arg or "")
            if _is_forbidden_identifier(kw_name) or kw_name in _FORBIDDEN_ATTRS:
                raise SandboxExecutionError(
                    f"Disallowed keyword name in sandboxed analysis: {kw_name!r}"
                )
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            for bound_name in node.names:
                if _is_forbidden_identifier(str(bound_name or "")):
                    raise SandboxExecutionError(
                        f"Disallowed identifier in sandboxed analysis: {bound_name!r}"
                    )
        if isinstance(node, ast.ExceptHandler) and node.name:
            if _is_forbidden_identifier(str(node.name)):
                raise SandboxExecutionError(
                    f"Disallowed identifier in sandboxed analysis: {node.name!r}"
                )
        if match_class_cls is not None and isinstance(node, match_class_cls):
            for kwd_attr in getattr(node, "kwd_attrs", ()) or ():
                attr_name = str(kwd_attr or "")
                if _is_forbidden_identifier(attr_name) or attr_name in _FORBIDDEN_ATTRS:
                    raise SandboxExecutionError(
                        f"Disallowed match attribute in sandboxed analysis: {attr_name!r}"
                    )


def execute_generated_analysis(
    code: str,
    inputs: dict[str, Any],
    *,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    """Execute generated code under strict AST and subprocess isolation.

    Allowed contract:
    - code may define a function named `run(inputs)` (plus pure helper functions)
    - no imports or dunder/frame introspection are allowed
    - executes in an isolated child process with a strict wall-clock timeout
    """
    source = str(code or "")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        raise SandboxExecutionError(
            f"Syntax error in generated analysis: {exc}"
        ) from exc
    _validate_sandbox_ast(tree)

    try:
        encoded_payload = json.dumps({"code": source, "inputs": dict(inputs)})
    except (TypeError, ValueError) as exc:
        raise SandboxExecutionError(
            f"Sandbox inputs must be JSON-serializable: {exc}"
        ) from exc

    effective_timeout = max(0.5, float(timeout_s))
    with tempfile.TemporaryDirectory(prefix="annolid_sandbox_") as tmp_dir:
        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    _CHILD_RUNNER,
                    str(int(effective_timeout + 1)),
                ],
                input=encoded_payload,
                capture_output=True,
                text=True,
                cwd=tmp_dir,
                env={"PYTHONHASHSEED": "0"},
                timeout=effective_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SandboxExecutionError(
                f"Sandboxed analysis timed out after {float(timeout_s):.1f}s."
            ) from exc

    if proc.returncode != 0:
        err_msg = (proc.stderr or proc.stdout or "").strip()
        if "Generated analysis must define run(inputs)." in err_msg:
            raise SandboxExecutionError("Generated analysis must define run(inputs).")
        raise SandboxExecutionError(
            f"Sandboxed analysis failed (exit {proc.returncode}): {err_msg[-500:]}"
        )

    stdout_text = proc.stdout or "{}"
    if len(stdout_text.encode("utf-8", errors="replace")) > _MAX_SANDBOX_STDOUT_BYTES:
        raise SandboxExecutionError("Sandboxed analysis output exceeded size limit.")
    try:
        parsed = json.loads(stdout_text)
    except json.JSONDecodeError as exc:
        raise SandboxExecutionError(
            f"Sandboxed analysis returned invalid JSON output: {exc}"
        ) from exc
    if isinstance(parsed, dict):
        return parsed
    return {"result": parsed}


__all__ = ["SandboxExecutionError", "execute_generated_analysis"]
