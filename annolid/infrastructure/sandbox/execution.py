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
    }
)

_CHILD_RUNNER = r"""
import ast, json, sys
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
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


def _validate_sandbox_ast(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SandboxExecutionError(
                "Imports are not allowed in sandboxed analysis."
            )
        if isinstance(node, ast.Attribute):
            attr = str(node.attr or "")
            if attr.startswith("__") or attr.endswith("__") or attr in _FORBIDDEN_ATTRS:
                raise SandboxExecutionError(
                    f"Disallowed attribute access in sandboxed analysis: {attr!r}"
                )
        if isinstance(node, ast.Name):
            name = str(node.id or "")
            if name.startswith("__") or name.endswith("__"):
                raise SandboxExecutionError(
                    f"Disallowed identifier in sandboxed analysis: {name!r}"
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

    with tempfile.TemporaryDirectory(prefix="annolid_sandbox_") as tmp_dir:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", _CHILD_RUNNER],
                input=encoded_payload,
                capture_output=True,
                text=True,
                cwd=tmp_dir,
                env={"PYTHONHASHSEED": "0"},
                timeout=max(0.5, float(timeout_s)),
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

    try:
        parsed = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise SandboxExecutionError(
            f"Sandboxed analysis returned invalid JSON output: {exc}"
        ) from exc
    if isinstance(parsed, dict):
        return parsed
    return {"result": parsed}


__all__ = ["SandboxExecutionError", "execute_generated_analysis"]
