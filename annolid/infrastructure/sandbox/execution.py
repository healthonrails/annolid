"""Deterministic sandbox execution helper for generated analysis code."""

from __future__ import annotations

import ast
from typing import Any


class SandboxExecutionError(RuntimeError):
    pass


def execute_generated_analysis(code: str, inputs: dict[str, Any]) -> dict[str, Any]:
    """Execute generated code under strict restrictions.

    Allowed contract:
    - code may define a function named `run(inputs)`
    - no imports are allowed
    - only a restricted builtin set is exposed
    """
    tree = ast.parse(str(code or ""), mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SandboxExecutionError(
                "Imports are not allowed in sandboxed analysis."
            )

    safe_globals: dict[str, Any] = {
        "__builtins__": {
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
        }
    }
    safe_locals: dict[str, Any] = {}
    exec(compile(tree, "<generated_analysis>", "exec"), safe_globals, safe_locals)
    run_callable = safe_locals.get("run") or safe_globals.get("run")
    if not callable(run_callable):
        raise SandboxExecutionError("Generated analysis must define run(inputs).")
    result = run_callable(dict(inputs))
    if isinstance(result, dict):
        return result
    return {"result": result}


__all__ = ["SandboxExecutionError", "execute_generated_analysis"]
