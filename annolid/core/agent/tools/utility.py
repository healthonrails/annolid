from __future__ import annotations

import ast
import datetime as dt
import re
from dataclasses import dataclass
from typing import Dict, Optional

from .base import Tool, ToolContext, ToolError
from .registry import ToolRegistry


@dataclass(frozen=True)
class CalculatorResult:
    expression: str
    result: float


@dataclass(frozen=True)
class DateTimeResult:
    iso: str
    unix: int
    utc_offset: str


@dataclass(frozen=True)
class TextStatsResult:
    characters: int
    words: int
    lines: int


class CalculatorTool(Tool[str, CalculatorResult]):
    """Safe arithmetic expression evaluator for agent workflows."""

    name = "calculator"

    def run(self, ctx: ToolContext, payload: str) -> CalculatorResult:
        _ = ctx
        expr = str(payload or "").strip()
        if not expr:
            raise ToolError("CalculatorTool requires a non-empty expression.")
        try:
            value = _safe_eval(expr)
        except Exception as exc:
            raise ToolError(f"Invalid expression: {exc}") from exc
        return CalculatorResult(expression=expr, result=float(value))


class DateTimeTool(Tool[Optional[Dict[str, object]], DateTimeResult]):
    """Return the current datetime in UTC or a requested UTC offset."""

    name = "datetime"

    def run(
        self, ctx: ToolContext, payload: Optional[Dict[str, object]]
    ) -> DateTimeResult:
        _ = ctx
        data = dict(payload or {})
        offset_raw = data.get("utc_offset")
        tz = _parse_utc_offset(str(offset_raw)) if offset_raw else dt.timezone.utc
        now = dt.datetime.now(tz=tz)
        offset = now.strftime("%z")
        offset_fmt = f"{offset[:3]}:{offset[3:]}" if len(offset) == 5 else "+00:00"
        return DateTimeResult(
            iso=now.isoformat(),
            unix=int(now.timestamp()),
            utc_offset=offset_fmt,
        )


class TextStatsTool(Tool[str, TextStatsResult]):
    """Basic text statistics helper."""

    name = "text_stats"

    def run(self, ctx: ToolContext, payload: str) -> TextStatsResult:
        _ = ctx
        text = str(payload or "")
        words = len([w for w in re.split(r"\s+", text.strip()) if w]) if text else 0
        lines = text.count("\n") + (1 if text else 0)
        return TextStatsResult(
            characters=len(text),
            words=int(words),
            lines=int(lines),
        )


def register_builtin_utility_tools(registry: ToolRegistry) -> None:
    """Register lightweight built-in utility tools."""

    registry.register("calculator", lambda cfg: CalculatorTool(config=cfg))
    registry.register("datetime", lambda cfg: DateTimeTool(config=cfg))
    registry.register("text_stats", lambda cfg: TextStatsTool(config=cfg))


def _parse_utc_offset(value: str) -> dt.timezone:
    if not value:
        return dt.timezone.utc
    raw = value.strip()
    match = re.fullmatch(r"([+-])(\d{2}):?(\d{2})", raw)
    if not match:
        raise ToolError(
            "utc_offset must be in format '+HH:MM' or '-HH:MM', e.g. '+08:00'."
        )
    sign, hh, mm = match.groups()
    hours = int(hh)
    minutes = int(mm)
    if hours > 23 or minutes > 59:
        raise ToolError("utc_offset hours/minutes are out of range.")
    delta = dt.timedelta(hours=hours, minutes=minutes)
    if sign == "-":
        delta = -delta
    return dt.timezone(delta)


def _safe_eval(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    return float(_eval_ast(tree.body))


def _eval_ast(node: ast.AST) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("only numeric literals are allowed")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_ast(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)
    ):
        left = _eval_ast(node.left)
        right = _eval_ast(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left**right
    raise ValueError("unsupported expression")
