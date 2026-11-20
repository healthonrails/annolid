"""External physiological and hardware time-series streams synchronized with behavior.

Supports BORIS ``plot_data`` external CSV/TSV streams (fiber photometry dF/F,
heart rate, temperature, rotary encoder speed, Bpod analog/TTL traces, USV power):
- 1-based BORIS column specifications (``"1,2"``) as well as named CSV/TSV headers
  (``"time,dff"``) or single-column uniform-rate traces (``sampling_rate_hz``).
- Millisecond ``time_offset_s`` alignment against primary video time.
- Baseline subtraction (``subtract_first_value`` / BORIS ``substract_first_value``).
- O(log N) binary-search playhead lookup and peak-preserving min-max envelope
  decimation for smooth real-time GUI rendering on high-rate recordings.
- Behavior-aligned bout signal summaries and deterministic SVG rendering.
"""

from __future__ import annotations

import ast
from collections import defaultdict
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from pathlib import Path
import re
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

from annolid.behavior.event_utils import parse_recording_seconds
from annolid.behavior.time_budget import BehaviorInterval
from annolid.utils.atomic_io import atomic_write_text

__all__ = [
    "ExternalDataError",
    "UnsafeConverterError",
    "ExternalDataStreamSpec",
    "LoadedExternalStream",
    "ExternalBoutSummaryRow",
    "compile_safe_boris_converter",
    "evaluate_safe_boris_converter",
    "inspect_external_data_columns",
    "load_external_data_stream",
    "summarize_external_stream_by_behavior",
    "render_external_data_svg",
    "write_external_data_svg",
]


class ExternalDataError(Exception):
    """Raised when an external time-series file cannot be parsed or aligned."""


class UnsafeConverterError(ExternalDataError):
    """Raised when a BORIS converter expression contains disallowed AST constructs."""


_ISO_DATE_PREFIX_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T\s]")


def _iso8601_to_seconds(value: object) -> float:
    txt = str(value or "").strip()
    if not txt:
        raise ValueError("Empty ISO-8601 timestamp.")
    if txt.endswith("Z") or txt.endswith("z"):
        txt = txt[:-1] + "+00:00"
    dt = datetime.fromisoformat(txt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return float(dt.timestamp())


def _hhmmss_to_seconds(value: object) -> float:
    parsed = parse_recording_seconds(value)
    if parsed is None:
        raise ValueError(f"Cannot parse HH:MM:SS timestamp: {value!r}")
    return float(parsed)


_BUILTIN_CONVERTERS: Dict[str, Callable[[str], float]] = {
    "hhmmss_2_seconds": _hhmmss_to_seconds,
    "hhmmss_to_seconds": _hhmmss_to_seconds,
    "iso8601_2_seconds": _iso8601_to_seconds,
    "iso8601_to_seconds": _iso8601_to_seconds,
    "epoch_ms_2_seconds": lambda s: float(s) / 1000.0,
    "ms_to_seconds": lambda s: float(s) / 1000.0,
    "epoch_us_2_seconds": lambda s: float(s) / 1_000_000.0,
    "us_to_seconds": lambda s: float(s) / 1_000_000.0,
    "invert": lambda s: -float(s),
    "invert_value": lambda s: -float(s),
}

_SAFE_FUNCTIONS: Dict[str, Callable[..., Any]] = {
    "float": float,
    "int": int,
    "str": str,
    "len": len,
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "hhmmss_2_seconds": _hhmmss_to_seconds,
    "iso8601_2_seconds": _iso8601_to_seconds,
}
_SAFE_STR_METHODS = frozenset({"split", "strip", "lstrip", "rstrip", "replace"})
_MAX_AST_STEPS = 128


def _validate_converter_ast(tree: ast.Module) -> None:
    node_count = 0
    for node in ast.walk(tree):
        node_count += 1
        if node_count > _MAX_AST_STEPS:
            raise UnsafeConverterError(
                f"Converter exceeds maximum AST complexity ({_MAX_AST_STEPS} nodes)."
            )
        if isinstance(
            node,
            (
                ast.Import,
                ast.ImportFrom,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Lambda,
                ast.While,
                ast.For,
                ast.AsyncFor,
                ast.With,
                ast.AsyncWith,
                ast.Try,
                ast.Raise,
                ast.Delete,
                ast.Global,
                ast.Nonlocal,
                ast.Yield,
                ast.YieldFrom,
                ast.Await,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
            ),
        ):
            raise UnsafeConverterError(
                f"Disallowed construct '{type(node).__name__}' in BORIS converter."
            )
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            raise UnsafeConverterError(
                f"Disallowed identifier '{node.id}' in BORIS converter."
            )
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr not in _SAFE_STR_METHODS:
                raise UnsafeConverterError(
                    f"Disallowed attribute access '.{node.attr}' in BORIS converter."
                )
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in _SAFE_FUNCTIONS:
                    raise UnsafeConverterError(
                        f"Disallowed function call '{node.func.id}()' in BORIS converter."
                    )
            elif isinstance(node.func, ast.Attribute):
                if node.func.attr not in _SAFE_STR_METHODS:
                    raise UnsafeConverterError(
                        f"Disallowed method call '.{node.func.attr}()' in BORIS converter."
                    )
            else:
                raise UnsafeConverterError(
                    "Dynamic call targets are disallowed in BORIS converters."
                )


def _eval_safe_expr(node: ast.AST, env: Dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, str, bool)) or node.value is None:
            return node.value
        raise UnsafeConverterError(
            f"Unsupported constant type: {type(node.value).__name__}"
        )
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ExternalDataError(f"Undefined variable '{node.id}' in converter.")
        return env[node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        items = [_eval_safe_expr(elt, env) for elt in node.elts]
        return tuple(items) if isinstance(node, ast.Tuple) else items
    if isinstance(node, ast.UnaryOp):
        val = _eval_safe_expr(node.operand, env)
        if isinstance(node.op, ast.UAdd):
            return +val
        if isinstance(node.op, ast.USub):
            return -val
        if isinstance(node.op, ast.Not):
            return not val
        raise UnsafeConverterError(
            f"Unsupported unary operator: {type(node.op).__name__}"
        )
    if isinstance(node, ast.BinOp):
        left = _eval_safe_expr(node.left, env)
        right = _eval_safe_expr(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            if isinstance(left, str) or isinstance(right, str):
                raise UnsafeConverterError(
                    "String repetition is disallowed in converters."
                )
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            if isinstance(left, str):
                raise UnsafeConverterError(
                    "String formatting is disallowed in converters."
                )
            return left % right
        if isinstance(node.op, ast.Pow):
            if abs(float(right)) > 100.0:
                raise UnsafeConverterError(
                    "Exponent magnitude exceeds safe limit (100)."
                )
            return left**right
        raise UnsafeConverterError(
            f"Unsupported binary operator: {type(node.op).__name__}"
        )
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            res: Any = True
            for v in node.values:
                res = _eval_safe_expr(v, env)
                if not res:
                    return res
            return res
        if isinstance(node.op, ast.Or):
            res = False
            for v in node.values:
                res = _eval_safe_expr(v, env)
                if res:
                    return res
            return res
        raise UnsafeConverterError(
            f"Unsupported boolean operator: {type(node.op).__name__}"
        )
    if isinstance(node, ast.Compare):
        cur = _eval_safe_expr(node.left, env)
        for op, comp in zip(node.ops, node.comparators):
            nxt = _eval_safe_expr(comp, env)
            if isinstance(op, ast.Eq):
                ok = cur == nxt
            elif isinstance(op, ast.NotEq):
                ok = cur != nxt
            elif isinstance(op, ast.Lt):
                ok = cur < nxt
            elif isinstance(op, ast.LtE):
                ok = cur <= nxt
            elif isinstance(op, ast.Gt):
                ok = cur > nxt
            elif isinstance(op, ast.GtE):
                ok = cur >= nxt
            else:
                raise UnsafeConverterError(
                    f"Unsupported comparison operator: {type(op).__name__}"
                )
            if not ok:
                return False
            cur = nxt
        return True
    if isinstance(node, ast.IfExp):
        cond = _eval_safe_expr(node.test, env)
        return (
            _eval_safe_expr(node.body, env)
            if cond
            else _eval_safe_expr(node.orelse, env)
        )
    if isinstance(node, ast.Subscript):
        container = _eval_safe_expr(node.value, env)
        if not isinstance(container, (str, list, tuple)):
            raise UnsafeConverterError(
                "Subscript is allowed only on strings, lists, or tuples."
            )
        sl = node.slice
        if isinstance(sl, ast.Slice):
            lo = _eval_safe_expr(sl.lower, env) if sl.lower is not None else None
            hi = _eval_safe_expr(sl.upper, env) if sl.upper is not None else None
            st = _eval_safe_expr(sl.step, env) if sl.step is not None else None
            return container[slice(lo, hi, st)]
        idx = _eval_safe_expr(sl, env)
        return container[int(idx)]
    if isinstance(node, ast.Call):
        args = [_eval_safe_expr(a, env) for a in node.args]
        if node.keywords:
            raise UnsafeConverterError(
                "Keyword arguments are disallowed in BORIS converters."
            )
        if isinstance(node.func, ast.Name):
            fn = _SAFE_FUNCTIONS[node.func.id]
            return fn(*args)
        if isinstance(node.func, ast.Attribute):
            recv = _eval_safe_expr(node.func.value, env)
            if not isinstance(recv, str) or node.func.attr not in _SAFE_STR_METHODS:
                raise UnsafeConverterError(
                    f"Method '.{node.func.attr}()' is only allowed on str values."
                )
            return getattr(recv, node.func.attr)(*args)
    raise UnsafeConverterError(f"Unsupported AST expression '{type(node).__name__}'.")


def compile_safe_boris_converter(
    spec_or_code: object,
    *,
    registry: Optional[Mapping[str, Any]] = None,
) -> Callable[[str], float]:
    """Compile a BORIS converter name, dict, or ``INPUT``/``OUTPUT`` snippet safely without ``eval``."""
    if isinstance(spec_or_code, Mapping):
        code_raw = str(spec_or_code.get("code") or "").strip()
        name_raw = str(spec_or_code.get("name") or "").strip()
        if not code_raw and name_raw:
            return compile_safe_boris_converter(name_raw, registry=registry)
        target_str = code_raw
    else:
        target_str = str(spec_or_code or "").strip()

    if not target_str:
        return lambda s: float(s)

    lower_key = target_str.lower()
    if lower_key in _BUILTIN_CONVERTERS:
        return _BUILTIN_CONVERTERS[lower_key]

    if registry:
        reg_lower = {str(k).strip().lower(): v for k, v in registry.items()}
        if lower_key in reg_lower:
            entry = reg_lower[lower_key]
            if isinstance(entry, Mapping) and entry.get("code"):
                target_str = str(entry["code"]).strip()
            elif isinstance(entry, str) and entry.strip().lower() != lower_key:
                target_str = entry.strip()
                if target_str.lower() in _BUILTIN_CONVERTERS:
                    return _BUILTIN_CONVERTERS[target_str.lower()]

    try:
        tree = ast.parse(target_str, mode="exec")
    except SyntaxError as exc:
        raise UnsafeConverterError(
            f"Syntax error in BORIS converter '{target_str}': {exc}"
        ) from exc

    _validate_converter_ast(tree)
    statements = list(tree.body)
    if not statements:
        return lambda s: float(s)

    for stmt in statements:
        if not isinstance(stmt, (ast.Assign, ast.Expr)):
            raise UnsafeConverterError(
                f"Disallowed statement '{type(stmt).__name__}' in BORIS converter."
            )
        if isinstance(stmt, ast.Assign):
            if len(stmt.targets) != 1:
                raise UnsafeConverterError(
                    "Chained assignments are disallowed in BORIS converters."
                )
            tgt = stmt.targets[0]
            if isinstance(tgt, ast.Name):
                continue
            if isinstance(tgt, (ast.Tuple, ast.List)) and all(
                isinstance(e, ast.Name) for e in tgt.elts
            ):
                continue
            raise UnsafeConverterError(
                "Converter assignments may only bind variable names."
            )

    def _runner(raw_input: str) -> float:
        text_in = str(raw_input).strip()
        num_in: Any = text_in
        try:
            num_in = float(text_in)
        except ValueError:
            num_in = text_in
        # Provide INPUT as raw string (matching BORIS contract where INPUT is str)
        # while x / val is pre-parsed float when numeric for convenience.
        env: Dict[str, Any] = {
            "INPUT": text_in,
            "x": num_in,
            "math_pi": math.pi,
            "math_e": math.e,
        }
        last_val: Any = None
        for stmt in statements:
            if isinstance(stmt, ast.Expr):
                last_val = _eval_safe_expr(stmt.value, env)
            elif isinstance(stmt, ast.Assign):
                val = _eval_safe_expr(stmt.value, env)
                tgt = stmt.targets[0]
                if isinstance(tgt, ast.Name):
                    env[tgt.id] = val
                elif isinstance(tgt, (ast.Tuple, ast.List)):
                    seq = list(val)
                    if len(seq) != len(tgt.elts):
                        raise ExternalDataError(
                            f"Tuple unpacking mismatch in converter: expected {len(tgt.elts)}, got {len(seq)}."
                        )
                    for name_node, item_val in zip(tgt.elts, seq):
                        assert isinstance(name_node, ast.Name)
                        env[name_node.id] = item_val
                last_val = val
        result = env["OUTPUT"] if "OUTPUT" in env else last_val
        if result is None:
            raise ExternalDataError("Converter produced None.")
        out_f = float(result)
        if not math.isfinite(out_f):
            raise ExternalDataError("Converter produced non-finite float.")
        return out_f

    return _runner


def evaluate_safe_boris_converter(
    spec_or_code: object,
    raw_input: object,
    *,
    registry: Optional[Mapping[str, Any]] = None,
) -> float:
    """Evaluate a BORIS converter safely on a single ``raw_input`` value."""
    fn = compile_safe_boris_converter(spec_or_code, registry=registry)
    return fn(str(raw_input))


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ExternalDataStreamSpec:
    """Configuration for a synchronized external CSV/TSV data trace (BORIS ``plot_data``)."""

    file_path: str
    columns: str = "1,2"
    title: str = ""
    variable_name: str = ""
    color: str = "#2563eb"
    time_offset_s: float = 0.0
    time_interval_s: float = 60.0
    subtract_first_value: bool = False
    sampling_rate_hz: Optional[float] = None
    converters: Dict[str, Any] = field(default_factory=dict)
    converter_definitions: Dict[str, Any] = field(default_factory=dict)

    @property
    def display_label(self) -> str:
        if self.variable_name.strip():
            return self.variable_name.strip()
        if self.title.strip():
            return self.title.strip()
        return Path(self.file_path).stem or "Stream"

    def with_offset(self, time_offset_s: float) -> "ExternalDataStreamSpec":
        return ExternalDataStreamSpec(
            file_path=self.file_path,
            columns=self.columns,
            title=self.title,
            variable_name=self.variable_name,
            color=self.color,
            time_offset_s=float(time_offset_s),
            time_interval_s=self.time_interval_s,
            subtract_first_value=self.subtract_first_value,
            sampling_rate_hz=self.sampling_rate_hz,
            converters=dict(self.converters),
            converter_definitions=dict(self.converter_definitions),
        )

    def to_boris_plot_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "file_path": self.file_path,
            "columns": self.columns or "1,2",
            "title": self.title or self.display_label,
            "variable_name": self.variable_name or self.display_label,
            "converters": dict(self.converters),
            "time_interval": int(round(max(1.0, float(self.time_interval_s)))),
            "time_offset": float(self.time_offset_s),
            "substract_first_value": "True" if self.subtract_first_value else "False",
            "color": self.color or "#2563eb",
        }
        if self.sampling_rate_hz is not None and self.sampling_rate_hz > 0:
            payload["sampling_rate_hz"] = float(self.sampling_rate_hz)
        return payload

    @classmethod
    def from_boris_plot_dict(
        cls,
        raw: Mapping[str, Any],
        *,
        project_converters: Optional[Mapping[str, Any]] = None,
    ) -> "ExternalDataStreamSpec":
        file_path = str(raw.get("file_path") or raw.get("path") or "").strip()
        columns = str(raw.get("columns") or "1,2").strip() or "1,2"
        title = str(raw.get("title") or "").strip()
        var_name = str(raw.get("variable_name") or raw.get("label") or "").strip()
        color = str(raw.get("color") or "#2563eb").strip() or "#2563eb"
        # Convert Qt GlobalColor names occasionally stored by older BORIS versions
        qt_color_map = {
            "b": "#2563eb",
            "g": "#16a34a",
            "r": "#dc2626",
            "c": "#0891b2",
            "m": "#9333ea",
            "y": "#ca8a04",
            "k": "#0f172a",
        }
        color = qt_color_map.get(color.lower(), color)
        try:
            offset_s = float(raw.get("time_offset", raw.get("offset", 0.0)) or 0.0)
        except (TypeError, ValueError):
            offset_s = 0.0
        try:
            interval_s = float(
                raw.get("time_interval", raw.get("window", 60.0)) or 60.0
            )
        except (TypeError, ValueError):
            interval_s = 60.0
        # Support both BORIS's historical spelling ("substract_first_value") and standard spelling
        sub_first = _parse_bool(
            raw.get("substract_first_value", raw.get("subtract_first_value", False))
        )
        sr_raw = raw.get("sampling_rate_hz")
        sr_val: Optional[float] = None
        if sr_raw is not None:
            try:
                sr_f = float(sr_raw)
                if sr_f > 0 and math.isfinite(sr_f):
                    sr_val = sr_f
            except (TypeError, ValueError):
                sr_val = None
        conv = raw.get("converters")
        conv_dict = dict(conv) if isinstance(conv, Mapping) else {}
        defs: Dict[str, Any] = {}
        if project_converters and conv_dict:
            proj_lower = {
                str(k).strip().lower(): (str(k), v)
                for k, v in project_converters.items()
            }
            for ref in conv_dict.values():
                if isinstance(ref, str) and ref.strip().lower() in proj_lower:
                    orig_k, orig_v = proj_lower[ref.strip().lower()]
                    defs[orig_k] = orig_v
        return cls(
            file_path=file_path,
            columns=columns,
            title=title,
            variable_name=var_name,
            color=color,
            time_offset_s=offset_s,
            time_interval_s=max(0.5, interval_s),
            subtract_first_value=sub_first,
            sampling_rate_hz=sr_val,
            converters=conv_dict,
            converter_definitions=defs,
        )


@dataclass
class LoadedExternalStream:
    """In-memory sorted NumPy representation of an external time-series stream."""

    spec: ExternalDataStreamSpec
    resolved_path: Path
    times_raw: np.ndarray
    values: np.ndarray

    @property
    def label(self) -> str:
        return self.spec.display_label

    @property
    def sample_count(self) -> int:
        return int(self.times_raw.shape[0])

    @property
    def time_bounds(self) -> Tuple[float, float]:
        if self.times_raw.size == 0:
            return (0.0, 0.0)
        off = float(self.spec.time_offset_s)
        return (float(self.times_raw[0]) + off, float(self.times_raw[-1]) + off)

    def with_offset(self, time_offset_s: float) -> "LoadedExternalStream":
        return LoadedExternalStream(
            spec=self.spec.with_offset(time_offset_s),
            resolved_path=self.resolved_path,
            times_raw=self.times_raw,
            values=self.values,
        )

    def value_at_time(self, video_time_s: float) -> Optional[float]:
        """Interpolate stream value at ``video_time_s`` in O(log N) time."""
        n = int(self.times_raw.size)
        if n == 0:
            return None
        t_query = float(video_time_s) - float(self.spec.time_offset_s)
        if n == 1:
            return (
                float(self.values[0])
                if abs(t_query - float(self.times_raw[0])) <= 1.0
                else None
            )

        t_min = float(self.times_raw[0])
        t_max = float(self.times_raw[-1])
        dt_est = max(1e-3, (t_max - t_min) / max(1, n - 1))
        if t_query < t_min - dt_est or t_query > t_max + dt_est:
            return None
        if t_query <= t_min:
            return float(self.values[0])
        if t_query >= t_max:
            return float(self.values[-1])

        idx = int(np.searchsorted(self.times_raw, t_query, side="left"))
        t0 = float(self.times_raw[idx - 1])
        t1 = float(self.times_raw[idx])
        v0 = float(self.values[idx - 1])
        v1 = float(self.values[idx])
        span = t1 - t0
        if span <= 1e-12:
            return v1
        alpha = (t_query - t0) / span
        return v0 + alpha * (v1 - v0)

    def window_slice(
        self,
        center_time_s: float,
        window_width_s: Optional[float] = None,
        *,
        max_points: int = 1200,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(video_times, values)`` within ``[center - W/2, center + W/2]``.

        Uses O(log N) binary search and peak-preserving min-max envelope decimation
        when the window exceeds ``max_points``.
        """
        if self.times_raw.size == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

        off = float(self.spec.time_offset_s)
        if window_width_s is None:
            window_width_s = float(self.spec.time_interval_s)
        if window_width_s <= 0:
            raw_t = self.times_raw
            raw_v = self.values
        else:
            half_w = 0.5 * float(window_width_s)
            q_start = float(center_time_s) - half_w - off
            q_end = float(center_time_s) + half_w - off
            i0 = max(0, int(np.searchsorted(self.times_raw, q_start, side="left")) - 1)
            i1 = min(
                int(self.times_raw.size),
                int(np.searchsorted(self.times_raw, q_end, side="right")) + 1,
            )
            raw_t = self.times_raw[i0:i1]
            raw_v = self.values[i0:i1]

        count = int(raw_t.size)
        if count <= max(4, int(max_points)):
            return raw_t + off, raw_v

        # Min-max envelope decimation: each bucket emits its chronological (min, max) pair
        n_buckets = max(2, int(max_points) // 2)
        edges = np.linspace(0, count, n_buckets + 1, dtype=int)
        out_t: List[float] = []
        out_v: List[float] = []
        for b in range(n_buckets):
            s_i = int(edges[b])
            e_i = int(edges[b + 1])
            if e_i <= s_i:
                continue
            seg_v = raw_v[s_i:e_i]
            min_rel = int(np.argmin(seg_v))
            max_rel = int(np.argmax(seg_v))
            if min_rel == max_rel:
                out_t.append(float(raw_t[s_i + min_rel]) + off)
                out_v.append(float(seg_v[min_rel]))
            elif min_rel < max_rel:
                out_t.append(float(raw_t[s_i + min_rel]) + off)
                out_v.append(float(seg_v[min_rel]))
                out_t.append(float(raw_t[s_i + max_rel]) + off)
                out_v.append(float(seg_v[max_rel]))
            else:
                out_t.append(float(raw_t[s_i + max_rel]) + off)
                out_v.append(float(seg_v[max_rel]))
                out_t.append(float(raw_t[s_i + min_rel]) + off)
                out_v.append(float(seg_v[min_rel]))
        return np.asarray(out_t, dtype=np.float64), np.asarray(out_v, dtype=np.float64)


@dataclass(frozen=True)
class ExternalBoutSummaryRow:
    """Summary of an external physiological/hardware trace within behavioral bouts."""

    stream_label: str
    subject: str
    behavior: str
    bouts: int
    samples: int
    mean_value: float
    std_value: float
    min_value: float
    max_value: float


def _detect_delimiter(sample_lines: Sequence[str], suffix: str) -> str:
    if suffix == ".tsv":
        return "\t"
    for line in sample_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        for cand in ("\t", ",", ";"):
            if cand in stripped:
                return cand
    return ","


def inspect_external_data_columns(path: Path | str) -> List[str]:
    """Inspect column headers (or synthetic 1-based labels) of an external CSV/TSV file."""
    target = Path(path).expanduser().resolve()
    if not target.is_file():
        raise ExternalDataError(f"External data file not found: {target}")
    lines = [
        ln
        for ln in target.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if not lines:
        return []
    delim = _detect_delimiter(lines[:10], target.suffix.lower())
    first_cells = [c.strip() for c in next(csv.reader([lines[0]], delimiter=delim), [])]
    has_non_numeric = any(
        parse_recording_seconds(c) is None and c != "" for c in first_cells
    )
    if has_non_numeric:
        return [
            f"{idx + 1}: {name or f'col_{idx + 1}'}"
            for idx, name in enumerate(first_cells)
        ]
    return [f"{idx + 1}: col_{idx + 1}" for idx in range(len(first_cells))]


def _resolve_column_index(
    token: str, header_cells: Optional[Sequence[str]], ncols: int
) -> int:
    cleaned = token.strip()
    if not cleaned:
        raise ExternalDataError(
            "Empty column specifier in external data configuration."
        )
    # Support "1: time" UI display format as well as BORIS 1-based integer index "1"
    if ":" in cleaned:
        prefix = cleaned.split(":", 1)[0].strip()
        if prefix.isdigit():
            cleaned = prefix
    if cleaned.isdigit():
        idx_1based = int(cleaned)
        if idx_1based < 1 or idx_1based > max(1, ncols):
            raise ExternalDataError(
                f"1-based column index {idx_1based} is out of range for file with {ncols} column(s)."
            )
        return idx_1based - 1
    if header_cells:
        lower_map = {str(h).strip().lower(): i for i, h in enumerate(header_cells)}
        if cleaned.lower() in lower_map:
            return lower_map[cleaned.lower()]
    raise ExternalDataError(
        f"Column '{token}' not found in external data headers {list(header_cells or ())}."
    )


def _looks_like_data_cell(cell: str) -> bool:
    if parse_recording_seconds(cell) is not None:
        return True
    return bool(_ISO_DATE_PREFIX_RE.match(cell.strip()))


def _resolve_column_converter(
    col_0based: Optional[int],
    role: str,
    header_cells: Optional[Sequence[str]],
    converters_map: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> Optional[Callable[[str], float]]:
    if col_0based is None or not converters_map:
        return None
    candidates = [str(col_0based + 1), role]
    if header_cells and col_0based < len(header_cells) and header_cells[col_0based]:
        candidates.append(header_cells[col_0based].strip())
    lower_conv = {str(k).strip().lower(): v for k, v in converters_map.items()}
    for cand in candidates:
        if cand.lower() in lower_conv and lower_conv[cand.lower()]:
            return compile_safe_boris_converter(
                lower_conv[cand.lower()], registry=registry
            )
    return None


def load_external_data_stream(
    spec_or_path: ExternalDataStreamSpec | Path | str,
    *,
    base_dir: Optional[Path | str] = None,
    project_converters: Optional[Mapping[str, Any]] = None,
    **overrides: Any,
) -> LoadedExternalStream:
    """Load and validate an external CSV/TSV data trace (executing safe BORIS converters)."""
    if isinstance(spec_or_path, ExternalDataStreamSpec):
        base_spec = spec_or_path
    else:
        base_spec = ExternalDataStreamSpec(file_path=str(spec_or_path))

    if overrides:
        spec_dict: Dict[str, Any] = {
            "file_path": base_spec.file_path,
            "columns": base_spec.columns,
            "title": base_spec.title,
            "variable_name": base_spec.variable_name,
            "color": base_spec.color,
            "time_offset_s": base_spec.time_offset_s,
            "time_interval_s": base_spec.time_interval_s,
            "subtract_first_value": base_spec.subtract_first_value,
            "sampling_rate_hz": base_spec.sampling_rate_hz,
            "converters": dict(base_spec.converters),
            "converter_definitions": dict(base_spec.converter_definitions),
        }
        spec_dict.update(overrides)
        spec = ExternalDataStreamSpec(**spec_dict)
    else:
        spec = base_spec

    raw_p = Path(spec.file_path).expanduser()
    candidates = [raw_p]
    if not raw_p.is_absolute() and base_dir is not None:
        candidates.insert(0, Path(base_dir).expanduser().resolve() / raw_p)
        candidates.append(Path(base_dir).expanduser().resolve() / raw_p.name)

    resolved: Optional[Path] = None
    for cand in candidates:
        if cand.is_file():
            resolved = cand.resolve()
            break
    if resolved is None:
        raise ExternalDataError(f"External data file not found: {spec.file_path}")

    text = resolved.read_text(encoding="utf-8-sig", errors="replace")
    lines = [
        ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if not lines:
        raise ExternalDataError(f"External data file is empty: {resolved}")

    delim = _detect_delimiter(lines[:15], resolved.suffix.lower())
    reader_rows = list(csv.reader(lines, delimiter=delim))
    if not reader_rows:
        raise ExternalDataError(f"No rows found in external data file: {resolved}")

    # Skip any leading metadata/header rows until data rows begin
    header_cells: Optional[List[str]] = None
    data_start = 0
    for idx, r in enumerate(reader_rows[:20]):
        if not r:
            continue
        non_empty = [c.strip() for c in r if c is not None and c.strip() != ""]
        if not non_empty:
            continue
        parsed_probe = [_looks_like_data_cell(c) for c in non_empty[:2]]
        if all(parsed_probe):
            data_start = idx
            break
        header_cells = [c.strip() for c in r]
        data_start = idx + 1

    data_rows = reader_rows[data_start:]
    if not data_rows:
        raise ExternalDataError(
            f"External data file contains no numeric data rows: {resolved}"
        )

    max_cols = max(len(r) for r in data_rows)
    col_tokens = [t.strip() for t in str(spec.columns or "1,2").split(",") if t.strip()]
    if len(col_tokens) == 1:
        if spec.sampling_rate_hz is None or spec.sampling_rate_hz <= 0:
            if max_cols >= 2:
                time_col_idx: Optional[int] = 0
                val_col_idx = _resolve_column_index(
                    col_tokens[0], header_cells, max_cols
                )
            else:
                raise ExternalDataError(
                    "Single-column external data requires positive sampling_rate_hz."
                )
        else:
            time_col_idx = None
            val_col_idx = _resolve_column_index(col_tokens[0], header_cells, max_cols)
    else:
        time_col_idx = _resolve_column_index(col_tokens[0], header_cells, max_cols)
        val_col_idx = _resolve_column_index(col_tokens[1], header_cells, max_cols)

    merged_registry: Dict[str, Any] = dict(spec.converter_definitions)
    if project_converters:
        merged_registry.update(project_converters)
    time_converter = _resolve_column_converter(
        time_col_idx, "time", header_cells, spec.converters, merged_registry
    )
    val_converter = _resolve_column_converter(
        val_col_idx, "value", header_cells, spec.converters, merged_registry
    )

    times_list: List[float] = []
    vals_list: List[float] = []
    sr = float(spec.sampling_rate_hz or 1.0)

    for sample_idx, row in enumerate(data_rows):
        if val_col_idx >= len(row):
            continue
        raw_val = row[val_col_idx].strip()
        if not raw_val:
            continue
        try:
            v_num = (
                float(val_converter(raw_val))
                if val_converter is not None
                else float(raw_val)
            )
        except (ValueError, ZeroDivisionError):
            continue
        if not math.isfinite(v_num):
            continue

        if time_col_idx is None:
            t_num = float(sample_idx) / sr
        else:
            if time_col_idx >= len(row):
                continue
            raw_t_cell = row[time_col_idx].strip()
            if not raw_t_cell:
                continue
            if time_converter is not None:
                try:
                    t_parsed: Optional[float] = float(time_converter(raw_t_cell))
                except (ValueError, ZeroDivisionError):
                    continue
            else:
                t_parsed = parse_recording_seconds(raw_t_cell)
                if t_parsed is None and _ISO_DATE_PREFIX_RE.match(raw_t_cell):
                    try:
                        t_parsed = _iso8601_to_seconds(raw_t_cell)
                    except ValueError:
                        t_parsed = None
            if t_parsed is None or not math.isfinite(t_parsed):
                continue
            t_num = float(t_parsed)

        times_list.append(t_num)
        vals_list.append(v_num)

    if not times_list:
        raise ExternalDataError(
            f"No valid numeric (time, value) samples parsed from {resolved}"
        )

    t_arr = np.asarray(times_list, dtype=np.float64)
    v_arr = np.asarray(vals_list, dtype=np.float64)
    if t_arr.size > 1 and np.any(np.diff(t_arr) < 0):
        order = np.argsort(t_arr, kind="mergesort")
        t_arr = t_arr[order]
        v_arr = v_arr[order]

    if spec.subtract_first_value and v_arr.size > 0:
        v_arr = v_arr - float(v_arr[0])

    # Auto-populate variable_name from header when left empty
    if (
        not spec.variable_name.strip()
        and not spec.title.strip()
        and header_cells
        and val_col_idx < len(header_cells)
        and header_cells[val_col_idx]
    ):
        spec = ExternalDataStreamSpec(
            file_path=spec.file_path,
            columns=spec.columns,
            title=spec.title,
            variable_name=header_cells[val_col_idx],
            color=spec.color,
            time_offset_s=spec.time_offset_s,
            time_interval_s=spec.time_interval_s,
            subtract_first_value=spec.subtract_first_value,
            sampling_rate_hz=spec.sampling_rate_hz,
            converters=dict(spec.converters),
            converter_definitions=dict(spec.converter_definitions),
        )

    return LoadedExternalStream(
        spec=spec,
        resolved_path=resolved,
        times_raw=t_arr,
        values=v_arr,
    )


def summarize_external_stream_by_behavior(
    stream: LoadedExternalStream,
    intervals: Iterable[BehaviorInterval],
) -> List[ExternalBoutSummaryRow]:
    """Summarize external signal statistics across behavioral intervals/points."""
    eff_t = stream.times_raw + float(stream.spec.time_offset_s)
    grouped: Dict[Tuple[str, str], List[BehaviorInterval]] = defaultdict(list)
    for iv in intervals:
        grouped[(iv.subject, iv.behavior)].append(iv)

    rows: List[ExternalBoutSummaryRow] = []
    for (subject, behavior), bouts in sorted(grouped.items()):
        collected: List[np.ndarray] = []
        for iv in bouts:
            if abs(iv.end - iv.start) <= 1e-9:
                pt_val = stream.value_at_time(iv.start)
                if pt_val is not None:
                    collected.append(np.asarray([pt_val], dtype=np.float64))
            else:
                i0 = int(np.searchsorted(eff_t, iv.start, side="left"))
                i1 = int(np.searchsorted(eff_t, iv.end, side="right"))
                if i1 > i0:
                    collected.append(stream.values[i0:i1])
                else:
                    mid_val = stream.value_at_time(0.5 * (iv.start + iv.end))
                    if mid_val is not None:
                        collected.append(np.asarray([mid_val], dtype=np.float64))
        if not collected:
            continue
        cat_vals = np.concatenate(collected)
        std_val = float(np.std(cat_vals, ddof=1)) if cat_vals.size >= 2 else 0.0
        rows.append(
            ExternalBoutSummaryRow(
                stream_label=stream.label,
                subject=subject,
                behavior=behavior,
                bouts=len(bouts),
                samples=int(cat_vals.size),
                mean_value=float(np.mean(cat_vals)),
                std_value=std_val,
                min_value=float(np.min(cat_vals)),
                max_value=float(np.max(cat_vals)),
            )
        )
    return rows


def _xml_escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_external_data_svg(
    streams: Sequence[LoadedExternalStream],
    *,
    intervals: Iterable[BehaviorInterval] = (),
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    playhead_time: Optional[float] = None,
    title: str = "Synchronized External Data Streams",
    width: int = 920,
    height: int = 280,
) -> str:
    """Render a deterministic, standalone SVG plot of external data traces and behavior bouts."""
    if not streams:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} 120" width="{width}" height="120">'
            f'<rect width="{width}" height="120" fill="#ffffff"/>'
            f'<text x="{width / 2:.1f}" y="65" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#64748b">No external data streams loaded.</text>'
            "</svg>"
        )

    bounds = [s.time_bounds for s in streams if s.sample_count > 0]
    t0 = (
        float(start_time)
        if start_time is not None
        else min((b[0] for b in bounds), default=0.0)
    )
    t1 = (
        float(end_time)
        if end_time is not None
        else max((b[1] for b in bounds), default=t0 + 1.0)
    )
    if t1 <= t0 + 1e-6:
        t1 = t0 + 1.0
    span = t1 - t0
    center = 0.5 * (t0 + t1)

    left_pad = 72.0
    right_pad = 24.0
    top_pad = 42.0
    bottom_pad = 40.0
    plot_w = max(120.0, float(width) - left_pad - right_pad)
    plot_h = max(80.0, float(height) - top_pad - bottom_pad)

    sliced: List[Tuple[LoadedExternalStream, np.ndarray, np.ndarray]] = []
    y_min = float("inf")
    y_max = float("-inf")
    for st in streams:
        ts, vs = st.window_slice(center, span, max_points=1200)
        mask = (ts >= t0) & (ts <= t1)
        ts = ts[mask]
        vs = vs[mask]
        if vs.size > 0:
            y_min = min(y_min, float(np.min(vs)))
            y_max = max(y_max, float(np.max(vs)))
        sliced.append((st, ts, vs))

    if not math.isfinite(y_min) or not math.isfinite(y_max):
        y_min, y_max = -1.0, 1.0
    elif abs(y_max - y_min) <= 1e-9:
        y_min -= 1.0
        y_max += 1.0
    else:
        pad = 0.06 * (y_max - y_min)
        y_min -= pad
        y_max += pad

    def _x_of(t_val: float) -> float:
        return left_pad + ((t_val - t0) / span) * plot_w

    def _y_of(v_val: float) -> float:
        return top_pad + (1.0 - (v_val - y_min) / (y_max - y_min)) * plot_h

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left_pad}" y="24" font-family="Helvetica, Arial, sans-serif" font-size="14" font-weight="bold" fill="#0f172a">{_xml_escape(title)}</text>',
        f'<rect x="{left_pad}" y="{top_pad}" width="{plot_w}" height="{plot_h}" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>',
    ]

    # Shaded behavioral intervals behind traces
    for iv in intervals:
        if iv.end <= t0 or iv.start >= t1 or iv.end <= iv.start:
            continue
        ix0 = _x_of(max(t0, iv.start))
        ix1 = _x_of(min(t1, iv.end))
        svg.append(
            f'<rect x="{ix0:.1f}" y="{top_pad:.1f}" width="{max(1.5, ix1 - ix0):.1f}" height="{plot_h:.1f}" '
            f'fill="#fde68a" fill-opacity="0.32">'
            f"<title>{_xml_escape(iv.subject)} - {_xml_escape(iv.behavior)}</title></rect>"
        )

    # Trace polylines + legend
    legend_x = left_pad + plot_w
    for idx, (st, ts, vs) in enumerate(sliced):
        col = st.spec.color or "#2563eb"
        if ts.size >= 2:
            pts = " ".join(
                f"{_x_of(float(tx)):.1f},{_y_of(float(vx)):.1f}"
                for tx, vx in zip(ts, vs)
            )
            svg.append(
                f'<polyline fill="none" stroke="{_xml_escape(col)}" stroke-width="1.6" points="{pts}"/>'
            )
        elif ts.size == 1:
            svg.append(
                f'<circle cx="{_x_of(float(ts[0])):.1f}" cy="{_y_of(float(vs[0])):.1f}" r="3" fill="{_xml_escape(col)}"/>'
            )
        lx = max(left_pad + 180.0, legend_x - idx * 130.0)
        svg.append(
            f'<text x="{lx:.1f}" y="24" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="11" font-weight="600" fill="{_xml_escape(col)}">● {_xml_escape(st.label)}</text>'
        )

    if playhead_time is not None and t0 <= float(playhead_time) <= t1:
        px = _x_of(float(playhead_time))
        svg.append(
            f'<line x1="{px:.1f}" y1="{top_pad:.1f}" x2="{px:.1f}" y2="{top_pad + plot_h:.1f}" stroke="#dc2626" stroke-width="1.5" stroke-dasharray="4,3"/>'
        )

    # Axes labels
    for frac in (0.0, 0.5, 1.0):
        vy = top_pad + (1.0 - frac) * plot_h
        v_lbl = y_min + frac * (y_max - y_min)
        svg.append(
            f'<text x="{left_pad - 6:.1f}" y="{vy + 3.5:.1f}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="10" fill="#475569">{v_lbl:.2f}</text>'
        )
    for ti in range(6):
        frac = ti / 5.0
        tx = left_pad + frac * plot_w
        t_lbl = t0 + frac * span
        svg.append(
            f'<text x="{tx:.1f}" y="{top_pad + plot_h + 16:.1f}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="10" fill="#475569">{t_lbl:.1f}s</text>'
        )
    svg.append("</svg>")
    return "\n".join(svg)


def write_external_data_svg(
    streams: Sequence[LoadedExternalStream],
    output_path: Path | str,
    **kwargs: Any,
) -> Path:
    dst = Path(output_path)
    atomic_write_text(
        dst, render_external_data_svg(streams, **kwargs), encoding="utf-8"
    )
    return dst
