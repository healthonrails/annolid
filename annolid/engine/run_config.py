from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml


def load_run_config(path: str) -> Dict[str, Any]:
    cfg_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid run config YAML (expected mapping): {cfg_path}")
    return payload


def get_cfg_value(payload: Dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if "." in key:
            current: Any = payload
            for token in key.split("."):
                if not isinstance(current, dict) or token not in current:
                    current = None
                    break
                current = current[token]
            if current is not None:
                return current
        elif key in payload:
            return payload[key]
    return None


def find_run_config_path(argv: Sequence[str]) -> Optional[str]:
    for i, token in enumerate(argv):
        value = str(token)
        if value == "--run-config" and i + 1 < len(argv):
            return str(argv[i + 1])
        if value.startswith("--run-config="):
            return value.split("=", 1)[1].strip()
    return None


def select_model_mode_config(
    payload: Dict[str, Any],
    *,
    model_name: str,
    mode: str,
) -> Dict[str, Any]:
    candidates: List[Any] = [
        get_cfg_value(payload, f"models.{model_name}.{mode}"),
        get_cfg_value(payload, f"models.{model_name}.train"),
        get_cfg_value(payload, f"{model_name}.{mode}"),
        get_cfg_value(payload, f"{mode}"),
        payload,
    ]
    for candidate in candidates:
        if isinstance(candidate, dict):
            return dict(candidate)
    return {}


def _iter_cli_options(parser: argparse.ArgumentParser) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for action in getattr(parser, "_actions", []):
        for opt in list(getattr(action, "option_strings", []) or []):
            if not str(opt).startswith("--"):
                continue
            out[str(opt)[2:]] = str(opt)
    return out


def _as_cli_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def build_cli_args_from_config(
    parser: argparse.ArgumentParser, config: Dict[str, Any]
) -> Tuple[List[str], List[str]]:
    option_map = _iter_cli_options(parser)
    args: List[str] = []
    unknown: List[str] = []
    for key, value in config.items():
        if value is None:
            continue
        normalized = str(key).strip().replace("_", "-")
        if normalized.startswith("--"):
            normalized = normalized[2:]
        opt = option_map.get(normalized)
        if opt is None:
            unknown.append(str(key))
            continue

        if isinstance(value, bool):
            if bool(value):
                args.append(opt)
                continue
            neg_opt = f"--no-{normalized}"
            if f"no-{normalized}" in option_map:
                args.append(neg_opt)
                continue
            # Skip false booleans if parser has no dedicated --no-* flag.
            continue

        args.extend([opt, _as_cli_value(value)])
    return args, unknown


def expand_argv_with_run_config(
    *,
    parser: argparse.ArgumentParser,
    argv: Sequence[str],
    model_name: str,
    mode: str,
) -> Sequence[str]:
    cfg_path = find_run_config_path(argv)
    if not cfg_path:
        return argv
    payload = load_run_config(cfg_path)
    selected = select_model_mode_config(payload, model_name=model_name, mode=mode)
    cfg_argv, _unknown = build_cli_args_from_config(parser, selected)
    # Keep raw argv last so explicit CLI arguments override config values.
    return [*cfg_argv, *list(argv)]
