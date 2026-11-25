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
    ]
    if mode == "train":
        candidates.append(get_cfg_value(payload, f"models.{model_name}.train"))
    candidates.extend(
        [
            get_cfg_value(payload, f"{model_name}.{mode}"),
            get_cfg_value(payload, f"{mode}"),
            payload,
        ]
    )
    for candidate in candidates:
        if isinstance(candidate, dict):
            return dict(candidate)
    return {}


def _iter_cli_actions(
    parser: argparse.ArgumentParser,
) -> Dict[str, Tuple[str, argparse.Action]]:
    out: Dict[str, Tuple[str, argparse.Action]] = {}
    for action in getattr(parser, "_actions", []):
        for opt in list(getattr(action, "option_strings", []) or []):
            if not str(opt).startswith("--"):
                continue
            out[str(opt)[2:]] = (str(opt), action)
    return out


def _as_cli_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def build_cli_args_from_config(
    parser: argparse.ArgumentParser, config: Dict[str, Any]
) -> Tuple[List[str], List[str]]:
    option_actions = _iter_cli_actions(parser)
    args: List[str] = []
    unknown: List[str] = []
    for key, value in config.items():
        if value is None:
            continue
        normalized = str(key).strip().replace("_", "-")
        if normalized.startswith("--"):
            normalized = normalized[2:]
        entry = option_actions.get(normalized)
        if entry is None:
            unknown.append(str(key))
            continue
        opt, action = entry

        if isinstance(value, bool):
            if bool(value):
                args.append(opt)
                continue
            neg_entry = option_actions.get(f"no-{normalized}")
            if neg_entry is not None:
                args.append(neg_entry[0])
                continue
            # Skip false booleans if parser has no dedicated --no-* flag.
            continue

        if isinstance(value, (list, tuple)):
            is_append = isinstance(action, argparse._AppendAction)
            nargs = getattr(action, "nargs", None)
            multi_token = nargs in ("+", "*") or (isinstance(nargs, int) and nargs > 1)
            if is_append:
                for item in value:
                    if multi_token and isinstance(item, (list, tuple)):
                        args.append(opt)
                        args.extend(str(v) for v in item)
                    else:
                        args.extend([opt, str(item)])
                continue
            if multi_token:
                args.append(opt)
                args.extend(str(v) for v in value)
                continue

        args.extend([opt, _as_cli_value(value)])
    return args, unknown


def _strip_run_config_args(argv: Sequence[str]) -> List[str]:
    stripped: List[str] = []
    i = 0
    tokens = list(argv)
    while i < len(tokens):
        tok = str(tokens[i])
        if tok == "--run-config":
            i += 2
            continue
        if tok.startswith("--run-config="):
            i += 1
            continue
        stripped.append(tokens[i])
        i += 1
    return stripped


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
    # Strip --run-config if the target parser did not register it.
    option_actions = _iter_cli_actions(parser)
    tail_argv = (
        list(argv) if "run-config" in option_actions else _strip_run_config_args(argv)
    )
    # Keep raw argv last so explicit CLI arguments override config values.
    return [*cfg_argv, *tail_argv]
