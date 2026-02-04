from __future__ import annotations

import os
import shutil
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence


def locate_yolo_cli() -> List[str]:
    """Return the command prefix used to invoke the Ultralytics YOLO CLI."""
    exe = shutil.which("yolo")
    if exe:
        return [exe]
    raise RuntimeError(
        "Ultralytics YOLO CLI not found. Install 'ultralytics' so the 'yolo' command is available."
    )


def _format_cli_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float, str)):
        return str(value)
    # Ultralytics CLI accepts Python-like literals for some args; JSON is widely compatible.
    import json

    return json.dumps(value)


def _append_kv(cmd: List[str], key: str, value: Any) -> None:
    if value is None:
        return
    rendered = _format_cli_value(value)
    if rendered == "":
        return
    cmd.append(f"{key}={rendered}")


def build_yolo_train_command(
    *,
    model: str,
    data: str,
    epochs: int,
    imgsz: int,
    batch: Optional[int] = None,
    device: Optional[str] = None,
    project: Optional[str] = None,
    name: Optional[str] = None,
    exist_ok: Optional[bool] = None,
    plots: Optional[bool] = None,
    workers: Optional[int] = None,
    overrides: Optional[Dict[str, Any]] = None,
    yolo_cmd: Optional[Sequence[str]] = None,
) -> List[str]:
    cmd: List[str] = list(yolo_cmd or locate_yolo_cli())
    cmd.append("train")
    _append_kv(cmd, "model", model)
    _append_kv(cmd, "data", data)
    _append_kv(cmd, "epochs", int(epochs))
    _append_kv(cmd, "imgsz", int(imgsz))
    _append_kv(cmd, "batch", int(batch) if batch is not None else None)

    device_str = str(device or "").strip()
    _append_kv(cmd, "device", device_str if device_str else None)
    _append_kv(cmd, "project", project)
    _append_kv(cmd, "name", name)
    _append_kv(cmd, "exist_ok", exist_ok)
    _append_kv(cmd, "plots", plots)
    _append_kv(cmd, "workers", workers)

    if overrides:
        for key in sorted(overrides.keys()):
            _append_kv(cmd, key, overrides[key])

    return cmd


def _terminate_process_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name == "nt":
            # type: ignore[attr-defined]
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass


@dataclass(frozen=True)
class YOLOCLICompleted:
    command: List[str]
    returncode: int
    output_tail: List[str]


def run_yolo_cli(
    command: Sequence[str],
    *,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[Path] = None,
    stop_event: Optional[object] = None,
    output_sink: Optional[Callable[[str], None]] = None,
    tail_lines: int = 200,
) -> YOLOCLICompleted:
    """Run a YOLO CLI command, streaming stdout/stderr to a sink while retaining a tail for errors."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update({str(k): str(v) for k, v in env.items()})

    # Prevent matplotlib backends from requiring a GUI environment in training.
    merged_env.setdefault("MPLBACKEND", "Agg")

    if output_sink is None:

        def output_sink(_line):
            return None

    cmd_list = [str(part) for part in command]
    proc = subprocess.Popen(
        cmd_list,
        cwd=str(cwd) if cwd is not None else None,
        env=merged_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        start_new_session=(os.name != "nt"),
        creationflags=(
            # type: ignore[attr-defined]
            subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        ),
    )

    tail: List[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if (
                stop_event is not None
                and getattr(stop_event, "is_set", None)
                and stop_event.is_set()
            ):
                _terminate_process_tree(proc)
                raise RuntimeError("YOLO training cancelled.")
            output_sink(line)
            tail.append(line.rstrip("\n"))
            if len(tail) > tail_lines:
                tail = tail[-tail_lines:]
    finally:
        try:
            proc.stdout and proc.stdout.close()
        except Exception:
            pass

    returncode = proc.wait()
    return YOLOCLICompleted(
        command=list(cmd_list), returncode=returncode, output_tail=tail
    )


def ensure_parent_dir(path_str: str) -> str:
    """Best-effort mkdir for parent directories when a CLI argument is an output or weight path."""
    try:
        p = Path(path_str).expanduser()
        parent = p.parent
        if parent and str(parent) not in {".", ""}:
            parent.mkdir(parents=True, exist_ok=True)
        return str(p)
    except Exception:
        return str(path_str)
