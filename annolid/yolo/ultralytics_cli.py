from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
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

    device_str = str(device).strip() if device is not None else ""
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


def build_yolo_val_command(
    *,
    model: str,
    data: str,
    imgsz: Optional[int] = None,
    batch: Optional[int] = None,
    device: Optional[str] = None,
    project: Optional[str] = None,
    name: Optional[str] = None,
    split: Optional[str] = None,
    plots: Optional[bool] = None,
    save_json: Optional[bool] = None,
    workers: Optional[int] = None,
    overrides: Optional[Dict[str, Any]] = None,
    yolo_cmd: Optional[Sequence[str]] = None,
) -> List[str]:
    cmd: List[str] = list(yolo_cmd or locate_yolo_cli())
    cmd.append("val")
    _append_kv(cmd, "model", model)
    _append_kv(cmd, "data", data)
    _append_kv(cmd, "imgsz", int(imgsz) if imgsz is not None else None)
    _append_kv(cmd, "batch", int(batch) if batch is not None else None)

    device_str = str(device).strip() if device is not None else ""
    _append_kv(cmd, "device", device_str if device_str else None)
    _append_kv(cmd, "project", project)
    _append_kv(cmd, "name", name)
    _append_kv(cmd, "split", split)
    _append_kv(cmd, "plots", plots)
    _append_kv(cmd, "save_json", save_json)
    _append_kv(cmd, "workers", workers)

    if overrides:
        for key in sorted(overrides.keys()):
            _append_kv(cmd, key, overrides[key])

    return cmd


def _terminate_process_tree(proc: subprocess.Popen[str]) -> None:
    # A POSIX parent can exit while its children still hold the output pipe.
    if os.name == "nt" and proc.poll() is not None:
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

    def check_cancelled() -> None:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("YOLO training cancelled.")

    if tail_lines < 1:
        raise ValueError("tail_lines must be positive")
    check_cancelled()
    merged_env = os.environ.copy()
    if env:
        merged_env.update({str(k): str(v) for k, v in env.items()})

    # Prevent matplotlib backends from requiring a GUI environment in training.
    merged_env.setdefault("MPLBACKEND", "Agg")
    merged_env.setdefault("PYTHONUNBUFFERED", "1")

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
        errors="replace",
        start_new_session=(os.name != "nt"),
        creationflags=(
            # type: ignore[attr-defined]
            subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        ),
    )

    # Reading a pipe can block indefinitely while a model is loading or stalled.
    # Keep cancellation in this thread and perform blocking reads separately.
    output: Queue[object] = Queue(maxsize=256)
    reader_stop = threading.Event()
    eof = object()

    def enqueue(item: object) -> None:
        while not reader_stop.is_set():
            try:
                output.put(item, timeout=0.1)
                return
            except Full:
                continue

    def read_output() -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                if reader_stop.is_set():
                    break
                enqueue(line)
        except Exception as exc:
            enqueue(exc)
        finally:
            enqueue(eof)

    reader = threading.Thread(
        target=read_output, name="annolid-yolo-output", daemon=True
    )
    tail = deque(maxlen=tail_lines)
    completed = False
    try:
        reader.start()
        reached_eof = False
        while not reached_eof or proc.poll() is None:
            check_cancelled()
            try:
                item = output.get(timeout=0.1)
            except Empty:
                continue
            if item is eof:
                reached_eof = True
            elif isinstance(item, Exception):
                raise item
            else:
                assert isinstance(item, str)
                output_sink(item)
                tail.append(item.rstrip("\n"))
        check_cancelled()
        completed = True
        return YOLOCLICompleted(
            command=cmd_list, returncode=proc.wait(), output_tail=list(tail)
        )
    finally:
        reader_stop.set()
        if not completed:
            _terminate_process_tree(proc)
        if proc.poll() is None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name != "nt":
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    proc.kill()
                proc.wait(timeout=5)
        if reader.ident is not None:
            reader.join(timeout=1)
        # An inherited pipe held by another process must not block cleanup.
        if not reader.is_alive() and proc.stdout is not None:
            proc.stdout.close()


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
