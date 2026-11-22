from __future__ import annotations

import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Callable, Dict, List, Optional, Sequence, Set


def locate_yolo_cli() -> List[str]:
    """Return the command prefix used to invoke the Ultralytics YOLO CLI.

    Prefers the ``yolo`` console script living alongside ``sys.executable`` so
    venv-based GUI/CLI launches work even when the venv's ``bin``/``Scripts``
    directory is not on ``PATH``. Falls back to ``PATH`` lookup and finally to
    invoking ``ultralytics.cfg.entrypoint`` with ``sys.executable``.
    """
    exe_names = ("yolo.exe", "yolo") if os.name == "nt" else ("yolo",)
    py_dir = Path(sys.executable).resolve().parent if sys.executable else None
    if py_dir is not None:
        for name in exe_names:
            candidate = py_dir / name
            try:
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return [str(candidate)]
            except OSError:
                continue

    exe = shutil.which("yolo")
    if exe:
        return [exe]

    if sys.executable and importlib.util.find_spec("ultralytics") is not None:
        return [
            sys.executable,
            "-c",
            "import sys; from ultralytics.cfg import entrypoint; entrypoint()",
        ]

    raise RuntimeError(
        "Ultralytics YOLO CLI not found. Install the optional YOLO runtime with "
        '`python -m pip install "annolid[yolo]"`.'
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


def _append_kv(
    cmd: List[str],
    key: str,
    value: Any,
    *,
    seen_keys: Optional[Set[str]] = None,
) -> None:
    if value is None:
        return
    rendered = _format_cli_value(value)
    if rendered == "":
        return
    norm_key = str(key).strip()
    if not norm_key:
        return
    if seen_keys is not None:
        if norm_key in seen_keys:
            return
        seen_keys.add(norm_key)
    cmd.append(f"{norm_key}={rendered}")


def _append_overrides(
    cmd: List[str],
    overrides: Optional[Dict[str, Any]],
    *,
    seen_keys: Set[str],
    reserved_keys: Set[str],
) -> None:
    if not overrides:
        return
    for key in sorted(overrides.keys()):
        norm_key = str(key).strip()
        if not norm_key or norm_key in reserved_keys or norm_key in seen_keys:
            continue
        _append_kv(cmd, norm_key, overrides[key], seen_keys=seen_keys)


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
    resume: Optional[bool] = None,
    freeze: Optional[int] = None,
    patience: Optional[int] = None,
    cache: Optional[Any] = None,
    overrides: Optional[Dict[str, Any]] = None,
    yolo_cmd: Optional[Sequence[str]] = None,
) -> List[str]:
    cmd: List[str] = list(yolo_cmd or locate_yolo_cli())
    cmd.append("train")
    seen: Set[str] = set()
    _append_kv(cmd, "model", model, seen_keys=seen)
    _append_kv(cmd, "data", data, seen_keys=seen)
    _append_kv(cmd, "epochs", int(epochs), seen_keys=seen)
    _append_kv(cmd, "imgsz", int(imgsz), seen_keys=seen)
    _append_kv(cmd, "batch", int(batch) if batch is not None else None, seen_keys=seen)

    device_str = str(device).strip() if device is not None else ""
    _append_kv(cmd, "device", device_str if device_str else None, seen_keys=seen)
    _append_kv(cmd, "project", project, seen_keys=seen)
    _append_kv(cmd, "name", name, seen_keys=seen)
    _append_kv(cmd, "exist_ok", exist_ok, seen_keys=seen)
    _append_kv(cmd, "plots", plots, seen_keys=seen)
    _append_kv(cmd, "workers", workers, seen_keys=seen)
    if resume:
        _append_kv(cmd, "resume", True, seen_keys=seen)
    if freeze is not None and int(freeze) > 0:
        _append_kv(cmd, "freeze", int(freeze), seen_keys=seen)
    if patience is not None:
        _append_kv(cmd, "patience", int(patience), seen_keys=seen)
    if cache is not None:
        _append_kv(cmd, "cache", cache, seen_keys=seen)

    # Primary explicit arguments (and model/data) must never be clobbered by overrides.
    _append_overrides(
        cmd,
        overrides,
        seen_keys=seen,
        reserved_keys={"model", "data"},
    )

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
    seen: Set[str] = set()
    _append_kv(cmd, "model", model, seen_keys=seen)
    _append_kv(cmd, "data", data, seen_keys=seen)
    _append_kv(cmd, "imgsz", int(imgsz) if imgsz is not None else None, seen_keys=seen)
    _append_kv(cmd, "batch", int(batch) if batch is not None else None, seen_keys=seen)

    device_str = str(device).strip() if device is not None else ""
    _append_kv(cmd, "device", device_str if device_str else None, seen_keys=seen)
    _append_kv(cmd, "project", project, seen_keys=seen)
    _append_kv(cmd, "name", name, seen_keys=seen)
    _append_kv(cmd, "split", split, seen_keys=seen)
    _append_kv(cmd, "plots", plots, seen_keys=seen)
    _append_kv(cmd, "save_json", save_json, seen_keys=seen)
    _append_kv(cmd, "workers", workers, seen_keys=seen)

    _append_overrides(
        cmd,
        overrides,
        seen_keys=seen,
        reserved_keys={"model", "data"},
    )

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
    # On Apple Silicon, allow unsupported PyTorch MPS ops to fall back to CPU
    # instead of crashing mid-training unless the user explicitly disabled it.
    if sys.platform == "darwin":
        merged_env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

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


def format_yolo_cli_error(
    completed: YOLOCLICompleted,
    *,
    model: Optional[str] = None,
    device: Optional[str] = None,
) -> str:
    """Format an actionable diagnostic message for a failed YOLO CLI invocation."""
    tail_lines = completed.output_tail[-50:]
    tail = "\n".join(tail_lines)
    header = f"YOLO CLI failed (exit {completed.returncode})"
    if model:
        header += f" while training {model!r}"
    header += "."

    hints: List[str] = []
    tail_lower = tail.lower()
    device_lower = str(device or "").strip().lower()
    if device_lower == "mps" or any(
        token in tail_lower
        for token in (
            "not currently implemented for the mps",
            "mps backend out of memory",
            "placeholder storage has not been allocated on mps",
            "mpsgraph",
        )
    ):
        hints.append(
            "Apple MPS backend error detected. Re-run with --device cpu (or select CPU in Device), "
            "or reduce batch size / image size."
        )
    elif "out of memory" in tail_lower or "cuda out of memory" in tail_lower:
        hints.append(
            "GPU out-of-memory detected. Try a smaller batch size (e.g. --batch 4) "
            "or smaller image size (e.g. --imgsz 640)."
        )

    parts = [header, f"Command: {' '.join(completed.command)}"]
    if hints:
        parts.append("Suggestion:\n- " + "\n- ".join(hints))
    parts.append(f"Last output:\n{tail}")
    return "\n\n".join(parts)


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
