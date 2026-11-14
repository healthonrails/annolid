"""Cooperative mid-inference cancellation utilities for ONNX Runtime and PyTorch."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterator
import contextlib
import hashlib
import inspect
import os
from pathlib import Path
import tempfile
import threading
from typing import Any
import urllib.request
import weakref

from annolid.utils.logger import logger


class InferenceCancelledError(RuntimeError):
    """Raised when an in-flight ONNX or PyTorch forward pass is cooperatively cancelled."""


class ObservableStopEvent(threading.Event):
    """``threading.Event`` subclass that invokes registered callbacks immediately on ``set()``."""

    def __init__(self) -> None:
        super().__init__()
        self._listener_lock = threading.Lock()
        self._listeners: list[Callable[[], None]] = []

    def set(self) -> None:
        with self._listener_lock:
            super().set()
            callbacks = list(self._listeners)
        for callback in callbacks:
            try:
                callback()
            except Exception:
                logger.debug("ObservableStopEvent listener failed.", exc_info=True)

    def clear(self) -> None:
        with self._listener_lock:
            super().clear()

    def add_listener(self, callback: Callable[[], None]) -> Callable[[], None]:
        fire_now = False
        with self._listener_lock:
            if self.is_set():
                fire_now = True
            else:
                self._listeners.append(callback)

        if fire_now:
            try:
                callback()
            except Exception:
                logger.debug(
                    "ObservableStopEvent immediate listener failed.", exc_info=True
                )

        def _unsubscribe() -> None:
            with self._listener_lock:
                try:
                    self._listeners.remove(callback)
                except ValueError:
                    pass

        return _unsubscribe


def is_stop_requested(stop_event: threading.Event | None) -> bool:
    """Return ``True`` when ``stop_event`` is non-None and set."""
    return bool(stop_event is not None and stop_event.is_set())


def _session_run_accepts_run_options(run_fn: Any) -> bool:
    try:
        sig = inspect.signature(run_fn)
    except (TypeError, ValueError):
        return True
    params = sig.parameters
    if "run_options" in params:
        return True
    return (
        any(
            p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
            for p in params.values()
        )
        or len(params) >= 3
    )


def onnx_cancellable_run(
    session: Any,
    output_names: Any,
    input_feed: dict[str, Any],
    *,
    stop_event: threading.Event | None = None,
) -> Any:
    """Execute ``session.run`` with C++ ``onnxruntime.RunOptions.terminate`` wired to ``stop_event``."""
    if stop_event is None:
        return session.run(output_names, input_feed)
    if stop_event.is_set():
        raise InferenceCancelledError("ONNX inference cancelled before execution.")

    run_opts = None
    if _session_run_accepts_run_options(session.run):
        try:
            import onnxruntime as ort

            run_opts = ort.RunOptions()
        except Exception:
            run_opts = None

    if run_opts is None:
        outputs = session.run(output_names, input_feed)
        if stop_event.is_set():
            raise InferenceCancelledError("ONNX inference cancelled.")
        return outputs

    def _abort_run() -> None:
        try:
            run_opts.terminate = True
        except Exception:
            pass

    add_listener = getattr(stop_event, "add_listener", None)
    if callable(add_listener):
        unsubscribe = add_listener(_abort_run)
        try:
            if stop_event.is_set():
                raise InferenceCancelledError(
                    "ONNX inference cancelled before execution."
                )
            outputs = session.run(output_names, input_feed, run_opts)
        except InferenceCancelledError:
            raise
        except Exception as exc:
            if stop_event.is_set() or "terminate" in str(exc).lower():
                raise InferenceCancelledError(
                    "ONNX inference cancelled during execution."
                ) from exc
            raise
        finally:
            unsubscribe()
    else:
        done_event = threading.Event()

        def _watch_plain_event() -> None:
            while not done_event.wait(0.015):
                if stop_event.is_set():
                    _abort_run()
                    break

        watcher = threading.Thread(
            target=_watch_plain_event,
            name="annolid-ort-cancel-watch",
            daemon=True,
        )
        watcher.start()
        try:
            if stop_event.is_set():
                _abort_run()
                raise InferenceCancelledError(
                    "ONNX inference cancelled before execution."
                )
            outputs = session.run(output_names, input_feed, run_opts)
        except InferenceCancelledError:
            raise
        except Exception as exc:
            if stop_event.is_set() or "terminate" in str(exc).lower():
                raise InferenceCancelledError(
                    "ONNX inference cancelled during execution."
                ) from exc
            raise
        finally:
            done_event.set()
            watcher.join(timeout=0.1)

    if stop_event.is_set():
        raise InferenceCancelledError("ONNX inference cancelled.")
    return outputs


@contextlib.contextmanager
def torch_cancellable_forward(
    module: Any,
    stop_event: threading.Event | None = None,
    *,
    max_depth: int = 3,
) -> Iterator[None]:
    """Abort a PyTorch ``nn.Module`` forward pass between coarse submodules when ``stop_event`` is set."""
    if stop_event is None:
        yield
        return
    if stop_event.is_set():
        raise InferenceCancelledError(
            "PyTorch inference cancelled before forward pass."
        )

    register_fn = getattr(module, "register_forward_pre_hook", None)
    if module is None or not callable(register_fn):
        yield
        if stop_event.is_set():
            raise InferenceCancelledError("PyTorch inference cancelled.")
        return

    targets: list[Any] = []
    seen_ids: set[int] = set()
    queue: deque[tuple[Any, int]] = deque([(module, 0)])
    while queue:
        current, depth = queue.popleft()
        obj_id = id(current)
        if obj_id in seen_ids:
            continue
        seen_ids.add(obj_id)
        if callable(getattr(current, "register_forward_pre_hook", None)):
            targets.append(current)
        if depth >= max_depth:
            continue
        children_fn = getattr(current, "children", None)
        if callable(children_fn):
            for child in children_fn():
                if child is not None:
                    queue.append((child, depth + 1))

    def _check_cancel_hook(_mod: Any, _inputs: Any) -> None:
        if stop_event.is_set():
            raise InferenceCancelledError(
                "PyTorch inference cancelled during forward pass."
            )

    handles = []
    try:
        for target in targets:
            handles.append(target.register_forward_pre_hook(_check_cancel_hook))
        yield
        if stop_event.is_set():
            raise InferenceCancelledError("PyTorch inference cancelled.")
    finally:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass


def default_model_cache_dir(cache_subdir: str = "models") -> Path:
    """Return the user-writable Annolid model cache directory for ``cache_subdir``."""
    override = os.environ.get("ANNOLID_CACHE_DIR", "").strip()
    root = Path(override) if override else (Path.home() / ".cache" / "annolid")
    return root / cache_subdir


_DOWNLOAD_LOCKS_GUARD = threading.Lock()
_DOWNLOAD_LOCKS: weakref.WeakValueDictionary[str, threading.Lock] = (
    weakref.WeakValueDictionary()
)


def _get_download_lock(dest_path: str | os.PathLike[str]) -> threading.Lock:
    key = os.path.normcase(os.path.abspath(os.fspath(dest_path)))
    with _DOWNLOAD_LOCKS_GUARD:
        lock = _DOWNLOAD_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _DOWNLOAD_LOCKS[key] = lock
        return lock


def _acquire_download_lock(
    lock: threading.Lock,
    stop_event: threading.Event | None,
) -> None:
    while not lock.acquire(timeout=0.05):
        if is_stop_requested(stop_event):
            raise InferenceCancelledError("Model checkpoint download cancelled.")


def _existing_nonempty_checkpoint(
    preferred: Path,
    fallback: Path,
) -> str | None:
    try:
        if preferred.is_file() and preferred.stat().st_size > 0:
            return str(preferred)
    except OSError:
        pass
    try:
        if fallback != preferred and fallback.is_file() and fallback.stat().st_size > 0:
            return str(fallback)
    except OSError:
        pass
    return None


def _open_temp_part_file(target_path: Path):
    target_path.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.NamedTemporaryFile(
        mode="wb",
        dir=str(target_path.parent),
        prefix=f".{target_path.name}.",
        suffix=".part",
        delete=False,
    )


def cancellable_download_file(
    url: str,
    dest_path: str | os.PathLike[str],
    *,
    md5: str | None = None,
    sha256: str | None = None,
    stop_event: threading.Event | None = None,
    timeout: float = 30.0,
    chunk_size: int = 65536,
    cache_subdir: str = "models",
    headers: dict[str, str] | None = None,
) -> str:
    """Download ``url`` atomically with cooperative cancellation, checksum verification, and read-only path fallback."""
    if is_stop_requested(stop_event):
        raise InferenceCancelledError("Model checkpoint download cancelled.")

    preferred = Path(dest_path)
    fallback = default_model_cache_dir(cache_subdir) / preferred.name
    existing = _existing_nonempty_checkpoint(preferred, fallback)
    if existing is not None:
        return existing

    dest_lock = _get_download_lock(preferred)
    _acquire_download_lock(dest_lock, stop_event)
    try:
        if is_stop_requested(stop_event):
            raise InferenceCancelledError("Model checkpoint download cancelled.")

        existing = _existing_nonempty_checkpoint(preferred, fallback)
        if existing is not None:
            return existing

        target = preferred
        try:
            part_fp = _open_temp_part_file(target)
        except OSError:
            target = fallback
            part_fp = _open_temp_part_file(target)

        tmp_path = Path(part_fp.name)
        req_headers = {"User-Agent": "Annolid/1.0"}
        if headers:
            req_headers.update(headers)
        request = urllib.request.Request(url, headers=req_headers)

        md5_hasher = hashlib.md5() if md5 else None
        sha256_hasher = hashlib.sha256() if sha256 else None
        resp = None
        unsub = None

        try:
            if is_stop_requested(stop_event):
                raise InferenceCancelledError("Model checkpoint download cancelled.")

            resp = urllib.request.urlopen(request, timeout=timeout)

            def _abort_stream() -> None:
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass

            if stop_event is not None:
                add_listener = getattr(stop_event, "add_listener", None)
                if callable(add_listener):
                    unsub = add_listener(_abort_stream)
                elif stop_event.is_set():
                    _abort_stream()
                    raise InferenceCancelledError(
                        "Model checkpoint download cancelled."
                    )

            expected_length: int | None = None
            raw_len = (
                getattr(resp, "headers", {}).get("Content-Length")
                if hasattr(resp, "headers")
                else None
            )
            if raw_len is not None:
                try:
                    parsed_len = int(raw_len)
                    if parsed_len > 0:
                        expected_length = parsed_len
                except (TypeError, ValueError):
                    expected_length = None

            bytes_written = 0
            with part_fp:
                while True:
                    if is_stop_requested(stop_event):
                        raise InferenceCancelledError(
                            "Model checkpoint download cancelled."
                        )
                    try:
                        chunk = resp.read(max(4096, int(chunk_size)))
                    except Exception as exc:
                        if is_stop_requested(stop_event):
                            raise InferenceCancelledError(
                                "Model checkpoint download cancelled."
                            ) from exc
                        raise
                    if is_stop_requested(stop_event):
                        raise InferenceCancelledError(
                            "Model checkpoint download cancelled."
                        )
                    if not chunk:
                        break
                    part_fp.write(chunk)
                    bytes_written += len(chunk)
                    if md5_hasher is not None:
                        md5_hasher.update(chunk)
                    if sha256_hasher is not None:
                        sha256_hasher.update(chunk)

            if bytes_written == 0:
                raise RuntimeError(f"Downloaded empty model file from {url}.")
            if expected_length is not None and bytes_written != expected_length:
                raise RuntimeError(
                    f"Incomplete download from {url}: expected {expected_length} bytes, got {bytes_written}."
                )
            if md5_hasher is not None:
                actual_md5 = md5_hasher.hexdigest().lower()
                if actual_md5 != str(md5).strip().lower():
                    raise RuntimeError(
                        f"MD5 checksum mismatch for {target.name}: expected {md5}, got {actual_md5}."
                    )
            if sha256_hasher is not None:
                actual_sha256 = sha256_hasher.hexdigest().lower()
                if actual_sha256 != str(sha256).strip().lower():
                    raise RuntimeError(
                        f"SHA256 checksum mismatch for {target.name}: expected {sha256}, got {actual_sha256}."
                    )

            os.replace(str(tmp_path), str(target))
            return str(target)
        finally:
            if callable(unsub):
                unsub()
            if resp is not None:
                try:
                    resp.close()
                except Exception:
                    pass
            try:
                part_fp.close()
            except Exception:
                pass
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
    finally:
        dest_lock.release()
