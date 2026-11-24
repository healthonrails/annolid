"""TensorBoard lifecycle management and embedded viewer utilities for the Annolid GUI.

This module keeps top-level imports lean (stdlib + ``qtpy`` + internal helpers)
so importing it during GUI startup does not pull ``tensorboard`` or
``QtWebEngineWidgets`` eagerly.
"""

from __future__ import annotations

import atexit
import collections
import contextlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional, Tuple

from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.widgets.webengine_lifecycle import release_webengine_view
from annolid.utils.logger import logger
from annolid.utils.runs import shared_runs_root


@dataclass(frozen=True)
class TensorBoardSession:
    """Metadata describing a running managed TensorBoard server."""

    process: subprocess.Popen
    url: str
    host: str
    port: int
    log_dir: Optional[Path] = None
    logdir_spec: Optional[str] = None
    started_at: float = 0.0
    has_projector: bool = False
    projector_ready: bool = False

    @property
    def projector_url(self) -> str:
        base = self.url.rstrip("/")
        return f"{base}/#projector"

    @property
    def scope_label(self) -> str:
        if self.logdir_spec:
            count = len([p for p in self.logdir_spec.split(",") if p.strip()])
            return f"{count} compared runs"
        if self.log_dir is not None:
            return str(self.log_dir)
        return "runs"


_LOCK = threading.RLock()
_STARTED_PROCESSES: list[subprocess.Popen] = []
_LAST_URL: Optional[str] = None
_PRIMARY: Optional[Tuple[subprocess.Popen, str, str]] = None  # (proc, url, scope_key)
_PRIMARY_SESSION: Optional[TensorBoardSession] = None
_ATEXIT_REGISTERED = False


def _scope_key(*, log_dir: Optional[Path], logdir_spec: Optional[str]) -> str:
    if logdir_spec:
        return f"spec:{logdir_spec}"
    if log_dir is not None:
        return f"dir:{Path(log_dir).expanduser().resolve()}"
    return f"dir:{shared_runs_root()}"


def _tensorboard_env() -> dict:
    env = dict(os.environ)
    sitecustomize_dir = (
        Path(__file__).resolve().parent.parent / "utils" / "tensorboard_sitecustomize"
    )
    if sitecustomize_dir.exists():
        existing = env.get("PYTHONPATH", "")
        prefix = str(sitecustomize_dir)
        env["PYTHONPATH"] = (
            prefix if not existing else f"{prefix}{os.pathsep}{existing}"
        )
    return env


def _can_connect(url: str, *, timeout: float = 0.5) -> bool:
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except urllib.error.HTTPError:
        # An HTTP status response still proves the server socket is reachable.
        return True
    except Exception:
        return False


def _plugin_enabled(
    url: str, plugin_name: str, *, timeout: float = 0.5
) -> Optional[bool]:
    base = url.rstrip("/")
    try:
        req = urllib.request.Request(f"{base}/data/plugins_listing", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", 200)
            if not (200 <= int(status) < 300):
                return None
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None
    try:
        payload = json.loads(raw)
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    info = payload.get(str(plugin_name))
    if isinstance(info, dict):
        enabled = info.get("enabled")
        if isinstance(enabled, bool):
            return enabled
    return None


def _has_projector_config(log_dir: Path, *, max_depth: int = 9) -> bool:
    """Return True if log_dir contains a projector_config.pbtxt (bounded walk)."""
    root = Path(log_dir).expanduser().resolve()
    if not root.exists():
        return False
    for candidate in (
        root / "projector_config.pbtxt",
        root / "tensorboard" / "projector_config.pbtxt",
        root / "tb" / "projector_config.pbtxt",
    ):
        if candidate.is_file():
            return True
    root_parts = len(root.parts)
    try:
        for dirpath, dirnames, filenames in os.walk(str(root), topdown=True):
            depth = len(Path(dirpath).parts) - root_parts
            if depth > int(max_depth):
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if d and not str(d).startswith(".")]
            dirnames.sort(key=lambda d: (d != "tensorboard", d != "runs", d))
            if "projector_config.pbtxt" in filenames:
                return True
    except Exception:
        return False
    return False


def _spec_has_projector_config(logdir_spec: str) -> bool:
    for entry in str(logdir_spec or "").split(","):
        if ":" not in entry:
            continue
        _label, raw_path = entry.split(":", 1)
        cand = Path(raw_path.strip())
        if _has_projector_config(cand, max_depth=4):
            return True
    return False


def _wait_for_projector(
    url: str,
    *,
    timeout_s: float = 18.0,
    poll_s: float = 0.25,
    cancel_event: Optional[threading.Event] = None,
) -> bool:
    """Wait until the TensorBoard projector plugin reports enabled=true."""
    t0 = time.time()
    while time.time() - t0 < float(timeout_s):
        if cancel_event is not None and cancel_event.is_set():
            return False
        enabled = _plugin_enabled(url, "projector", timeout=0.5)
        if enabled is True:
            return True
        if cancel_event is not None:
            if cancel_event.wait(float(poll_s)):
                return False
        else:
            time.sleep(float(poll_s))
    return False


def _pick_free_port(*, preferred: int = 6006, host: str = "127.0.0.1") -> int:
    def port_free(port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((host, int(port)))
            return True
        except OSError:
            return False

    start = max(1024, int(preferred))
    if port_free(start):
        return start
    for port in range(start + 1, start + 50):
        if port_free(port):
            return port

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _terminate_process(proc: subprocess.Popen, *, grace_s: float = 1.5) -> None:
    try:
        if proc.poll() is not None:
            return
    except Exception:
        return
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=max(0.1, float(grace_s)))
        return
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass
    with contextlib.suppress(Exception):
        proc.wait(timeout=1.0)


def _prune_dead_processes_locked() -> None:
    alive: list[subprocess.Popen] = []
    for proc in _STARTED_PROCESSES:
        try:
            if proc.poll() is None:
                alive.append(proc)
        except Exception:
            continue
    _STARTED_PROCESSES[:] = alive


def _register_atexit_once() -> None:
    global _ATEXIT_REGISTERED
    with _LOCK:
        if not _ATEXIT_REGISTERED:
            atexit.register(stop_tensorboard)
            _ATEXIT_REGISTERED = True


def _start_output_drain(proc: subprocess.Popen) -> Deque[str]:
    tail: Deque[str] = collections.deque(maxlen=60)
    stream = getattr(proc, "stdout", None)
    if stream is None:
        return tail

    def _drain() -> None:
        try:
            for raw in stream:
                line = str(raw).rstrip()
                if line:
                    tail.append(line)
        except Exception:
            pass
        finally:
            with contextlib.suppress(Exception):
                stream.close()

    threading.Thread(target=_drain, name="annolid-tensorboard-io", daemon=True).start()
    return tail


def get_primary_tensorboard_session() -> Optional[TensorBoardSession]:
    """Return the currently running primary TensorBoard session, if alive."""
    with _LOCK:
        sess = _PRIMARY_SESSION
        if sess is None:
            return None
        try:
            if sess.process.poll() is not None:
                return None
        except Exception:
            return None
        return sess


def ensure_tensorboard_session(
    *,
    log_dir: Optional[Path] = None,
    logdir_spec: Optional[str] = None,
    preferred_port: int = 6006,
    host: str = "127.0.0.1",
    startup_timeout_s: float = 15.0,
    cancel_event: Optional[threading.Event] = None,
    ensure_runtime: bool = True,
) -> TensorBoardSession:
    """Start (or reuse) a managed TensorBoard server and return its session metadata.

    Supports either a single ``log_dir`` or a multi-run ``logdir_spec``
    (``name1:/path1,name2:/path2``). Captures startup output for actionable
    diagnostics and cleans up the spawned process on timeout or cancellation.
    """
    global _LAST_URL, _PRIMARY, _PRIMARY_SESSION
    spec_clean = str(logdir_spec or "").strip() or None
    resolved_dir: Optional[Path] = None
    if spec_clean is None:
        resolved_dir = (
            Path(log_dir).expanduser().resolve()
            if log_dir is not None
            else shared_runs_root()
        )
        resolved_dir.mkdir(parents=True, exist_ok=True)

    if ensure_runtime:
        from annolid.infrastructure.capabilities import ensure_capability

        ensure_capability("training", reason="TensorBoard training dashboard")

    scope = _scope_key(log_dir=resolved_dir, logdir_spec=spec_clean)
    has_proj = (
        _spec_has_projector_config(spec_clean)
        if spec_clean is not None
        else (
            _has_projector_config(resolved_dir) if resolved_dir is not None else False
        )
    )

    existing_candidate: Optional[Tuple[subprocess.Popen, str, TensorBoardSession]] = (
        None
    )
    with _LOCK:
        _prune_dead_processes_locked()
        if _PRIMARY is not None and _PRIMARY_SESSION is not None:
            proc, url, existing_scope = _PRIMARY
            if proc.poll() is None and existing_scope == scope:
                existing_candidate = (proc, url, _PRIMARY_SESSION)

    # Probe reachability and projector readiness outside _LOCK so concurrent
    # callers (e.g. GUI closeEvent -> stop_tensorboard) are never blocked on I/O.
    if existing_candidate is not None:
        proc, url, prev_session = existing_candidate
        if _can_connect(url, timeout=0.5):
            proj_ready = prev_session.projector_ready
            if has_proj and not proj_ready:
                proj_ready = _wait_for_projector(
                    url,
                    timeout_s=max(6.0, float(startup_timeout_s)),
                    cancel_event=cancel_event,
                )
            reused = TensorBoardSession(
                process=proc,
                url=url,
                host=prev_session.host,
                port=prev_session.port,
                log_dir=resolved_dir,
                logdir_spec=spec_clean,
                started_at=prev_session.started_at,
                has_projector=has_proj,
                projector_ready=proj_ready,
            )
            with _LOCK:
                if _PRIMARY is not None and _PRIMARY[0] is proc:
                    _PRIMARY_SESSION = reused
            return reused

    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("TensorBoard startup cancelled.")

    port = _pick_free_port(preferred=int(preferred_port), host=str(host))
    url = f"http://{host}:{int(port)}/"

    cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        f"--logdir_spec={spec_clean}"
        if spec_clean is not None
        else f"--logdir={str(resolved_dir)}",
        f"--host={host}",
        f"--port={int(port)}",
        "--reload_multifile=true",
    ]
    process = subprocess.Popen(
        cmd,
        env=_tensorboard_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_tail = _start_output_drain(process)
    _register_atexit_once()
    with _LOCK:
        _STARTED_PROCESSES.append(process)

    t0 = time.time()
    try:
        while time.time() - t0 < float(startup_timeout_s):
            if cancel_event is not None and cancel_event.is_set():
                stop_tensorboard(process)
                raise InterruptedError("TensorBoard startup cancelled.")
            rc = process.poll()
            if rc is not None:
                # Give output drain thread a brief moment to collect final stderr lines.
                time.sleep(0.05)
                tail_text = "\n".join(list(output_tail)[-12:]).strip()
                detail = f"\n\nTensorBoard output:\n{tail_text}" if tail_text else ""
                stop_tensorboard(process)
                raise RuntimeError(
                    f"TensorBoard exited unexpectedly while starting (exit code {rc}).{detail}"
                )
            if _can_connect(url, timeout=0.5):
                proj_ready = False
                if has_proj:
                    proj_ready = _wait_for_projector(
                        url,
                        timeout_s=max(6.0, float(startup_timeout_s)),
                        cancel_event=cancel_event,
                    )
                session = TensorBoardSession(
                    process=process,
                    url=url,
                    host=str(host),
                    port=int(port),
                    log_dir=resolved_dir,
                    logdir_spec=spec_clean,
                    started_at=time.time(),
                    has_projector=has_proj,
                    projector_ready=proj_ready,
                )
                with _LOCK:
                    _LAST_URL = url
                    _PRIMARY = (process, url, scope)
                    _PRIMARY_SESSION = session
                return session
            if cancel_event is not None:
                if cancel_event.wait(0.25):
                    stop_tensorboard(process)
                    raise InterruptedError("TensorBoard startup cancelled.")
            else:
                time.sleep(0.25)

        tail_text = "\n".join(list(output_tail)[-10:]).strip()
        detail = f"\n\nLast output:\n{tail_text}" if tail_text else ""
        stop_tensorboard(process)
        raise TimeoutError(
            f"TensorBoard did not respond at {url} within {startup_timeout_s}s.{detail}"
        )
    except Exception:
        if process.poll() is None:
            stop_tensorboard(process)
        raise


def ensure_tensorboard(
    *,
    log_dir: Optional[Path] = None,
    logdir_spec: Optional[str] = None,
    preferred_port: int = 6006,
    host: str = "127.0.0.1",
    startup_timeout_s: float = 15.0,
    cancel_event: Optional[threading.Event] = None,
) -> Tuple[Optional[subprocess.Popen], str]:
    """Ensure a managed TensorBoard server is running and return ``(process, url)``."""
    session = ensure_tensorboard_session(
        log_dir=log_dir,
        logdir_spec=logdir_spec,
        preferred_port=preferred_port,
        host=host,
        startup_timeout_s=startup_timeout_s,
        cancel_event=cancel_event,
    )
    return session.process, session.url


def start_tensorboard(
    log_dir: Optional[Path] = None,
    tensorboard_url: str = "http://localhost:6006",
) -> Optional[subprocess.Popen]:
    """Backward-compatible wrapper: starts TensorBoard and returns the process."""
    preferred_port = 6006
    try:
        parsed = QtCore.QUrl(tensorboard_url)
        if parsed.port() != -1:
            preferred_port = int(parsed.port())
    except Exception:
        pass
    process, _url = ensure_tensorboard(
        log_dir=log_dir,
        preferred_port=int(preferred_port),
        host="127.0.0.1",
    )
    return process


def stop_tensorboard(process: Optional[subprocess.Popen] = None) -> None:
    """Stop one or all managed TensorBoard processes gracefully."""
    global _PRIMARY, _PRIMARY_SESSION, _LAST_URL
    with _LOCK:
        if process is not None:
            targets = [process]
        else:
            targets = list(_STARTED_PROCESSES)

    for proc in targets:
        _terminate_process(proc, grace_s=1.5)

    with _LOCK:
        if process is None:
            _STARTED_PROCESSES.clear()
            _PRIMARY = None
            _PRIMARY_SESSION = None
            _LAST_URL = None
        else:
            with contextlib.suppress(ValueError):
                _STARTED_PROCESSES.remove(process)
            if _PRIMARY is not None and _PRIMARY[0] is process:
                _PRIMARY = None
                _PRIMARY_SESSION = None


class TensorBoardLaunchWorker(QtCore.QObject):
    """Background worker starting TensorBoard off the Qt UI thread."""

    launched = QtCore.Signal(int, object)  # (generation, TensorBoardSession)
    failed = QtCore.Signal(int, str)  # (generation, error_text)
    cancelled = QtCore.Signal(int)  # (generation,)

    def __init__(
        self,
        *,
        generation: int = 0,
        log_dir: Optional[Path] = None,
        logdir_spec: Optional[str] = None,
        preferred_port: int = 6006,
        host: str = "127.0.0.1",
        startup_timeout_s: float = 18.0,
        ensure_runtime: bool = False,
    ) -> None:
        super().__init__()
        self._generation = int(generation)
        self._log_dir = log_dir
        self._logdir_spec = logdir_spec
        self._preferred_port = int(preferred_port)
        self._host = str(host)
        self._startup_timeout_s = float(startup_timeout_s)
        self._ensure_runtime = bool(ensure_runtime)
        self._cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self._cancel_event.set()

    @QtCore.Slot()
    def run(self) -> None:
        try:
            session = ensure_tensorboard_session(
                log_dir=self._log_dir,
                logdir_spec=self._logdir_spec,
                preferred_port=self._preferred_port,
                host=self._host,
                startup_timeout_s=self._startup_timeout_s,
                cancel_event=self._cancel_event,
                ensure_runtime=self._ensure_runtime,
            )
            if self._cancel_event.is_set():
                stop_tensorboard(session.process)
                self.cancelled.emit(self._generation)
                return
            self.launched.emit(self._generation, session)
        except InterruptedError:
            self.cancelled.emit(self._generation)
        except Exception as exc:
            self.failed.emit(self._generation, str(exc))


def _create_ephemeral_web_profile() -> Optional[QtCore.QObject]:
    """Create an unmanaged off-the-record QWebEngineProfile for TensorBoard."""
    try:
        from qtpy import QtWebEngineWidgets  # type: ignore

        profile_cls = getattr(QtWebEngineWidgets, "QWebEngineProfile", None)
        if profile_cls is None:
            from qtpy import QtWebEngineCore  # type: ignore

            profile_cls = getattr(QtWebEngineCore, "QWebEngineProfile", None)
        if profile_cls is None:
            return None
        profile = profile_cls()
        cookie_policy = getattr(profile_cls, "NoPersistentCookies", None)
        if cookie_policy is not None:
            profile.setPersistentCookiesPolicy(cookie_policy)
        cache_kind = getattr(profile_cls, "MemoryHttpCache", None)
        if cache_kind is not None:
            profile.setHttpCacheType(cache_kind)
        return profile
    except Exception:
        return None


class TensorBoardEmbedWidget(QtWidgets.QWidget):
    """Non-blocking embedded TensorBoard viewer with safe WebEngine teardown.

    Shared by ``TrainingDashboardWidget`` and ``VisualizationWindow`` so both
    surfaces reuse one lifecycle-safe implementation.
    """

    status_changed = QtCore.Signal(str, str)  # (state, message)
    session_started = QtCore.Signal(object)  # TensorBoardSession
    session_stopped = QtCore.Signal()
    start_requested = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._session: Optional[TensorBoardSession] = None
        self._worker_thread: Optional[QtCore.QThread] = None
        self._worker: Optional[TensorBoardLaunchWorker] = None
        self._web_view: Optional[QtWidgets.QWidget] = None
        self._web_profile: Optional[QtCore.QObject] = None
        self._launch_generation: int = 0
        self._pending_initial_plugin: Optional[str] = None
        self._pending_open_external_fallback: bool = True
        self._state: str = "idle"
        self._build_ui()

    @property
    def session(self) -> Optional[TensorBoardSession]:
        return self._session

    @property
    def state(self) -> str:
        return self._state

    @property
    def current_url(self) -> Optional[str]:
        return self._session.url if self._session is not None else None

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        nav = QtWidgets.QHBoxLayout()
        nav.setSpacing(6)
        self.reload_btn = QtWidgets.QPushButton(self.tr("Reload"))
        self.reload_btn.setEnabled(False)
        self.reload_btn.clicked.connect(self.reload)

        self.scalars_btn = QtWidgets.QPushButton(self.tr("Scalars"))
        self.scalars_btn.setEnabled(False)
        self.scalars_btn.clicked.connect(lambda: self.navigate_plugin("scalars"))

        self.images_btn = QtWidgets.QPushButton(self.tr("Images"))
        self.images_btn.setEnabled(False)
        self.images_btn.clicked.connect(lambda: self.navigate_plugin("images"))

        self.projector_btn = QtWidgets.QPushButton(self.tr("Projector"))
        self.projector_btn.setEnabled(False)
        self.projector_btn.clicked.connect(lambda: self.navigate_plugin("projector"))

        self.browser_btn = QtWidgets.QPushButton(self.tr("Open in Browser"))
        self.browser_btn.setEnabled(False)
        self.browser_btn.clicked.connect(self.open_in_browser)

        self.url_label = QtWidgets.QLabel(self.tr("TensorBoard is idle"))
        self.url_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.url_label.setProperty("muted", True)

        nav.addWidget(self.reload_btn)
        nav.addWidget(self.scalars_btn)
        nav.addWidget(self.images_btn)
        nav.addWidget(self.projector_btn)
        nav.addWidget(self.browser_btn)
        nav.addWidget(self.url_label, 1)
        layout.addLayout(nav)

        self._stack = QtWidgets.QStackedWidget(self)
        layout.addWidget(self._stack, 1)

        # Page 0: Clean idle / starting / fallback / error card (never ERR_CONNECTION_REFUSED)
        placeholder = QtWidgets.QWidget(self)
        p_layout = QtWidgets.QVBoxLayout(placeholder)
        p_layout.setContentsMargins(24, 24, 24, 24)
        p_layout.setSpacing(10)
        p_layout.addStretch(1)

        self.placeholder_title = QtWidgets.QLabel(
            self.tr("TensorBoard Interactive Viewer")
        )
        t_font = self.placeholder_title.font()
        t_font.setPointSize(max(12, t_font.pointSize() + 2))
        t_font.setBold(True)
        self.placeholder_title.setFont(t_font)
        self.placeholder_title.setAlignment(QtCore.Qt.AlignCenter)
        p_layout.addWidget(self.placeholder_title)

        self.placeholder_detail = QtWidgets.QLabel(
            self.tr(
                "Click 'Start TensorBoard' to inspect TensorBoard scalars, images, "
                "graphs, and embedding projectors for the selected run or runs.\n\n"
                "Tip: Live loss and metric curves are already available natively in "
                "the 'Monitor & Curves' tab without starting TensorBoard."
            )
        )
        self.placeholder_detail.setWordWrap(True)
        self.placeholder_detail.setAlignment(QtCore.Qt.AlignCenter)
        self.placeholder_detail.setProperty("muted", True)
        p_layout.addWidget(self.placeholder_detail)

        self.placeholder_progress = QtWidgets.QProgressBar(self)
        self.placeholder_progress.setRange(0, 0)
        self.placeholder_progress.setMaximumWidth(280)
        self.placeholder_progress.setVisible(False)
        p_layout.addWidget(self.placeholder_progress, 0, QtCore.Qt.AlignHCenter)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        self.placeholder_start_btn = QtWidgets.QPushButton(self.tr("Start TensorBoard"))
        self.placeholder_start_btn.clicked.connect(self.start_requested.emit)
        btn_row.addWidget(self.placeholder_start_btn)

        self.placeholder_browser_btn = QtWidgets.QPushButton(
            self.tr("Open in External Browser")
        )
        self.placeholder_browser_btn.setVisible(False)
        self.placeholder_browser_btn.clicked.connect(self.open_in_browser)
        btn_row.addWidget(self.placeholder_browser_btn)
        btn_row.addStretch(1)
        p_layout.addLayout(btn_row)
        p_layout.addStretch(1)

        self._stack.addWidget(placeholder)

        # Page 1: Host container for lazy QWebEngineView
        self._web_host = QtWidgets.QWidget(self)
        self._web_layout = QtWidgets.QVBoxLayout(self._web_host)
        self._web_layout.setContentsMargins(0, 0, 0, 0)
        self._stack.addWidget(self._web_host)
        self._stack.setCurrentIndex(0)

    def _set_state(self, state: str, message: str) -> None:
        self._state = state
        self.url_label.setText(message)
        self.status_changed.emit(state, message)

    def _ensure_web_view(self) -> Optional[QtWidgets.QWidget]:
        if self._web_view is not None:
            return self._web_view
        try:
            from qtpy import QtWebEngineWidgets  # type: ignore

            view_cls = QtWebEngineWidgets.QWebEngineView
            page_cls = getattr(QtWebEngineWidgets, "QWebEnginePage", None)
            if page_cls is None:
                from qtpy import QtWebEngineCore  # type: ignore

                page_cls = getattr(QtWebEngineCore, "QWebEnginePage", None)
        except Exception:
            return None
        view = view_cls(self._web_host)
        profile = _create_ephemeral_web_profile()
        if profile is not None and page_cls is not None:
            try:
                page = page_cls(profile, view)
                view.setPage(page)
                self._web_profile = profile
            except Exception:
                with contextlib.suppress(Exception):
                    profile.deleteLater()
                self._web_profile = None
        with contextlib.suppress(Exception):
            view.loadFinished.connect(self._on_web_load_finished)
        self._web_layout.addWidget(view, 1)
        self._web_view = view
        return view

    @QtCore.Slot(bool)
    def _on_web_load_finished(self, ok: bool) -> None:
        if ok or self._state != "running" or self._session is None:
            return
        cur_url = ""
        if self._web_view is not None:
            with contextlib.suppress(Exception):
                cur_url = self._web_view.url().toString()
        if not cur_url or cur_url == "about:blank":
            return
        self.placeholder_title.setText(self.tr("Embedded Viewer Could Not Load Page"))
        self.placeholder_detail.setText(
            self.tr(
                "TensorBoard is running at %s (%s), but the embedded viewer failed "
                "to render it.\n\nClick 'Open in External Browser' or 'Reload'."
            )
            % (self._session.url, self._session.scope_label)
        )
        self.placeholder_progress.setVisible(False)
        self.placeholder_browser_btn.setVisible(True)
        self._stack.setCurrentIndex(0)

    def _release_web_view(self) -> None:
        view = self._web_view
        profile = self._web_profile
        if view is None and profile is None:
            return
        self._web_view = None
        self._web_profile = None
        if view is not None:
            with contextlib.suppress(Exception):
                view.loadFinished.disconnect(self._on_web_load_finished)
            with contextlib.suppress(Exception):
                self._web_layout.removeWidget(view)
        release_webengine_view(view, profile)

    def _cancel_worker(self) -> None:
        self._launch_generation += 1
        worker = self._worker
        thread = self._worker_thread
        self._worker = None
        self._worker_thread = None
        if worker is not None:
            with contextlib.suppress(Exception):
                worker.launched.disconnect(self._on_worker_launched)
            with contextlib.suppress(Exception):
                worker.failed.disconnect(self._on_worker_failed)
            with contextlib.suppress(Exception):
                worker.cancelled.disconnect(self._on_worker_cancelled)
            with contextlib.suppress(Exception):
                worker.request_cancel()
        still_running = False
        if thread is not None:
            with contextlib.suppress(Exception):
                if thread.isRunning():
                    thread.quit()
                    thread.wait(3000)
                still_running = bool(thread.isRunning())
            if still_running:
                # Defer QThread/worker destruction until the thread finishes so Qt
                # never aborts with "QThread: Destroyed while thread is still running".
                with contextlib.suppress(Exception):
                    thread.finished.connect(thread.deleteLater)
                if worker is not None:
                    with contextlib.suppress(Exception):
                        thread.finished.connect(worker.deleteLater)
            else:
                with contextlib.suppress(Exception):
                    thread.deleteLater()
        if worker is not None and not still_running:
            with contextlib.suppress(Exception):
                worker.deleteLater()

    def start_server(
        self,
        *,
        log_dir: Optional[Path] = None,
        logdir_spec: Optional[str] = None,
        preferred_port: int = 6006,
        host: str = "127.0.0.1",
        open_projector: bool = False,
        initial_plugin: Optional[str] = None,
        open_external_if_no_webengine: bool = True,
        check_capability: bool = True,
    ) -> bool:
        """Start TensorBoard asynchronously without blocking the Qt UI thread."""
        if check_capability:
            from annolid.gui.runtime_installer import ensure_gui_capability

            if not ensure_gui_capability(
                self.window(),
                "training",
                feature_title=self.tr("TensorBoard Dashboard"),
            ):
                self._set_state("idle", self.tr("TensorBoard runtime not installed"))
                return False

        self.stop_server(emit_signal=False)
        self._launch_generation += 1
        generation = self._launch_generation
        plugin_clean = str(initial_plugin or "").strip().lstrip("#") or None
        if plugin_clean is None and open_projector:
            plugin_clean = "projector"
        self._pending_initial_plugin = plugin_clean
        self._pending_open_external_fallback = bool(open_external_if_no_webengine)

        scope_desc = (
            "selected runs"
            if logdir_spec
            else str(
                Path(log_dir).expanduser().resolve() if log_dir else shared_runs_root()
            )
        )
        self._stack.setCurrentIndex(0)
        self.placeholder_title.setText(self.tr("Starting TensorBoard…"))
        self.placeholder_detail.setText(
            self.tr("Launching TensorBoard server for %s") % scope_desc
        )
        self.placeholder_progress.setVisible(True)
        self.placeholder_start_btn.setEnabled(False)
        self.placeholder_browser_btn.setVisible(False)
        self._set_state("starting", self.tr("Starting TensorBoard…"))

        worker = TensorBoardLaunchWorker(
            generation=generation,
            log_dir=log_dir,
            logdir_spec=logdir_spec,
            preferred_port=preferred_port,
            host=host,
            ensure_runtime=False,
        )
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.launched.connect(self._on_worker_launched)
        worker.failed.connect(self._on_worker_failed)
        worker.cancelled.connect(self._on_worker_cancelled)
        worker.launched.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.cancelled.connect(thread.quit)
        thread.finished.connect(self._on_worker_thread_finished)

        self._worker = worker
        self._worker_thread = thread
        thread.start()
        return True

    def _on_worker_thread_finished(self) -> None:
        thread = self.sender()
        if thread is self._worker_thread:
            worker = self._worker
            self._worker = None
            self._worker_thread = None
            if worker is not None:
                with contextlib.suppress(Exception):
                    worker.deleteLater()
        if thread is not None:
            with contextlib.suppress(Exception):
                thread.deleteLater()

    def _target_url_for_session(self, session: TensorBoardSession) -> str:
        if self._pending_initial_plugin:
            base = session.url.rstrip("/")
            return f"{base}/#{self._pending_initial_plugin}"
        return session.url

    @QtCore.Slot(int, object)
    def _on_worker_launched(self, generation: int, session: object) -> None:
        if not isinstance(session, TensorBoardSession):
            return
        if int(generation) != self._launch_generation:
            # Stale completion from a superseded or stopped launch — clean up immediately.
            with contextlib.suppress(Exception):
                stop_tensorboard(session.process)
            return

        self._session = session
        target_url = self._target_url_for_session(session)

        self.placeholder_progress.setVisible(False)
        self.placeholder_start_btn.setEnabled(True)
        self.reload_btn.setEnabled(True)
        self.scalars_btn.setEnabled(True)
        self.images_btn.setEnabled(True)
        self.projector_btn.setEnabled(True)
        self.browser_btn.setEnabled(True)

        view = self._ensure_web_view()
        if view is not None:
            view.setUrl(QtCore.QUrl(target_url))
            self._stack.setCurrentIndex(1)
        else:
            self.placeholder_title.setText(self.tr("TensorBoard Running"))
            self.placeholder_detail.setText(
                self.tr(
                    "TensorBoard is serving at %s (%s).\n\n"
                    "Embedded QtWebEngine is not installed in this environment. "
                    "Use 'Open in External Browser' to view TensorBoard."
                )
                % (target_url, session.scope_label)
            )
            self.placeholder_browser_btn.setVisible(True)
            self._stack.setCurrentIndex(0)
            if self._pending_open_external_fallback:
                QtGui.QDesktopServices.openUrl(QtCore.QUrl(target_url))

        self._set_state("running", f"{session.url} ({session.scope_label})")
        self.session_started.emit(session)

    @QtCore.Slot(int, str)
    def _on_worker_failed(self, generation: int, error_text: str) -> None:
        if int(generation) != self._launch_generation:
            return
        self._session = None
        self.placeholder_progress.setVisible(False)
        self.placeholder_start_btn.setEnabled(True)
        self.placeholder_browser_btn.setVisible(False)
        self.placeholder_title.setText(self.tr("TensorBoard Could Not Start"))
        self.placeholder_detail.setText(str(error_text))
        self._stack.setCurrentIndex(0)
        self._set_state("error", self.tr("TensorBoard failed to start"))
        logger.warning("TensorBoard startup failed: %s", error_text)

    @QtCore.Slot(int)
    def _on_worker_cancelled(self, generation: int) -> None:
        if int(generation) != self._launch_generation:
            return
        self.placeholder_progress.setVisible(False)
        self.placeholder_start_btn.setEnabled(True)
        self._stack.setCurrentIndex(0)
        self._set_state("idle", self.tr("TensorBoard: idle"))

    def navigate_plugin(self, plugin: str) -> None:
        if self._session is None:
            return
        clean = str(plugin).strip().lstrip("#")
        self._pending_initial_plugin = clean or None
        base = self._session.url.rstrip("/")
        url = f"{base}/#{clean}" if clean else self._session.url
        if self._web_view is not None:
            self._web_view.setUrl(QtCore.QUrl(url))
            self._stack.setCurrentIndex(1)
        else:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))

    def reload(self) -> None:
        if self._web_view is not None and self._session is not None:
            with contextlib.suppress(Exception):
                self._stack.setCurrentIndex(1)
                self._web_view.reload()

    def open_in_browser(self) -> None:
        if self._session is None:
            return
        url = self._target_url_for_session(self._session)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))

    def stop_server(self, *, emit_signal: bool = True) -> None:
        self._cancel_worker()
        sess = self._session
        self._session = None
        if sess is not None:
            with contextlib.suppress(Exception):
                stop_tensorboard(sess.process)
        if self._web_view is not None:
            with contextlib.suppress(Exception):
                self._web_view.setUrl(QtCore.QUrl("about:blank"))
        self._stack.setCurrentIndex(0)
        self.placeholder_progress.setVisible(False)
        self.placeholder_start_btn.setEnabled(True)
        self.placeholder_browser_btn.setVisible(False)
        self.placeholder_title.setText(self.tr("TensorBoard Interactive Viewer"))
        self.placeholder_detail.setText(
            self.tr(
                "Click 'Start TensorBoard' to inspect TensorBoard scalars, images, "
                "graphs, and embedding projectors for the selected run or runs."
            )
        )
        for btn in (
            self.reload_btn,
            self.scalars_btn,
            self.images_btn,
            self.projector_btn,
            self.browser_btn,
        ):
            btn.setEnabled(False)
        self._set_state("idle", self.tr("TensorBoard: idle"))
        if emit_signal:
            self.session_stopped.emit()

    def cleanup(self) -> None:
        self.stop_server(emit_signal=False)
        self._release_web_view()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.cleanup()
        super().closeEvent(event)


class VisualizationWindow(QtWidgets.QDialog):
    """Embedded TensorBoard viewer dialog (non-blocking, lifecycle-safe).

    Retained for backward compatibility with existing callers; prefers the
    unified ``TrainingDashboardDialog`` for full multi-run monitoring.
    """

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        log_dir: Optional[Path] = None,
        tensorboard_url: str = "http://localhost:6006",
        *,
        logdir_spec: Optional[str] = None,
        open_projector: bool = False,
        auto_start: bool = True,
    ) -> None:
        super().__init__(parent=parent)
        self.setWindowTitle(self.tr("TensorBoard Visualization"))
        self.setModal(False)
        self.resize(1040, 700)

        self._log_dir = (
            Path(log_dir).expanduser().resolve() if log_dir is not None else None
        )
        self._logdir_spec = logdir_spec
        self._open_projector = bool(open_projector)
        parsed = QtCore.QUrl(tensorboard_url)
        self._preferred_port = parsed.port() if parsed.port() != -1 else 6006
        self.tensorboard_url = str(tensorboard_url)

        layout = QtWidgets.QVBoxLayout(self)
        self.embed = TensorBoardEmbedWidget(self)
        self.embed.start_requested.connect(self._launch)
        self.embed.session_started.connect(self._on_session_started)
        layout.addWidget(self.embed, 1)

        if auto_start:
            QtCore.QTimer.singleShot(0, self._launch)

    @property
    def process(self) -> Optional[subprocess.Popen]:
        sess = self.embed.session
        return sess.process if sess is not None else None

    def _on_session_started(self, session: object) -> None:
        if isinstance(session, TensorBoardSession):
            self.tensorboard_url = session.url

    def _launch(self) -> None:
        self.embed.start_server(
            log_dir=self._log_dir,
            logdir_spec=self._logdir_spec,
            preferred_port=self._preferred_port,
            open_projector=self._open_projector,
            open_external_if_no_webengine=True,
            check_capability=True,
        )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.embed.cleanup()
        event.accept()


__all__ = [
    "TensorBoardEmbedWidget",
    "TensorBoardLaunchWorker",
    "TensorBoardSession",
    "VisualizationWindow",
    "_can_connect",
    "_has_projector_config",
    "_pick_free_port",
    "_plugin_enabled",
    "_tensorboard_env",
    "_wait_for_projector",
    "ensure_tensorboard",
    "ensure_tensorboard_session",
    "get_primary_tensorboard_session",
    "start_tensorboard",
    "stop_tensorboard",
]
