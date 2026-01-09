"""TensorBoard launch utilities used by the Annolid GUI."""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import requests
from qtpy import QtCore, QtWidgets, QtGui

from annolid.utils.runs import shared_runs_root

_STARTED_PROCESSES: list[subprocess.Popen] = []
_LAST_URL: Optional[str] = None
_PRIMARY: Optional[Tuple[subprocess.Popen, str, str]
                   ] = None  # (proc, url, logdir)


def _tensorboard_env() -> dict:
    env = dict(os.environ)
    sitecustomize_dir = (
        Path(__file__).resolve().parent.parent
        / "utils"
        / "tensorboard_sitecustomize"
    )
    if sitecustomize_dir.exists():
        existing = env.get("PYTHONPATH", "")
        prefix = str(sitecustomize_dir)
        env["PYTHONPATH"] = prefix if not existing else f"{prefix}{os.pathsep}{existing}"
    return env


def _can_connect(url: str, *, timeout: float = 0.5) -> bool:
    try:
        requests.get(url, timeout=timeout)
        return True
    except Exception:
        return False


def _plugin_enabled(url: str, plugin_name: str, *, timeout: float = 0.5) -> Optional[bool]:
    base = url.rstrip("/")
    try:
        resp = requests.get(f"{base}/data/plugins_listing", timeout=timeout)
    except Exception:
        return None
    if not resp.ok:
        return None
    try:
        payload = resp.json()
    except Exception:
        return None
    info = payload.get(str(plugin_name))
    if isinstance(info, dict):
        enabled = info.get("enabled")
        if isinstance(enabled, bool):
            return enabled
    return None


def _has_projector_config(log_dir: Path, *, max_depth: int = 9) -> bool:
    """Return True if log_dir contains a projector_config.pbtxt (bounded walk).

    Fast path checks common locations first, then falls back to a bounded walk.
    """
    root = Path(log_dir).expanduser().resolve()
    if not root.exists():
        return False
    for candidate in (
        root / "projector_config.pbtxt",
        root / "tensorboard" / "projector_config.pbtxt",
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
            # Prefer traversing likely folders first and skip hidden folders.
            dirnames[:] = [
                d for d in dirnames if d and not str(d).startswith(".")
            ]
            dirnames.sort(key=lambda d: (d != "tensorboard", d != "runs", d))
            if "projector_config.pbtxt" in filenames:
                return True
    except Exception:
        return False
    return False


def _wait_for_projector(
    url: str,
    *,
    timeout_s: float = 18.0,
    poll_s: float = 0.25,
) -> bool:
    """Wait until the TensorBoard projector plugin reports enabled=true."""
    t0 = time.time()
    while time.time() - t0 < float(timeout_s):
        enabled = _plugin_enabled(url, "projector", timeout=0.5)
        if enabled is True:
            return True
        time.sleep(float(poll_s))
    return False


def _pick_free_port(*, preferred: int = 6006, host: str = "127.0.0.1") -> int:
    import socket

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

    # Fallback: ask OS for an ephemeral port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def ensure_tensorboard(
    *,
    log_dir: Optional[Path] = None,
    preferred_port: int = 6006,
    host: str = "127.0.0.1",
    startup_timeout_s: float = 15.0,
) -> Tuple[Optional[subprocess.Popen], str]:
    """
    Ensure a TensorBoard server started by Annolid is available.

    Always starts a new TensorBoard instance on an available local port to avoid
    reusing an external (possibly incompatible) server.
    """
    if log_dir is None:
        log_dir = shared_runs_root()

    global _PRIMARY
    if _PRIMARY is not None:
        proc, url, existing_logdir = _PRIMARY
        if proc.poll() is None and str(Path(existing_logdir)) == str(Path(log_dir)):
            if _can_connect(url, timeout=0.5):
                if _has_projector_config(Path(log_dir)):
                    _wait_for_projector(
                        url, timeout_s=max(8.0, float(startup_timeout_s)))
                return proc, url

    port = _pick_free_port(preferred=int(preferred_port), host=str(host))
    url = f"http://{host}:{int(port)}/"

    cmd = [
        sys.executable,
        "-m",
        "tensorboard.main",
        f"--logdir={str(log_dir)}",
        f"--host={host}",
        f"--port={int(port)}",
    ]
    process = subprocess.Popen(cmd, env=_tensorboard_env())
    _STARTED_PROCESSES.append(process)

    t0 = time.time()
    while time.time() - t0 < float(startup_timeout_s):
        if process.poll() is not None:
            raise RuntimeError(
                "TensorBoard exited unexpectedly while starting.")
        if _can_connect(url, timeout=0.5):
            global _LAST_URL
            _LAST_URL = url
            _PRIMARY = (process, url, str(Path(log_dir)))
            if _has_projector_config(Path(log_dir)):
                _wait_for_projector(
                    url, timeout_s=max(8.0, float(startup_timeout_s)))
            return process, url
        time.sleep(0.25)

    raise TimeoutError(
        f"TensorBoard did not respond at {url} within {startup_timeout_s}s")


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
    """Stop a TensorBoard process started via this module (best-effort)."""
    targets = []
    if process is not None:
        targets = [process]
    else:
        targets = list(_STARTED_PROCESSES)

    for proc in targets:
        try:
            proc.kill()
        except Exception:
            pass

    if process is None:
        _STARTED_PROCESSES.clear()
        global _PRIMARY, _LAST_URL
        _PRIMARY = None
        _LAST_URL = None
    else:
        try:
            _STARTED_PROCESSES.remove(process)
        except ValueError:
            pass
        if _PRIMARY is not None and _PRIMARY[0] is process:
            _PRIMARY = None


class VisualizationWindow(QtWidgets.QDialog):
    """Embedded TensorBoard viewer dialog."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        log_dir: Optional[Path] = None,
        tensorboard_url: str = "http://localhost:6006",
    ) -> None:
        super().__init__(parent=parent)
        from qtpy.QtWebEngineWidgets import QWebEngineView

        self.setWindowTitle("Visualization Tensorboard")
        self.process, self.tensorboard_url = ensure_tensorboard(
            log_dir=log_dir,
            preferred_port=QtCore.QUrl(tensorboard_url).port() if QtCore.QUrl(
                tensorboard_url).port() != -1 else 6006,
            host="127.0.0.1",
        )
        self.browser = QWebEngineView()
        self.browser.setUrl(QtCore.QUrl(self.tensorboard_url))
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self.browser)
        self.setLayout(vbox)
        self.show()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.process is not None:
            time.sleep(3)
            stop_tensorboard(self.process)
        event.accept()
