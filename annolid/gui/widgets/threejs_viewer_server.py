from __future__ import annotations

import mimetypes
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

from annolid.utils.logger import logger

_THREEJS_HTTP_SERVER: Optional[ThreadingHTTPServer] = None
_THREEJS_HTTP_PORT: Optional[int] = None
_THREEJS_HTTP_THREAD: Optional[threading.Thread] = None
_THREEJS_HTTP_LOCK = threading.Lock()
_THREEJS_HTTP_TOKENS: dict[str, Path] = {}
_THREEJS_HTTP_ASSET_CACHE: dict[str, tuple[int, bytes]] = {}


def _threejs_asset_path(filename: str) -> Optional[Path]:
    try:
        root = Path(__file__).resolve().parents[1] / "assets" / "threejs"
        candidate = (root / filename).resolve()
        if candidate.exists() and candidate.is_file():
            return candidate
    except Exception:
        return None
    return None


def _ensure_threejs_http_server() -> str:
    global _THREEJS_HTTP_SERVER, _THREEJS_HTTP_PORT, _THREEJS_HTTP_THREAD
    with _THREEJS_HTTP_LOCK:
        if _THREEJS_HTTP_SERVER is not None and _THREEJS_HTTP_PORT is not None:
            return f"http://127.0.0.1:{_THREEJS_HTTP_PORT}"

        class _Handler(BaseHTTPRequestHandler):
            server_version = "AnnolidThreeJsServer/1.0"

            def log_message(self, fmt: str, *args: object) -> None:  # noqa: D401
                return

            def do_HEAD(self) -> None:  # noqa: N802
                self._serve(send_body=False)

            def do_GET(self) -> None:  # noqa: N802
                self._serve(send_body=True)

            def do_OPTIONS(self) -> None:  # noqa: N802
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")
                self.send_header(
                    "Access-Control-Expose-Headers",
                    "Accept-Ranges, Content-Range, Content-Length",
                )
                self.end_headers()

            def _serve(self, *, send_body: bool) -> None:
                try:
                    parsed = urlparse(self.path)
                    path = parsed.path or ""
                except Exception:
                    self.send_error(400)
                    return

                if path.startswith("/threejs/"):
                    name = unquote(path[len("/threejs/") :]).strip().lstrip("/")
                    if name not in {
                        "annolid_threejs_viewer.js",
                        "annolid_threejs_viewer.css",
                    }:
                        self.send_error(404)
                        return
                    try:
                        asset = _threejs_asset_path(name)
                        if asset is None:
                            self.send_error(404)
                            return
                        mtime_ns = int(asset.stat().st_mtime_ns)
                        cached = _THREEJS_HTTP_ASSET_CACHE.get(name)
                        if cached is None or cached[0] != mtime_ns:
                            payload = asset.read_bytes()
                            cached = (mtime_ns, payload)
                            _THREEJS_HTTP_ASSET_CACHE[name] = cached
                    except Exception:
                        self.send_error(404)
                        return
                    payload = cached[1]
                    content_type = (
                        "text/css"
                        if name.endswith(".css")
                        else "application/javascript"
                    )
                    self.send_response(200)
                    self.send_header("Content-Type", content_type)
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    if send_body:
                        try:
                            self.wfile.write(payload)
                        except Exception:
                            return
                    return

                if not path.startswith("/model/"):
                    self.send_error(404)
                    return

                token = unquote(path[len("/model/") :]).strip().split("/", 1)[0]
                file_path = _THREEJS_HTTP_TOKENS.get(token)
                if file_path is None or not file_path.exists():
                    self.send_error(404)
                    return
                try:
                    payload = file_path.read_bytes()
                except Exception:
                    self.send_error(404)
                    return

                content_type, _ = mimetypes.guess_type(str(file_path))
                if not content_type:
                    suffix = file_path.suffix.lower()
                    if suffix == ".stl":
                        content_type = "model/stl"
                    elif suffix == ".obj":
                        content_type = "text/plain"
                    elif suffix == ".ply":
                        content_type = "application/octet-stream"
                    else:
                        content_type = "application/octet-stream"

                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if send_body:
                    try:
                        self.wfile.write(payload)
                    except Exception:
                        return

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        _THREEJS_HTTP_SERVER = httpd
        _THREEJS_HTTP_PORT = int(getattr(httpd, "server_port", 0) or 0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        _THREEJS_HTTP_THREAD = thread
        logger.info(
            "Three.js local HTTP server started on 127.0.0.1:%s", _THREEJS_HTTP_PORT
        )
        return f"http://127.0.0.1:{_THREEJS_HTTP_PORT}"


def _register_threejs_http_model(path: Path) -> str:
    base = _ensure_threejs_http_server()
    token = uuid.uuid4().hex
    _THREEJS_HTTP_TOKENS[token] = path
    return f"{base}/model/{token}"


__all__ = ["_ensure_threejs_http_server", "_register_threejs_http_model"]
