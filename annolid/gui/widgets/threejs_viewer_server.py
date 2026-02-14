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
                        "points_3d.html",
                        "parser.worker.js",
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
                    if name.endswith(".css"):
                        content_type = "text/css"
                    elif name.endswith(".html"):
                        content_type = "text/html"
                    else:
                        content_type = "application/javascript"

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

                token_and_path = unquote(path[len("/model/") :]).strip()
                parts = token_and_path.split("/", 1)
                token = parts[0]

                # Get the base model file path
                base_file_path = _THREEJS_HTTP_TOKENS.get(token)
                if base_file_path is None:
                    logger.warning(f"Token not found: {token}")
                    self.send_error(404)
                    return

                logger.info(
                    f"Model request: token={token}, base_file={base_file_path}, parts={parts}"
                )

                # If no additional path, serve the base file
                if len(parts) == 1:
                    file_path = base_file_path
                else:
                    # Serve a related file in the same directory
                    requested_filename = parts[1]
                    model_dir = base_file_path.parent
                    file_path = model_dir / requested_filename

                    logger.info(
                        f"Related file request: {requested_filename}, full_path={file_path}, exists={file_path.exists()}"
                    )

                    # Security check: only serve files in the same directory or subdirectories
                    try:
                        file_path.resolve().relative_to(model_dir.resolve())
                    except ValueError:
                        # File is outside the model directory
                        logger.warning(
                            f"Security violation: requested file {file_path} is outside model directory {model_dir}"
                        )
                        self.send_error(403)
                        return

                if not file_path.exists():
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
                        content_type = "model/obj"
                    elif suffix == ".mtl":
                        content_type = "model/mtl"
                    elif suffix == ".ply":
                        content_type = "model/ply"
                    elif suffix == ".glb":
                        content_type = "model/gltf-binary"
                    elif suffix == ".gltf":
                        content_type = "model/gltf+json"
                    elif suffix in (".png", ".jpg", ".jpeg"):
                        content_type = f"image/{suffix[1:]}"
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
    return f"{base}/model/{token}/{path.name}"


__all__ = ["_ensure_threejs_http_server", "_register_threejs_http_model"]
