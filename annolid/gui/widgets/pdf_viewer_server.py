from __future__ import annotations

import os
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urlparse

from annolid.utils.logger import logger

_PDFJS_HTTP_SERVER: Optional[ThreadingHTTPServer] = None
_PDFJS_HTTP_PORT: Optional[int] = None
_PDFJS_HTTP_THREAD: Optional[threading.Thread] = None
_PDFJS_HTTP_LOCK = threading.Lock()
_PDFJS_HTTP_TOKENS: dict[str, Path] = {}
_PDFJS_HTTP_ASSET_CACHE: dict[str, tuple[int, bytes]] = {}


def _pdfjs_asset_path(filename: str) -> Optional[Path]:
    try:
        root = Path(__file__).resolve().parents[1] / "assets" / "pdfjs"
        candidate = (root / filename).resolve()
        if candidate.exists() and candidate.is_file():
            return candidate
    except Exception:
        return None
    return None


def _ensure_pdfjs_http_server() -> str:
    global _PDFJS_HTTP_SERVER, _PDFJS_HTTP_PORT, _PDFJS_HTTP_THREAD
    with _PDFJS_HTTP_LOCK:
        if _PDFJS_HTTP_SERVER is not None and _PDFJS_HTTP_PORT is not None:
            return f"http://127.0.0.1:{_PDFJS_HTTP_PORT}"

        class _Handler(BaseHTTPRequestHandler):
            server_version = "AnnolidPdfServer/1.0"

            def log_message(self, fmt: str, *args: object) -> None:  # noqa: D401
                return

            def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler
                self._serve(send_body=False)

            def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler
                self._serve(send_body=True)

            def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler
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

                if path.startswith("/pdfjs/"):
                    name = unquote(path[len("/pdfjs/") :]).strip().lstrip("/")
                    if name not in {
                        "pdf.worker.min.js",
                        "pdf.min.js",
                        "annolid.worker.js",
                        "annolid_viewer.js",
                        "annolid_viewer_polyfills.js",
                        "annolid_viewer.css",
                    }:
                        self.send_error(404)
                        return
                    try:
                        asset = _pdfjs_asset_path(name)
                        if asset is None:
                            self.send_error(404)
                            return
                        mtime_ns = int(asset.stat().st_mtime_ns)
                        cached = _PDFJS_HTTP_ASSET_CACHE.get(name)
                        if cached is None or cached[0] != mtime_ns:
                            payload = asset.read_bytes()
                            cached = (mtime_ns, payload)
                            _PDFJS_HTTP_ASSET_CACHE[name] = cached
                    except Exception:
                        self.send_error(404)
                        return
                    payload = cached[1]
                    if name.endswith(".css"):
                        content_type = "text/css"
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

                if not path.startswith("/pdf/"):
                    self.send_error(404)
                    return

                token = unquote(path[len("/pdf/") :]).strip().split("/", 1)[0]
                file_path = _PDFJS_HTTP_TOKENS.get(token)
                if file_path is None or not file_path.exists():
                    self.send_error(404)
                    return
                try:
                    size = file_path.stat().st_size
                except Exception:
                    self.send_error(404)
                    return

                range_header = self.headers.get("Range", "")
                start = 0
                end = max(0, size - 1)
                status = 200
                if range_header.startswith("bytes="):
                    try:
                        value = range_header[len("bytes=") :].strip()
                        start_str, end_str = (value.split("-", 1) + [""])[:2]
                        if start_str == "" and end_str:
                            length = int(end_str)
                            length = max(0, min(size, length))
                            start = max(0, size - length)
                            end = max(0, size - 1)
                        else:
                            start = int(start_str) if start_str else 0
                            end = int(end_str) if end_str else end
                        start = max(0, min(start, max(0, size - 1)))
                        end = max(start, min(end, max(0, size - 1)))
                        status = 206
                    except Exception:
                        start = 0
                        end = max(0, size - 1)
                        status = 200

                length = max(0, end - start + 1)
                self.send_response(status)
                self.send_header("Content-Type", "application/pdf")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header(
                    "Access-Control-Expose-Headers",
                    "Accept-Ranges, Content-Range, Content-Length",
                )
                self.send_header("Content-Length", str(length))
                if status == 206:
                    self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.end_headers()
                if not send_body:
                    return
                try:
                    with open(file_path, "rb") as f:
                        if start:
                            f.seek(start, os.SEEK_SET)
                        remaining = length
                        while remaining > 0:
                            chunk = f.read(min(1024 * 256, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except Exception:
                    return

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        _PDFJS_HTTP_SERVER = httpd
        _PDFJS_HTTP_PORT = int(getattr(httpd, "server_port", 0) or 0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        _PDFJS_HTTP_THREAD = thread
        logger.info(f"PDF.js local HTTP server started on 127.0.0.1:{_PDFJS_HTTP_PORT}")
        return f"http://127.0.0.1:{_PDFJS_HTTP_PORT}"


def _register_pdfjs_http_pdf(path: Path) -> str:
    base = _ensure_pdfjs_http_server()
    token = uuid.uuid4().hex
    _PDFJS_HTTP_TOKENS[token] = path
    logger.debug(f"PDF.js serving {path} via token {token}")
    return f"{base}/pdf/{token}"


__all__ = ["_ensure_pdfjs_http_server", "_register_pdfjs_http_pdf"]
