from __future__ import annotations

from collections import OrderedDict
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
_PDFJS_HTTP_TOKENS: OrderedDict[str, Path] = OrderedDict()
_PDFJS_HTTP_VIEWERS: OrderedDict[str, str] = OrderedDict()
_PDFJS_HTTP_ASSET_CACHE: dict[str, tuple[int, bytes]] = {}

_MAX_HTTP_PDF_TOKENS = 32
_MAX_HTTP_VIEWER_TOKENS = 16

_ALLOWED_PDFJS_ASSETS = frozenset(
    {
        "pdf.worker.min.js",
        "pdf.min.js",
        "annolid.worker.js",
        "annolid_pdf_speech.js",
        "annolid_viewer.js",
        "annolid_viewer_polyfills.js",
        "annolid_viewer.css",
    }
)


def _pdfjs_asset_path(filename: str) -> Optional[Path]:
    try:
        root = Path(__file__).resolve().parents[1] / "assets" / "pdfjs"
        candidate = (root / filename).resolve()
        if candidate.parent != root.resolve():
            return None
        if candidate.exists() and candidate.is_file():
            return candidate
    except Exception:
        return None
    return None


def _resolve_pdf_byte_range(size: int, range_header: str) -> tuple[int, int, int, int]:
    """Resolve an HTTP Range header for a PDF of ``size`` bytes.

    Returns ``(status_code, start, end, content_length)``. For empty files
    (``size <= 0``), returns ``(200, 0, -1, 0)``.
    """
    file_size = max(0, int(size))
    if file_size <= 0:
        return (200, 0, -1, 0)

    start = 0
    end = file_size - 1
    status = 200
    header = str(range_header or "")
    if header.startswith("bytes="):
        try:
            value = header[len("bytes=") :].strip()
            start_str, end_str = (value.split("-", 1) + [""])[:2]
            if start_str == "" and end_str:
                length = max(0, min(file_size, int(end_str)))
                start = max(0, file_size - length)
                end = file_size - 1
            else:
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else end
            start = max(0, min(start, file_size - 1))
            end = max(start, min(end, file_size - 1))
            status = 206
        except Exception:
            start = 0
            end = file_size - 1
            status = 200
    return (status, start, end, max(0, end - start + 1))


def _store_pdfjs_http_token(path: Path) -> str:
    """Register a PDF path in the bounded token cache and return its token."""
    token = uuid.uuid4().hex
    with _PDFJS_HTTP_LOCK:
        _PDFJS_HTTP_TOKENS[token] = path
        while len(_PDFJS_HTTP_TOKENS) > _MAX_HTTP_PDF_TOKENS:
            _PDFJS_HTTP_TOKENS.popitem(last=False)
    return token


def _store_pdfjs_http_viewer_html(html: str) -> str:
    """Register viewer HTML in the bounded token cache and return its token."""
    token = uuid.uuid4().hex
    with _PDFJS_HTTP_LOCK:
        _PDFJS_HTTP_VIEWERS[token] = str(html)
        while len(_PDFJS_HTTP_VIEWERS) > _MAX_HTTP_VIEWER_TOKENS:
            _PDFJS_HTTP_VIEWERS.popitem(last=False)
    return token


def _allowed_loopback_cors_origin(
    origin: str, server_port: Optional[int]
) -> Optional[str]:
    """Allow CORS only for exact loopback origins matching this local server."""
    raw = str(origin or "").strip()
    if not raw:
        return None
    try:
        parsed = urlparse(raw)
    except Exception:
        return None
    if parsed.scheme.lower() != "http":
        return None
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        return None
    host = str(parsed.hostname or "").strip().lower()
    if host not in {"127.0.0.1", "localhost", "::1"}:
        return None
    if server_port is not None and server_port > 0:
        try:
            if parsed.port != int(server_port):
                return None
        except ValueError:
            return None
    return f"{parsed.scheme.lower()}://{parsed.netloc}"


class _PdfJsHttpRequestHandler(BaseHTTPRequestHandler):
    """Local HTTP request handler serving PDF.js assets, viewer HTML, and PDF bytes."""

    server_version = "AnnolidPdfServer/1.0"

    def log_message(self, fmt: str, *args: object) -> None:  # noqa: D401
        return

    def _send_security_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        headers = getattr(self, "headers", None)
        origin = headers.get("Origin", "") if headers is not None else ""
        server_port = getattr(getattr(self, "server", None), "server_port", None)
        if server_port is None:
            server_port = _PDFJS_HTTP_PORT
        allowed_origin = _allowed_loopback_cors_origin(origin, server_port)
        if allowed_origin:
            self.send_header("Access-Control-Allow-Origin", allowed_origin)
            self.send_header("Vary", "Origin")

    def do_HEAD(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler
        self._serve(send_body=False)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler
        self._serve(send_body=True)

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler
        self.send_response(204)
        self._send_security_headers()
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
            if name not in _ALLOWED_PDFJS_ASSETS:
                self.send_error(404)
                return
            try:
                asset = _pdfjs_asset_path(name)
                if asset is None:
                    self.send_error(404)
                    return
                mtime_ns = int(asset.stat().st_mtime_ns)
                with _PDFJS_HTTP_LOCK:
                    cached = _PDFJS_HTTP_ASSET_CACHE.get(name)
                if cached is None or cached[0] != mtime_ns:
                    payload = asset.read_bytes()
                    cached = (mtime_ns, payload)
                    with _PDFJS_HTTP_LOCK:
                        _PDFJS_HTTP_ASSET_CACHE[name] = cached
            except Exception:
                self.send_error(404)
                return
            payload = cached[1]
            content_type = (
                "text/css" if name.endswith(".css") else "application/javascript"
            )
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self._send_security_headers()
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if send_body:
                try:
                    self.wfile.write(payload)
                except Exception:
                    return
            return

        if path.startswith("/viewer/"):
            token = unquote(path[len("/viewer/") :]).strip().split("/", 1)[0]
            with _PDFJS_HTTP_LOCK:
                html = _PDFJS_HTTP_VIEWERS.get(token)
            if html is None:
                self.send_error(404)
                return
            payload = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self._send_security_headers()
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
        with _PDFJS_HTTP_LOCK:
            file_path = _PDFJS_HTTP_TOKENS.get(token)
        if file_path is None or not file_path.exists():
            self.send_error(404)
            return
        try:
            size = int(file_path.stat().st_size)
        except Exception:
            self.send_error(404)
            return

        if size <= 0:
            self.send_response(200)
            self.send_header("Content-Type", "application/pdf")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Cache-Control", "no-store")
            self._send_security_headers()
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        status, start, end, length = _resolve_pdf_byte_range(
            size, self.headers.get("Range", "")
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/pdf")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        self._send_security_headers()
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


def _ensure_pdfjs_http_server() -> str:
    global _PDFJS_HTTP_SERVER, _PDFJS_HTTP_PORT, _PDFJS_HTTP_THREAD
    with _PDFJS_HTTP_LOCK:
        if _PDFJS_HTTP_SERVER is not None and _PDFJS_HTTP_PORT is not None:
            return f"http://127.0.0.1:{_PDFJS_HTTP_PORT}"

        httpd = ThreadingHTTPServer(("127.0.0.1", 0), _PdfJsHttpRequestHandler)
        _PDFJS_HTTP_SERVER = httpd
        _PDFJS_HTTP_PORT = int(getattr(httpd, "server_port", 0) or 0)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        _PDFJS_HTTP_THREAD = thread
        logger.info(f"PDF.js local HTTP server started on 127.0.0.1:{_PDFJS_HTTP_PORT}")
        return f"http://127.0.0.1:{_PDFJS_HTTP_PORT}"


def _register_pdfjs_http_pdf(path: Path) -> str:
    base = _ensure_pdfjs_http_server()
    token = _store_pdfjs_http_token(path)
    logger.debug(f"PDF.js serving {path} via token {token}")
    return f"{base}/pdf/{token}"


def _register_pdfjs_http_viewer_html(html: str) -> str:
    base = _ensure_pdfjs_http_server()
    token = _store_pdfjs_http_viewer_html(html)
    return f"{base}/viewer/{token}"


__all__ = [
    "_ALLOWED_PDFJS_ASSETS",
    "_MAX_HTTP_PDF_TOKENS",
    "_MAX_HTTP_VIEWER_TOKENS",
    "_PdfJsHttpRequestHandler",
    "_ensure_pdfjs_http_server",
    "_pdfjs_asset_path",
    "_register_pdfjs_http_pdf",
    "_register_pdfjs_http_viewer_html",
    "_resolve_pdf_byte_range",
    "_store_pdfjs_http_token",
    "_store_pdfjs_http_viewer_html",
]
