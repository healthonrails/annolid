"""Network target guards for agent tools."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
import ipaddress
import re
import socket
import weakref
from contextlib import contextmanager
from typing import Any, Iterator
from urllib.parse import urlparse

try:
    import httpx as _httpx
except ImportError:  # pragma: no cover - web tooling is optional
    _httpx = None

try:
    import httpcore as _httpcore
except ImportError:  # pragma: no cover - web tooling is optional
    _httpcore = None

_BLOCKED_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "::/128",
        "::1/128",
        "fc00::/7",
        "fe80::/10",
    )
)
_URL_RE = re.compile(r"https?://[^\s\"'`;|<>]+", re.IGNORECASE)
_UNSAFE_URL_CHARACTER_RE = re.compile(r"[\x00-\x20\x7f]")
_MAX_AGENT_URL_LENGTH = 8192


def _normalize_address(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def _is_blocked_address(address: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    normalized = _normalize_address(address)
    if not normalized.is_global:
        return True
    return any(normalized in network for network in _BLOCKED_NETWORKS)


def validate_http_url_shape(url: str) -> tuple[bool, str]:
    """Validate the syntax of an agent-controlled HTTP URL without DNS access."""
    raw_url = str(url or "").strip()
    if not raw_url:
        return False, "URL is required"
    if len(raw_url) > _MAX_AGENT_URL_LENGTH:
        return False, f"URL exceeds maximum length ({_MAX_AGENT_URL_LENGTH})"
    if _UNSAFE_URL_CHARACTER_RE.search(raw_url) or "\\" in raw_url:
        return False, "URL contains unsafe whitespace or control characters"
    try:
        parsed = urlparse(raw_url)
    except Exception:
        return False, "Invalid URL"

    if parsed.scheme.lower() not in {"http", "https"}:
        return (
            False,
            f"Only http/https allowed, got '{parsed.scheme or 'none'}'",
        )
    if not parsed.netloc:
        return False, "Missing domain"
    if parsed.username is not None or parsed.password is not None:
        return False, "Embedded URL credentials are not allowed"
    hostname = str(parsed.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        return False, "Missing hostname"
    try:
        port = parsed.port
    except ValueError:
        return False, "Invalid URL port"
    if port is not None and not 1 <= port <= 65535:
        return False, "Invalid URL port"

    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and _is_blocked_address(address):
        return False, f"Blocked private or internal address: {address}"
    return True, ""


def resolve_public_url_target(
    url: str,
) -> tuple[bool, str, tuple[str, ...]]:
    """Validate a public URL and return the exact public IPs that were checked.

    This blocks local, private, link-local, and cloud metadata targets even when
    they are hidden behind a hostname.
    """
    raw_url = str(url or "").strip()
    ok, error = validate_http_url_shape(raw_url)
    if not ok:
        return False, error, ()
    parsed = urlparse(raw_url)
    hostname = str(parsed.hostname or "").strip().lower().rstrip(".")

    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and _is_blocked_address(address):
        return False, f"Blocked private or internal address: {address}", ()

    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except OSError:
        return False, f"Cannot resolve hostname: {hostname}", ()

    resolved_addresses: list[str] = []
    for info in infos:
        try:
            resolved = ipaddress.ip_address(info[4][0])
        except (IndexError, ValueError):
            continue
        if _is_blocked_address(resolved):
            return (
                False,
                f"Blocked private or internal resolved address: {resolved}",
                (),
            )
        normalized = str(_normalize_address(resolved))
        if normalized not in resolved_addresses:
            resolved_addresses.append(normalized)
    if not resolved_addresses:
        return False, f"Hostname did not resolve to a usable address: {hostname}", ()
    return True, "", tuple(resolved_addresses)


def validate_public_url_target(url: str) -> tuple[bool, str]:
    """Validate a URL before tool-initiated network access."""
    ok, error, _ = resolve_public_url_target(url)
    return ok, error


@contextmanager
def pin_public_url_dns(
    url: str,
    resolved_ips: tuple[str, ...],
) -> Iterator[None]:
    """Pin the URL hostname to the public IPs validated immediately beforehand.

    The process-global resolver override is only safe while serialized. HTTP
    clients should use :class:`PinnedPublicAsyncTransport`, which owns that
    serialization.
    """
    hostname = str(urlparse(str(url or "")).hostname or "").rstrip(".").lower()
    if not hostname or not resolved_ips:
        yield
        return

    original_getaddrinfo = socket.getaddrinfo

    def _pinned_getaddrinfo(
        host: object,
        port: object,
        family: int = 0,
        type: int = 0,  # noqa: A002
        proto: int = 0,
        flags: int = 0,
    ) -> list[tuple[object, ...]]:
        if str(host).rstrip(".").lower() != hostname:
            return original_getaddrinfo(host, port, family, type, proto, flags)
        pinned: list[tuple[object, ...]] = []
        for raw_ip in resolved_ips:
            address = ipaddress.ip_address(raw_ip)
            address_family = socket.AF_INET6 if address.version == 6 else socket.AF_INET
            if family not in (0, socket.AF_UNSPEC, address_family):
                continue
            socket_address: tuple[object, ...]
            if address_family == socket.AF_INET6:
                socket_address = (raw_ip, port or 0, 0, 0)
            else:
                socket_address = (raw_ip, port or 0)
            pinned.append(
                (
                    address_family,
                    type or socket.SOCK_STREAM,
                    proto,
                    "",
                    socket_address,
                )
            )
        return pinned

    socket.getaddrinfo = _pinned_getaddrinfo
    try:
        yield
    finally:
        socket.getaddrinfo = original_getaddrinfo


_PINNED_HOST_IPS: ContextVar[dict[str, tuple[str, ...]]] = ContextVar(
    "annolid_pinned_public_host_ips",
    default={},
)
_LEGACY_PINNED_DNS_LOCKS: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = weakref.WeakKeyDictionary()


def _get_legacy_pinned_dns_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    lock = _LEGACY_PINNED_DNS_LOCKS.get(loop)
    if lock is None:
        lock = asyncio.Lock()
        _LEGACY_PINNED_DNS_LOCKS[loop] = lock
    return lock


def _resolve_public_host(host: str) -> tuple[bool, str, tuple[str, ...]]:
    """Resolve and validate a bare hostname or IP literal to public IP(s)."""
    normalized_host = str(host or "").strip().lower().rstrip(".")
    if not normalized_host:
        return False, "Missing hostname", ()
    # Strip optional IPv6 brackets.
    if normalized_host.startswith("[") and normalized_host.endswith("]"):
        normalized_host = normalized_host[1:-1]
    try:
        address = ipaddress.ip_address(normalized_host)
    except ValueError:
        address = None
    if address is not None:
        if _is_blocked_address(address):
            return False, f"Blocked private or internal address: {address}", ()
        return True, "", (str(_normalize_address(address)),)

    try:
        infos = socket.getaddrinfo(
            normalized_host, None, socket.AF_UNSPEC, socket.SOCK_STREAM
        )
    except OSError:
        return False, f"Cannot resolve hostname: {normalized_host}", ()

    resolved_addresses: list[str] = []
    for info in infos:
        try:
            resolved = ipaddress.ip_address(info[4][0])
        except (IndexError, ValueError):
            continue
        if _is_blocked_address(resolved):
            return (
                False,
                f"Blocked private or internal resolved address: {resolved}",
                (),
            )
        normalized = str(_normalize_address(resolved))
        if normalized not in resolved_addresses:
            resolved_addresses.append(normalized)
    if not resolved_addresses:
        return (
            False,
            f"Hostname did not resolve to a usable address: {normalized_host}",
            (),
        )
    return True, "", tuple(resolved_addresses)


if _httpcore is not None:

    class PinnedPublicAsyncNetworkBackend(_httpcore.AsyncNetworkBackend):
        """HTTPCore network backend that dials pre-validated public IPs directly.

        Because ``connect_tcp`` receives the request hostname while
        ``AsyncHTTPConnection`` separately passes the original hostname as
        ``server_hostname`` to ``start_tls``, dialing the validated IP here
        prevents DNS rebinding without mutating process-global
        ``socket.getaddrinfo`` and preserves TLS SNI + certificate validation.
        """

        def __init__(self, inner: Any | None = None) -> None:
            self._inner = inner or _httpcore.AnyIOBackend()

        async def connect_tcp(
            self,
            host: str,
            port: int,
            timeout: float | None = None,
            local_address: str | None = None,
            socket_options: Any | None = None,
        ) -> Any:
            host_key = str(host or "").strip().lower().rstrip(".")
            if host_key.startswith("[") and host_key.endswith("]"):
                host_key = host_key[1:-1]
            pinned_map = _PINNED_HOST_IPS.get()
            candidate_ips = pinned_map.get(host_key)
            if not candidate_ips:
                ok, error, candidate_ips = _resolve_public_host(host_key)
                if not ok:
                    raise _httpcore.ConnectError(error)

            last_exc: Exception | None = None
            for ip in candidate_ips:
                try:
                    address = ipaddress.ip_address(ip)
                except ValueError as exc:
                    last_exc = _httpcore.ConnectError(
                        f"Invalid pinned IP address: {ip}"
                    )
                    last_exc.__cause__ = exc
                    continue
                if _is_blocked_address(address):
                    raise _httpcore.ConnectError(
                        f"Blocked private or internal address: {address}"
                    )
                try:
                    return await self._inner.connect_tcp(
                        str(_normalize_address(address)),
                        port,
                        timeout=timeout,
                        local_address=local_address,
                        socket_options=socket_options,
                    )
                except (
                    _httpcore.ConnectError,
                    _httpcore.ConnectTimeout,
                    OSError,
                ) as exc:
                    last_exc = exc
            if last_exc is not None:
                raise last_exc
            raise _httpcore.ConnectError(
                f"No validated public IP addresses available for host: {host}"
            )

        async def connect_unix_socket(
            self,
            path: str,
            timeout: float | None = None,
            socket_options: Any | None = None,
        ) -> Any:
            del path, timeout, socket_options
            raise _httpcore.ConnectError(
                "Unix domain sockets are not allowed for public HTTP requests."
            )

        async def sleep(self, seconds: float) -> None:
            await self._inner.sleep(seconds)

else:

    class PinnedPublicAsyncNetworkBackend:  # pragma: no cover - optional dependency
        def __init__(self, inner: Any | None = None) -> None:
            del inner
            raise RuntimeError(
                "Pinned HTTP network backend requires the `httpcore` package."
            )


if _httpx is not None:

    class PinnedPublicAsyncTransport(_httpx.AsyncBaseTransport):
        """HTTPX transport that connects using the IPs validated for each URL.

        When using the default transport pool, connections dial the validated IP
        directly through :class:`PinnedPublicAsyncNetworkBackend` with per-task
        ``ContextVar`` isolation (no process-global ``socket.getaddrinfo``
        monkeypatch and full concurrency). Custom ``inner`` transports remain
        supported for tests and wrappers.
        """

        def __init__(
            self,
            inner: object | None = None,
            *,
            network_backend: Any | None = None,
        ) -> None:
            self._network_backend: PinnedPublicAsyncNetworkBackend | None = None
            if inner is None and _httpcore is not None:
                transport = _httpx.AsyncHTTPTransport(trust_env=False)
                base_pool = transport._pool
                pinned_backend = PinnedPublicAsyncNetworkBackend(
                    network_backend or getattr(base_pool, "_network_backend", None)
                )
                transport._pool = _httpcore.AsyncConnectionPool(
                    ssl_context=getattr(base_pool, "_ssl_context", None),
                    max_connections=getattr(base_pool, "_max_connections", None),
                    max_keepalive_connections=getattr(
                        base_pool, "_max_keepalive_connections", None
                    ),
                    keepalive_expiry=getattr(base_pool, "_keepalive_expiry", None),
                    http1=bool(getattr(base_pool, "_http1", True)),
                    http2=bool(getattr(base_pool, "_http2", False)),
                    retries=int(getattr(base_pool, "_retries", 0) or 0),
                    network_backend=pinned_backend,
                )
                self._inner = transport
                self._network_backend = pinned_backend
                self._uses_network_backend = True
            else:
                self._inner = inner or _httpx.AsyncHTTPTransport(trust_env=False)
                self._uses_network_backend = False

        async def handle_async_request(self, request: object) -> object:
            url = str(getattr(request, "url", "") or "")
            ok, error, resolved_ips = resolve_public_url_target(url)
            if not ok:
                raise _httpx.RequestError(error, request=request)
            hostname = str(urlparse(url).hostname or "").strip().lower().rstrip(".")
            if hostname.startswith("[") and hostname.endswith("]"):
                hostname = hostname[1:-1]
            updated_map = dict(_PINNED_HOST_IPS.get())
            if hostname:
                updated_map[hostname] = resolved_ips
            token = _PINNED_HOST_IPS.set(updated_map)
            try:
                if self._uses_network_backend:
                    return await self._inner.handle_async_request(request)
                # Backward-compatible shim for caller-provided custom inner transports.
                async with _get_legacy_pinned_dns_lock():
                    with pin_public_url_dns(url, resolved_ips):
                        return await self._inner.handle_async_request(request)
            finally:
                _PINNED_HOST_IPS.reset(token)

        async def aclose(self) -> None:
            await self._inner.aclose()

else:

    class PinnedPublicAsyncTransport:  # pragma: no cover - optional dependency
        def __init__(
            self,
            inner: object | None = None,
            *,
            network_backend: Any | None = None,
        ) -> None:
            del inner, network_backend
            raise RuntimeError("Pinned HTTP transport requires the `httpx` package.")


def contains_private_url_target(text: str) -> tuple[bool, str]:
    """Return whether text embeds an http(s) URL targeting a private network."""
    for match in _URL_RE.finditer(str(text or "")):
        url = match.group(0)
        ok, error = validate_public_url_target(url)
        if not ok:
            return True, error
    return False, ""


async def guard_public_httpx_request(request: object) -> None:
    """Reject an httpx request before it follows an unsafe redirect target."""
    url = str(getattr(request, "url", "") or "")
    ok, error = validate_public_url_target(url)
    if not ok:
        raise ValueError(f"Blocked private or internal request target: {error}")


def public_httpx_event_hooks() -> dict[str, list[object]]:
    """Return redirect-aware httpx hooks for agent-controlled requests."""
    return {"request": [guard_public_httpx_request]}


def public_httpx_client_kwargs() -> dict[str, object]:
    """Return fail-closed HTTPX settings for public agent-controlled requests."""
    return {
        "event_hooks": public_httpx_event_hooks(),
        "transport": PinnedPublicAsyncTransport(),
        # Environment proxies resolve the destination outside this process, so
        # the validated address cannot be pinned to the actual connection.
        "trust_env": False,
    }


__all__ = [
    "contains_private_url_target",
    "guard_public_httpx_request",
    "pin_public_url_dns",
    "PinnedPublicAsyncNetworkBackend",
    "PinnedPublicAsyncTransport",
    "public_httpx_client_kwargs",
    "public_httpx_event_hooks",
    "resolve_public_url_target",
    "validate_http_url_shape",
    "validate_public_url_target",
]
