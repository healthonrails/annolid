"""Hardened stdlib HTTP helpers for URL validation, safe redirects, and bounded reads.

This module uses only the Python standard library so it remains safe to import
from lean ``annolid[gui]`` and frozen desktop entry points.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Collection

_ORIGINAL_URLOPEN = urllib.request.urlopen

_UNSAFE_URL_CHAR_RE = re.compile(r"[\x00-\x20\x7f]")
_MAX_URL_LENGTH = 8192

_BLOCKED_NETWORKS = tuple(
    ipaddress.ip_network(cidr)
    for cidr in (
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

_SENSITIVE_REDIRECT_HEADERS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "cookie",
        "x-colab-runtime-proxy-token",
    }
)


def _normalize_ip(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        return address.ipv4_mapped
    return address


def is_loopback_host(hostname: str) -> bool:
    """Return whether ``hostname`` refers to a local loopback interface."""
    normalized = str(hostname or "").strip().lower().rstrip(".")
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    if not normalized:
        return False
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return True
    try:
        addr = _normalize_ip(ipaddress.ip_address(normalized))
    except ValueError:
        return False
    return bool(addr.is_loopback)


def _is_blocked_ip(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
    *,
    allow_loopback: bool = False,
) -> bool:
    normalized = _normalize_ip(address)
    if allow_loopback and normalized.is_loopback:
        return False
    if not normalized.is_global:
        return True
    return any(normalized in net for net in _BLOCKED_NETWORKS)


def _url_origin(url: str) -> tuple[str, str, int]:
    parsed = urllib.parse.urlparse(url)
    scheme = str(parsed.scheme or "").lower()
    host = str(parsed.hostname or "").lower().rstrip(".")
    port = parsed.port
    if port is None:
        port = 443 if scheme == "https" else 80
    return (scheme, host, int(port))


def validate_http_url(
    url: str,
    *,
    allowed_schemes: Collection[str] = ("http", "https"),
    allowed_hosts: Collection[str] | None = None,
    allow_private: bool = False,
    allow_loopback: bool = False,
    allow_http_only_for_loopback: bool = False,
    resolve_dns: bool = False,
) -> str:
    """Validate an HTTP(S) URL against scheme, credential, host, and IP policies."""
    raw = str(url or "").strip()
    if not raw:
        raise ValueError("URL is required.")
    if len(raw) > _MAX_URL_LENGTH:
        raise ValueError(f"URL exceeds maximum length ({_MAX_URL_LENGTH}).")
    if _UNSAFE_URL_CHAR_RE.search(raw) or "\\" in raw:
        raise ValueError("URL contains unsafe whitespace or control characters.")

    parsed = urllib.parse.urlparse(raw)
    scheme = str(parsed.scheme or "").strip().lower()
    normalized_schemes = {str(s).strip().lower() for s in allowed_schemes}
    if scheme not in normalized_schemes or not parsed.netloc:
        raise ValueError(f"Unsupported URL scheme or host: {scheme or 'none'!r}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Embedded URL credentials are not allowed.")

    hostname = str(parsed.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        raise ValueError("URL is missing a hostname.")

    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("URL contains an invalid port.") from exc
    if port is not None and not (1 <= port <= 65535):
        raise ValueError("URL port is out of valid range.")

    if (
        allow_http_only_for_loopback
        and scheme == "http"
        and not is_loopback_host(hostname)
    ):
        raise ValueError(
            f"Plaintext http:// is only allowed for loopback hosts, not {hostname!r}."
        )

    if allowed_hosts is not None:
        allowed_set = {str(h).strip().lower().rstrip(".") for h in allowed_hosts}
        if hostname not in allowed_set:
            raise ValueError(f"Untrusted host: {hostname!r}")

    if not allow_private:
        if not allow_loopback and (
            hostname == "localhost" or hostname.endswith(".localhost")
        ):
            raise ValueError(f"Blocked local hostname: {hostname!r}")
        literal_ip: ipaddress.IPv4Address | ipaddress.IPv6Address | None
        try:
            literal_ip = ipaddress.ip_address(hostname)
        except ValueError:
            literal_ip = None
        if literal_ip is not None:
            if _is_blocked_ip(literal_ip, allow_loopback=allow_loopback):
                raise ValueError(
                    f"Blocked private or internal IP address: {literal_ip}"
                )
        elif resolve_dns:
            try:
                infos = socket.getaddrinfo(
                    hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM
                )
            except OSError as exc:
                raise ValueError(f"Cannot resolve hostname: {hostname!r}") from exc
            resolved_any = False
            for info in infos:
                try:
                    addr = ipaddress.ip_address(info[4][0])
                except (IndexError, ValueError):
                    continue
                resolved_any = True
                if _is_blocked_ip(addr, allow_loopback=allow_loopback):
                    raise ValueError(
                        f"Blocked private or internal resolved address: {addr}"
                    )
            if not resolved_any:
                raise ValueError(
                    f"Hostname did not resolve to any valid IP address: {hostname!r}"
                )

    return raw


class _SafeHTTPRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirect handler that re-validates every hop and strips cross-origin auth."""

    def __init__(
        self,
        *,
        allowed_schemes: Collection[str],
        allowed_hosts: Collection[str] | None,
        allow_private: bool,
        allow_loopback: bool,
        allow_http_only_for_loopback: bool,
        resolve_dns: bool,
        max_redirects: int,
    ) -> None:
        super().__init__()
        self._allowed_schemes = tuple(allowed_schemes)
        self._allowed_hosts = (
            frozenset(allowed_hosts) if allowed_hosts is not None else None
        )
        self._allow_private = bool(allow_private)
        self._allow_loopback = bool(allow_loopback)
        self._allow_http_only_for_loopback = bool(allow_http_only_for_loopback)
        self._resolve_dns = bool(resolve_dns)
        self._max_redirects = max(0, int(max_redirects))

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        hops = int(getattr(req, "_annolid_redirect_hops", 0) or 0) + 1
        if hops > self._max_redirects:
            raise urllib.error.HTTPError(
                newurl,
                code,
                f"Too many redirects (max {self._max_redirects})",
                headers,
                fp,
            )

        resolved_target = urllib.parse.urljoin(req.full_url, str(newurl or "").strip())
        validate_http_url(
            resolved_target,
            allowed_schemes=self._allowed_schemes,
            allowed_hosts=self._allowed_hosts,
            allow_private=self._allow_private,
            allow_loopback=self._allow_loopback,
            allow_http_only_for_loopback=self._allow_http_only_for_loopback,
            resolve_dns=self._resolve_dns,
        )

        redirected = super().redirect_request(
            req, fp, code, msg, headers, resolved_target
        )
        if redirected is None:
            return None

        setattr(redirected, "_annolid_redirect_hops", hops)
        if _url_origin(req.full_url) != _url_origin(resolved_target):
            for header_name in list(redirected.headers.keys()):
                if header_name.lower() in _SENSITIVE_REDIRECT_HEADERS:
                    del redirected.headers[header_name]
            for header_name in list(redirected.unredirected_hdrs.keys()):
                if header_name.lower() in _SENSITIVE_REDIRECT_HEADERS:
                    del redirected.unredirected_hdrs[header_name]
        return redirected


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirect handler that refuses all HTTP 3xx redirects."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        del req
        raise urllib.error.HTTPError(
            newurl,
            code,
            f"HTTP redirects are disabled ({msg})",
            headers,
            fp,
        )


def _build_safe_opener(
    *,
    allowed_schemes: Collection[str],
    allowed_hosts: Collection[str] | None,
    allow_private: bool,
    allow_loopback: bool,
    allow_http_only_for_loopback: bool,
    allow_redirects: bool,
    max_redirects: int,
    resolve_dns: bool,
) -> urllib.request.OpenerDirector:
    """Build an OpenerDirector without FileHandler, FTPHandler, or DataHandler."""
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.ProxyHandler())
    opener.add_handler(urllib.request.UnknownHandler())
    opener.add_handler(urllib.request.HTTPHandler())
    opener.add_handler(urllib.request.HTTPSHandler())
    opener.add_handler(urllib.request.HTTPDefaultErrorHandler())
    if allow_redirects:
        opener.add_handler(
            _SafeHTTPRedirectHandler(
                allowed_schemes=allowed_schemes,
                allowed_hosts=allowed_hosts,
                allow_private=allow_private,
                allow_loopback=allow_loopback,
                allow_http_only_for_loopback=allow_http_only_for_loopback,
                resolve_dns=resolve_dns,
                max_redirects=max_redirects,
            )
        )
    else:
        opener.add_handler(_NoRedirectHandler())
    opener.add_handler(urllib.request.HTTPErrorProcessor())
    return opener


def build_http_request(
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    method: str | None = None,
    allowed_schemes: Collection[str] = ("http", "https"),
    allowed_hosts: Collection[str] | None = None,
    allow_private: bool = False,
    allow_loopback: bool = False,
    allow_http_only_for_loopback: bool = False,
) -> urllib.request.Request:
    """Validate ``url`` shape/scheme/host/IP and construct a ``urllib`` Request."""
    safe_url = validate_http_url(
        url,
        allowed_schemes=allowed_schemes,
        allowed_hosts=allowed_hosts,
        allow_private=allow_private,
        allow_loopback=allow_loopback,
        allow_http_only_for_loopback=allow_http_only_for_loopback,
        resolve_dns=False,
    )
    return urllib.request.Request(  # noqa: S310 - URL validated above
        safe_url,
        data=data,
        headers=dict(headers or {}),
        method=method,
    )


def safe_urlopen(
    request_or_url: urllib.request.Request | str,
    *,
    timeout: float = 30.0,
    allowed_schemes: Collection[str] = ("http", "https"),
    allowed_hosts: Collection[str] | None = None,
    allow_private: bool = False,
    allow_loopback: bool = False,
    allow_http_only_for_loopback: bool = False,
    allow_redirects: bool = True,
    max_redirects: int = 5,
    resolve_dns: bool = True,
    urlopen_func: Callable[..., Any] | None = None,
) -> Any:
    """Open an HTTP(S) URL safely with scheme, host, IP, and redirect checks.

    If ``urllib.request.urlopen`` has been monkeypatched (for example in unit
    tests), URL shape/scheme/host/literal-IP validation still runs before
    delegating to the patched callable.
    """
    active_urlopen = (
        urlopen_func if urlopen_func is not None else urllib.request.urlopen
    )
    is_mocked = active_urlopen is not _ORIGINAL_URLOPEN

    url = (
        request_or_url.full_url
        if isinstance(request_or_url, urllib.request.Request)
        else str(request_or_url)
    )
    validate_http_url(
        url,
        allowed_schemes=allowed_schemes,
        allowed_hosts=allowed_hosts,
        allow_private=allow_private,
        allow_loopback=allow_loopback,
        allow_http_only_for_loopback=allow_http_only_for_loopback,
        resolve_dns=(resolve_dns and not is_mocked),
    )

    if is_mocked:
        return active_urlopen(request_or_url, timeout=timeout)

    opener = _build_safe_opener(
        allowed_schemes=allowed_schemes,
        allowed_hosts=allowed_hosts,
        allow_private=allow_private,
        allow_loopback=allow_loopback,
        allow_http_only_for_loopback=allow_http_only_for_loopback,
        allow_redirects=allow_redirects,
        max_redirects=max_redirects,
        resolve_dns=resolve_dns,
    )
    return opener.open(request_or_url, timeout=timeout)  # noqa: S310 - validated opener


def read_bounded(
    stream: Any,
    max_bytes: int,
    *,
    label: str = "HTTP response",
) -> bytes:
    """Read at most ``max_bytes`` from a stream, raising ValueError if exceeded."""
    limit = max(1, int(max_bytes))
    try:
        data = stream.read(limit + 1)
    except TypeError:
        # Compatibility fallback for minimal unit-test fakes implementing read().
        data = stream.read()
    raw = bytes(data or b"")
    if len(raw) > limit:
        raise ValueError(f"{label} exceeded {limit} byte limit.")
    return raw


__all__ = [
    "build_http_request",
    "is_loopback_host",
    "read_bounded",
    "safe_urlopen",
    "validate_http_url",
]
