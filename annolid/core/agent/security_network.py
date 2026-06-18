"""Network target guards for agent tools."""

from __future__ import annotations

import ipaddress
import re
import socket
from urllib.parse import urlparse

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


def validate_public_url_target(url: str) -> tuple[bool, str]:
    """Validate a URL before tool-initiated network access.

    This blocks local, private, link-local, and cloud metadata targets even when
    they are hidden behind a hostname.
    """
    try:
        parsed = urlparse(str(url or "").strip())
    except Exception as exc:
        return False, str(exc)

    if parsed.scheme not in {"http", "https"}:
        return False, f"Only http/https allowed, got '{parsed.scheme or 'none'}'"
    if not parsed.netloc:
        return False, "Missing domain"
    hostname = str(parsed.hostname or "").strip().lower().rstrip(".")
    if not hostname:
        return False, "Missing hostname"

    try:
        address = ipaddress.ip_address(hostname)
    except ValueError:
        address = None
    if address is not None and _is_blocked_address(address):
        return False, f"Blocked private or internal address: {address}"

    try:
        infos = socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except OSError:
        return False, f"Cannot resolve hostname: {hostname}"

    for info in infos:
        try:
            resolved = ipaddress.ip_address(info[4][0])
        except (IndexError, ValueError):
            continue
        if _is_blocked_address(resolved):
            return (
                False,
                f"Blocked private or internal resolved address: {resolved}",
            )
    return True, ""


def contains_private_url_target(text: str) -> tuple[bool, str]:
    """Return whether text embeds an http(s) URL targeting a private network."""
    for match in _URL_RE.finditer(str(text or "")):
        url = match.group(0)
        ok, error = validate_public_url_target(url)
        if not ok:
            return True, error
    return False, ""


__all__ = ["contains_private_url_target", "validate_public_url_target"]
