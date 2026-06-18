from __future__ import annotations

import socket

from annolid.core.agent.security_network import (
    contains_private_url_target,
    validate_public_url_target,
)


def _fake_resolve(host: str, results: list[str]):
    def _resolver(hostname, port, family=0, socktype=0):
        del port, family, socktype
        if hostname == host:
            entries = []
            for ip in results:
                if ":" in ip:
                    entries.append(
                        (socket.AF_INET6, socket.SOCK_STREAM, 0, "", (ip, 0, 0, 0))
                    )
                else:
                    entries.append((socket.AF_INET, socket.SOCK_STREAM, 0, "", (ip, 0)))
            return entries
        raise socket.gaierror(f"cannot resolve {hostname}")

    return _resolver


def test_validate_public_url_target_blocks_ipv6_mapped_metadata(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "annolid.core.agent.security_network.socket.getaddrinfo",
        _fake_resolve("evil.example", ["::ffff:169.254.169.254"]),
    )

    ok, err = validate_public_url_target("http://evil.example/latest/meta-data/")

    assert ok is False
    assert "private or internal" in err


def test_validate_public_url_target_allows_public_ipv6_mapped_ipv4(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "annolid.core.agent.security_network.socket.getaddrinfo",
        _fake_resolve("example.org", ["::ffff:93.184.216.34"]),
    )

    ok, err = validate_public_url_target("https://example.org/page")

    assert ok is True
    assert err == ""


def test_contains_private_url_target_detects_private_shell_url() -> None:
    blocked, err = contains_private_url_target(
        "curl http://169.254.169.254/latest/meta-data/"
    )

    assert blocked is True
    assert "169.254.169.254" in err
