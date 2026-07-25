from __future__ import annotations

import asyncio
import socket

import pytest

from annolid.core.agent.security_network import (
    PinnedPublicAsyncTransport,
    contains_private_url_target,
    guard_public_httpx_request,
    pin_public_url_dns,
    public_httpx_client_kwargs,
    public_httpx_event_hooks,
    resolve_public_url_target,
    validate_public_url_target,
)


def _fake_resolve(host: str, results: list[str]):
    def _resolver(hostname, port, family=0, socktype=0, proto=0, flags=0):
        del port, family, socktype, proto, flags
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


def test_httpx_request_hook_blocks_private_redirect_target(monkeypatch) -> None:
    monkeypatch.setattr(
        "annolid.core.agent.security_network.socket.getaddrinfo",
        _fake_resolve("redirect.example", ["127.0.0.1"]),
    )

    class _Request:
        url = "http://redirect.example/admin"

    with pytest.raises(ValueError, match="private or internal request target"):
        asyncio.run(guard_public_httpx_request(_Request()))


def test_httpx_event_hooks_install_request_guard() -> None:
    hooks = public_httpx_event_hooks()

    assert hooks == {"request": [guard_public_httpx_request]}


def test_resolve_public_url_target_returns_validated_addresses(monkeypatch) -> None:
    monkeypatch.setattr(
        "annolid.core.agent.security_network.socket.getaddrinfo",
        _fake_resolve("public.example", ["93.184.216.34", "2606:2800:220:1::1"]),
    )

    ok, error, addresses = resolve_public_url_target("https://public.example/resource")

    assert ok is True
    assert error == ""
    assert addresses == ("93.184.216.34", "2606:2800:220:1::1")


def test_pin_public_url_dns_prevents_second_resolution_rebind(monkeypatch) -> None:
    calls = 0

    def _rebinding_resolver(
        hostname,
        port,
        family=0,
        socktype=0,
        proto=0,
        flags=0,
    ):
        del hostname, port, family, socktype, proto, flags
        nonlocal calls
        calls += 1
        address = "93.184.216.34" if calls == 1 else "169.254.169.254"
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (address, 0))]

    monkeypatch.setattr(
        "annolid.core.agent.security_network.socket.getaddrinfo",
        _rebinding_resolver,
    )
    ok, error, addresses = resolve_public_url_target("https://rebind.example/resource")
    assert ok is True, error

    with pin_public_url_dns("https://rebind.example/resource", addresses):
        infos = socket.getaddrinfo(
            "rebind.example",
            443,
            socket.AF_UNSPEC,
            socket.SOCK_STREAM,
        )

    assert infos[0][4][0] == "93.184.216.34"
    assert calls == 1


def test_pinned_public_transport_connects_with_validated_address(
    monkeypatch,
) -> None:
    import httpx

    calls = 0

    def _rebinding_resolver(
        hostname,
        port,
        family=0,
        socktype=0,
        proto=0,
        flags=0,
    ):
        del hostname, port, family, socktype, proto, flags
        nonlocal calls
        calls += 1
        address = "93.184.216.34" if calls == 1 else "127.0.0.1"
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, "", (address, 0))]

    class _ResolvingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            infos = socket.getaddrinfo(
                request.url.host,
                request.url.port,
                socket.AF_UNSPEC,
                socket.SOCK_STREAM,
            )
            assert infos[0][4][0] == "93.184.216.34"
            return httpx.Response(200, request=request, text="ok")

    monkeypatch.setattr(
        "annolid.core.agent.security_network.socket.getaddrinfo",
        _rebinding_resolver,
    )

    async def _run() -> None:
        async with httpx.AsyncClient(
            transport=PinnedPublicAsyncTransport(inner=_ResolvingTransport())
        ) as client:
            response = await client.get("https://rebind.example/resource")
        assert response.text == "ok"

    asyncio.run(_run())
    assert calls == 1


def test_public_httpx_client_kwargs_disable_environment_proxies() -> None:
    kwargs = public_httpx_client_kwargs()

    assert kwargs["trust_env"] is False
    assert isinstance(kwargs["transport"], PinnedPublicAsyncTransport)
    assert kwargs["event_hooks"] == {"request": [guard_public_httpx_request]}
    asyncio.run(kwargs["transport"].aclose())


def test_httpx_request_hook_runs_before_following_redirect(monkeypatch) -> None:
    import httpx

    monkeypatch.setattr(
        "annolid.core.agent.security_network.socket.getaddrinfo",
        _fake_resolve("public.example", ["93.184.216.34"]),
    )
    requested_urls: list[str] = []

    def _transport(request):
        requested_urls.append(str(request.url))
        return httpx.Response(
            302,
            headers={"Location": "http://169.254.169.254/latest/meta-data/"},
        )

    async def _run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(_transport),
            follow_redirects=True,
            event_hooks=public_httpx_event_hooks(),
        ) as client:
            await client.get("https://public.example/start")

    with pytest.raises(ValueError, match="private or internal request target"):
        asyncio.run(_run())

    assert requested_urls == ["https://public.example/start"]
