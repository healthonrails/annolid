from __future__ import annotations

import asyncio
import socket

import pytest

from annolid.core.agent.security_network import (
    PinnedPublicAsyncNetworkBackend,
    PinnedPublicAsyncTransport,
    contains_private_url_target,
    guard_public_httpx_request,
    pin_public_url_dns,
    public_httpx_client_kwargs,
    public_httpx_event_hooks,
    resolve_public_url_target,
    validate_http_url_shape,
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


@pytest.mark.parametrize(
    ("url", "error_fragment"),
    [
        ("https://user:secret@example.org/file", "credentials"),
        ("https://example.org\\@127.0.0.1/file", "unsafe"),
        ("https://example.org/\nHost: internal", "unsafe"),
        ("https://example.org:99999/file", "port"),
        ("https://example.org/" + ("a" * 8192), "maximum length"),
    ],
    ids=["credentials", "backslash", "control-character", "port", "length"],
)
def test_validate_http_url_shape_rejects_ambiguous_or_sensitive_urls(
    url: str,
    error_fragment: str,
) -> None:
    ok, error = validate_http_url_shape(url)

    assert ok is False
    assert error_fragment in error.lower()
    assert "secret" not in error


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
    httpx = pytest.importorskip("httpx")

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
    pytest.importorskip("httpx")
    kwargs = public_httpx_client_kwargs()

    assert kwargs["trust_env"] is False
    assert isinstance(kwargs["transport"], PinnedPublicAsyncTransport)
    assert kwargs["event_hooks"] == {"request": [guard_public_httpx_request]}
    asyncio.run(kwargs["transport"].aclose())


def test_httpx_request_hook_runs_before_following_redirect(monkeypatch) -> None:
    httpx = pytest.importorskip("httpx")

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


def test_pinned_network_backend_dials_validated_ip_without_mutating_socket_getaddrinfo(
    monkeypatch,
) -> None:
    httpx = pytest.importorskip("httpx")
    httpcore = pytest.importorskip("httpcore")

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
    dialed_targets: list[tuple[str, int]] = []
    getaddrinfo_was_unmodified: list[bool] = []
    tls_server_hostnames: list[str] = []

    class _FakeStream(httpcore.AsyncNetworkStream):
        def __init__(self) -> None:
            self._sent_response = False

        async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
            del max_bytes, timeout
            if not self._sent_response:
                self._sent_response = True
                return (
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Length: 2\r\n"
                    b"Connection: close\r\n\r\nok"
                )
            return b""

        async def write(self, buffer: bytes, timeout: float | None = None) -> None:
            del buffer, timeout

        async def aclose(self) -> None:
            return None

        async def start_tls(
            self,
            ssl_context,
            server_hostname: str | None = None,
            timeout: float | None = None,
        ):
            del ssl_context, timeout
            tls_server_hostnames.append(str(server_hostname or ""))
            return self

        def get_extra_info(self, info: str):
            del info
            return None

    class _RecordingBackend(httpcore.AsyncNetworkBackend):
        async def connect_tcp(
            self,
            host: str,
            port: int,
            timeout: float | None = None,
            local_address: str | None = None,
            socket_options=None,
        ):
            del timeout, local_address, socket_options
            dialed_targets.append((host, port))
            getaddrinfo_was_unmodified.append(socket.getaddrinfo is _rebinding_resolver)
            return _FakeStream()

        async def connect_unix_socket(
            self, path: str, timeout: float | None = None, socket_options=None
        ):
            raise AssertionError("unix socket should not be called")

        async def sleep(self, seconds: float) -> None:
            del seconds

    async def _run() -> None:
        transport = PinnedPublicAsyncTransport(network_backend=_RecordingBackend())
        assert isinstance(transport._network_backend, PinnedPublicAsyncNetworkBackend)
        async with httpx.AsyncClient(transport=transport, trust_env=False) as client:
            response = await client.get("https://rebind.example/resource")
        assert response.status_code == 200
        assert response.text == "ok"

    asyncio.run(_run())
    assert calls == 1
    assert dialed_targets == [("93.184.216.34", 443)]
    assert getaddrinfo_was_unmodified == [True]
    assert tls_server_hostnames == ["rebind.example"]


def test_pinned_network_backend_blocks_private_hosts_and_unix_sockets(
    monkeypatch,
) -> None:
    httpcore = pytest.importorskip("httpcore")
    monkeypatch.setattr(
        "annolid.core.agent.security_network.socket.getaddrinfo",
        _fake_resolve("internal.example", ["127.0.0.1"]),
    )
    backend = PinnedPublicAsyncNetworkBackend()
    with pytest.raises(httpcore.ConnectError, match="private or internal"):
        asyncio.run(backend.connect_tcp("internal.example", 80))
    with pytest.raises(httpcore.ConnectError, match="Unix domain sockets"):
        asyncio.run(backend.connect_unix_socket("/var/run/docker.sock"))


def test_pinned_transport_custom_inner_works_across_multiple_event_loops(
    monkeypatch,
) -> None:
    httpx = pytest.importorskip("httpx")
    monkeypatch.setattr(
        "annolid.core.agent.security_network.socket.getaddrinfo",
        _fake_resolve("public.example", ["93.184.216.34"]),
    )

    class _Inner(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            return httpx.Response(200, request=request, text="loop-ok")

    async def _once() -> str:
        async with httpx.AsyncClient(
            transport=PinnedPublicAsyncTransport(inner=_Inner())
        ) as client:
            return (await client.get("https://public.example/")).text

    assert asyncio.run(_once()) == "loop-ok"
    assert asyncio.run(_once()) == "loop-ok"
