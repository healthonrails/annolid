from __future__ import annotations

from pathlib import Path

from annolid.gui.widgets import threejs_viewer_server as srv


def test_prune_threejs_http_tokens_removes_expired_and_limits_size() -> None:
    original_tokens = dict(srv._THREEJS_HTTP_TOKENS)  # noqa: SLF001
    original_created = dict(srv._THREEJS_HTTP_TOKEN_CREATED_AT)  # noqa: SLF001
    try:
        srv._THREEJS_HTTP_TOKENS.clear()  # noqa: SLF001
        srv._THREEJS_HTTP_TOKEN_CREATED_AT.clear()  # noqa: SLF001
        now = 1000.0
        # one expired
        srv._THREEJS_HTTP_TOKENS["old"] = Path("/tmp/a.obj")  # noqa: SLF001
        srv._THREEJS_HTTP_TOKEN_CREATED_AT["old"] = (  # noqa: SLF001
            now - srv._THREEJS_HTTP_TOKEN_TTL_SECONDS - 1  # noqa: SLF001
        )
        # many fresh tokens exceeding cap
        for i in range(srv._THREEJS_HTTP_MAX_TOKENS + 10):  # noqa: SLF001
            token = f"t{i}"
            srv._THREEJS_HTTP_TOKENS[token] = Path("/tmp/a.obj")  # noqa: SLF001
            srv._THREEJS_HTTP_TOKEN_CREATED_AT[token] = now + i  # noqa: SLF001

        srv._prune_threejs_http_tokens(now_ts=now)  # noqa: SLF001
        assert "old" not in srv._THREEJS_HTTP_TOKENS  # noqa: SLF001
        assert len(srv._THREEJS_HTTP_TOKENS) <= srv._THREEJS_HTTP_MAX_TOKENS  # noqa: SLF001
    finally:
        srv._THREEJS_HTTP_TOKENS.clear()  # noqa: SLF001
        srv._THREEJS_HTTP_TOKENS.update(original_tokens)  # noqa: SLF001
        srv._THREEJS_HTTP_TOKEN_CREATED_AT.clear()  # noqa: SLF001
        srv._THREEJS_HTTP_TOKEN_CREATED_AT.update(original_created)  # noqa: SLF001


def test_threejs_allowed_model_suffixes_include_core_formats() -> None:
    for suffix in {".obj", ".mtl", ".ply", ".stl", ".glb", ".gltf", ".png", ".jpg"}:
        assert suffix in srv._THREEJS_ALLOWED_MODEL_SUFFIXES  # noqa: SLF001


def test_threejs_allowed_assets_include_runtime_modules() -> None:
    assert "annolid_threejs_runtime.js" in srv._THREEJS_ALLOWED_ASSETS  # noqa: SLF001
    assert "annolid_threejs_css2d.js" in srv._THREEJS_ALLOWED_ASSETS  # noqa: SLF001
