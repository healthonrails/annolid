from __future__ import annotations

import contextlib
import gc
import json
import os
from pathlib import Path
from urllib.request import urlopen

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
os.environ.setdefault("QTWEBENGINE_DISABLE_SANDBOX", "1")

from annolid.infrastructure.runtime import configure_qt_runtime

configure_qt_runtime()

from qtpy import QtCore, QtWidgets  # noqa: E402

try:
    from qtpy import QtWebEngineWidgets  # type: ignore
except Exception:  # pragma: no cover - depends on environment packaging
    QtWebEngineWidgets = None  # type: ignore

from annolid.gui.widgets import threejs_viewer_server as srv  # noqa: E402
from annolid.gui.widgets.webengine_lifecycle import (  # noqa: E402
    release_webengine_view,
)

_QAPP = None


def test_prune_threejs_http_tokens_removes_expired_and_limits_size() -> None:
    original_tokens = dict(srv._THREEJS_HTTP_TOKENS)  # noqa: SLF001
    original_created = dict(srv._THREEJS_HTTP_TOKEN_CREATED_AT)  # noqa: SLF001
    original_roots = dict(srv._THREEJS_HTTP_TOKEN_ASSET_ROOTS)  # noqa: SLF001
    try:
        srv._THREEJS_HTTP_TOKENS.clear()  # noqa: SLF001
        srv._THREEJS_HTTP_TOKEN_ASSET_ROOTS.clear()  # noqa: SLF001
        srv._THREEJS_HTTP_TOKEN_CREATED_AT.clear()  # noqa: SLF001
        now = 1000.0
        # one expired
        srv._THREEJS_HTTP_TOKENS["old"] = Path("/tmp/a.obj")  # noqa: SLF001
        srv._THREEJS_HTTP_TOKEN_ASSET_ROOTS["old"] = {  # noqa: SLF001
            "atlas_meshes": Path("/tmp/meshes")
        }
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
        assert "old" not in srv._THREEJS_HTTP_TOKEN_ASSET_ROOTS  # noqa: SLF001
        assert len(srv._THREEJS_HTTP_TOKENS) <= srv._THREEJS_HTTP_MAX_TOKENS  # noqa: SLF001
    finally:
        srv._THREEJS_HTTP_TOKENS.clear()  # noqa: SLF001
        srv._THREEJS_HTTP_TOKENS.update(original_tokens)  # noqa: SLF001
        srv._THREEJS_HTTP_TOKEN_ASSET_ROOTS.clear()  # noqa: SLF001
        srv._THREEJS_HTTP_TOKEN_ASSET_ROOTS.update(original_roots)  # noqa: SLF001
        srv._THREEJS_HTTP_TOKEN_CREATED_AT.clear()  # noqa: SLF001
        srv._THREEJS_HTTP_TOKEN_CREATED_AT.update(original_created)  # noqa: SLF001


def test_register_threejs_http_model_records_token_scoped_asset_roots(
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "payload.json"
    payload_path.write_text("{}", encoding="utf-8")
    mesh_dir = tmp_path / "meshes"
    mesh_dir.mkdir()

    url = srv._register_threejs_http_model(  # noqa: SLF001
        payload_path,
        asset_roots={
            "atlas_meshes": mesh_dir,
            "../bad": mesh_dir,
            "missing": tmp_path / "missing",
        },
    )
    token = url.split("/model/", 1)[1].split("/", 1)[0]

    try:
        assert srv._THREEJS_HTTP_TOKENS[token] == payload_path  # noqa: SLF001
        assert srv._THREEJS_HTTP_TOKEN_ASSET_ROOTS[token] == {  # noqa: SLF001
            "atlas_meshes": mesh_dir
        }
    finally:
        srv._THREEJS_HTTP_TOKENS.pop(token, None)  # noqa: SLF001
        srv._THREEJS_HTTP_TOKEN_ASSET_ROOTS.pop(token, None)  # noqa: SLF001
        srv._THREEJS_HTTP_TOKEN_CREATED_AT.pop(token, None)  # noqa: SLF001


def test_threejs_allowed_model_suffixes_include_core_formats() -> None:
    for suffix in {
        ".obj",
        ".mtl",
        ".ply",
        ".stl",
        ".glb",
        ".gltf",
        ".json",
        ".png",
        ".jpg",
    }:
        assert suffix in srv._THREEJS_ALLOWED_MODEL_SUFFIXES  # noqa: SLF001


def test_threejs_allowed_assets_include_runtime_modules() -> None:
    assert "annolid_threejs_runtime.js" in srv._THREEJS_ALLOWED_ASSETS  # noqa: SLF001
    assert "annolid_threejs_css2d.js" in srv._THREEJS_ALLOWED_ASSETS  # noqa: SLF001


@pytest.mark.parametrize(
    "filename", ["two_mice_scene.js", "two_mice_model.js", "two_mice_simulation.js"]
)
def test_two_mice_modules_are_served_by_the_desktop_viewer(filename):
    base_url = srv._ensure_threejs_http_server()  # noqa: SLF001
    with urlopen(f"{base_url}/threejs/{filename}", timeout=5) as response:
        assert response.status == 200
        assert "javascript" in response.headers["Content-Type"]
        assert response.read().decode("utf-8") == srv._threejs_asset_path(
            filename
        ).read_text(encoding="utf-8")


def test_update_swarm_node_preserves_turn_latency() -> None:
    original_state = srv.get_swarm_state()
    try:
        srv.clear_swarm_state()
        srv.update_swarm_node(
            "planner",
            "idle",
            "Awaiting Tasks",
            turn_latency_ms=123.4,
        )
        state = srv.get_swarm_state()
        assert state["planner"]["turn_latency_ms"] == 123.4
    finally:
        srv.clear_swarm_state()
        for node_id, node_state in original_state.items():
            srv.update_swarm_node(
                node_id,
                str(node_state.get("status", "")),
                str(node_state.get("task", "")),
                thinking=str(node_state.get("thinking", "")),
                output=str(node_state.get("output", "")),
                parent=str(node_state.get("parent", "")),
                turn_latency_ms=node_state.get("turn_latency_ms"),
            )


@pytest.mark.skipif(QtWebEngineWidgets is None, reason="QtWebEngine is unavailable")
def test_swarm_visualizer_smoke_loads_latency_dom() -> None:
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        _QAPP = QtWidgets.QApplication([])
        app = _QAPP

    srv.clear_swarm_state()
    srv.update_swarm_node(
        "planner",
        "idle",
        "Awaiting Tasks",
        turn_latency_ms=123.4,
    )

    base_url = srv._ensure_threejs_http_server()  # noqa: SLF001
    with urlopen(f"{base_url}/threejs/swarm_visualizer.html", timeout=5) as response:
        visualizer_html = response.read().decode("utf-8")
    assert 'id="detail-latency"' in visualizer_html

    profile = QtWebEngineWidgets.QWebEngineProfile()
    cookie_policy = getattr(
        QtWebEngineWidgets.QWebEngineProfile, "NoPersistentCookies", None
    )
    if cookie_policy is not None:
        profile.setPersistentCookiesPolicy(cookie_policy)
    cache_kind = getattr(QtWebEngineWidgets.QWebEngineProfile, "MemoryHttpCache", None)
    if cache_kind is not None:
        profile.setHttpCacheType(cache_kind)

    page = QtWebEngineWidgets.QWebEnginePage(profile)
    load_loop = QtCore.QEventLoop()
    load_result: dict[str, bool] = {}

    def _on_load_finished(ok: bool) -> None:
        load_result["ok"] = ok
        load_loop.quit()

    load_connected = False
    try:
        page.loadFinished.connect(_on_load_finished)
        load_connected = True
        page.setHtml(
            '<div id="detail-latency">-</div>', QtCore.QUrl(f"{base_url}/threejs/")
        )
        load_timer = QtCore.QTimer()
        load_timer.setSingleShot(True)
        load_timer.timeout.connect(load_loop.quit)
        load_timer.start(20000)
        try:
            load_loop.exec()
        finally:
            load_timer.stop()
            load_timer.deleteLater()
        page.loadFinished.disconnect(_on_load_finished)
        load_connected = False

        assert load_result.get("ok") is True

        js_loop = QtCore.QEventLoop()
        js_result: dict[str, str] = {}

        def _on_js(value: object) -> None:
            js_result["value"] = str(value or "")
            js_loop.quit()

        page.runJavaScript(
            """
(() => {
  const detail = document.getElementById('detail-latency');
  const xhr = new XMLHttpRequest();
  xhr.open('GET', '/swarm/status', false);
  xhr.send(null);
  return JSON.stringify({
    detailExists: !!detail,
    detailText: detail ? detail.textContent.trim() : '',
    statusText: xhr.status,
    payload: xhr.responseText
  });
})()
            """.strip(),
            _on_js,
        )
        js_timer = QtCore.QTimer()
        js_timer.setSingleShot(True)
        js_timer.timeout.connect(js_loop.quit)
        js_timer.start(10000)
        try:
            js_loop.exec()
        finally:
            js_timer.stop()
            js_timer.deleteLater()

        payload = json.loads(js_result.get("value", "{}"))
        assert payload["detailExists"] is True
        assert payload["detailText"] == "-"
        assert payload["statusText"] == 200
        inner = json.loads(payload["payload"])
        assert inner["planner"]["turn_latency_ms"] == 123.4
    finally:
        if load_connected:
            with contextlib.suppress(Exception):
                page.loadFinished.disconnect(_on_load_finished)
        release_webengine_view(None, profile, page=page)
        del page
        del profile
        for _ in range(3):
            app.sendPostedEvents(None, QtCore.QEvent.Type.DeferredDelete)
            app.processEvents()
        gc.collect()
