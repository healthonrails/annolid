from __future__ import annotations

import os
import json
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
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app

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

    page = QtWebEngineWidgets.QWebEnginePage()
    load_loop = QtCore.QEventLoop()
    load_result: dict[str, bool] = {}

    def _on_load_finished(ok: bool) -> None:
        load_result["ok"] = ok
        load_loop.quit()

    page.loadFinished.connect(_on_load_finished)
    page.setHtml(
        '<div id="detail-latency">-</div>', QtCore.QUrl(f"{base_url}/threejs/")
    )
    QtCore.QTimer.singleShot(20000, load_loop.quit)
    load_loop.exec()
    page.loadFinished.disconnect(_on_load_finished)

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
    QtCore.QTimer.singleShot(10000, js_loop.quit)
    js_loop.exec()

    payload = json.loads(js_result.get("value", "{}"))
    assert payload["detailExists"] is True
    assert payload["detailText"] == "-"
    assert payload["statusText"] == 200
    inner = json.loads(payload["payload"])
    assert inner["planner"]["turn_latency_ms"] == 123.4
