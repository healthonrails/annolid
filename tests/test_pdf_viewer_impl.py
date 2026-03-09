from __future__ import annotations

from pathlib import Path

from qtpy import QtCore, QtWidgets

from annolid.gui.widgets.pdf_viewer_impl import (
    PdfViewerWidget,
    _build_pdfjs_viewer_html,
)
from annolid.gui.widgets.pdf_viewer_bridge import _PdfReaderBridge


class _DummyStack:
    def __init__(self) -> None:
        self.index = -1

    def setCurrentIndex(self, index: int) -> None:  # noqa: N802 - Qt-style
        self.index = int(index)


class _DummyLabel:
    def __init__(self) -> None:
        self.cleared = False

    def clear(self) -> None:
        self.cleared = True


class _DummyTextView:
    def __init__(self) -> None:
        self.text = ""

    def setPlainText(self, text: str) -> None:  # noqa: N802 - Qt-style
        self.text = str(text)


class _DummyPage:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def runJavaScript(self, _script: str, callback) -> None:  # noqa: N802
        callback(self.payload)


class _DummyWebView:
    def __init__(self, payload: object) -> None:
        self._page = _DummyPage(payload)
        self.html: str | None = None
        self.base_url = None
        self.loaded_url = None

    def page(self) -> _DummyPage:
        return self._page

    def setHtml(self, html: str, base_url) -> None:  # noqa: N802 - Qt-style
        self.html = html
        self.base_url = base_url

    def load(self, url) -> None:  # noqa: N802 - Qt-style
        self.loaded_url = url


class _MenuCapableWebView(_DummyWebView):
    def createStandardContextMenu(self):  # noqa: N802 - Qt-style
        return QtWidgets.QMenu()


class _DummyBridgeViewer(QtCore.QObject):
    def _handle_reader_click(self, _payload) -> None:
        pass

    def _get_pdf_user_state(self, _pdf_key: str):
        return {}

    def _handle_pdf_user_state_save(self, _payload) -> None:
        pass

    def _clear_pdf_user_state(self, _pdf_key: str) -> None:
        pass

    def _handle_pdf_log_event(self, _payload) -> None:
        pass

    def _open_scholar_citations(self, _payload) -> None:
        pass


def test_maybe_repair_pdf_skips_repair_when_no_hint(
    monkeypatch, tmp_path: Path
) -> None:
    widget = PdfViewerWidget.__new__(PdfViewerWidget)
    path = tmp_path / "valid.pdf"
    path.write_bytes(b"%PDF-1.7\n%%EOF\n")

    monkeypatch.setattr(
        widget,
        "_pdf_needs_repair_hint",
        lambda _path: False,
    )
    monkeypatch.setattr(
        widget,
        "_attempt_repair_pdf",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("repair should not be attempted")
        ),
    )

    out = widget._maybe_repair_pdf(path, reason="test")
    assert out == path


def test_fallback_from_web_preserves_source_path_for_pymupdf(tmp_path: Path) -> None:
    widget = PdfViewerWidget.__new__(PdfViewerWidget)
    opened: list[tuple[Path, Path | None]] = []

    source = tmp_path / "source.pdf"
    source.write_bytes(b"%PDF-1.7\n%%EOF\n")
    repaired = tmp_path / "source.repaired.pdf"
    repaired.write_bytes(b"%PDF-1.7\n%%EOF\n")

    widget._web_view = None
    widget._pdfjs_active = False
    widget._use_web_engine = True
    widget._stack = _DummyStack()
    widget.image_label = _DummyLabel()
    widget.text_view = _DummyTextView()
    widget._web_loading_path = repaired
    widget._web_loading_source_path = source
    widget._load_pdfjs_viewer = lambda _path: False
    widget._set_controls_for_web = lambda _web_mode: None
    widget._attempt_repair_pdf = lambda _path, reason="": None
    widget._open_with_pymupdf = lambda path, *, source_path=None: opened.append(
        (path, source_path)
    )

    widget._fallback_from_web(repaired, "PDF.js error", source_path=source)

    assert opened == [(repaired, source)]
    assert widget._web_loading_path is None
    assert widget._web_loading_source_path is None


def test_on_web_load_finished_pdfjs_success_sets_source_path(tmp_path: Path) -> None:
    widget = PdfViewerWidget.__new__(PdfViewerWidget)
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.7\n%%EOF\n")
    repaired = tmp_path / "paper.repaired.pdf"
    repaired.write_bytes(b"%PDF-1.7\n%%EOF\n")

    widget._web_loading_path = repaired
    widget._web_loading_source_path = source
    widget._pdfjs_active = True
    widget._web_view = _DummyWebView(
        {
            "err": "",
            "spans": 0,
            "renderedPages": 0,
            "pdfLoaded": True,
            "ready": True,
            "hasPdfjs": True,
            "state": "complete",
        }
    )
    widget._pdf_path = None
    widget._apply_reader_enabled_to_web = lambda: None
    widget.fit_to_width = lambda: None
    widget._emit_reader_availability = lambda: None
    widget._fallback_from_web = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("fallback should not be called on success")
    )

    widget._on_web_load_finished(True)

    assert widget._pdf_path == source
    assert widget._web_loading_path is None
    assert widget._web_loading_source_path is None


def test_on_web_load_finished_pdfjs_inconclusive_keeps_web_view(
    monkeypatch, tmp_path: Path
) -> None:
    widget = PdfViewerWidget.__new__(PdfViewerWidget)
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.7\n%%EOF\n")
    repaired = tmp_path / "paper.repaired.pdf"
    repaired.write_bytes(b"%PDF-1.7\n%%EOF\n")

    widget._web_loading_path = repaired
    widget._web_loading_source_path = source
    widget._pdfjs_active = True
    widget._web_view = _DummyWebView(
        {
            "err": "",
            "spans": 0,
            "renderedPages": 0,
            "pdfLoaded": False,
            "ready": False,
            "hasPdfjs": False,
            "state": "",
        }
    )
    widget._pdf_path = None
    widget._apply_reader_enabled_to_web = lambda: None
    widget._emit_reader_availability = lambda: None
    widget._fallback_from_web = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("fallback should not be called on inconclusive PDF.js probe")
    )

    monkeypatch.setattr(
        "annolid.gui.widgets.pdf_viewer_impl.QtCore.QTimer.singleShot",
        lambda _ms, fn: fn(),
    )

    widget._on_web_load_finished(True)

    assert widget._pdf_path == source
    assert widget._web_loading_path is None
    assert widget._web_loading_source_path is None


def test_load_pdfjs_viewer_escapes_bootstrap_values(
    monkeypatch, tmp_path: Path
) -> None:
    widget = PdfViewerWidget.__new__(PdfViewerWidget)
    pdf_path = tmp_path / 'paper "draft".pdf'
    pdf_path.write_bytes(b"%PDF-1.7\n%%EOF\n")
    web_view = _DummyWebView(payload={})

    widget._web_view = web_view
    widget._pdf_key = 'key "quoted" </script>'
    widget._pdf_user_state = {"note": 'line "one" </script>'}

    monkeypatch.setattr(
        "annolid.gui.widgets.pdf_viewer_impl._ensure_pdfjs_http_server",
        lambda: "http://127.0.0.1:9999",
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.pdf_viewer_impl._register_pdfjs_http_pdf",
        lambda _path: 'http://127.0.0.1:9999/pdf?id="x"&name=</script>',
    )
    captured_html: dict[str, str] = {}

    def _capture_viewer_html(html: str) -> str:
        captured_html["html"] = html
        return "http://127.0.0.1:9999/viewer/test"

    monkeypatch.setattr(
        "annolid.gui.widgets.pdf_viewer_impl._register_pdfjs_http_viewer_html",
        _capture_viewer_html,
    )

    assert widget._load_pdfjs_viewer(pdf_path) is True
    html = captured_html["html"]
    assert str(web_view.loaded_url.toString()) == "http://127.0.0.1:9999/viewer/test"
    assert 'window.__annolidAssetBaseUrl = bootstrap.assetBaseUrl || "";' in html
    assert "window.__annolidPdfUrl = bootstrap.pdfUrl;" in html
    assert "window.__annolidPdfTitle = bootstrap.pdfTitle;" in html
    assert 'src="http://127.0.0.1:9999/pdfjs/pdf.min.js"' in html
    assert 'src="http://127.0.0.1:9999/pdfjs/annolid_viewer.js"' in html
    assert 'href="http://127.0.0.1:9999/pdfjs/annolid_viewer.css"' in html
    assert "<\\/script>" in html
    assert 'http://127.0.0.1:9999/pdf?id="x"&name=</script>' not in html


def test_build_pdfjs_viewer_html_uses_absolute_asset_urls() -> None:
    html = _build_pdfjs_viewer_html(
        asset_base_url="http://127.0.0.1:9999",
        pdf_key="k",
        initial_state={},
        pdf_url="http://127.0.0.1:9999/pdf/abc",
        pdf_b64="",
        pdf_title="paper.pdf",
    )

    assert 'href="http://127.0.0.1:9999/pdfjs/annolid_viewer.css"' in html
    assert 'src="http://127.0.0.1:9999/pdfjs/annolid_viewer_polyfills.js"' in html
    assert 'src="http://127.0.0.1:9999/pdfjs/pdf.min.js"' in html
    assert 'src="http://127.0.0.1:9999/pdfjs/annolid_viewer.js"' in html
    assert 'href="pdfjs/annolid_viewer.css"' not in html


def test_build_web_context_menu_falls_back_to_view_menu() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    widget = PdfViewerWidget.__new__(PdfViewerWidget)
    widget._web_view = _MenuCapableWebView(payload={})

    menu = widget._build_web_context_menu()

    assert isinstance(menu, QtWidgets.QMenu)


def test_pdf_reader_bridge_exposes_qvariant_slots() -> None:
    bridge = _PdfReaderBridge(_DummyBridgeViewer())
    meta = bridge.metaObject()

    assert meta.indexOfSlot("saveUserState(QVariant)") >= 0
    assert meta.indexOfSlot("logEvent(QVariant)") >= 0
    assert meta.indexOfSlot("openScholarCitations(QVariant)") >= 0
