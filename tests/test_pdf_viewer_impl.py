from __future__ import annotations

from pathlib import Path

from annolid.gui.widgets.pdf_viewer_impl import PdfViewerWidget


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

    def page(self) -> _DummyPage:
        return self._page


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
