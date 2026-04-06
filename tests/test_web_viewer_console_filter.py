from pathlib import Path
import sys

from annolid.gui.widgets.web_viewer import (
    _build_local_markdown_view_html,
    _clamp_text,
    _console_source_domain,
    _format_suppressed_console_summary,
    _is_ignorable_js_console_message,
    _resolve_existing_local_file,
    _sanitize_context_menu_image_src,
    _sanitize_image_data_url,
)
from qtpy import QtWidgets


def test_ignorable_console_noise_markers() -> None:
    shader_warning = (
        "THREE.WebGLProgram: gl.getProgramInfoLog() WARNING: "
        "Output of vertex shader 'worldPosition' not read by fragment shader"
    )
    integrity_warning = (
        "Failed to find a valid digest in the 'integrity' attribute for resource "
        "'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'."
    )
    preload_warning = (
        "The resource https://accounts.google.com/gsi/client was preloaded "
        "using link preload but not used within a few seconds from the window's load event."
    )
    cors_warning = (
        "Access to font at "
        "'https://static.arxiv.org/MathJax-2.7.3/fonts/HTML-CSS/TeX/woff/MathJax_Math-Italic.woff?V=2.7.3' "
        "from origin 'https://arxiv.org' has been blocked by CORS policy: "
        "No 'Access-Control-Allow-Origin' header is present on the requested resource."
    )

    assert _is_ignorable_js_console_message("Deprecated API for given entry type.")
    assert _is_ignorable_js_console_message(
        "Unrecognized feature: 'attribution-reporting'."
    )
    assert _is_ignorable_js_console_message(shader_warning)
    assert _is_ignorable_js_console_message(integrity_warning)
    assert _is_ignorable_js_console_message(
        "Uncaught ReferenceError: solveSimpleChallenge is not defined"
    )
    assert _is_ignorable_js_console_message(
        "Uncaught TypeError: Cannot read property 'style' of undefined"
    )
    assert _is_ignorable_js_console_message(
        "Atom change detected, updating - store value: undefined new value: usa"
    )
    assert _is_ignorable_js_console_message(
        "RangeError: Value longOffset out of range for Intl.DateTimeFormat options property timeZoneName"
    )
    assert _is_ignorable_js_console_message(preload_warning)
    assert _is_ignorable_js_console_message("Error")
    assert _is_ignorable_js_console_message("[object Object]")
    # CORS errors for external site fonts - these are external site issues, not actionable
    assert _is_ignorable_js_console_message(cors_warning)
    assert _is_ignorable_js_console_message(
        "Access to font at 'https://static.example.com/font.woff' from origin "
        "'https://example.com' has been blocked by CORS policy"
    )
    assert _is_ignorable_js_console_message(
        "Access to font at 'https://cdn.example.com/fonts/MyFont.ttf' from "
        "origin 'https://arxiv.org' has been blocked by cors policy"
    )


def test_non_ignorable_console_errors() -> None:
    assert not _is_ignorable_js_console_message(
        "Uncaught ReferenceError: $ is not defined"
    )
    assert not _is_ignorable_js_console_message(
        "Uncaught (in promise) TypeError: this.o.at is not a function"
    )
    assert not _is_ignorable_js_console_message(
        "Uncaught TypeError: crypto.randomUUID is not a function"
    )
    assert not _is_ignorable_js_console_message(
        "Uncaught SyntaxError: Failed to execute 'matches' on 'Element': '[open]:not(:modal)' is not a valid selector."
    )


def test_ignorable_weather_com_console_noise() -> None:
    src = "https://weather.com/weather/today/l/Ithaca+NY"
    enrolled_warning = (
        "For 'enrolled-in-experiment', the corresponding attribute value of "
        "'schemaVersion' must be a string, number, boolean, or null."
    )
    page_view_warning = (
        "For 'page-viewed', the corresponding attribute value of "
        "'premiumContent' must be a string, number, boolean, or null."
    )
    react_418 = (
        "Error: Minified React error #418; visit "
        "https://reactjs.org/docs/error-decoder.html?invariant=418"
    )

    assert _is_ignorable_js_console_message(
        "Identity value for 'email' is falsy (). This value will be removed from the request.",
        src,
    )
    assert _is_ignorable_js_console_message(enrolled_warning, src)
    assert _is_ignorable_js_console_message(
        "Position latitude and/or longitude must both be of type number",
        src,
    )
    assert _is_ignorable_js_console_message(page_view_warning, src)
    assert _is_ignorable_js_console_message(
        "[object Object] Google Accounts SDK is not available",
        src,
    )
    assert _is_ignorable_js_console_message(react_418, src)
    assert _is_ignorable_js_console_message(
        "Geolocation failed in saga: Error: Geolocation request timed out.",
        src,
    )


def test_weather_like_message_not_ignored_for_non_weather_source() -> None:
    src = "https://example.org/app"
    react_418 = (
        "Error: Minified React error #418; visit "
        "https://reactjs.org/docs/error-decoder.html?invariant=418"
    )
    assert not _is_ignorable_js_console_message(
        react_418,
        src,
    )


def test_console_source_domain_parsing() -> None:
    assert _console_source_domain("https://weather.com/weather/today/l/Ithaca+NY") == (
        "weather.com"
    )
    assert _console_source_domain("") == "unknown"
    assert _console_source_domain("not-a-url") == "unknown"


def test_format_suppressed_console_summary_top_order() -> None:
    summary = _format_suppressed_console_summary(
        {"weather.com": 7, "unknown": 2, "example.org": 5}
    )
    assert summary == "weather.com=7, example.org=5, unknown=2"


def test_build_local_markdown_view_html_renders_and_sets_base_href(
    tmp_path: Path,
) -> None:
    source = tmp_path / "notes.md"
    source.write_text(
        "# Title\n\nSee [details](details.md) and `inline code`.\n",
        encoding="utf-8",
    )

    html = _build_local_markdown_view_html(
        source_path=source,
        markdown_text=source.read_text(encoding="utf-8"),
    )

    assert "<h1" in html
    assert "Title" in html
    assert "details.md" in html
    assert source.name in html
    assert str(source.parent) in html
    assert "annolid-markdown-toc" in html
    assert 'class="toc"' in html
    assert "slugify" not in html
    assert "Last modified:" in html


def test_resolve_existing_local_file_accepts_file_url(tmp_path: Path) -> None:
    source = tmp_path / "README.md"
    source.write_text("# Demo\n", encoding="utf-8")

    resolved = _resolve_existing_local_file(source.as_uri())
    assert resolved == source.resolve()


def test_resolve_existing_local_file_handles_case_mismatch_file_url(
    tmp_path: Path,
) -> None:
    mixed_dir = tmp_path / "MixedCaseDir"
    mixed_dir.mkdir()
    source = mixed_dir / "DocFile.md"
    source.write_text("# Demo\n", encoding="utf-8")
    lowered_url = (
        source.as_uri()
        .replace("MixedCaseDir", "mixedcasedir")
        .replace("DocFile.md", "docfile.md")
    )

    resolved = _resolve_existing_local_file(lowered_url)
    if sys.platform == "darwin":
        assert resolved is not None
        assert resolved.exists()
        assert resolved.samefile(source)
    else:
        assert resolved is None


# ---------------------------------------------------------------------------
# _is_pdf_url tests (uses the method directly via a minimal instance)
# ---------------------------------------------------------------------------


def test_is_pdf_url_arxiv_renders_inline() -> None:
    """ArXiv /pdf/ URLs should NOT be flagged as PDF downloads."""
    from annolid.gui.widgets.web_viewer import WebViewerWidget

    # Call the unbound method with a dummy self (only url_lower is used)
    _is_pdf = WebViewerWidget._is_pdf_url
    assert _is_pdf(None, "https://arxiv.org/pdf/2512.21586") is False
    assert _is_pdf(None, "https://arxiv.org/pdf/2512.21586v1") is False
    assert _is_pdf(None, "https://ARXIV.org/PDF/1234.56789") is False


def test_is_pdf_url_real_pdf_files() -> None:
    """Actual .pdf file URLs should still be detected."""
    from annolid.gui.widgets.web_viewer import WebViewerWidget

    _is_pdf = WebViewerWidget._is_pdf_url
    assert _is_pdf(None, "https://example.com/paper.pdf") is True
    assert _is_pdf(None, "https://example.com/file.pdf?v=1") is True
    assert _is_pdf(None, "https://example.com/file.pdf#page=2") is True


def test_is_pdf_url_generic_pdf_path_not_flagged() -> None:
    """Generic /pdf/ in path (non-arxiv) should no longer trigger."""
    from annolid.gui.widgets.web_viewer import WebViewerWidget

    _is_pdf = WebViewerWidget._is_pdf_url
    assert _is_pdf(None, "https://example.com/pdf/something") is False
    assert _is_pdf(None, "https://example.com/viewer/pdf/12345") is False


def test_is_saveable_pdf_url_includes_arxiv_inline_pdf() -> None:
    from annolid.gui.widgets.web_viewer import WebViewerWidget

    _is_saveable = WebViewerWidget._is_saveable_pdf_url
    assert _is_saveable(None, "https://arxiv.org/pdf/2512.21586") is True
    assert _is_saveable(None, "https://example.com/file.pdf") is True
    assert _is_saveable(None, "https://example.com/viewer/pdf/12345") is False


def test_resolve_pdf_download_url_for_arxiv_abs() -> None:
    from annolid.gui.widgets.web_viewer import WebViewerWidget

    resolve = WebViewerWidget._resolve_pdf_download_url
    assert (
        resolve("https://arxiv.org/abs/2602.17594")
        == "https://arxiv.org/pdf/2602.17594.pdf"
    )


def test_resolve_pdf_download_url_for_arxiv_pdf_without_extension() -> None:
    from annolid.gui.widgets.web_viewer import WebViewerWidget

    resolve = WebViewerWidget._resolve_pdf_download_url
    assert (
        resolve("https://arxiv.org/pdf/2602.17594")
        == "https://arxiv.org/pdf/2602.17594.pdf"
    )


def test_resolve_pdf_download_url_non_arxiv_unchanged() -> None:
    from annolid.gui.widgets.web_viewer import WebViewerWidget

    resolve = WebViewerWidget._resolve_pdf_download_url
    url = "https://example.com/paper.pdf?download=1"
    assert resolve(url) == url


def test_pmc_pdf_url_is_saveable() -> None:
    from annolid.gui.widgets.web_viewer import WebViewerWidget

    _is_saveable = WebViewerWidget._is_saveable_pdf_url
    url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC8219259/pdf/nihms-1556781.pdf"
    assert _is_saveable(None, url) is True


def test_save_pdf_task_adds_pmc_download_fallback() -> None:
    from annolid.gui.widgets.web_viewer import _SavePdfTask

    url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC8219259/pdf/nihms-1556781.pdf"
    candidates = _SavePdfTask._candidate_download_urls(url)
    assert candidates[0].startswith(
        "https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMC8219259"
    )
    assert candidates[1] == url
    assert candidates[2].endswith("nihms-1556781.pdf?download=1")


def test_web_viewer_context_menu_slot_registered() -> None:
    from annolid.gui.widgets.web_viewer import WebViewerWidget

    meta = WebViewerWidget.staticMetaObject
    assert meta.indexOfSlot("_show_context_menu(QPoint)") >= 0


def test_build_web_context_menu_falls_back_to_view_menu(monkeypatch) -> None:
    from annolid.gui.widgets.web_viewer import WebViewerWidget

    created = []

    class _FakeMenu:
        pass

    class _FakeView:
        def page(self):
            return object()

        def createStandardContextMenu(self):
            created.append("view")
            return _FakeMenu()

    class _FakeWidget:
        def __init__(self):
            self._web_view = _FakeView()

    monkeypatch.setattr(QtWidgets, "QMenu", lambda *_args, **_kwargs: _FakeMenu())

    menu = WebViewerWidget._build_web_context_menu(_FakeWidget())

    assert isinstance(menu, _FakeMenu)
    assert created == ["view"]


def test_context_menu_payload_sanitizers_restrict_unsafe_input() -> None:
    assert _clamp_text("abc", max_chars=2) == "ab"
    assert _sanitize_context_menu_image_src("javascript:alert(1)") == ""
    assert _sanitize_context_menu_image_src("file:///tmp/demo.png") == ""
    assert _sanitize_context_menu_image_src("https://example.com/a.png") == (
        "https://example.com/a.png"
    )
    assert _sanitize_image_data_url("data:text/html;base64,Zm9v") == ""
    assert _sanitize_image_data_url("data:image/png;base64,Zm9v") == (
        "data:image/png;base64,Zm9v"
    )


def test_save_image_from_context_rejects_non_http_non_data_scheme() -> None:
    from annolid.gui.widgets.web_viewer import WebViewerWidget

    path, error = WebViewerWidget._save_image_from_context(
        None, image_src="file:///tmp/demo.png", image_data_url=""
    )

    assert path == ""
    assert "Unsupported image URL scheme" in error


def test_is_inline_pdf_context_uses_saveable_pdf_url() -> None:
    from annolid.gui.widgets.web_viewer import WebViewerWidget

    class _FakeWidget:
        _current_url = "https://example.com/paper.pdf"

        @staticmethod
        def _is_saveable_pdf_url(url: str) -> bool:
            return url.endswith(".pdf")

    assert WebViewerWidget._is_inline_pdf_context(_FakeWidget()) is True


def test_resolve_context_selection_uses_text_immediately() -> None:
    from annolid.gui.widgets.web_viewer import WebViewerWidget

    captured = []

    class _FakeWidget:
        def _is_inline_pdf_context(self) -> bool:
            return False

    WebViewerWidget._resolve_context_selection(
        _FakeWidget(),
        " chosen text ",
        lambda text: captured.append(text),
    )

    assert captured == ["chosen text"]


def test_resolve_context_selection_uses_pdf_copy_fallback(monkeypatch) -> None:
    from annolid.gui.widgets.web_viewer import WebViewerWidget
    from qtpy import QtCore, QtWebEngineWidgets

    captured = []

    class _Clipboard:
        def __init__(self):
            self._text = "before"

        def text(self):
            return self._text

        def setText(self, value):
            self._text = value

    clipboard = _Clipboard()

    class _Page:
        def triggerAction(self, action):
            assert action == QtWebEngineWidgets.QWebEnginePage.Copy
            clipboard.setText("pdf selected text")

    class _WebView:
        def page(self):
            return _Page()

    class _FakeWidget:
        _web_view = _WebView()

        def _is_inline_pdf_context(self) -> bool:
            return True

    monkeypatch.setattr(QtWidgets.QApplication, "clipboard", lambda: clipboard)
    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _ms, fn: fn())

    WebViewerWidget._resolve_context_selection(
        _FakeWidget(),
        "",
        lambda text: captured.append(text),
    )

    assert captured == ["pdf selected text"]
    assert clipboard.text() == "before"
