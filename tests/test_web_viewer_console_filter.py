from annolid.gui.widgets.web_viewer import (
    _console_source_domain,
    _format_suppressed_console_summary,
    _is_ignorable_js_console_message,
)


def test_ignorable_console_noise_markers() -> None:
    assert _is_ignorable_js_console_message("Deprecated API for given entry type.")
    assert _is_ignorable_js_console_message(
        "Unrecognized feature: 'attribution-reporting'."
    )
    assert _is_ignorable_js_console_message(
        "THREE.WebGLProgram: gl.getProgramInfoLog() WARNING: Output of vertex shader 'worldPosition' not read by fragment shader"
    )
    assert _is_ignorable_js_console_message(
        "Failed to find a valid digest in the 'integrity' attribute for resource 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css'."
    )
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
    assert _is_ignorable_js_console_message(
        "The resource https://accounts.google.com/gsi/client was preloaded using link preload but not used within a few seconds from the window's load event."
    )
    assert _is_ignorable_js_console_message("Error")
    assert _is_ignorable_js_console_message("[object Object]")
    # CORS errors for external site fonts - these are external site issues, not actionable
    assert _is_ignorable_js_console_message(
        "Access to font at 'https://static.arxiv.org/MathJax-2.7.3/fonts/HTML-CSS/TeX/woff/MathJax_Math-Italic.woff?V=2.7.3' from origin 'https://arxiv.org' has been blocked by CORS policy: No 'Access-Control-Allow-Origin' header is present on the requested resource."
    )
    assert _is_ignorable_js_console_message(
        "Access to font at 'https://static.example.com/font.woff' from origin 'https://example.com' has been blocked by CORS policy"
    )
    assert _is_ignorable_js_console_message(
        "Access to font at 'https://cdn.example.com/fonts/MyFont.ttf' from origin 'https://arxiv.org' has been blocked by cors policy"
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
    assert _is_ignorable_js_console_message(
        "Identity value for 'email' is falsy (). This value will be removed from the request.",
        src,
    )
    assert _is_ignorable_js_console_message(
        "For 'enrolled-in-experiment', the corresponding attribute value of 'schemaVersion' must be a string, number, boolean, or null.",
        src,
    )
    assert _is_ignorable_js_console_message(
        "Position latitude and/or longitude must both be of type number",
        src,
    )
    assert _is_ignorable_js_console_message(
        "For 'page-viewed', the corresponding attribute value of 'premiumContent' must be a string, number, boolean, or null.",
        src,
    )
    assert _is_ignorable_js_console_message(
        "[object Object] Google Accounts SDK is not available",
        src,
    )
    assert _is_ignorable_js_console_message(
        "Error: Minified React error #418; visit https://reactjs.org/docs/error-decoder.html?invariant=418",
        src,
    )
    assert _is_ignorable_js_console_message(
        "Geolocation failed in saga: Error: Geolocation request timed out.",
        src,
    )


def test_weather_like_message_not_ignored_for_non_weather_source() -> None:
    src = "https://example.org/app"
    assert not _is_ignorable_js_console_message(
        "Error: Minified React error #418; visit https://reactjs.org/docs/error-decoder.html?invariant=418",
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
