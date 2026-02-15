from annolid.gui.widgets.web_viewer import _is_ignorable_js_console_message


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
