from annolid.gui.widgets.pdf_viewer_impl import _is_benign_pdfjs_console_message


def test_benign_pdfjs_font_warnings_are_ignored() -> None:
    assert _is_benign_pdfjs_console_message(
        'Warning: loadFont - translateFont failed: "FormatError: invalid font name".'
    )
    assert _is_benign_pdfjs_console_message(
        "Warning: Error during font loading: invalid font name"
    )


def test_non_benign_pdfjs_errors_not_ignored() -> None:
    assert not _is_benign_pdfjs_console_message(
        "PDF.js render failed Error: Invalid PDF structure."
    )
    assert not _is_benign_pdfjs_console_message(
        "Uncaught ReferenceError: foo is not defined"
    )
