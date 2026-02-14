from annolid.gui.widgets.web_viewer import _is_ignorable_js_console_message


def test_ignorable_console_noise_markers() -> None:
    assert _is_ignorable_js_console_message("Deprecated API for given entry type.")
    assert _is_ignorable_js_console_message(
        "Unrecognized feature: 'attribution-reporting'."
    )
    assert _is_ignorable_js_console_message(
        "THREE.WebGLProgram: gl.getProgramInfoLog() WARNING: Output of vertex shader 'worldPosition' not read by fragment shader"
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
