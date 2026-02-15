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
