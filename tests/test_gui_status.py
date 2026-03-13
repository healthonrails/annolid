from annolid.gui.status import post_window_status


class _StatusOnlyWindow:
    def __init__(self) -> None:
        self.calls = []

    def status(self, message: str, timeout: int) -> None:
        self.calls.append((message, timeout))


class _LegacyStatusWindow:
    def __init__(self) -> None:
        self.calls = []

    def status(self, message: str) -> None:
        self.calls.append((message, None))


class _PostStatusWindow:
    def __init__(self) -> None:
        self.calls = []

    def post_status_message(self, message: str, timeout: int) -> None:
        self.calls.append((message, timeout))


class _StatusBar:
    def __init__(self) -> None:
        self.calls = []

    def showMessage(self, message: str, timeout: int) -> None:
        self.calls.append((message, timeout))


class _StatusBarWindow:
    def __init__(self) -> None:
        self.bar = _StatusBar()

    def statusBar(self):
        return self.bar


def test_post_window_status_prefers_status_timeout_signature() -> None:
    window = _StatusOnlyWindow()
    post_window_status(window, "hello", 1234)
    assert window.calls == [("hello", 1234)]


def test_post_window_status_falls_back_to_legacy_status_signature() -> None:
    window = _LegacyStatusWindow()
    post_window_status(window, "hello", 1234)
    assert window.calls == [("hello", None)]


def test_post_window_status_uses_post_status_message_when_available() -> None:
    window = _PostStatusWindow()
    post_window_status(window, "hello", 1234)
    assert window.calls == [("hello", 1234)]


def test_post_window_status_falls_back_to_status_bar() -> None:
    window = _StatusBarWindow()
    post_window_status(window, "hello", 1234)
    assert window.bar.calls == [("hello", 1234)]
