from __future__ import annotations

from annolid.gui.widgets.webengine_lifecycle import release_webengine_view


class _DummySignal:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self) -> None:
        for callback in list(self._callbacks):
            callback()


class _DummyPage:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.destroyed = _DummySignal()

    def setWebChannel(self, channel) -> None:  # noqa: N802
        self.events.append(f"setWebChannel:{channel}")

    def deleteLater(self) -> None:
        self.events.append("page.deleteLater")
        self.destroyed.emit()


class _DummyView:
    def __init__(self, page: _DummyPage, events: list[str]) -> None:
        self._page = page
        self.events = events

    def stop(self) -> None:
        self.events.append("view.stop")

    def setUrl(self, url) -> None:  # noqa: N802
        self.events.append(f"view.setUrl:{url.toString()}")

    def page(self) -> _DummyPage:
        return self._page

    def setPage(self, page) -> None:  # noqa: N802
        self.events.append(f"view.setPage:{page}")

    def deleteLater(self) -> None:
        self.events.append("view.deleteLater")


class _DummyProfile:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def setParent(self, value) -> None:  # noqa: N802
        self.events.append(f"profile.setParent:{value}")

    def deleteLater(self) -> None:
        self.events.append("profile.deleteLater")


def test_release_webengine_view_deletes_profile_after_page() -> None:
    events: list[str] = []
    page = _DummyPage(events)
    view = _DummyView(page, events)
    profile = _DummyProfile(events)

    release_webengine_view(
        view,
        profile,
        before_page_delete=lambda value: value.setWebChannel(None),
    )

    assert events == [
        "view.stop",
        "view.setUrl:about:blank",
        "setWebChannel:None",
        "profile.setParent:None",
        "view.setPage:None",
        "page.deleteLater",
        "profile.deleteLater",
        "view.deleteLater",
    ]


def test_release_webengine_view_deletes_profile_immediately_without_page() -> None:
    events: list[str] = []
    profile = _DummyProfile(events)

    release_webengine_view(None, profile)

    assert events == ["profile.deleteLater"]
