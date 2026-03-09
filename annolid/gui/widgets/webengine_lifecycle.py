from __future__ import annotations

import contextlib
from typing import Callable, Optional

from qtpy import QtCore


def release_webengine_view(
    view,
    profile,
    *,
    blank_url: str = "about:blank",
    before_page_delete: Optional[Callable[[object], None]] = None,
) -> None:
    """Tear down a WebEngine view/page/profile in a safe order.

    The key invariant is that a custom QWebEngineProfile must outlive any
    QWebEnginePage using it. Deleting the profile in the same close event as the
    page can trigger Qt warnings because `deleteLater()` is deferred.
    """
    page = None
    if view is not None:
        with contextlib.suppress(Exception):
            view.stop()
        with contextlib.suppress(Exception):
            view.setUrl(QtCore.QUrl(blank_url))
        with contextlib.suppress(Exception):
            page = view.page()

    if page is not None and before_page_delete is not None:
        with contextlib.suppress(Exception):
            before_page_delete(page)

    if page is not None and profile is not None:
        with contextlib.suppress(Exception):
            # Detach parent ownership so profile lifetime is controlled solely
            # by page-destroy ordering below.
            profile.setParent(None)

        def _delete_profile_later(*_args) -> None:
            with contextlib.suppress(Exception):
                profile.deleteLater()

        with contextlib.suppress(Exception):
            page.destroyed.connect(_delete_profile_later)

    if view is not None:
        with contextlib.suppress(Exception):
            view.setPage(None)  # type: ignore[arg-type]

    if page is not None:
        with contextlib.suppress(Exception):
            page.deleteLater()

    if view is not None:
        with contextlib.suppress(Exception):
            view.deleteLater()

    if profile is not None and page is None:
        with contextlib.suppress(Exception):
            profile.deleteLater()
