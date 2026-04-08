from __future__ import annotations


def post_window_status(target, message: str, timeout: int = 4000) -> None:
    """Post a status message across the supported Annolid window APIs."""
    post_status = getattr(target, "post_status_message", None)
    if callable(post_status):
        post_status(message, timeout)
        return
    status = getattr(target, "status", None)
    if callable(status):
        try:
            status(message, timeout)
            return
        except TypeError:
            status(message)
            return
    status_bar = getattr(target, "statusBar", None)
    if callable(status_bar):
        try:
            bar = status_bar()
            if bar is not None:
                bar.showMessage(str(message), int(timeout))
        except Exception:
            pass
