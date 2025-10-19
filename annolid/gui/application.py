from typing import Optional, Sequence

from qtpy import QtWidgets


def create_qapp(argv: Optional[Sequence[str]] = None) -> QtWidgets.QApplication:
    """Create (or return) the singleton QApplication instance."""
    existing_app = QtWidgets.QApplication.instance()
    if existing_app is not None:
        return existing_app
    return QtWidgets.QApplication(list(argv) if argv is not None else None)
