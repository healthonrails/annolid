"""PDF viewer public API.

This is a small wrapper module that re-exports `PdfViewerWidget` from the
implementation module to keep import paths stable while allowing the
implementation to be refactored into smaller units.
"""

from __future__ import annotations

from annolid.gui.widgets.pdf_viewer_impl import PdfViewerWidget

__all__ = ["PdfViewerWidget"]
