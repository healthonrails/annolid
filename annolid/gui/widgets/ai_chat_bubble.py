from __future__ import annotations

import sys
from datetime import datetime
from typing import Optional

from qtpy import QtCore, QtGui, QtWidgets
from annolid.gui.qt_compat import palette_color_role


def _emoji_supported(text: str) -> bool:
    candidate = str(text or "")
    if not candidate:
        return False
    font = QtWidgets.QApplication.font()
    metrics = QtGui.QFontMetrics(font)
    for ch in candidate:
        codepoint = ord(ch)
        if codepoint < 128:
            continue
        if not metrics.inFontUcs4(codepoint):
            return False
    return True


def _symbol_text(preferred: str, fallback: str) -> str:
    # Ubuntu/Linux font stacks can report emoji coverage but still render
    # placeholder glyphs in Qt widgets. Prefer stable ASCII labels there.
    if sys.platform.startswith("linux"):
        return str(fallback)
    return str(preferred) if _emoji_supported(preferred) else str(fallback)


class _ChatBubble(QtWidgets.QFrame):
    """Single chat bubble row with timestamp, collapsible thinking, and theme-aware styles."""

    def __init__(
        self,
        sender: str,
        text: str,
        *,
        is_user: bool,
        on_speak=None,
        on_copy=None,
        on_regenerate=None,
        on_stop=None,
        on_delete=None,
        on_open_link=None,
        on_open_link_in_browser=None,
        allow_regenerate: bool = False,
        allow_stop: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._sender = sender
        self._is_user = is_user
        self._ts = datetime.now().strftime("%H:%M")
        self._on_speak = on_speak
        self._on_copy = on_copy
        self._on_regenerate = on_regenerate
        self._on_stop = on_stop
        self._on_delete = on_delete
        self._on_open_link = on_open_link
        self._on_open_link_in_browser = on_open_link_in_browser
        self._allow_regenerate = bool(allow_regenerate)
        self._allow_stop = bool(allow_stop)
        self._raw_text = str(text or "")
        self._syncing_message_height = False
        self._syncing_thinking_height = False
        self._thinking_text = ""
        self._thinking_finished = False
        self._is_dark_theme = self._detect_dark_theme()

        self.setObjectName("chatBubble")
        self.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.setAutoFillBackground(True)
        self.setProperty("role", "user" if is_user else "assistant")

        self.main_layout = QtWidgets.QVBoxLayout(self)
        self.main_layout.setContentsMargins(12, 10, 12, 10)
        self.main_layout.setSpacing(4)

        if not self._is_user:
            self.sender_label = QtWidgets.QLabel(sender, self)
            self.sender_label.setObjectName("sender")
            self.main_layout.addWidget(self.sender_label)

        self.message_view = QtWidgets.QTextBrowser(self)
        self.message_view.setObjectName("messageView")
        self.message_view.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.message_view.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        self.message_view.setAutoFillBackground(False)
        self.message_view.setOpenExternalLinks(False)
        self.message_view.setOpenLinks(False)
        self.message_view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.message_view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.message_view.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContents
        )
        self.message_view.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse | QtCore.Qt.LinksAccessibleByMouse
        )
        self.message_view.anchorClicked.connect(self._handle_anchor_clicked)
        self.message_view.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.message_view.customContextMenuRequested.connect(
            self._show_message_context_menu
        )

        self.message_doc = QtGui.QTextDocument(self.message_view)
        self.message_doc.setDocumentMargin(0)
        text_option = self.message_doc.defaultTextOption()
        text_option.setWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.message_doc.setDefaultTextOption(text_option)
        self._apply_message_doc_styles(is_dark=self._is_dark_theme)
        self.message_view.setDocument(self.message_doc)

        viewport = self.message_view.viewport()
        viewport.setAutoFillBackground(False)
        viewport.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        viewport.setStyleSheet("background: transparent;")

        doc_layout = self.message_doc.documentLayout()
        if doc_layout is not None:
            doc_layout.documentSizeChanged.connect(self._sync_message_height)

        self.main_layout.addWidget(self.message_view)
        self._render_markdown(self._raw_text)

        self.thinking_toggle_button: Optional[QtWidgets.QToolButton] = None
        self.thinking_view: Optional[QtWidgets.QTextBrowser] = None
        self.thinking_doc: Optional[QtGui.QTextDocument] = None
        if not self._is_user:
            thinking_insert_idx = 1 if hasattr(self, "sender_label") else 0
            self.thinking_toggle_button = QtWidgets.QToolButton(self)
            self.thinking_toggle_button.setObjectName("thinkingToggleButton")
            self.thinking_toggle_button.setCursor(QtCore.Qt.PointingHandCursor)
            self.thinking_toggle_button.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
            self.thinking_toggle_button.clicked.connect(self.toggle_thinking_content)
            self.thinking_toggle_button.hide()
            self.main_layout.insertWidget(
                thinking_insert_idx, self.thinking_toggle_button
            )

            self.thinking_view = QtWidgets.QTextBrowser(self)
            self.thinking_view.setObjectName("thinkingView")
            self.thinking_view.setFrameShape(QtWidgets.QFrame.NoFrame)
            self.thinking_view.setOpenExternalLinks(False)
            self.thinking_view.setOpenLinks(False)
            self.thinking_view.setHorizontalScrollBarPolicy(
                QtCore.Qt.ScrollBarAlwaysOff
            )
            self.thinking_view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
            self.thinking_view.setSizeAdjustPolicy(
                QtWidgets.QAbstractScrollArea.AdjustToContents
            )
            self.thinking_view.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            self.thinking_doc = QtGui.QTextDocument(self.thinking_view)
            self.thinking_doc.setDocumentMargin(0)
            thinking_option = self.thinking_doc.defaultTextOption()
            thinking_option.setWrapMode(QtGui.QTextOption.WrapAtWordBoundaryOrAnywhere)
            self.thinking_doc.setDefaultTextOption(thinking_option)
            self._apply_thinking_doc_styles(is_dark=self._is_dark_theme)
            self.thinking_view.setDocument(self.thinking_doc)
            thinking_layout = self.thinking_doc.documentLayout()
            if thinking_layout is not None:
                thinking_layout.documentSizeChanged.connect(self._sync_thinking_height)
            self.thinking_view.hide()
            self.main_layout.insertWidget(thinking_insert_idx + 1, self.thinking_view)

        self.footer_layout = QtWidgets.QHBoxLayout()
        self.footer_layout.setContentsMargins(0, 4, 0, 0)
        self.footer_layout.setSpacing(8)

        self.actions_layout = QtWidgets.QHBoxLayout()
        self.actions_layout.setSpacing(4)

        self.speak_button = self._create_action_button(
            _symbol_text("🔊", "S"), "Read aloud"
        )
        self.copy_button = self._create_action_button(
            _symbol_text("📋", "C"), "Copy text"
        )
        self.regenerate_button = self._create_action_button(
            _symbol_text("🔄", "R"), "Regenerate"
        )
        self.stop_button = self._create_action_button(
            _symbol_text("⏹", "X"), "Stop running"
        )
        self.delete_button = self._create_action_button(
            _symbol_text("🗑", "D"), "Delete message"
        )
        self.regenerate_button.setVisible(
            (not self._is_user) and self._allow_regenerate
        )
        self.stop_button.setVisible((not self._is_user) and self._allow_stop)

        self.speak_button.clicked.connect(self._speak)
        self.copy_button.clicked.connect(self._copy_text)
        self.regenerate_button.clicked.connect(self._regenerate)
        self.stop_button.clicked.connect(self._stop)
        self.delete_button.clicked.connect(self._delete)

        self.actions_layout.addWidget(self.speak_button)
        self.actions_layout.addWidget(self.copy_button)
        self.actions_layout.addWidget(self.regenerate_button)
        self.actions_layout.addWidget(self.stop_button)
        self.actions_layout.addWidget(self.delete_button)
        self.actions_layout.addStretch(1)

        self.footer_layout.addLayout(self.actions_layout)

        self.meta_label = QtWidgets.QLabel(self._ts, self)
        self.meta_label.setObjectName("meta")
        self.meta_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        self.footer_layout.addWidget(self.meta_label)

        self.main_layout.addLayout(self.footer_layout)
        self.apply_theme(is_dark=self._is_dark_theme)

    def _detect_dark_theme(self) -> bool:
        try:
            palette = self.palette()
            return bool(palette.color(palette_color_role("Window")).lightness() < 128)
        except Exception:
            return True

    def _create_action_button(self, text_icon: str, tooltip: str):
        btn = QtWidgets.QPushButton(text_icon, self)
        btn.setObjectName("bubbleActionButton")
        btn.setToolTip(tooltip)
        btn.setCursor(QtCore.Qt.PointingHandCursor)
        btn.setFixedSize(28, 28)
        return btn

    def _apply_action_button_styles(self, *, is_dark: bool) -> None:
        if is_dark:
            fg = "#8f96a3"
            hover_bg = "rgba(255, 255, 255, 0.1)"
            hover_fg = "#E0E0E0"
            pressed_bg = "rgba(255, 255, 255, 0.2)"
        else:
            fg = "#5f6368"
            hover_bg = "rgba(0, 0, 0, 0.08)"
            hover_fg = "#202124"
            pressed_bg = "rgba(0, 0, 0, 0.14)"
        qss = f"""
            QPushButton#bubbleActionButton {{
                background: transparent;
                border: none;
                color: {fg};
                font-size: 14px;
                padding: 4px;
                border-radius: 4px;
            }}
            QPushButton#bubbleActionButton:hover {{
                background: {hover_bg};
                color: {hover_fg};
            }}
            QPushButton#bubbleActionButton:pressed {{
                background: {pressed_bg};
            }}
        """
        for btn in (
            self.speak_button,
            self.copy_button,
            self.regenerate_button,
            self.stop_button,
            self.delete_button,
        ):
            btn.setStyleSheet(qss)

    def _apply_thinking_doc_styles(self, *, is_dark: bool) -> None:
        if self.thinking_doc is None:
            return
        thinking_fg = "#AAB2BD" if is_dark else "#5f6368"
        self.thinking_doc.setDefaultStyleSheet(
            f"""
            body, p, li, ul, ol {{
                color: {thinking_fg};
                font-size: 12px;
                line-height: 1.35;
            }}
            p {{ margin: 0 0 6px 0; }}
            p:last-child {{ margin-bottom: 0; }}
            """
        )

    def apply_theme(self, *, is_dark: bool) -> None:
        self._is_dark_theme = bool(is_dark)
        self._apply_message_doc_styles(is_dark=self._is_dark_theme)
        self._apply_thinking_doc_styles(is_dark=self._is_dark_theme)
        self._apply_action_button_styles(is_dark=self._is_dark_theme)
        # Re-render so QTextDocument reapplies the updated default stylesheet.
        self._render_markdown(self._raw_text)
        if self._thinking_text and self.thinking_doc is not None:
            if hasattr(self.thinking_doc, "setMarkdown"):
                self.thinking_doc.setMarkdown(self._thinking_text)
            else:
                self.thinking_doc.setPlainText(self._thinking_text)
            self._sync_thinking_height()

    def _apply_message_doc_styles(self, *, is_dark: Optional[bool] = None) -> None:
        dark = self._is_dark_theme if is_dark is None else bool(is_dark)
        base_css = """
            body, p, li, ul, ol, blockquote {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                font-size: 13px;
                line-height: 1.4;
            }
            p { margin: 0 0 8px 0; }
            p:last-child { margin-bottom: 0; }
            a { text-decoration: none; font-weight: 500; }
            pre {
                font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
                font-size: 12px;
                border-radius: 6px;
                padding: 10px;
                margin: 8px 0;
            }
            code {
                font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
                font-size: 12px;
                padding: 2px 4px;
                border-radius: 4px;
            }
        """

        if dark:
            if self._is_user:
                theme_css = """
                body, p, li, ul, ol, blockquote { color: #FFFFFF; }
                a { color: #d0e8ff; }
                pre { background: rgba(0,0,0,0.2); color: #FFFFFF; border: 1px solid rgba(255,255,255,0.2); }
                code { background: rgba(0,0,0,0.2); color: #FFFFFF; }
                """
            else:
                theme_css = """
                body, p, li, ul, ol, blockquote { color: #E0E0E0; }
                a { color: #64B5F6; }
                pre { background: #2b2d31; color: #E0E0E0; border: 1px solid #3f4148; }
                code { background: #2f3136; color: #E0E0E0; }
                """
        else:
            if self._is_user:
                theme_css = """
                body, p, li, ul, ol, blockquote { color: #172b4d; }
                a { color: #1a73e8; }
                pre { background: rgba(26, 115, 232, 0.08); color: #172b4d; border: 1px solid #c6dafc; }
                code { background: rgba(26, 115, 232, 0.1); color: #172b4d; }
                """
            else:
                theme_css = """
                body, p, li, ul, ol, blockquote { color: #202124; }
                a { color: #1a73e8; }
                pre { background: #e8eaed; color: #202124; border: 1px solid #dadce0; }
                code { background: #e8eaed; color: #202124; }
                """

        self.message_doc.setDefaultStyleSheet(base_css + theme_css)

    def append_text(self, chunk: str) -> None:
        self._raw_text += str(chunk or "")
        self._render_markdown(self._raw_text)

    def set_text(self, text: str) -> None:
        self._raw_text = str(text or "")
        self._render_markdown(self._raw_text)

    def text(self) -> str:
        return str(self.message_doc.toPlainText() or "").strip()

    def raw_text(self) -> str:
        return str(self._raw_text or "").strip()

    def _render_markdown(self, text: str) -> None:
        content = str(text or "")
        if hasattr(self.message_doc, "setMarkdown"):
            self.message_doc.setMarkdown(content)
        else:
            self.message_doc.setPlainText(content)
        self._sync_message_height()

    def _sync_message_height(self, _doc_size: Optional[QtCore.QSizeF] = None) -> None:
        if self._syncing_message_height:
            return
        self._syncing_message_height = True
        try:
            vp_width = self.message_view.viewport().width()
            if vp_width > 0:
                self.message_doc.setTextWidth(float(vp_width))

            doc_height = int(self.message_doc.size().height())
            target_height = doc_height + 25

            parent_window = self.window()
            max_limit = 600
            if parent_window:
                max_limit = int(max(parent_window.height() * 0.6, 200))

            if target_height > max_limit:
                self.message_view.setFixedHeight(max_limit)
                self.message_view.setVerticalScrollBarPolicy(
                    QtCore.Qt.ScrollBarAsNeeded
                )
            else:
                self.message_view.setFixedHeight(target_height)
                self.message_view.setVerticalScrollBarPolicy(
                    QtCore.Qt.ScrollBarAlwaysOff
                )
        finally:
            self._syncing_message_height = False

    def _sync_thinking_height(self, _doc_size: Optional[QtCore.QSizeF] = None) -> None:
        if self._syncing_thinking_height:
            return
        if self.thinking_view is None or self.thinking_doc is None:
            return
        if not self.thinking_view.isVisible():
            return
        self._syncing_thinking_height = True
        try:
            vp_width = self.thinking_view.viewport().width()
            if vp_width > 0:
                self.thinking_doc.setTextWidth(float(vp_width))
            doc_height = int(self.thinking_doc.size().height())
            target_height = max(28, doc_height + 12)
            parent_window = self.window()
            max_limit = 240
            if parent_window:
                max_limit = int(max(parent_window.height() * 0.35, 120))
            if target_height > max_limit:
                self.thinking_view.setFixedHeight(max_limit)
                self.thinking_view.setVerticalScrollBarPolicy(
                    QtCore.Qt.ScrollBarAsNeeded
                )
            else:
                self.thinking_view.setFixedHeight(target_height)
                self.thinking_view.setVerticalScrollBarPolicy(
                    QtCore.Qt.ScrollBarAlwaysOff
                )
        finally:
            self._syncing_thinking_height = False

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._sync_message_height()
        self._sync_thinking_height()

    def set_message_width(self, width: int) -> None:
        if self.message_doc:
            self.message_doc.setTextWidth(float(width))

    def apply_layout_width(self, text_width: int, bubble_max_width: int) -> None:
        del bubble_max_width
        if self.message_doc:
            self.message_doc.setTextWidth(float(text_width))
        if self.thinking_doc:
            self.thinking_doc.setTextWidth(float(text_width))
        self._sync_message_height()
        self._sync_thinking_height()

    def set_thinking_text(self, text: str, *, in_progress: bool = True) -> None:
        if self._is_user:
            return
        value = str(text or "").strip()
        self._thinking_text = value
        if in_progress:
            self._thinking_finished = False
        if self.thinking_toggle_button is None or self.thinking_view is None:
            return
        if not value:
            self.thinking_toggle_button.hide()
            self.thinking_view.hide()
            return
        if self.thinking_doc is not None:
            if hasattr(self.thinking_doc, "setMarkdown"):
                self.thinking_doc.setMarkdown(value)
            else:
                self.thinking_doc.setPlainText(value)
        self.thinking_toggle_button.show()
        self.thinking_view.show()
        self._sync_thinking_height()
        self._refresh_thinking_toggle_text()

    def finish_thinking(self, collapse: bool = True) -> None:
        if self._is_user:
            return
        self._thinking_finished = True
        if self.thinking_toggle_button is None or self.thinking_view is None:
            return
        if not self._thinking_text:
            self.thinking_toggle_button.hide()
            self.thinking_view.hide()
            return
        if collapse:
            self.thinking_view.hide()
        self._refresh_thinking_toggle_text()

    def toggle_thinking_content(self) -> None:
        if self.thinking_view is None or self.thinking_toggle_button is None:
            return
        if not self._thinking_text:
            return
        self.thinking_view.setVisible(not self.thinking_view.isVisible())
        self._refresh_thinking_toggle_text()
        self._sync_thinking_height()

    def _refresh_thinking_toggle_text(self) -> None:
        if self.thinking_toggle_button is None:
            return
        is_open = bool(
            self.thinking_view is not None and self.thinking_view.isVisible()
        )
        chevron = "▼" if is_open else "▶"
        status = "finished" if self._thinking_finished else "running"
        self.thinking_toggle_button.setText(f"{chevron} Thinking ({status})")

    def _speak(self) -> None:
        if callable(self._on_speak):
            self._on_speak(self.text())

    def _copy_text(self) -> None:
        if callable(self._on_copy):
            self._on_copy(self.text())

    def _regenerate(self) -> None:
        if callable(self._on_regenerate):
            self._on_regenerate(self.text())

    def _stop(self) -> None:
        if callable(self._on_stop):
            self._on_stop()

    def _delete(self) -> None:
        if callable(self._on_delete):
            self._on_delete(self)

    def set_stop_visible(self, visible: bool) -> None:
        self.stop_button.setVisible(bool(visible))

    def _handle_anchor_clicked(self, url: QtCore.QUrl) -> None:
        target = str(url.toString() if isinstance(url, QtCore.QUrl) else url).strip()
        if not target:
            return
        if callable(self._on_open_link):
            self._on_open_link(target)

    def _show_message_context_menu(self, position: QtCore.QPoint) -> None:
        menu = self.message_view.createStandardContextMenu()
        link = str(self.message_view.anchorAt(position) or "").strip()
        if link:
            menu.addSeparator()

            open_viewer_action = menu.addAction("Open Link in Annolid Web Viewer")
            open_viewer_action.triggered.connect(
                lambda _checked=False, value=link: self._open_link_in_viewer(value)
            )

            open_browser_action = menu.addAction("Open Link in Default Browser")
            open_browser_action.triggered.connect(
                lambda _checked=False, value=link: self._open_link_in_browser(
                    value, "default"
                )
            )

            open_tab_action = menu.addAction("Open Link in Default Browser (New Tab)")
            open_tab_action.triggered.connect(
                lambda _checked=False, value=link: self._open_link_in_browser(
                    value, "new_tab"
                )
            )

            open_window_action = menu.addAction(
                "Open Link in Default Browser (New Window)"
            )
            open_window_action.triggered.connect(
                lambda _checked=False, value=link: self._open_link_in_browser(
                    value, "new_window"
                )
            )

            copy_link_action = menu.addAction("Copy Link Address")
            copy_link_action.triggered.connect(
                lambda _checked=False,
                value=link: QtGui.QGuiApplication.clipboard().setText(value)
            )
        if callable(self._on_delete):
            menu.addSeparator()
            delete_action = menu.addAction("Delete Message")
            delete_action.triggered.connect(self._delete)

        menu.exec_(self.message_view.viewport().mapToGlobal(position))

    def _open_link_in_viewer(self, url: str) -> None:
        target = str(url or "").strip()
        if not target:
            return
        if callable(self._on_open_link):
            self._on_open_link(target)

    def _open_link_in_browser(self, url: str, mode: str) -> None:
        target = str(url or "").strip()
        if not target:
            return
        if callable(self._on_open_link_in_browser):
            self._on_open_link_in_browser(target, str(mode or "default"))


__all__ = ["_ChatBubble", "_emoji_supported", "_symbol_text"]
