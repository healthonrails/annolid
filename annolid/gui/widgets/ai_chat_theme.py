"""Theme tokens, WCAG contrast helpers, and stylesheet builder for AIChatWidget."""

from __future__ import annotations

from typing import Dict


def _srgb_channel(value: int) -> float:
    scaled = max(0.0, min(1.0, float(value) / 255.0))
    if scaled <= 0.03928:
        return scaled / 12.92
    return ((scaled + 0.055) / 1.055) ** 2.4


def relative_luminance(hex_color: str) -> float:
    """Compute WCAG 2.1 relative luminance for a `#RRGGBB` color string."""
    raw = str(hex_color or "").strip().lstrip("#")
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) != 6:
        raise ValueError(f"Expected #RRGGBB hex color, got {hex_color!r}")
    r = _srgb_channel(int(raw[0:2], 16))
    g = _srgb_channel(int(raw[2:4], 16))
    b = _srgb_channel(int(raw[4:6], 16))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(color_a: str, color_b: str) -> float:
    """Compute WCAG 2.1 contrast ratio (>= 1.0, <= 21.0) between two hex colors."""
    lum_a = relative_luminance(color_a)
    lum_b = relative_luminance(color_b)
    lighter = max(lum_a, lum_b)
    darker = min(lum_a, lum_b)
    return (lighter + 0.05) / (darker + 0.05)


def chat_theme_tokens(is_dark: bool) -> Dict[str, str]:
    """Return theme-aware color tokens with WCAG AA (>= 4.5:1) contrast pairs."""
    if is_dark:
        return {
            "bg_main": "#111317",
            "bg_input": "#1a1d23",
            "fg_main": "#e7e8ea",
            "border_main": "#31353c",
            "title_fg": "#f4f5f6",
            "subtitle_fg": "#9ea4af",
            "bubble_user_bg": "#244f48",
            "bubble_user_fg": "#f4f7f6",
            "bubble_user_border": "#3a756b",
            "bubble_assistant_bg": "#1f2228",
            "bubble_assistant_border": "#31353e",
            "send_hover_bg": "#2e655c",
            "stop_bg": "#4f1e1e",
            "stop_fg": "#ffe0e0",
            "stop_border": "#993d3d",
            "stop_hover_bg": "#682828",
            "attachment_bg": "#1f2937",
            "attachment_fg": "#e5ecf6",
            "attachment_border": "#3b4c63",
            "danger_fg": "#ff8a80",
            "hover_subtle": "rgba(255, 255, 255, 0.12)",
        }
    return {
        "bg_main": "#ffffff",
        "bg_input": "#f1f3f4",
        "fg_main": "#202124",
        "border_main": "#dadce0",
        "title_fg": "#1a73e8",
        "subtitle_fg": "#5f6368",
        "bubble_user_bg": "#e8f0fe",
        "bubble_user_fg": "#172b4d",
        "bubble_user_border": "#c6dafc",
        "bubble_assistant_bg": "#f1f3f4",
        "bubble_assistant_border": "#e0e0e0",
        "send_hover_bg": "#d2e3fc",
        "stop_bg": "#fce8e6",
        "stop_fg": "#a50e0e",
        "stop_border": "#e0a0a0",
        "stop_hover_bg": "#f8d0cc",
        "attachment_bg": "#e8f0fe",
        "attachment_fg": "#174ea6",
        "attachment_border": "#c6dafc",
        "danger_fg": "#b3261e",
        "hover_subtle": "rgba(0, 0, 0, 0.08)",
    }


def build_chat_widget_stylesheet(is_dark: bool) -> str:
    """Build the complete QSS stylesheet for AIChatWidget."""
    t = chat_theme_tokens(is_dark)
    return f"""
        QWidget#AIChatWidget {{
            background: {t["bg_main"]};
            color: {t["fg_main"]};
            border: none;
            outline: none;
        }}
        /* Specific containers to avoid broad QWidget styling */
        #chatContainer, #headerBar, #inputBarContainer {{
            background: {t["bg_main"]};
            color: {t["fg_main"]};
            border: none;
        }}
        QComboBox {{
            border: 1px solid {t["border_main"]};
            border-radius: 8px;
            background: {t["bg_input"]};
            color: {t["fg_main"]};
            min-height: 24px;
            padding: 3px 8px;
        }}
        QLineEdit {{
            border: 1px solid {t["border_main"]};
            border-radius: 8px;
            background: {t["bg_input"]};
            color: {t["fg_main"]};
            min-height: 24px;
            padding: 3px 8px;
        }}
        QScrollArea {{
            border: none;
            background: {t["bg_main"]};
        }}
        QScrollArea > QWidget > QWidget {{
            border: none;
            background: {t["bg_main"]};
        }}
        QPlainTextEdit {{
            border: 1px solid {t["border_main"]};
            border-radius: 12px;
            background: {t["bg_input"]};
            padding: 8px;
            font-size: 14px;
            color: {t["fg_main"]};
        }}
        QPlainTextEdit:focus {{
            border: 1px solid {t["title_fg"]};
        }}
        QLabel#chatTitleLabel {{
            color: {t["title_fg"]};
            font-size: 18px;
            font-weight: 700;
            background: transparent;
        }}
        QLabel#chatSubtitleLabel {{
            color: {t["subtitle_fg"]};
            font-size: 12px;
            background: transparent;
        }}
        QLabel#chatEmptyState {{
            color: {t["subtitle_fg"]};
            font-size: 13px;
            background: transparent;
        }}
        QLabel#chatCapabilitySummaryLabel {{
            color: {t["fg_main"]};
            font-size: 11px;
            background: rgba(128, 128, 128, 0.10);
            border: 1px solid {t["border_main"]};
            border-radius: 10px;
            padding: 2px 8px;
        }}
        QToolButton#chatTopIconButton {{
            background: transparent;
            border: 1px solid transparent;
            border-radius: 6px;
            padding: 4px;
            color: {t["fg_main"]};
        }}
        QToolButton#chatTopIconButton:hover {{
            background: {t["hover_subtle"]};
            border: 1px solid {t["border_main"]};
        }}
        QPushButton#quickActionButton {{
            border: 1px solid {t["border_main"]};
            border-radius: 14px;
            padding: 6px 12px;
            font-size: 12px;
            background: {t["bg_input"]};
            color: {t["fg_main"]};
        }}
        QPushButton#quickActionButton:hover {{
            background: {t["bubble_assistant_border"]};
        }}
        QPushButton#sendButton, QToolButton#sendButton {{
            border: 1px solid {t["bubble_user_border"]};
            border-radius: 10px;
            padding: 4px 10px;
            background: {t["bubble_user_bg"]};
            color: {t["bubble_user_fg"]};
            font-weight: 600;
        }}
        QPushButton#sendButton:hover, QToolButton#sendButton:hover {{
            background: {t["send_hover_bg"]};
        }}
        QPushButton#sendButton:disabled, QToolButton#sendButton:disabled {{
            color: {t["subtitle_fg"]};
            background: {t["bg_input"]};
            border: 1px solid {t["border_main"]};
        }}
        QToolButton#talkButton {{
            border: 1px solid {t["border_main"]};
            border-radius: 10px;
            padding: 4px 6px;
            background: {t["bg_main"]};
            color: {t["fg_main"]};
        }}
        QToolButton#talkButton:hover {{
            background: {t["hover_subtle"]};
        }}
        QToolButton#stopButton {{
            border: 1px solid {t["stop_border"]};
            border-radius: 10px;
            padding: 4px 8px;
            background: {t["stop_bg"]};
            color: {t["stop_fg"]};
            font-weight: 600;
        }}
        QToolButton#stopButton:hover {{
            background: {t["stop_hover_bg"]};
        }}
        QFrame#attachmentChipBar {{
            background: {t["attachment_bg"]};
            border: 1px solid {t["attachment_border"]};
            border-radius: 10px;
        }}
        QLabel#attachmentChipLabel {{
            color: {t["attachment_fg"]};
            font-size: 11px;
            font-weight: 600;
            background: transparent;
        }}
        QToolButton#attachmentClearButton {{
            background: transparent;
            border: none;
            border-radius: 6px;
            color: {t["attachment_fg"]};
            font-weight: 700;
        }}
        QToolButton#attachmentClearButton:hover {{
            background: {t["hover_subtle"]};
        }}
        QPushButton#jumpToLatestButton {{
            border: 1px solid {t["border_main"]};
            border-radius: 12px;
            padding: 4px 12px;
            font-size: 11px;
            font-weight: 600;
            background: {t["bg_input"]};
            color: {t["fg_main"]};
        }}
        QPushButton#jumpToLatestButton:hover {{
            background: {t["bubble_assistant_border"]};
        }}
        /* Input Bar Icons */
        QToolButton#chatInputButton {{
            background: transparent;
            border: none;
            border-radius: 6px;
            color: {t["fg_main"]};
        }}
        QToolButton#chatInputButton:hover {{
            background: {t["hover_subtle"]};
        }}
        /* Chat Bubbles */
        QFrame#chatBubble {{
            border-radius: 12px;
        }}
        QFrame#chatBubble[role="user"] {{
            background-color: {t["bubble_user_bg"]};
            border: 1px solid {t["bubble_user_border"]};
            margin-left: 40px;
        }}
        QFrame#chatBubble[role="assistant"] {{
            background-color: {t["bubble_assistant_bg"]};
            border: 1px solid {t["bubble_assistant_border"]};
            margin-right: 40px;
        }}
        /* Bubble Actions */
        QPushButton#bubbleActionButton {{
            background: transparent;
            border: none;
            border-radius: 4px;
            padding: 2px;
        }}
        QPushButton#bubbleActionButton:hover {{
            background: {t["hover_subtle"]};
        }}
        QToolButton#thinkingToggleButton {{
            border: none;
            background: transparent;
            color: {t["subtitle_fg"]};
            font-size: 11px;
            font-weight: 600;
            text-align: left;
            padding: 2px 0;
        }}
        QToolButton#thinkingToggleButton:hover {{
            color: {t["fg_main"]};
        }}
        QTextBrowser#thinkingView {{
            border: 1px solid {t["border_main"]};
            border-radius: 8px;
            background: rgba(128, 128, 128, 0.08);
            padding: 6px 8px;
        }}
        QLabel#sender {{
            color: {t["subtitle_fg"]};
            font-size: 11px;
            font-weight: 600;
            margin-bottom: 2px;
            background: transparent;
        }}
        QLabel#meta {{
            color: {t["subtitle_fg"]};
            font-size: 10px;
            background: transparent;
        }}
        /* Floating Input Bar */
        QFrame#inputBarContainer {{
            background: {t["bg_input"]};
            border-top: 1px solid {t["border_main"]};
            border-bottom-left-radius: 12px;
            border-bottom-right-radius: 12px;
        }}
        QFrame#zulipDraftPanel {{
            background: {t["bg_main"]};
            border: 1px solid {t["border_main"]};
            border-radius: 12px;
        }}
        QLabel#chatStatusLabel {{
            color: {t["subtitle_fg"]};
            font-size: 11px;
            background: transparent;
        }}
        QLabel#promptCountLabel {{
            color: {t["subtitle_fg"]};
            font-size: 11px;
            background: transparent;
        }}
        QLabel#promptCountLabel[limitReached="true"] {{
            color: {t["danger_fg"]};
            font-weight: 700;
        }}
    """
