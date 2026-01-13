from __future__ import annotations

from qtpy import QtWidgets


def apply_modern_theme(app: QtWidgets.QApplication) -> None:
    """Apply the 'modern' app-wide dark theme with accent color.

    This was previously embedded inside the training wizard. It has been
    extracted so callers can opt-in via settings.
    """
    app.setStyle("Fusion")

    # Slightly nicer default font
    f = app.font()
    if f.pointSize() < 11:
        f.setPointSize(11)
    app.setFont(f)

    qss = """
    /* ========= Global ========= */
    QWidget {
        color: #E9EEF5;
        background: #0F141A;
        font-size: 11pt;
    }

    QWizard {
        background: #0F141A;
    }

    /* Wizard page title/subtitle (we add these labels ourselves) */
    QLabel[role="title"] {
        font-size: 20pt;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: 0.3px;
    }
    QLabel[role="subtitle"] {
        color: #A7B3C2;
        font-size: 10.7pt;
    }

    QLabel[muted="true"] { color: #A7B3C2; }
    QLabel[good="true"]  { color: #4CD97B; }
    QLabel[bad="true"]   { color: #FF5C7A; }

    /* ========= Containers ========= */
    QGroupBox {
        border: 1px solid #283241;
        border-radius: 14px;
        margin-top: 14px;
        padding: 12px;
        background: #121922;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 10px;
        margin-left: 6px;
        color: #CFE3FF;
        font-weight: 700;
    }

    /* “Card” frames */
    QFrame[card="true"] {
        background: #121922;
        border: 1px solid #283241;
        border-radius: 16px;
    }
    QFrame[card="true"]:hover {
        border: 1px solid #2B7FFF;
    }
    QFrame[cardSelected="true"] {
        border: 1px solid #2B7FFF;
        background: #121E2C;
    }

    /* ========= Inputs ========= */
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextBrowser {
        background: #0D1218;
        border: 1px solid #283241;
        border-radius: 12px;
        padding: 8px 10px;
        selection-background-color: #2B7FFF;
    }
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextBrowser:focus {
        border: 1px solid #2B7FFF;
    }

    QTextBrowser {
        padding: 12px;
    }

    /* ========= Buttons ========= */
    QPushButton {
        background: #182230;
        border: 1px solid #2A3A50;
        border-radius: 12px;
        padding: 8px 12px;
        font-weight: 700;
    }
    QPushButton:hover { border-color: #2B7FFF; }
    QPushButton:pressed { background: #142033; }

    /* Primary action */
    QPushButton[primary="true"] {
        background: #2B7FFF;
        border: none;
        color: white;
        padding: 9px 14px;
        border-radius: 12px;
    }
    QPushButton[primary="true"]:hover {
        background: #3B8BFF;
    }

    /* ========= Tabs ========= */
    QTabWidget::pane {
        border: 1px solid #283241;
        border-radius: 14px;
        top: -1px;
        background: #121922;
    }
    QTabBar::tab {
        background: #121922;
        border: 1px solid #283241;
        padding: 8px 14px;
        border-top-left-radius: 12px;
        border-top-right-radius: 12px;
        margin-right: 6px;
        color: #A7B3C2;
        font-weight: 700;
    }
    QTabBar::tab:selected {
        color: #FFFFFF;
        border-bottom-color: #121922;
        background: #121E2C;
    }

    /* ========= Checkbox ========= */
    QCheckBox { spacing: 10px; }
    """

    app.setStyleSheet(qss)


def apply_dark_theme(app: QtWidgets.QApplication) -> None:
    """Apply a simple dark theme (OS-like dark preference)."""
    try:
        app.setStyle("Fusion")
    except Exception:
        pass
    f = app.font()
    if f.pointSize() < 11:
        f.setPointSize(11)
    app.setFont(f)

    # Dark theme: slightly different from 'modern' but simpler
    dark_qss = """
    QWidget { color: #E9EEF5; background: #121316; font-size: 11pt; }
    QGroupBox { background: #15171A; border: 1px solid #22262A; border-radius: 10px; }
    QLabel[muted="true"] { color: #9CA6B2; }
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextBrowser { background: #0E1113; border: 1px solid #23282C; color: #E9EEF5; }
    QPushButton { background: #1B1F23; border: 1px solid #2B2F33; color: #E9EEF5; }
    QPushButton[primary="true"] { background: #2B7FFF; color: white; }
    """
    app.setStyleSheet(dark_qss)


def apply_light_theme(app: QtWidgets.QApplication) -> None:
    """Apply the 'light' app theme.

    This is the default app theme (light mode) used when `ui/theme` is
    set to "light".
    """
    try:
        app.setStyle("Fusion")
    except Exception:
        pass

    # Slightly nicer default font
    f = app.font()
    if f.pointSize() < 11:
        f.setPointSize(11)
    app.setFont(f)

    qss = """
    /* ========= Global (light) ========= */
    QWidget {
        color: #1B2630;
        background: #FFFFFF;
        font-size: 11pt;
    }

    QGroupBox {
        border: 1px solid #E3E8EE;
        border-radius: 8px;
        margin-top: 10px;
        padding: 10px;
        background: #FAFBFC;
    }
    QGroupBox::title {
        color: #0F1720;
        font-weight: 700;
    }

    QLabel[muted="true"] { color: #6B7785; }
    QLabel[good="true"]  { color: #0A8A3C; }
    QLabel[bad="true"]   { color: #C92A2A; }

    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextBrowser {
        background: #FFFFFF;
        border: 1px solid #D8E1EA;
        border-radius: 6px;
        padding: 6px 8px;
        selection-background-color: #BEE0FF;
        color: #0F1720;
    }
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QTextBrowser:focus {
        border: 1px solid #3B82F6;
    }

    QTextBrowser { padding: 8px; }

    QPushButton {
        background: #F3F6F9;
        border: 1px solid #D0DAE6;
        border-radius: 8px;
        padding: 6px 10px;
        color: #0F1720;
        font-weight: 600;
    }
    QPushButton:hover { border-color: #3B82F6; }

    QPushButton[primary="true"] {
        background: #2B7FFF;
        border: none;
        color: white;
        padding: 7px 12px;
        border-radius: 8px;
    }
    QPushButton[primary="true"]:hover { background: #3B82FF; }

    QTabWidget::pane { border: 1px solid #E3E8EE; background: #FAFBFC; }
    QTabBar::tab { background: #FAFBFC; border: 1px solid #E3E8EE; padding: 6px 10px; margin-right: 6px; }
    QTabBar::tab:selected { background: #FFFFFF; border-bottom-color: #FFFFFF; }

    QCheckBox { spacing: 6px; }
    """

    app.setStyleSheet(qss)


def refresh_app_styles(app: QtWidgets.QApplication) -> None:
    """Force a style refresh on all top-level widgets so theme changes take effect.

    This calls the style's unpolish/polish cycle and updates widgets. It is
    safe to call after `setStyleSheet` to ensure existing windows repaint.
    """
    if app is None:
        return
    try:
        style = app.style()
        for w in list(app.topLevelWidgets()):
            try:
                style.unpolish(w)
                style.polish(w)
                w.update()
            except Exception:
                # Best-effort; do not fail the whole operation for one widget
                continue
    except Exception:
        pass
