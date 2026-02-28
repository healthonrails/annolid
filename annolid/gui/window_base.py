from __future__ import annotations

import os.path as osp
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

from qtpy import QtCore, QtGui, QtWidgets

from annolid.configs import get_config
from annolid.gui.file_dock import FileDockMixin
from annolid.gui.label_file import LabelFile, LabelFileError
from annolid.utils.annotation_compat import (
    AI_MODELS,
    PY2,
    QT4,
    QT5,
    addActions,
    newAction,
    utils,
)
from annolid.version import __version__
from annolid.utils.logger import logger


def format_tool_button_text(text: str) -> str:
    """Format toolbar labels into a compact stacked form.

    Example: "&Open Video" -> "Open\\nVideo"
    """
    base = (text or "").replace("&", "").replace("/", " ").replace("…", "").strip()
    if not base or "\n" in base:
        return base
    parts = base.split()
    if len(parts) >= 2:
        first = parts[0]
        rest = " ".join(parts[1:])
        return f"{first}\n{rest}"
    return base


class AnnolidLabelListItem(QtWidgets.QListWidgetItem):
    """List item that keeps a direct reference to its Shape object."""

    def __init__(self, text: str, shape=None):
        super().__init__(text)
        self._shape = shape

    def shape(self):
        return self._shape

    def setShape(self, shape) -> None:
        self._shape = shape


class AnnolidLabelListWidget(QtWidgets.QListWidget):
    """QListWidget with LabelMe-like iteration helpers."""

    VISIBILITY_STATE_ROLE = int(QtCore.Qt.UserRole) + 10

    shapeVisibilityChanged = QtCore.Signal(object, bool)
    shapeDeleteRequested = QtCore.Signal(object)
    shapesDeleteRequested = QtCore.Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        self.itemChanged.connect(self._on_item_changed)

    def __iter__(self):
        for idx in range(self.count()):
            yield self.item(idx)

    def _on_item_changed(self, item: QtWidgets.QListWidgetItem) -> None:
        if not isinstance(item, AnnolidLabelListItem):
            return
        if not (item.flags() & QtCore.Qt.ItemIsUserCheckable):
            return
        shape = item.shape()
        if shape is None:
            return

        is_visible = item.checkState() == QtCore.Qt.Checked
        prev = item.data(self.VISIBILITY_STATE_ROLE)
        if prev is not None and bool(prev) == bool(is_visible):
            return

        # Store the last-known state so non-checkbox item changes (text/color)
        # don't repeatedly trigger visibility wiring downstream.
        item.setData(self.VISIBILITY_STATE_ROLE, bool(is_visible))
        self.shapeVisibilityChanged.emit(shape, bool(is_visible))

    def _is_checkbox_click(
        self, item: QtWidgets.QListWidgetItem, pos: QtCore.QPoint
    ) -> bool:
        if not (item.flags() & QtCore.Qt.ItemIsUserCheckable):
            return False
        rect = self.visualItemRect(item)
        opt = QtWidgets.QStyleOptionViewItem()
        try:
            opt.initFrom(self)
        except Exception:
            pass
        opt.rect = rect
        try:
            opt.features |= QtWidgets.QStyleOptionViewItem.HasCheckIndicator
        except Exception:
            # Older Qt bindings may not expose this flag; best-effort.
            pass
        try:
            opt.checkState = item.checkState()
        except Exception:
            pass
        check_rect = self.style().subElementRect(
            QtWidgets.QStyle.SE_ItemViewItemCheckIndicator, opt, self
        )
        return check_rect.contains(pos)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # pragma: no cover
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # pragma: no cover
        # User-friendly behavior: left-click should only select/list-focus.
        # Deletion is available via explicit shortcuts and right-click menu.
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(
        self, event: QtGui.QMouseEvent
    ) -> None:  # pragma: no cover
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(
        self, event: QtGui.QContextMenuEvent
    ) -> None:  # pragma: no cover
        item = self.itemAt(event.pos())
        if item is not None and not item.isSelected():
            self.clearSelection()
            item.setSelected(True)
        self._show_delete_menu(self.viewport().mapToGlobal(event.pos()))
        event.accept()

    def _show_delete_menu(
        self, global_pos: QtCore.QPoint, clicked_item: QtWidgets.QListWidgetItem = None
    ) -> None:
        selected_shapes = []
        for item in self.selectedItems():
            if not isinstance(item, AnnolidLabelListItem):
                continue
            shape = item.shape()
            if shape is not None:
                selected_shapes.append(shape)

        # Fallback for single-click flow where selection may lag behind.
        if not selected_shapes and isinstance(clicked_item, AnnolidLabelListItem):
            shape = clicked_item.shape()
            if shape is not None:
                selected_shapes = [shape]

        if not selected_shapes:
            return

        menu = QtWidgets.QMenu(self)
        if len(selected_shapes) > 1:
            delete_action = menu.addAction(self.tr("Delete selected shapes"))
        else:
            delete_action = menu.addAction(self.tr("Delete shape"))
        chosen = menu.exec_(global_pos)
        if chosen is delete_action:
            if len(selected_shapes) > 1:
                self.shapesDeleteRequested.emit(selected_shapes)
            else:
                self.shapeDeleteRequested.emit(selected_shapes[0])


class AnnolidUniqLabelListWidget(QtWidgets.QListWidget):
    """Label summary list with helper methods used by AnnolidWindow (LabelMe-style)."""

    def findItemByLabel(self, label: str):
        target = str(label or "")
        if not target:
            return None
        for idx in range(self.count()):
            item = self.item(idx)
            try:
                if item.data(QtCore.Qt.UserRole) == target:
                    return item
            except Exception:
                continue
        return None

    def createItemFromLabel(self, label: str) -> QtWidgets.QListWidgetItem:
        item = QtWidgets.QListWidgetItem()
        item.setData(QtCore.Qt.UserRole, str(label or ""))
        item.setText(str(label or ""))
        return item

    def setItemLabel(self, item: QtWidgets.QListWidgetItem, text: str, rgb) -> None:
        """Update the visible text and color while preserving the stored label (UserRole)."""
        try:
            r, g, b = rgb
            item.setForeground(QtGui.QBrush(QtGui.QColor(int(r), int(g), int(b))))
        except Exception:
            pass
        item.setText(str(text or ""))


class AnnolidToolButton(QtWidgets.QToolButton):
    """Toolbar button that renders its text as multi-line under the icon.

    Qt's native toolbar toolbutton label painting is typically single-line on
    macOS styles, which ignores '\n' and prevents compact 2-row labels such as
    "Open\\nVideo" or "Edit\\nPolygons". We draw the label ourselves to ensure
    consistent stacked text rendering across platforms.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAutoRaise(True)
        self.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)

    def sizeHint(self) -> QtCore.QSize:  # pragma: no cover - UI sizing
        hint = super().sizeHint()
        fm = QtGui.QFontMetrics(self.font())
        text = ""
        act = self.defaultAction()
        if act is not None:
            text = act.iconText() or act.text() or ""
        else:
            text = self.text() or ""
        lines = max(1, text.count("\n") + 1)
        icon_h = self.iconSize().height()
        # Heuristic: give enough height for icon + two lines + small padding.
        h = max(hint.height(), icon_h + (fm.lineSpacing() * lines) + 10)
        w = max(hint.width(), self.minimumWidth())
        return QtCore.QSize(w, h)

    def paintEvent(
        self, event: QtGui.QPaintEvent
    ) -> None:  # pragma: no cover - UI painting
        painter = QtGui.QPainter(self)
        opt = QtWidgets.QStyleOptionToolButton()
        self.initStyleOption(opt)

        style = self.style()

        # Draw the toolbutton chrome (hover/pressed/checked) but suppress native
        # icon/text painting. macOS styles often paint labels as single-line.
        opt_chrome = QtWidgets.QStyleOptionToolButton(opt)
        opt_chrome.text = ""
        opt_chrome.icon = QtGui.QIcon()
        style.drawComplexControl(
            QtWidgets.QStyle.CC_ToolButton, opt_chrome, painter, self
        )

        rect = opt.rect
        margin = 2
        spacing = 1
        content = rect.adjusted(margin, margin, -margin, -margin)

        # Icon rect at the top, centered.
        icon_sz = opt.iconSize if not opt.iconSize.isEmpty() else self.iconSize()
        icon_w, icon_h = icon_sz.width(), icon_sz.height()
        icon_x = content.x() + max(0, (content.width() - icon_w) // 2)
        icon_y = content.y()
        icon_rect = QtCore.QRect(icon_x, icon_y, icon_w, icon_h)

        # Text rect below the icon.
        text_rect = QtCore.QRect(
            content.x(),
            icon_rect.bottom() + spacing,
            content.width(),
            max(0, content.bottom() - icon_rect.bottom() - spacing),
        )

        enabled = bool(opt.state & QtWidgets.QStyle.State_Enabled)
        checked = bool(opt.state & QtWidgets.QStyle.State_On)
        mode = QtGui.QIcon.Normal if enabled else QtGui.QIcon.Disabled
        state = QtGui.QIcon.On if checked else QtGui.QIcon.Off
        opt.icon.paint(painter, icon_rect, QtCore.Qt.AlignCenter, mode, state)

        # Multiline, centered label.
        # Use QAction.iconText() when available: QToolButton may reset its own
        # text to QAction.text() when the action changes (enabled/checked/etc).
        act = self.defaultAction()
        text = (act.iconText() if act is not None else "") or self.text()
        if text:
            pal = opt.palette
            group = QtGui.QPalette.Active if enabled else QtGui.QPalette.Disabled
            painter.setPen(pal.color(group, QtGui.QPalette.ButtonText))
            painter.drawText(
                text_rect,
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop | QtCore.Qt.TextWordWrap,
                text,
            )


class AnnolidToolBar(QtWidgets.QToolBar):
    def add_stacked_action(
        self,
        action: QtWidgets.QAction,
        stacked_text: str,
        *,
        width: int = 76,
        min_height: int = 72,
        icon_size: QtCore.QSize = QtCore.QSize(32, 32),
    ) -> QtWidgets.QAction:
        """Add an action as an AnnolidToolButton with 2-row text under the icon."""
        button = AnnolidToolButton(self)
        button.setIconSize(icon_size)
        button.setDefaultAction(action)
        # Keep the QAction's text intact for menus; use iconText for toolbar label.
        try:
            action.setIconText(stacked_text)
        except Exception:
            pass
        button.setMinimumWidth(width)
        button.setMaximumWidth(width)
        button.setMinimumHeight(min_height)
        button.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        return self.addWidget(button)


class AnnolidWindowBase(FileDockMixin, QtWidgets.QMainWindow):
    """In-tree replacement for the LabelMe MainWindow API used by Annolid."""

    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = 0, 1, 2

    def __init__(self, config=None):
        super().__init__()

        if isinstance(config, dict):
            self._config = get_config(config_from_args=config)
        else:
            self._config = get_config(config)

        self.image = QtGui.QImage()
        self.imagePath: Optional[str] = None
        self.imageData = None
        self.filename: Optional[str] = None
        self.labelFile = None
        self.otherData = {}
        self.dirty = False
        self.output_dir: Optional[str] = None
        self.lastOpenDir: Optional[str] = None
        self.recentFiles: list[str] = []
        self.imageList: list[str] = []
        self.zoomMode = self.FIT_WINDOW
        self.zoom_values: dict[str, tuple[int, int]] = {}
        self.scroll_values = {
            QtCore.Qt.Vertical: {},
            QtCore.Qt.Horizontal: {},
        }
        self.brightnessContrast_values: dict[
            str, tuple[Optional[int], Optional[int]]
        ] = {}
        self._icons_dir = Path(__file__).resolve().parent / "icons"

        self._selectAiModelComboBox = QtWidgets.QComboBox()
        self._suppress_unsaved_prompt = False

        self.labelList = AnnolidLabelListWidget()
        self.uniqLabelList = AnnolidUniqLabelListWidget()
        self._init_file_dock_ui()

        self.tools = AnnolidToolBar(self.tr("Tools"))
        self.tools.setObjectName("mainTools")
        self.tools.setMovable(False)
        self.tools.setFloatable(False)
        self.tools.setIconSize(QtCore.QSize(32, 32))
        self.tools.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        # Tighten toolbar spacing so more tools fit before overflow ("»").
        self.tools.setStyleSheet(
            "QToolBar { spacing: 0px; }QToolButton { margin: 0px; padding: 0px; }"
        )
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.tools)

        self.flag_widget = QtWidgets.QWidget(self)
        self.flag_dock = QtWidgets.QDockWidget(self.tr("Flags"), self)
        self.flag_dock.setObjectName("flagDock")
        self.flag_dock.setWidget(self.flag_widget)

        # Unique labels summary dock.
        self.label_dock = QtWidgets.QDockWidget(self.tr("Labels"), self)
        self.label_dock.setObjectName("labelDock")
        self.label_dock.setWidget(self.uniqLabelList)

        # Per-shape label instances dock.
        self.shape_dock = QtWidgets.QDockWidget(self.tr("Label Instances"), self)
        self.shape_dock.setObjectName("shapeDock")
        self.shape_dock.setWidget(self.labelList)

        # Add flag_dock first (top position), then tabify file_dock with it
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.flag_dock)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.file_dock)
        # Tabify Files and Flags docks together (Flags on top, Files as tab)
        self.tabifyDockWidget(self.flag_dock, self.file_dock)

        # Add label and shape docks below the tabified Flags/Files.
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.label_dock)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.shape_dock)
        # Tabify Labels + Label Instances, with Label Instances active by default.
        self.tabifyDockWidget(self.shape_dock, self.label_dock)
        self.shape_dock.raise_()

        self.zoomWidget = QtWidgets.QSpinBox(self)
        self.zoomWidget.setRange(1, 1000)
        self.zoomWidget.setValue(100)
        self.zoomWidget.setSuffix("%")
        self.zoomWidget.valueChanged.connect(self._on_zoom_changed)

        self.menus = SimpleNamespace(
            file=self.menuBar().addMenu(self.tr("&File")),
            edit=self.menuBar().addMenu(self.tr("&Edit")),
            view=self.menuBar().addMenu(self.tr("&View")),
            help=self.menuBar().addMenu(self.tr("&Help")),
            labelList=QtWidgets.QMenu(self.tr("&Label List"), self),
        )

        self.actions = SimpleNamespace()
        self._init_actions()
        self.toggleActions(False)

    def _mk_action(
        self,
        text: str,
        slot=None,
        *,
        checkable: bool = False,
        checked: bool = False,
        enabled: bool = True,
        shortcut=None,
    ) -> QtWidgets.QAction:
        return newAction(
            self,
            text,
            slot=slot,
            shortcut=shortcut,
            checkable=checkable,
            checked=checked,
            enabled=enabled,
        )

    def _init_actions(self) -> None:
        self.actions.open = self._mk_action(
            self.tr("Open"),
            getattr(self, "openFile", None),
            shortcut=self._shortcut("open"),
        )
        self.actions.open.setIcon(self._icon("open_file.svg"))
        self.actions.openDir = self._mk_action(
            self.tr("Open Dir"),
            getattr(self, "openDir", None),
            shortcut=self._shortcut("open_dir"),
        )
        self.actions.openDir.setIcon(self._icon("open_folder.svg"))
        self.actions.close = self._mk_action(
            self.tr("Close"),
            getattr(self, "closeFile", None),
            shortcut=self._shortcut("close"),
        )
        self.actions.close.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogCloseButton)
        )
        self.actions.openVideo = self._mk_action(
            self.tr("Open Video"),
            shortcut=self._shortcut("open_video"),
        )
        self.actions.openVideo.setIcon(self._icon("open_video.png"))
        self.actions.openNextImg = self._mk_action(
            self.tr("Next Image"),
            getattr(self, "openNextImg", None),
            shortcut=self._shortcut("open_next"),
        )
        self.actions.openNextImg.setIcon(self._icon("next_frame.svg"))
        self.actions.openPrevImg = self._mk_action(
            self.tr("Prev Image"),
            getattr(self, "openPrevImg", None),
            shortcut=self._shortcut("open_prev"),
        )
        self.actions.openPrevImg.setIcon(self._icon("prev_frame.svg"))
        self.actions.save = self._mk_action(
            self.tr("Save"),
            getattr(self, "saveFile", None),
            shortcut=self._shortcut("save"),
        )
        self.actions.save.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton)
        )
        self.actions.saveAs = self._mk_action(
            self.tr("Save As"),
            getattr(self, "saveFileAs", None),
            shortcut=self._shortcut("save_as"),
        )
        self.actions.saveAuto = self._mk_action(
            self.tr("Auto Save"),
            checkable=True,
            checked=bool(self._config.get("auto_save", True)),
            enabled=True,
            shortcut=self._shortcut("toggle_auto_save"),
        )
        if hasattr(self, "_on_auto_save_toggled"):
            try:
                self.actions.saveAuto.toggled.connect(self._on_auto_save_toggled)
            except Exception:
                pass
        self.actions.keepPrevMode = self._mk_action(
            self.tr("Keep Prev"),
            self.toggleKeepPrevMode,
            checkable=True,
            checked=bool(self._config.get("keep_prev", False)),
            shortcut=self._shortcut("toggle_keep_prev_mode"),
        )
        self.actions.keepPrevMode.setIcon(self._icon("keep_prev.svg"))
        self.actions.zoomIn = self._mk_action(
            self.tr("Zoom In"),
            lambda: self._step_zoom(+10),
            shortcut=self._shortcut("zoom_in"),
        )
        self.actions.zoomIn.setIcon(self._icon("zoom_in.svg"))
        self.actions.zoomOut = self._mk_action(
            self.tr("Zoom Out"),
            lambda: self._step_zoom(-10),
            shortcut=self._shortcut("zoom_out"),
        )
        self.actions.zoomOut.setIcon(self._icon("zoom_out.svg"))
        self.actions.zoomOrg = self._mk_action(
            self.tr("Original Size"),
            self.setZoomToOriginal,
            shortcut=self._shortcut("zoom_to_original"),
        )
        self.actions.zoomOrg.setIcon(self._icon("zoom_original.svg"))
        self.actions.fitWindow = self._mk_action(
            self.tr("Fit Window"),
            self.setFitWindow,
            checkable=True,
            checked=True,
            shortcut=self._shortcut("fit_window"),
        )
        self.actions.fitWindow.setIcon(self._icon("fit_window.svg"))
        self.actions.fitWidth = self._mk_action(
            self.tr("Fit Width"),
            self.setFitWidth,
            checkable=True,
            checked=False,
            shortcut=self._shortcut("fit_width"),
        )
        self.actions.fitWidth.setIcon(self._icon("fit_width.svg"))
        self.actions.brightnessContrast = self._mk_action(
            self.tr("Brightness/Contrast"),
            self.brightnessContrast,
        )
        self.actions.brightnessContrast.setIcon(self._icon("contrast.svg"))
        self.actions.deleteFile = self._mk_action(
            self.tr("Delete File"), getattr(self, "deleteFile", None)
        )
        self.actions.deleteFile.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_TrashIcon)
        )
        self.actions.deleteShapes = self._mk_action(
            self.tr("Delete Polygons"),
            self.deleteSelectedShapes,
            shortcut=self._shortcut("delete_polygon"),
        )
        self.actions.deleteShapes.setIcon(self._icon("delete_polygons.svg"))
        self.actions.deleteShapes.setEnabled(False)
        self.actions.duplicateShapes = self._mk_action(
            self.tr("Duplicate Polygons"),
            self.duplicateSelectedShapes,
            shortcut=self._shortcut("duplicate_polygon"),
        )
        self.actions.duplicateShapes.setIcon(self._icon("duplicate_polygons.svg"))
        self.actions.duplicateShapes.setEnabled(False)
        self.actions.removePoint = self._mk_action(
            self.tr("Remove Point"), getattr(self, "removeSelectedPoint", None)
        )
        self.actions.undo = self._mk_action(self.tr("Undo"), self.undoShapeEdit)
        self.actions.undo.setIcon(self._icon("undo.svg"))
        self.actions.undoLastPoint = self._mk_action(self.tr("Undo Last Point"))

        self.actions.createMode = self._mk_action(
            self.tr("Create Polygons"),
            checkable=True,
            shortcut=self._shortcut("create_polygon"),
        )
        self.actions.createMode.setIcon(self._icon("create_polygons.svg"))
        self.actions.createRectangleMode = self._mk_action(
            self.tr("Rectangle"), checkable=True
        )
        self.actions.createRectangleMode.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView)
        )
        self.actions.createCircleMode = self._mk_action(
            self.tr("Circle"), checkable=True
        )
        self.actions.createCircleMode.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload)
        )
        self.actions.createLineMode = self._mk_action(self.tr("Line"), checkable=True)
        self.actions.createLineMode.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_ArrowRight)
        )
        self.actions.createPointMode = self._mk_action(
            self.tr("Point"),
            checkable=True,
            shortcut=self._shortcut("create_point"),
        )
        self.actions.createPointMode.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogApplyButton)
        )
        self.actions.createLineStripMode = self._mk_action(
            self.tr("Line Strip"), checkable=True
        )
        self.actions.createLineStripMode.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_MediaSeekForward)
        )
        self.actions.createAiPolygonMode = self._mk_action(
            self.tr("AI Polygon"), checkable=True, enabled=False
        )
        self.actions.createAiPolygonMode.setIcon(self._icon("ai_polygons.svg"))
        self.actions.createAiMaskMode = self._mk_action(
            self.tr("AI Mask"), checkable=True, enabled=False
        )
        self.actions.createAiMaskMode.setIcon(self._icon("ai_polygons.svg"))
        self.actions.editMode = self._mk_action(
            self.tr("Edit Polygons"),
            checkable=True,
            shortcut=self._shortcut("edit_polygon"),
        )
        self.actions.editMode.setIcon(self._icon("edit_polygons.svg"))
        self.actions.editMode.setChecked(True)

        self.actions.tool = (
            self.actions.open,
            self.actions.openDir,
            self.actions.openPrevImg,
            self.actions.openNextImg,
            self.actions.fitWindow,
            self.actions.fitWidth,
            self.actions.zoomOut,
            self.actions.zoomIn,
            self.actions.zoomOrg,
            self.actions.brightnessContrast,
            self.actions.keepPrevMode,
            self.actions.save,
        )
        self.actions.menu = (
            self.actions.createMode,
            self.actions.createRectangleMode,
            self.actions.createCircleMode,
            self.actions.createLineMode,
            self.actions.createPointMode,
            self.actions.createLineStripMode,
            None,
            self.actions.createAiPolygonMode,
            self.actions.createAiMaskMode,
        )
        self.actions.editMenu = (
            self.actions.undo,
            self.actions.undoLastPoint,
            self.actions.removePoint,
        )
        self.actions.onShapesPresent = (
            self.actions.save,
            self.actions.saveAs,
            self.actions.deleteFile,
        )

    def addRecentFile(self, filename: str) -> None:
        if not filename:
            return
        path = str(filename)
        recent = [p for p in self.recentFiles if p != path]
        recent.insert(0, path)
        self.recentFiles = recent[:20]

    def _shortcut(self, key: str):
        shortcuts = self._config.get("shortcuts", {}) or {}
        if key in shortcuts and shortcuts.get(key):
            return shortcuts.get(key)
        defaults = {
            # Qt handles Ctrl/Cmd platform mapping for standard Save.
            "save": QtGui.QKeySequence.Save,
            "save_as": QtGui.QKeySequence.SaveAs,
            "toggle_auto_save": "Ctrl+Shift+A",
            "toggle_keypoint_sequence": "Ctrl+Shift+K",
            "create_polygon": "Ctrl+N",
            "create_point": "Ctrl+I",
            "edit_polygon": "Ctrl+J",
            "delete_polygon": "Delete",
            "duplicate_polygon": "Ctrl+D",
        }
        return defaults.get(key)

    def _icon(self, filename: str) -> QtGui.QIcon:
        path = self._icons_dir / filename
        if path.exists():
            return QtGui.QIcon(str(path))
        return QtGui.QIcon()

    def errorMessage(self, title: str, text: str) -> None:
        QtWidgets.QMessageBox.critical(self, title, text)

    def hasLabelFile(self) -> bool:
        if not self.filename:
            return False
        try:
            label_file = self._getLabelFile(self.filename)
        except Exception:
            return False
        return bool(label_file and osp.exists(label_file))

    def mayContinue(self) -> bool:
        if bool(getattr(self, "_suppress_unsaved_prompt", False)):
            return True
        if not self.dirty:
            return True
        response = QtWidgets.QMessageBox.warning(
            self,
            self.tr("Unsaved Changes"),
            self.tr("You have unsaved changes. Save before continuing?"),
            QtWidgets.QMessageBox.Save
            | QtWidgets.QMessageBox.Discard
            | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Save,
        )
        if response == QtWidgets.QMessageBox.Cancel:
            return False
        if response == QtWidgets.QMessageBox.Save:
            self.saveFile()
            return not self.dirty
        return True

    def noShapes(self) -> bool:
        canvas = getattr(self, "canvas", None)
        if canvas is None:
            return True
        return len(getattr(canvas, "shapes", []) or []) == 0

    def remLabels(self, shapes: Iterable) -> None:
        # Shapes are mutable and implement custom __eq__/__hash__ semantics, so
        # using them directly in sets/dicts is fragile (hash can change when
        # points move; __eq__ may consider near-overlapping polygons equal).
        # Use object identity instead.
        shape_ids = {id(s) for s in (shapes or [])}
        for row in reversed(range(self.labelList.count())):
            item = self.labelList.item(row)
            item_shape = item.shape() if hasattr(item, "shape") else None
            if item_shape is not None and id(item_shape) in shape_ids:
                self.labelList.takeItem(row)

    def saveFile(self) -> None:
        if not self.filename:
            return
        try:
            target = self._getLabelFile(self.filename)
        except Exception:
            target = None
        if target and hasattr(self, "_saveFile"):
            prev = bool(getattr(self, "_suppress_unsaved_prompt", False))
            self._suppress_unsaved_prompt = True
            try:
                self._saveFile(target)
            finally:
                self._suppress_unsaved_prompt = prev

    def setScroll(self, orientation, value: int) -> None:
        bars = getattr(self, "scrollBars", None)
        if not isinstance(bars, dict):
            return
        bar = bars.get(orientation)
        if bar is not None:
            bar.setValue(int(value))

    def setZoom(self, value: int) -> None:
        self.zoomWidget.setValue(int(value))

    def _on_zoom_changed(self, _value: int) -> None:
        self.zoomMode = self.MANUAL_ZOOM

    def _step_zoom(self, delta: int) -> None:
        self.setZoom(max(1, min(1000, self.zoomWidget.value() + int(delta))))
        if self._has_renderable_image() and hasattr(self, "paintCanvas"):
            self.paintCanvas()

    def zoomRequest(self, delta: int, pos) -> None:
        step = 10 if delta > 0 else -10
        self._step_zoom(step)

    def setZoomToOriginal(self) -> None:
        self.zoomMode = self.MANUAL_ZOOM
        self.setZoom(100)
        if self._has_renderable_image() and hasattr(self, "paintCanvas"):
            self.paintCanvas()

    def setFitWindow(self, value=True) -> None:
        enabled = bool(value)
        self.zoomMode = self.FIT_WINDOW if enabled else self.MANUAL_ZOOM
        self.actions.fitWindow.setChecked(enabled)
        self.actions.fitWidth.setChecked(
            False if enabled else self.actions.fitWidth.isChecked()
        )
        if self._has_renderable_image():
            self.adjustScale()

    def setFitWidth(self, value=True) -> None:
        enabled = bool(value)
        self.zoomMode = self.FIT_WIDTH if enabled else self.MANUAL_ZOOM
        self.actions.fitWidth.setChecked(enabled)
        self.actions.fitWindow.setChecked(
            False if enabled else self.actions.fitWindow.isChecked()
        )
        if self._has_renderable_image():
            self.adjustScale()

    def _canvas_pixmap_size(self) -> tuple[int, int]:
        canvas = getattr(self, "canvas", None)
        pixmap = getattr(canvas, "pixmap", None)
        if pixmap is None or pixmap.isNull():
            return (0, 0)
        return (pixmap.width(), pixmap.height())

    def _viewport_size(self) -> tuple[int, int]:
        central = self.centralWidget()
        if isinstance(central, QtWidgets.QScrollArea):
            viewport = central.viewport()
            return (max(1, viewport.width() - 2), max(1, viewport.height() - 2))
        return (max(1, self.width() - 2), max(1, self.height() - 2))

    def scaleFitWindow(self) -> int:
        img_w, img_h = self._canvas_pixmap_size()
        if img_w <= 0 or img_h <= 0:
            return self.zoomWidget.value()
        view_w, view_h = self._viewport_size()
        ratio = min(view_w / img_w, view_h / img_h)
        return max(1, min(1000, int(ratio * 100)))

    def scaleFitWidth(self) -> int:
        img_w, _img_h = self._canvas_pixmap_size()
        if img_w <= 0:
            return self.zoomWidget.value()
        view_w, _view_h = self._viewport_size()
        ratio = view_w / img_w
        return max(1, min(1000, int(ratio * 100)))

    def adjustScale(self, initial=False):
        if self.zoomMode == self.FIT_WINDOW:
            self.zoomWidget.blockSignals(True)
            self.zoomWidget.setValue(self.scaleFitWindow())
            self.zoomWidget.blockSignals(False)
        elif self.zoomMode == self.FIT_WIDTH:
            self.zoomWidget.blockSignals(True)
            self.zoomWidget.setValue(self.scaleFitWidth())
            self.zoomWidget.blockSignals(False)
        if self._has_renderable_image() and hasattr(self, "paintCanvas"):
            self.paintCanvas()

    def _has_renderable_image(self) -> bool:
        image = getattr(self, "image", None)
        return image is not None and hasattr(image, "isNull") and not image.isNull()

    def brightnessContrast(self, _value=False):
        # Hook for subclasses. AnnolidWindow overrides this with runtime conversion.
        return None

    def deleteSelectedShapes(self, _value=False) -> None:
        canvas = getattr(self, "canvas", None)
        if canvas is None:
            return
        deleted = canvas.deleteSelected() or []
        if deleted and hasattr(self, "remLabels"):
            self.remLabels(deleted)
        if deleted and hasattr(self, "setDirty"):
            self.setDirty()

    def duplicateSelectedShapes(self, _value=False) -> None:
        canvas = getattr(self, "canvas", None)
        if canvas is None:
            return
        duplicated = canvas.duplicateSelectedShapes() or []
        if duplicated and hasattr(self, "setDirty"):
            self.setDirty()

    def undoShapeEdit(self, _value=False) -> None:
        canvas = getattr(self, "canvas", None)
        if canvas is None or not getattr(canvas, "isShapeRestorable", False):
            return
        canvas.restoreShape()
        if hasattr(self, "loadShapes"):
            self.loadShapes(canvas.shapes, replace=True)
        if hasattr(self, "setDirty"):
            self.setDirty()

    def toggleKeepPrevMode(self, value=False):
        enabled = bool(value)
        self._config["keep_prev"] = enabled
        if getattr(self.actions, "keepPrevMode", None) is not None:
            self.actions.keepPrevMode.setChecked(enabled)

    def toggleActions(self, value: bool) -> None:
        enabled = bool(value)
        names = (
            "close",
            "openNextImg",
            "openPrevImg",
            "save",
            "saveAs",
            "deleteFile",
            "deleteShapes",
            "duplicateShapes",
            "removePoint",
            "undo",
            "undoLastPoint",
            "createMode",
            "createRectangleMode",
            "createCircleMode",
            "createLineMode",
            "createPointMode",
            "createLineStripMode",
            "createAiPolygonMode",
            "createAiMaskMode",
            "createGroundingSAMMode",
            "editMode",
            "fitWindow",
            "fitWidth",
            "zoomIn",
            "zoomOut",
            "zoomOrg",
            "brightnessContrast",
            "keepPrevMode",
        )
        for name in names:
            action = getattr(self.actions, name, None)
            if action is not None:
                action.setEnabled(enabled)

        # Some mode actions live outside `self.actions` (historical compatibility).
        try:
            act = getattr(self, "createPolygonSAMMode", None)
            if act is not None:
                act.setEnabled(enabled)
        except Exception:
            pass

    def validateLabel(self, text: str) -> bool:
        text = str(text or "").strip()
        if not text:
            return False
        if self._config.get("validate_label") == "exact":
            labels = set(self._config.get("labels") or [])
            if labels and text not in labels:
                return False
        return True

    def currentItem(self):
        items = self.labelList.selectedItems()
        return items[0] if items else None

    def shapeSelectionChanged(self, selected_shapes) -> None:
        selected_list = list(selected_shapes or [])
        selected_ids = {id(s) for s in selected_list}
        has_selected = len(selected_ids) > 0
        if getattr(self.actions, "deleteShapes", None) is not None:
            self.actions.deleteShapes.setEnabled(has_selected)
        if getattr(self.actions, "duplicateShapes", None) is not None:
            self.actions.duplicateShapes.setEnabled(has_selected)
        # Avoid feedback loops: selecting shapes on the canvas updates the list,
        # which would otherwise re-select shapes via list selection handlers.
        prev_no_slot = bool(getattr(self, "_noSelectionSlot", False))
        try:
            setattr(self, "_noSelectionSlot", True)
            self.labelList.blockSignals(True)
            for idx in range(self.labelList.count()):
                item = self.labelList.item(idx)
                item_shape = item.shape() if hasattr(item, "shape") else None
                item.setSelected(
                    item_shape is not None and id(item_shape) in selected_ids
                )
        finally:
            self.labelList.blockSignals(False)
            setattr(self, "_noSelectionSlot", prev_no_slot)

    def toggleDrawingSensitive(self, drawing: bool) -> None:
        self.actions.editMode.setEnabled(not bool(drawing))
        self.actions.undoLastPoint.setEnabled(bool(drawing))

    def openFile(self, _value=False) -> None:
        start_dir = self.lastOpenDir or str(Path.home())
        filters = self.tr("Images/JSON (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.json)")
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Open Image/Annotation"),
            start_dir,
            filters,
        )
        if not filename:
            return
        self.lastOpenDir = str(Path(filename).parent)
        self.loadFile(filename)

    def openDir(self, _value=False) -> None:
        start_dir = self.lastOpenDir or str(Path.home())
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Open Folder"),
            start_dir,
        )
        if not directory:
            return
        self.lastOpenDir = directory
        if hasattr(self, "importDirImages"):
            self.importDirImages(directory, load=True)

    def loadFile(self, filename: str) -> None:
        path = Path(filename)
        if not path.exists():
            self.errorMessage(
                self.tr("File Error"), self.tr("Missing file: %s") % filename
            )
            return

        image_path = path
        shapes = []

        if path.suffix.lower() == ".json":
            try:
                label_file = LabelFile(str(path))
            except LabelFileError as exc:
                self.errorMessage(self.tr("Load Error"), str(exc))
                return
            self.labelFile = label_file
            shapes = label_file.shapes
            image_rel = label_file.imagePath or ""
            if image_rel:
                image_path = (path.parent / image_rel).resolve()
        else:
            candidate = path.with_suffix(".json")
            if candidate.exists():
                try:
                    label_file = LabelFile(str(candidate))
                    self.labelFile = label_file
                    shapes = label_file.shapes
                except Exception:
                    shapes = []

        image = QtGui.QImage(str(image_path))
        if image.isNull():
            self.errorMessage(
                self.tr("Load Error"),
                self.tr("Failed to load image: %s") % str(image_path),
            )
            return

        self.image = image
        self.filename = str(path)
        self.imagePath = str(image_path)
        self.imageData = None

        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.setEnabled(True)
            pixmap = QtGui.QPixmap.fromImage(image)
            self.canvas.loadPixmap(pixmap, clear_shapes=True)
            # AnnolidWindow.loadLabels expects LabelFile JSON dict payloads and
            # materializes Shape objects before forwarding to loadShapes.
            if shapes and isinstance(shapes[0], dict) and hasattr(self, "loadLabels"):
                self.loadLabels(shapes)
            elif hasattr(self, "loadShapes"):
                self.loadShapes(shapes, replace=True)

        # Ensure image workflows (Open / Open Dir) activate toolbar actions.
        if hasattr(self, "toggleActions"):
            self.toggleActions(True)

        self.setWindowTitle(f"Annolid - {osp.basename(self.filename)}")
        try:
            matches = self.fileListWidget.findItems(
                self.filename, QtCore.Qt.MatchExactly
            )
            if matches:
                blocker = QtCore.QSignalBlocker(self.fileListWidget)
                try:
                    self.fileListWidget.setCurrentItem(matches[0])
                finally:
                    del blocker
        except Exception:
            pass

        # A freshly loaded file is the new baseline; only user edits should
        # mark it dirty afterwards.
        if hasattr(self, "setClean"):
            self.setClean()

    def closeFile(self, _value=False):
        return None


# Aliases kept so existing import sites can be migrated incrementally.
MainWindow = AnnolidWindowBase
ToolBar = AnnolidToolBar
LabelListWidgetItem = AnnolidLabelListItem


__all__ = [
    "AI_MODELS",
    "AnnolidLabelListItem",
    "AnnolidToolBar",
    "AnnolidWindowBase",
    "PY2",
    "QT4",
    "QT5",
    "__version__",
    "addActions",
    "LabelListWidgetItem",
    "logger",
    "MainWindow",
    "newAction",
    "ToolBar",
    "utils",
]
