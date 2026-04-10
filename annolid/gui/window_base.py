from __future__ import annotations

import os.path as osp
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, Optional

from qtpy import QtCore, QtGui, QtWidgets
from annolid.gui.qt_compat import (
    normalize_orientation,
    palette_color_group,
    palette_color_role,
)

from annolid.configs import get_config
from annolid.gui.file_dock import FileDockMixin
from annolid.gui.large_image import (
    DEFAULT_LARGE_IMAGE_CACHE_MAX_ENTRIES,
    DEFAULT_LARGE_IMAGE_CACHE_MAX_SIZE_BYTES,
    TIFF_SUFFIXES,
    clear_all_large_image_caches,
    format_large_image_cache_size,
    is_large_tiff_path,
    large_image_cache_root,
    large_image_cache_size_bytes,
    list_large_image_cache_entries,
    load_image_with_backends,
    optimize_large_tiff_for_viewing,
    open_large_image,
    pyvips_optimization_available,
    probe_large_image,
    remove_large_image_cache_file,
    resolve_fresh_optimized_large_image_path,
)
from annolid.gui.large_image_modes import large_image_draw_mode_label
from annolid.gui.large_image_document import (
    LargeImageDocument,
    LargeImageSelectionState,
    LargeImageViewport,
)
from annolid.io.large_image.base import LargeImageBackendCapabilities
from annolid.gui.viewer_layers import (
    AffineTransform,
    RasterImageLayer,
    ViewerLayerModel,
)
from annolid.gui.status import post_window_status
from annolid.gui.label_file import LabelFile, LabelFileError
from annolid.gui.workers import FlexibleWorker
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

LARGE_IMAGE_CACHE_MAX_ENTRIES_KEY = "large_image_cache/max_entries"
LARGE_IMAGE_CACHE_MAX_SIZE_GB_KEY = "large_image_cache/max_size_gb"


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
            group = (
                palette_color_group("Active")
                if enabled
                else palette_color_group("Disabled")
            )
            painter.setPen(pal.color(group, palette_color_role("ButtonText")))
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
        self.large_image_backend = None
        self.large_image_document: LargeImageDocument | None = None
        self._active_image_view = "canvas"
        self._large_image_surface_reason: str = ""
        self._large_image_hidden_docks_states: dict[QtWidgets.QDockWidget, bool] = {}
        self._play_button_owner: str | None = None
        self._large_image_surface_label = None
        self._large_image_mode_label = None
        self._large_image_return_button = None
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
        self.video_brightness_contrast_values: dict[
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
        self.actions.startAdjoiningPolygon = self._mk_action(
            self.tr("Start Adjoining Polygon"),
            getattr(self, "startAdjoiningPolygonFromSelection", None),
        )
        self.actions.startAdjoiningPolygon.setIcon(self._icon("duplicate_polygons.svg"))
        self.actions.startAdjoiningPolygon.setEnabled(False)
        self.actions.inferPagePolygons = self._mk_action(
            self.tr("Infer Page Polygons"),
            getattr(self, "inferCurrentLargeImagePagePolygons", None),
        )
        self.actions.inferPagePolygons.setIcon(self._icon("duplicate_polygons.svg"))
        self.actions.inferPagePolygons.setEnabled(False)
        self.actions.collapsePolygons = self._mk_action(
            self.tr("Collapse Selected Polygons"),
            getattr(self, "collapseSelectedPolygons", None),
        )
        self.actions.collapsePolygons.setIcon(self._icon("delete_polygons.svg"))
        self.actions.collapsePolygons.setEnabled(False)
        self.actions.restorePolygons = self._mk_action(
            self.tr("Restore Selected Polygons"),
            getattr(self, "restoreSelectedPolygons", None),
        )
        self.actions.restorePolygons.setIcon(self._icon("undo.svg"))
        self.actions.restorePolygons.setEnabled(False)
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
            self.actions.inferPagePolygons,
            self.actions.collapsePolygons,
            self.actions.restorePolygons,
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
        orientation = normalize_orientation(orientation)
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
            hook = getattr(self, "_onFitModeApplied", None)
            if enabled and callable(hook):
                try:
                    hook("fit_window")
                except Exception:
                    pass

    def setFitWidth(self, value=True) -> None:
        enabled = bool(value)
        self.zoomMode = self.FIT_WIDTH if enabled else self.MANUAL_ZOOM
        self.actions.fitWidth.setChecked(enabled)
        self.actions.fitWindow.setChecked(
            False if enabled else self.actions.fitWindow.isChecked()
        )
        if self._has_renderable_image():
            self.adjustScale()
            hook = getattr(self, "_onFitModeApplied", None)
            if enabled and callable(hook):
                try:
                    hook("fit_width")
                except Exception:
                    pass

    def _canvas_pixmap_size(self) -> tuple[int, int]:
        if getattr(self, "_active_image_view", "canvas") == "tiled":
            tiled = getattr(self, "large_image_view", None)
            if tiled is not None:
                return tiled.content_size()
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
        if getattr(self, "_active_image_view", "canvas") == "tiled":
            return getattr(self, "large_image_backend", None) is not None
        image = getattr(self, "image", None)
        return image is not None and hasattr(image, "isNull") and not image.isNull()

    def brightnessContrast(self, _value=False):
        # Hook for subclasses. AnnolidWindow overrides this with runtime conversion.
        return None

    def deleteSelectedShapes(self, _value=False) -> None:
        canvas = getattr(self, "canvas", None)
        if canvas is None:
            return
        active_editor = canvas
        if getattr(self, "_active_image_view", "canvas") == "tiled":
            tiled = getattr(self, "large_image_view", None)
            if tiled is not None:
                active_editor = tiled
                editor_selection = list(getattr(tiled, "selectedShapes", []) or [])
                if editor_selection:
                    try:
                        canvas.selectShapes(editor_selection)
                    except Exception:
                        pass

        deleted = canvas.deleteSelected() or []
        if deleted and active_editor is not canvas:
            try:
                if hasattr(active_editor, "set_selected_shapes"):
                    active_editor.set_selected_shapes([])
                elif hasattr(active_editor, "selectedShapes"):
                    active_editor.selectedShapes = []
            except Exception:
                pass
            try:
                if hasattr(active_editor, "set_shapes"):
                    active_editor.set_shapes(list(getattr(canvas, "shapes", []) or []))
            except Exception:
                pass
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
            "startAdjoiningPolygon",
            "inferPagePolygons",
            "collapsePolygons",
            "restorePolygons",
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
        if enabled:
            self._update_adjoining_polygon_action_state()
            self._update_polygon_tool_action_state()
        else:
            action = getattr(self.actions, "startAdjoiningPolygon", None)
            if action is not None:
                action.setEnabled(False)
            for name in (
                "inferPagePolygons",
                "collapsePolygons",
                "restorePolygons",
            ):
                action = getattr(self.actions, name, None)
                if action is not None:
                    action.setEnabled(False)

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
        self._update_adjoining_polygon_action_state()
        self._update_polygon_tool_action_state()
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
        sync_brain3d = getattr(self, "_syncBrain3DSelectionFromShapes", None)
        if callable(sync_brain3d):
            try:
                sync_brain3d(selected_list)
            except Exception:
                pass

    def toggleDrawingSensitive(self, drawing: bool) -> None:
        self.actions.editMode.setEnabled(not bool(drawing))
        self.actions.undoLastPoint.setEnabled(bool(drawing))
        self._update_adjoining_polygon_action_state()
        self._update_polygon_tool_action_state()

    def _update_adjoining_polygon_action_state(self) -> None:
        action = getattr(self.actions, "startAdjoiningPolygon", None)
        if action is None:
            return
        editor = None
        tiled_editor = getattr(self, "large_image_view", None)
        try:
            editor = self._active_shape_editor()
        except Exception:
            editor = None
        can_start = False
        if editor is not None:
            can_start = bool(
                getattr(editor, "canStartAdjoiningPolygon", lambda: False)()
            )
        if not can_start and tiled_editor is not None:
            can_start = bool(
                getattr(tiled_editor, "canStartAdjoiningPolygon", lambda: False)()
            )
        action.setEnabled(can_start)

    def _update_polygon_tool_action_state(self) -> None:
        collapse_action = getattr(self.actions, "collapsePolygons", None)
        restore_action = getattr(self.actions, "restorePolygons", None)
        infer_action = getattr(self.actions, "inferPagePolygons", None)
        try:
            can_collapse = bool(
                getattr(self, "canCollapseSelectedPolygons", lambda: False)()
            )
        except Exception:
            can_collapse = False
        try:
            can_restore = bool(
                getattr(self, "canRestoreSelectedPolygons", lambda: False)()
            )
        except Exception:
            can_restore = False
        try:
            can_infer = bool(
                getattr(self, "canInferCurrentLargeImagePagePolygons", lambda: False)()
            )
        except Exception:
            can_infer = False
        if collapse_action is not None:
            collapse_action.setEnabled(can_collapse)
        if restore_action is not None:
            restore_action.setEnabled(can_restore)
        if infer_action is not None:
            infer_action.setEnabled(can_infer)

    def openFile(self, _value=False) -> None:
        start_dir = self.lastOpenDir or str(Path.home())
        filters = self.tr("Images/JSON (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.json)")
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            self.tr("Open Image/Annotation"),
            start_dir,
            filters,
        )
        selected_paths = [
            str(path or "").strip()
            for path in list(filenames or [])
            if str(path or "").strip()
        ]
        if not selected_paths:
            return
        first_path = selected_paths[0]
        self.lastOpenDir = str(Path(first_path).parent)

        def _is_large_tiff(path_text: str) -> bool:
            try:
                return bool(is_large_tiff_path(Path(path_text)))
            except Exception:
                return False

        # User-friendly behavior: when a large TIFF is already open in tiled mode,
        # let users add another TIFF as a layer instead of always replacing.
        if (
            len(selected_paths) == 1
            and _is_large_tiff(first_path)
            and getattr(self, "_active_image_view", "") == "tiled"
            and bool(
                is_large_tiff_path(Path(str(getattr(self, "imagePath", "") or "")))
            )
            and hasattr(self, "addRasterImageLayersFromPaths")
            and str(Path(first_path))
            != str(Path(str(getattr(self, "imagePath", "") or "")))
        ):
            answer = QtWidgets.QMessageBox.question(
                self,
                self.tr("Open Large TIFF"),
                self.tr(
                    "A large TIFF is already open.\n\n"
                    "Choose Yes to add this TIFF as a new layer,\n"
                    "No to replace the current base image."
                ),
                QtWidgets.QMessageBox.Yes
                | QtWidgets.QMessageBox.No
                | QtWidgets.QMessageBox.Cancel,
                QtWidgets.QMessageBox.Yes,
            )
            if answer == QtWidgets.QMessageBox.Cancel:
                return
            if answer == QtWidgets.QMessageBox.Yes:
                added = int(self.addRasterImageLayersFromPaths([first_path]))  # type: ignore[attr-defined]
                if added > 0 and hasattr(self, "status"):
                    self.status(self.tr("Loaded %d TIFF layer(s)") % int(added))
                return

        self.loadFile(first_path)

        # Multi-select large TIFF workflow: first file is base image, rest become layers.
        if (
            len(selected_paths) > 1
            and all(_is_large_tiff(path) for path in selected_paths)
            and getattr(self, "_active_image_view", "") == "tiled"
            and hasattr(self, "addRasterImageLayersFromPaths")
        ):
            added = int(self.addRasterImageLayersFromPaths(selected_paths[1:]))  # type: ignore[attr-defined]
            if added > 0 and hasattr(self, "status"):
                self.status(self.tr("Loaded %d TIFF layer(s)") % int(added))

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

    def _activate_tiled_image_view(self, image_path: Path, backend=None) -> bool:
        if not hasattr(self, "canvas") or self.canvas is None:
            return False
        if not bool(is_large_tiff_path(image_path)) or not hasattr(
            self, "large_image_view"
        ):
            return False
        self.large_image_backend = (
            backend if backend is not None else open_large_image(image_path)
        )
        self.large_image_view.set_backend(self.large_image_backend)
        if hasattr(self, "_viewer_stack") and self._viewer_stack is not None:
            self._viewer_stack.setCurrentWidget(self.large_image_view)
        self._active_image_view = "tiled"
        self._large_image_surface_reason = ""
        self.canvas.setEnabled(False)
        self._syncLargeImageDocument()
        return True

    def _large_tiff_annotation_root(
        self, image_path: str | Path | None = None
    ) -> Path | None:
        candidate = (
            image_path if image_path is not None else getattr(self, "imagePath", None)
        )
        if not candidate:
            return None
        path = Path(candidate)
        if not bool(is_large_tiff_path(path)):
            return None
        path_str = str(path)
        lower_path = path_str.lower()
        matched_suffix = ""
        for suffix in sorted(TIFF_SUFFIXES, key=len, reverse=True):
            if lower_path.endswith(suffix):
                matched_suffix = suffix
                break
        if matched_suffix:
            return Path(path_str[: -len(matched_suffix)])
        return path.with_suffix("")

    def _large_image_stack_annotation_dir(
        self, image_path: str | Path | None = None
    ) -> Path | None:
        return self._large_tiff_annotation_root(image_path=image_path)

    def _large_image_stack_label_path(
        self,
        page_index: int | None = None,
        image_path: str | Path | None = None,
    ) -> str | None:
        backend = getattr(self, "large_image_backend", None)
        page_count = int(getattr(backend, "get_page_count", lambda: 1)() or 1)
        if image_path is None:
            if not self._has_large_image_page_navigation() and page_count <= 1:
                return None
            annotation_dir = self._large_image_stack_annotation_dir(
                image_path=image_path
            )
        else:
            if page_count <= 1:
                return None
            current_image_path = getattr(self, "imagePath", None)
            if not current_image_path:
                return None
            try:
                same_image = (
                    Path(current_image_path).expanduser().resolve()
                    == Path(image_path).expanduser().resolve()
                )
            except Exception:
                same_image = str(current_image_path) == str(image_path)
            if not same_image:
                return None
            annotation_dir = self._large_image_stack_annotation_dir(
                image_path=current_image_path
            )
        if annotation_dir is None:
            return None
        target_page = (
            int(page_index) if page_index is not None else self._largeImageCurrentPage()
        )
        return str(annotation_dir / f"{annotation_dir.name}_{target_page:09}.json")

    def _has_large_image_page_navigation(self) -> bool:
        return bool(getattr(self, "_seekbar_owner", "") == "large_image_stack")

    def _clear_large_image_stack_navigation(self) -> None:
        stack_annotation_dir = self._large_image_stack_annotation_dir()
        if not self._has_large_image_page_navigation():
            if getattr(self, "_play_button_owner", "") == "large_image_stack":
                play_button = getattr(self, "playButton", None)
                if play_button is not None:
                    try:
                        self.statusBar().removeWidget(play_button)
                    except Exception:
                        pass
                self.playButton = None
                self._play_button_owner = None
            if stack_annotation_dir is not None and str(
                getattr(self, "annotation_dir", "") or ""
            ) == str(stack_annotation_dir):
                self.annotation_dir = None
            return
        seekbar = getattr(self, "seekbar", None)
        if seekbar is not None:
            try:
                self.statusBar().removeWidget(seekbar)
            except Exception:
                pass
        self.seekbar = None
        if getattr(self, "_play_button_owner", "") == "large_image_stack":
            play_button = getattr(self, "playButton", None)
            if play_button is not None:
                try:
                    self.statusBar().removeWidget(play_button)
                except Exception:
                    pass
            self.playButton = None
            self._play_button_owner = None
        self._seekbar_owner = None
        self._large_image_page_count = 0
        if stack_annotation_dir is not None and str(
            getattr(self, "annotation_dir", "") or ""
        ) == str(stack_annotation_dir):
            self.annotation_dir = None

    def _set_play_button_state(
        self, is_playing: bool, *, enabled: bool | None = None
    ) -> None:
        play_button = getattr(self, "playButton", None)
        if play_button is None:
            return
        try:
            icon_style = (
                QtWidgets.QStyle.SP_MediaStop
                if bool(is_playing)
                else QtWidgets.QStyle.SP_MediaPlay
            )
            play_button.setIcon(QtWidgets.QApplication.style().standardIcon(icon_style))
            play_button.setText("Pause" if bool(is_playing) else "Play")
            if enabled is not None:
                play_button.setEnabled(bool(enabled))
        except Exception:
            pass

    def _clear_status_bar_media_controls(self) -> None:
        for attr_name in ("seekbar", "playButton", "saveButton"):
            widget = getattr(self, attr_name, None)
            if widget is None:
                continue
            try:
                self.statusBar().removeWidget(widget)
            except Exception:
                pass
            setattr(self, attr_name, None)
        self._seekbar_owner = None
        self._play_button_owner = None
        self._large_image_seekbar_drag_active = False

    def _on_large_image_seekbar_pressed(self, *_args) -> None:
        self._large_image_seekbar_drag_active = True

    def _on_large_image_seekbar_released(self, *_args) -> None:
        seekbar = getattr(self, "seekbar", None)
        if seekbar is None or not self._has_large_image_page_navigation():
            self._large_image_seekbar_drag_active = False
            return
        self._large_image_seekbar_drag_active = False
        self.requestLargeImagePageNumber(int(seekbar.value()))

    def _on_large_image_seekbar_value_changed(self, value: int) -> None:
        if getattr(self, "_large_image_seekbar_drag_active", False):
            return
        self.requestLargeImagePageNumber(int(value))

    def _setup_large_image_stack_navigation(self, backend) -> None:
        from annolid.gui.widgets.video_slider import VideoSlider

        if not self._largeImageSupportsPages(backend):
            self._clear_large_image_stack_navigation()
            return
        page_count = max(1, int(getattr(backend, "get_page_count", lambda: 1)() or 1))
        self._clear_status_bar_media_controls()
        seekbar = VideoSlider()
        seekbar.setMinimum(0)
        seekbar.setMaximum(page_count - 1)
        seekbar.setEnabled(True)
        seekbar.resizeEvent()
        if hasattr(self, "jump_to_frame"):
            seekbar.input_value.returnPressed.connect(self.jump_to_frame)
        if hasattr(self, "keyPressEvent"):
            seekbar.keyPress.connect(self.keyPressEvent)
        if hasattr(self, "keyReleaseEvent"):
            seekbar.keyRelease.connect(self.keyReleaseEvent)
        seekbar.mousePressed.connect(self._on_large_image_seekbar_pressed)
        seekbar.mouseReleased.connect(self._on_large_image_seekbar_released)
        seekbar.valueChanged.connect(self._on_large_image_seekbar_value_changed)
        play_button = QtWidgets.QPushButton("Play", self)
        toggle_play = getattr(self, "togglePlay", None)
        if callable(toggle_play):
            play_button.clicked.connect(toggle_play)
        self.statusBar().addPermanentWidget(play_button)
        self.statusBar().addPermanentWidget(seekbar, stretch=1)
        self.playButton = play_button
        self.seekbar = seekbar
        self._play_button_owner = "large_image_stack"
        self._seekbar_owner = "large_image_stack"
        self._set_play_button_state(False, enabled=callable(toggle_play))
        self._large_image_page_count = page_count
        annotation_dir = self._large_image_stack_annotation_dir()
        if annotation_dir is not None:
            self.annotation_dir = str(annotation_dir)
        current_page = int(getattr(backend, "get_current_page", lambda: 0)() or 0)
        try:
            with QtCore.QSignalBlocker(seekbar):
                seekbar.setValue(current_page)
        except Exception:
            pass

    def requestLargeImagePageNumber(self, page_index: int) -> bool:
        target = int(page_index)
        current = self._largeImageCurrentPage()
        if target == current:
            return True
        if not self.mayContinue():
            seekbar = getattr(self, "seekbar", None)
            if seekbar is not None and self._has_large_image_page_navigation():
                try:
                    with QtCore.QSignalBlocker(seekbar):
                        seekbar.setValue(current)
                except Exception:
                    pass
            return False
        try:
            return self.setLargeImagePageNumber(target)
        except Exception as exc:
            seekbar = getattr(self, "seekbar", None)
            if seekbar is not None and self._has_large_image_page_navigation():
                try:
                    with QtCore.QSignalBlocker(seekbar):
                        seekbar.setValue(current)
                except Exception:
                    pass
            message = str(exc)
            if hasattr(self, "errorMessage"):
                self.errorMessage(
                    self.tr("TIFF Page Navigation"),
                    self.tr("Failed to open TIFF page: %s") % message,
                )
            else:
                self._post_window_status(
                    self.tr("Failed to open TIFF page: %s") % message,
                    timeout=5000,
                )
            return False

    def setLargeImagePageNumber(self, page_index: int) -> bool:
        backend = getattr(self, "large_image_backend", None)
        large_view = getattr(self, "large_image_view", None)
        canvas = getattr(self, "canvas", None)
        if backend is None or large_view is None:
            return False
        if not self._largeImageSupportsPages(backend):
            return False
        page_count = max(1, int(getattr(backend, "get_page_count", lambda: 1)() or 1))
        if page_count <= 1:
            return False
        target = int(page_index)
        if target < 0 or target >= page_count:
            return False
        current = int(getattr(backend, "get_current_page", lambda: 0)() or 0)
        if target != current:
            backend.set_page(target)
            try:
                large_view.set_backend(backend)
            except Exception:
                try:
                    backend.set_page(current)
                except Exception:
                    pass
                raise
            label_backend = None
            if hasattr(large_view, "label_layer_backend"):
                label_backend = large_view.label_layer_backend()
            if label_backend is not None:
                try:
                    label_page_count = int(
                        getattr(label_backend, "get_page_count", lambda: 1)() or 1
                    )
                    if label_page_count == page_count and page_count > 1:
                        label_backend.set_page(target)
                except Exception:
                    pass
            if hasattr(self, "_syncRasterImageLayerPages"):
                try:
                    self._syncRasterImageLayerPages()
                except Exception:
                    pass
            if hasattr(self, "_restoreBaseRasterImageVisibilityFromState"):
                try:
                    self._restoreBaseRasterImageVisibilityFromState()
                except Exception:
                    pass
        self.frame_number = target
        self.num_frames = page_count
        if hasattr(self, "_loadLargeImagePageAnnotations"):
            self._loadLargeImagePageAnnotations(target)
        elif canvas is not None:
            large_view.set_shapes(getattr(canvas, "shapes", []) or [])
        if hasattr(self, "_restoreLabelImageOverlayFromState"):
            self._restoreLabelImageOverlayFromState()
        if hasattr(self, "_restoreBaseRasterImageVisibilityFromState"):
            self._restoreBaseRasterImageVisibilityFromState()
        if hasattr(self, "_restoreRasterImageLayersFromState"):
            self._restoreRasterImageLayersFromState()
        seekbar = getattr(self, "seekbar", None)
        if seekbar is not None and self._has_large_image_page_navigation():
            try:
                with QtCore.QSignalBlocker(seekbar):
                    seekbar.setValue(target)
            except Exception:
                pass
        self._syncLargeImageDocument()
        if hasattr(self, "_refreshBrain3DSessionDock"):
            try:
                self._refreshBrain3DSessionDock()
            except Exception:
                pass
        self._post_window_status(
            self.tr("Viewing TIFF page %d of %d") % (target + 1, page_count)
        )
        return True

    def _settings_value(self, key: str, default, value_type):
        settings = getattr(self, "settings", None)
        if settings is None:
            return default
        try:
            return settings.value(key, default, type=value_type)
        except Exception:
            return default

    def status(self, message: str, timeout: int = 4000) -> None:
        """Compat status updater for mixins expecting LabelMe-like status()."""
        try:
            bar = self.statusBar()
        except Exception:
            bar = None
        if bar is None:
            return
        try:
            bar.showMessage(str(message), int(timeout))
        except Exception:
            pass

    def _post_window_status(self, message: str, timeout: int = 4000) -> None:
        image_path = str(getattr(self, "imagePath", "") or "")
        if (
            (image_path and bool(is_large_tiff_path(image_path)))
            or getattr(self, "large_image_backend", None) is not None
            or getattr(self, "_active_image_view", "canvas") == "tiled"
        ):
            return
        post_window_status(self, message, timeout)

    def _ensureLargeImageModeWidgets(self) -> None:
        if getattr(self, "_large_image_surface_label", None) is not None:
            return
        surface_label = QtWidgets.QLabel(self)
        surface_label.setObjectName("largeImageSurfaceIndicator")
        surface_label.setStyleSheet("font-weight: 600;")
        surface_label.hide()
        mode_label = QtWidgets.QLabel(self)
        mode_label.setObjectName("largeImageModeIndicator")
        mode_label.hide()
        return_button = QtWidgets.QToolButton(self)
        return_button.setObjectName("largeImageReturnButton")
        return_button.setText(self.tr("Return to Tiled Viewer"))
        return_button.clicked.connect(self.returnToLargeImageTiledView)
        return_button.hide()
        self._large_image_surface_label = surface_label
        self._large_image_mode_label = mode_label
        self._large_image_return_button = return_button

    def _updateLargeImageModeWidgets(self) -> None:
        self._ensureLargeImageModeWidgets()
        surface_label = self._large_image_surface_label
        mode_label = self._large_image_mode_label
        return_button = self._large_image_return_button
        document = getattr(self, "large_image_document", None)
        if document is not None:
            surface = str(getattr(document, "surface", "canvas") or "canvas")
            draw_mode = str(getattr(document, "draw_mode", "polygon") or "polygon")
            if surface == "tiled":
                surface_label.setText(self.tr("Large Image: Tiled Viewer"))
                mode_label.setText(
                    self.tr("Mode: Editing")
                    if bool(getattr(document, "editing", True))
                    else self.tr("Mode: %s") % large_image_draw_mode_label(draw_mode)
                )
            else:
                surface_label.setText(self.tr("Large Image: Canvas Preview"))
                fallback_reason = str(
                    getattr(self, "_large_image_surface_reason", "") or ""
                ).strip()
                mode_label.setText(
                    self.tr("Fallback: %s") % fallback_reason
                    if fallback_reason
                    else self.tr("Mode: %s") % large_image_draw_mode_label(draw_mode)
                )
        for widget in (surface_label, mode_label, return_button):
            try:
                widget.hide()
            except Exception:
                pass

    def returnToLargeImageTiledView(self, _value: bool = False) -> bool:
        backend = getattr(self, "large_image_backend", None)
        large_view = getattr(self, "large_image_view", None)
        if backend is None or large_view is None:
            return False
        if hasattr(self, "toggleDrawMode"):
            try:
                self.toggleDrawMode(True)
            except Exception:
                pass
        try:
            large_view.set_shapes(list(getattr(self.canvas, "shapes", []) or []))
        except Exception:
            pass
        if hasattr(self, "_viewer_stack") and self._viewer_stack is not None:
            self._viewer_stack.setCurrentWidget(large_view)
        self._active_image_view = "tiled"
        self._large_image_surface_reason = ""
        if hasattr(self, "canvas") and self.canvas is not None:
            self.canvas.setEnabled(False)
        self._syncLargeImageDocument()
        return True

    def _captureLargeImageViewport(self) -> LargeImageViewport:
        tiled = getattr(self, "large_image_view", None)
        zoom_widget = getattr(self, "zoomWidget", None)
        default_zoom = int(zoom_widget.value()) if zoom_widget is not None else 100
        if tiled is not None and hasattr(tiled, "viewport_state"):
            try:
                state = dict(tiled.viewport_state() or {})
                return LargeImageViewport(
                    zoom_percent=int(
                        state.get("zoom_percent", default_zoom) or default_zoom
                    ),
                    center_x=float(state.get("center_x", 0.0) or 0.0),
                    center_y=float(state.get("center_y", 0.0) or 0.0),
                    fit_mode=str(state.get("fit_mode", "fit_window") or "fit_window"),
                )
            except Exception:
                pass
        return LargeImageViewport(
            zoom_percent=default_zoom,
            center_x=0.0,
            center_y=0.0,
            fit_mode="fit_window",
        )

    def _captureLargeImageSelection(self) -> LargeImageSelectionState:
        canvas = getattr(self, "canvas", None)
        large_view = getattr(self, "large_image_view", None)
        selected_shapes = list(getattr(canvas, "selectedShapes", []) or [])
        selected_overlay_id = None
        for shape in selected_shapes:
            other = dict(getattr(shape, "other_data", {}) or {})
            overlay_id = str(other.get("overlay_id") or "")
            if overlay_id:
                selected_overlay_id = overlay_id
                break
        selected_label_value = None
        if large_view is not None and hasattr(large_view, "selected_label_value"):
            try:
                selected_label_value = large_view.selected_label_value()
            except Exception:
                selected_label_value = None
        return LargeImageSelectionState(
            selected_shape_count=len(selected_shapes),
            selected_shape_labels=[
                str(getattr(shape, "label", "") or "") for shape in selected_shapes
            ],
            selected_overlay_id=selected_overlay_id or None,
            selected_landmark_pair_id=(
                str(getattr(self, "_selected_overlay_landmark_pair_id", "") or "")
                or None
            ),
            selected_label_value=selected_label_value,
        )

    def _captureLargeImageCacheMetadata(self) -> dict:
        other_data = getattr(self, "otherData", None)
        if not isinstance(other_data, dict):
            return {}
        return dict(other_data.get("large_image") or {})

    def _captureLargeImageDrawMode(self) -> tuple[str, bool]:
        if getattr(self, "_active_image_view", "canvas") == "tiled":
            tiled = getattr(self, "large_image_view", None)
            if tiled is not None:
                try:
                    editing = bool(tiled.editing())
                except Exception:
                    editing = True
                return str(
                    getattr(tiled, "createMode", "polygon") or "polygon"
                ), editing
        canvas = getattr(self, "canvas", None)
        editing = True
        if canvas is not None and hasattr(canvas, "editing"):
            try:
                editing = bool(canvas.editing())
            except Exception:
                editing = True
        create_mode = (
            str(getattr(canvas, "createMode", "polygon") or "polygon")
            if canvas is not None
            else "polygon"
        )
        return create_mode, editing

    def _syncLargeImageDocument(self) -> LargeImageDocument | None:
        image_path = str(getattr(self, "imagePath", "") or "")
        backend = getattr(self, "large_image_backend", None)
        if (
            not image_path
            or not bool(is_large_tiff_path(image_path))
            or backend is None
        ):
            self.large_image_document = None
            self._updateLargeImageModeWidgets()
            return None
        page_count = max(1, int(getattr(backend, "get_page_count", lambda: 1)() or 1))
        current_page = int(getattr(backend, "get_current_page", lambda: 0)() or 0)
        draw_mode, editing = self._captureLargeImageDrawMode()
        large_view = getattr(self, "large_image_view", None)
        active_label_layer_id = None
        if large_view is not None and hasattr(large_view, "label_layer_backend"):
            try:
                if large_view.label_layer_backend() is not None:
                    active_label_layer_id = "label_image_overlay"
            except Exception:
                active_label_layer_id = None
        label_overlay_state = {}
        if large_view is not None and hasattr(large_view, "label_overlay_state"):
            try:
                label_overlay_state = dict(large_view.label_overlay_state() or {})
            except Exception:
                label_overlay_state = {}
        capabilities = LargeImageBackendCapabilities()
        if hasattr(backend, "capabilities"):
            try:
                capabilities = backend.capabilities()
            except Exception:
                capabilities = LargeImageBackendCapabilities()
        self.large_image_document = LargeImageDocument(
            image_path=image_path,
            backend=backend,
            backend_name=str(getattr(backend, "name", "") or ""),
            backend_capabilities=capabilities,
            current_page=current_page,
            page_count=page_count,
            surface=str(getattr(self, "_active_image_view", "canvas") or "canvas"),
            draw_mode=draw_mode,
            editing=bool(editing),
            viewport=self._captureLargeImageViewport(),
            active_layers=self.viewerLayerModels(),
            active_label_layer_id=active_label_layer_id,
            label_overlay_state=label_overlay_state,
            cache_metadata=self._captureLargeImageCacheMetadata(),
            selection=self._captureLargeImageSelection(),
        )
        self._updateLargeImageModeWidgets()
        refresh_layer_dock = getattr(self, "_refreshViewerLayerDock", None)
        if callable(refresh_layer_dock):
            try:
                refresh_layer_dock()
            except Exception:
                pass
        return self.large_image_document

    def currentLargeImageDocument(self) -> LargeImageDocument | None:
        return self._syncLargeImageDocument()

    def currentLargeImageBackendCapabilities(self) -> LargeImageBackendCapabilities:
        return self._largeImageBackendCapabilities()

    def _largeImageBackendCapabilities(
        self, backend=None
    ) -> LargeImageBackendCapabilities:
        backend = (
            backend
            if backend is not None
            else getattr(self, "large_image_backend", None)
        )
        document = getattr(self, "large_image_document", None)
        if (
            document is not None
            and backend is not None
            and backend is getattr(document, "backend", None)
        ):
            return document.backend_capabilities
        if backend is None:
            document = self.currentLargeImageDocument()
            if document is not None:
                return document.backend_capabilities
        if backend is not None and hasattr(backend, "capabilities"):
            try:
                return backend.capabilities()
            except Exception:
                pass
        return LargeImageBackendCapabilities()

    def _largeImageSupportsPages(self, backend=None) -> bool:
        backend = (
            backend
            if backend is not None
            else getattr(self, "large_image_backend", None)
        )
        if backend is None:
            return False
        capabilities = self._largeImageBackendCapabilities(backend)
        if not bool(capabilities.supports_pages):
            return False
        if not hasattr(backend, "get_page_count"):
            return False
        try:
            return int(backend.get_page_count() or 1) > 1
        except Exception:
            return False

    def _largeImageCurrentPage(self) -> int:
        document = getattr(self, "large_image_document", None)
        if document is not None:
            return int(document.current_page or 0)
        return int(getattr(self, "frame_number", 0) or 0)

    def _largeImagePageCount(self) -> int:
        document = getattr(self, "large_image_document", None)
        if document is not None:
            return max(1, int(document.page_count or 1))
        backend = getattr(self, "large_image_backend", None)
        if backend is not None and hasattr(backend, "get_page_count"):
            try:
                return max(1, int(backend.get_page_count() or 1))
            except Exception:
                pass
        return max(1, int(getattr(self, "num_frames", 1) or 1))

    def currentRasterImageLayer(self) -> RasterImageLayer | None:
        image_path = str(getattr(self, "imagePath", "") or "")
        if not image_path:
            return None
        backend = getattr(self, "large_image_backend", None)
        page_index = 0
        if backend is not None and hasattr(backend, "get_current_page"):
            try:
                page_index = int(backend.get_current_page() or 0)
            except Exception:
                page_index = 0
        base_visible = bool(
            getattr(self, "_active_image_view", "canvas") in {"canvas", "tiled"}
        )
        large_view = getattr(self, "large_image_view", None)
        if (
            large_view is not None
            and getattr(self, "_active_image_view", "canvas") == "tiled"
            and hasattr(large_view, "base_raster_visible")
        ):
            try:
                base_visible = bool(large_view.base_raster_visible())
            except Exception:
                pass
        return RasterImageLayer(
            id="raster_image",
            name=Path(image_path).name,
            visible=base_visible,
            opacity=1.0,
            locked=True,
            z_index=-100,
            transform=AffineTransform(),
            backend_page_index=page_index,
            channel=None,
        )

    def viewerLayerModels(self) -> list[ViewerLayerModel]:
        layers: list[ViewerLayerModel] = []
        raster_layer = self.currentRasterImageLayer()
        if raster_layer is not None:
            layers.append(raster_layer)
        if hasattr(self, "currentLabelImageLayer"):
            try:
                label_layer = self.currentLabelImageLayer()
            except Exception:
                label_layer = None
            if label_layer is not None:
                layers.append(label_layer)
        if hasattr(self, "rasterOverlayLayers"):
            try:
                layers.extend(list(self.rasterOverlayLayers() or []))
            except Exception:
                pass
        if hasattr(self, "vectorOverlayLayers"):
            try:
                layers.extend(list(self.vectorOverlayLayers() or []))
            except Exception:
                pass
        if hasattr(self, "vectorOverlayLandmarkLayers"):
            try:
                layers.extend(list(self.vectorOverlayLandmarkLayers() or []))
            except Exception:
                pass
        if hasattr(self, "currentAnnotationLayer"):
            try:
                annotation_layer = self.currentAnnotationLayer()
            except Exception:
                annotation_layer = None
            if annotation_layer is not None:
                layers.append(annotation_layer)
        return sorted(layers, key=lambda layer: int(getattr(layer, "z_index", 0)))

    def _large_image_irrelevant_docks(self) -> list[QtWidgets.QDockWidget]:
        docks = []
        for name in (
            "video_dock",
            "timeline_dock",
            "behavior_log_dock",
            "behavior_controls_dock",
            "audio_dock",
            "caption_dock",
            "embedding_search_dock",
            "florence_dock",
            # Keep the keypoint sequencer out of the default large-image/overlay
            # workflow. Users can still open it manually when needed.
            "keypoint_sequence_dock",
        ):
            dock = getattr(self, name, None)
            if isinstance(dock, QtWidgets.QDockWidget):
                docks.append(dock)
        return docks

    def _large_image_related_docks(self) -> list[QtWidgets.QDockWidget]:
        docks = []
        for name in (
            "file_dock",
            "flag_dock",
            "label_dock",
            "shape_dock",
            "viewer_layer_dock",
            "vector_overlay_dock",
        ):
            dock = getattr(self, name, None)
            if isinstance(dock, QtWidgets.QDockWidget):
                docks.append(dock)
        return docks

    def setLargeImageDocksActive(self, active: bool) -> None:
        if active:
            related_states = getattr(self, "_large_image_related_docks_states", None)
            if related_states is None:
                related_states = {}
                self._large_image_related_docks_states = related_states
            for dock in self._large_image_irrelevant_docks():
                try:
                    if dock not in self._large_image_hidden_docks_states:
                        self._large_image_hidden_docks_states[dock] = dock.isHidden()
                    dock.hide()
                except Exception:
                    continue
            for dock in self._large_image_related_docks():
                try:
                    if dock not in related_states:
                        related_states[dock] = dock.isHidden()
                    dock.show()
                except Exception:
                    continue
            return

        for dock, was_hidden in list(self._large_image_hidden_docks_states.items()):
            try:
                if was_hidden:
                    dock.hide()
                else:
                    dock.show()
            except Exception:
                continue
        self._large_image_hidden_docks_states.clear()
        related_states = getattr(self, "_large_image_related_docks_states", {})
        for dock, was_hidden in list(related_states.items()):
            try:
                if was_hidden:
                    dock.hide()
                else:
                    dock.show()
            except Exception:
                continue
        related_states.clear()

    def largeImageCachePolicy(self) -> dict[str, int]:
        max_entries = self._settings_value(
            LARGE_IMAGE_CACHE_MAX_ENTRIES_KEY,
            DEFAULT_LARGE_IMAGE_CACHE_MAX_ENTRIES,
            int,
        )
        max_size_gb = self._settings_value(
            LARGE_IMAGE_CACHE_MAX_SIZE_GB_KEY,
            max(1, DEFAULT_LARGE_IMAGE_CACHE_MAX_SIZE_BYTES // (1024**3)),
            int,
        )
        max_entries = max(1, int(max_entries))
        max_size_gb = max(1, int(max_size_gb))
        return {
            "max_entries": max_entries,
            "max_size_gb": max_size_gb,
            "max_size_bytes": max_size_gb * 1024 * 1024 * 1024,
        }

    def setLargeImageCachePolicy(
        self, *, max_entries: int, max_size_gb: int
    ) -> dict[str, int]:
        policy = {
            "max_entries": max(1, int(max_entries)),
            "max_size_gb": max(1, int(max_size_gb)),
        }
        settings = getattr(self, "settings", None)
        if settings is not None:
            try:
                settings.setValue(
                    LARGE_IMAGE_CACHE_MAX_ENTRIES_KEY, policy["max_entries"]
                )
                settings.setValue(
                    LARGE_IMAGE_CACHE_MAX_SIZE_GB_KEY, policy["max_size_gb"]
                )
            except Exception:
                pass
        policy["max_size_bytes"] = policy["max_size_gb"] * 1024 * 1024 * 1024
        return policy

    def largeImageCacheOptimizeOptions(self) -> dict[str, int]:
        policy = self.largeImageCachePolicy()
        return {
            "max_cache_entries": policy["max_entries"],
            "max_cache_size_bytes": policy["max_size_bytes"],
        }

    def configureLargeImageCachePolicy(
        self, _value: bool = False
    ) -> dict[str, int] | None:
        current = self.largeImageCachePolicy()
        max_entries, ok_entries = QtWidgets.QInputDialog.getInt(
            self,
            self.tr("Large TIFF Cache Limits"),
            self.tr("Maximum optimized TIFF cache files to keep:"),
            current["max_entries"],
            1,
            1000,
            1,
        )
        if not ok_entries:
            return None
        max_size_gb, ok_size = QtWidgets.QInputDialog.getInt(
            self,
            self.tr("Large TIFF Cache Limits"),
            self.tr("Maximum total optimized TIFF cache size (GB):"),
            current["max_size_gb"],
            1,
            4096,
            1,
        )
        if not ok_size:
            return None
        policy = self.setLargeImageCachePolicy(
            max_entries=max_entries, max_size_gb=max_size_gb
        )
        if hasattr(self, "status"):
            self.status(
                self.tr("Updated large TIFF cache limits to %d file(s), %d GB")
                % (policy["max_entries"], policy["max_size_gb"])
            )
        return policy

    def optimizeLargeImageForViewing(self, _value: bool = False) -> str | None:
        source_path = Path(str(getattr(self, "imagePath", "") or "")).expanduser()
        if not source_path.exists() or not bool(is_large_tiff_path(source_path)):
            if hasattr(self, "errorMessage"):
                self.errorMessage(
                    self.tr("Optimize Large Image"),
                    self.tr(
                        "Open a TIFF-family image before creating an optimized viewing cache."
                    ),
                )
            return None
        available, reason = pyvips_optimization_available()
        if not available:
            if hasattr(self, "errorMessage"):
                self.errorMessage(
                    self.tr("Optimize Large Image"),
                    self.tr(
                        "Fast-view optimization requires a working pyvips/libvips runtime. Current error: %s"
                    )
                    % str(reason or self.tr("unknown error")),
                )
            return None
        if getattr(self, "_large_image_opt_thread", None) is not None:
            if hasattr(self, "status"):
                self.status(self.tr("Large image optimization is already running"))
            return None

        progress = QtWidgets.QProgressDialog(
            self.tr("Building optimized pyramidal TIFF cache…"),
            "",
            0,
            0,
            self,
        )
        progress.setWindowTitle(self.tr("Optimize Large Image"))
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)

        def _task() -> str:
            return str(
                optimize_large_tiff_for_viewing(
                    source_path,
                    **self.largeImageCacheOptimizeOptions(),
                )
            )

        thread = QtCore.QThread(self)
        worker = FlexibleWorker(_task)
        worker.moveToThread(thread)
        self._large_image_opt_thread = thread
        self._large_image_opt_worker = worker
        self._large_image_opt_progress = progress
        self._large_image_opt_source_path = source_path

        def _finish(result) -> None:
            self._finishLargeImageOptimization(result)

        thread.started.connect(worker.run)
        worker.finished_signal.connect(_finish)
        progress.show()
        thread.start()
        if hasattr(self, "status"):
            self.status(self.tr("Optimizing large TIFF for faster future viewing…"))
        return None

    def _finishLargeImageOptimization(self, result) -> None:
        progress = getattr(self, "_large_image_opt_progress", None)
        thread = getattr(self, "_large_image_opt_thread", None)
        worker = getattr(self, "_large_image_opt_worker", None)
        source_path = getattr(self, "_large_image_opt_source_path", None)
        try:
            if progress is not None:
                progress.close()
        except Exception:
            pass
        try:
            if thread is not None:
                thread.quit()
                thread.wait(2000)
        except Exception:
            pass
        try:
            if worker is not None:
                worker.deleteLater()
        except Exception:
            pass
        try:
            if thread is not None:
                thread.deleteLater()
        except Exception:
            pass
        self._large_image_opt_progress = None
        self._large_image_opt_thread = None
        self._large_image_opt_worker = None
        self._large_image_opt_source_path = None

        if isinstance(result, Exception):
            if hasattr(self, "errorMessage"):
                self.errorMessage(
                    self.tr("Optimize Large Image"),
                    self.tr("Failed to build optimized viewing cache: %s")
                    % str(result),
                )
            return
        if source_path is None:
            return
        self._applyOptimizedLargeImageCache(source_path, Path(str(result)))

    def _applyOptimizedLargeImageCache(
        self, source_path: Path, cache_path: Path
    ) -> str:
        if not isinstance(getattr(self, "otherData", None), dict):
            self.otherData = {}
        large_info = dict(self.otherData.get("large_image") or {})
        large_info["optimized_cache_path"] = str(cache_path)
        self.otherData["large_image"] = large_info

        try:
            backend = open_large_image(cache_path)
            self._activate_tiled_image_view(source_path, backend=backend)
            if hasattr(self, "large_image_view"):
                self.large_image_view.set_shapes(getattr(self.canvas, "shapes", []))
        except Exception:
            pass
        if hasattr(self, "status"):
            self.status(
                self.tr("Created optimized pyramidal TIFF cache at %s")
                % str(cache_path)
            )
        if hasattr(self, "setDirty"):
            self.setDirty()
        return str(cache_path)

    def _currentLargeImageCachePath(self) -> Path | None:
        large_info = (
            self.otherData.get("large_image")
            if isinstance(getattr(self, "otherData", None), dict)
            else None
        )
        if not isinstance(large_info, dict):
            return None
        cache_path = large_info.get("optimized_cache_path")
        if not cache_path:
            return None
        return Path(str(cache_path)).expanduser()

    def _reloadCurrentLargeImageFromSource(self) -> None:
        source_path = Path(str(getattr(self, "imagePath", "") or "")).expanduser()
        if not source_path.exists() or not bool(is_large_tiff_path(source_path)):
            return
        try:
            backend = open_large_image(source_path)
            self._activate_tiled_image_view(source_path, backend=backend)
            if hasattr(self, "large_image_view"):
                self.large_image_view.set_shapes(getattr(self.canvas, "shapes", []))
        except Exception as exc:
            logger.warning(
                "Failed to reload large image source after cache cleanup: %s", exc
            )

    def showLargeImageCacheInfo(self, _value: bool = False) -> dict[str, object]:
        entries = list_large_image_cache_entries()
        total_size = large_image_cache_size_bytes()
        cache_root = large_image_cache_root()
        policy = self.largeImageCachePolicy()
        current_cache_path = self._currentLargeImageCachePath()
        current_cache_exists = bool(current_cache_path and current_cache_path.exists())
        current_cache_size = 0
        if current_cache_exists and current_cache_path is not None:
            try:
                current_cache_size = int(current_cache_path.stat().st_size)
            except OSError:
                current_cache_exists = False
        message_lines = [
            self.tr("Cache folder: %s") % str(cache_root),
            self.tr("Cached TIFFs: %d") % len(entries),
            self.tr("Disk usage: %s") % format_large_image_cache_size(total_size),
            self.tr("Cache limit: %d file(s), %d GB")
            % (policy["max_entries"], policy["max_size_gb"]),
        ]
        if current_cache_path is not None:
            message_lines.append(
                self.tr("Current image cache: %s") % str(current_cache_path)
            )
            if current_cache_exists:
                message_lines.append(
                    self.tr("Current cache size: %s")
                    % format_large_image_cache_size(current_cache_size)
                )
            else:
                message_lines.append(
                    self.tr("Current image cache is not present on disk")
                )
        QtWidgets.QMessageBox.information(
            self,
            self.tr("Large Image Cache"),
            "\n".join(message_lines),
        )
        return {
            "cache_root": str(cache_root),
            "entry_count": len(entries),
            "total_size_bytes": total_size,
            "policy": policy,
            "current_cache_path": str(current_cache_path)
            if current_cache_path
            else None,
            "current_cache_exists": current_cache_exists,
        }

    def openLargeImageCacheFolder(self, _value: bool = False) -> str | None:
        cache_root = large_image_cache_root()
        cache_root.mkdir(parents=True, exist_ok=True)
        ok = QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(cache_root)))
        if not ok:
            if hasattr(self, "errorMessage"):
                self.errorMessage(
                    self.tr("Large Image Cache"),
                    self.tr("Could not open cache folder: %s") % str(cache_root),
                )
            return None
        if hasattr(self, "status"):
            self.status(
                self.tr("Opened large image cache folder: %s") % str(cache_root)
            )
        return str(cache_root)

    def clearCurrentLargeImageCache(self, _value: bool = False) -> int:
        cache_path = self._currentLargeImageCachePath()
        if cache_path is None:
            if hasattr(self, "status"):
                self.status(
                    self.tr("No optimized cache is recorded for the current image")
                )
            return 0
        answer = QtWidgets.QMessageBox.question(
            self,
            self.tr("Clear Current Cache"),
            self.tr("Delete the optimized TIFF cache for the current image?\n\n%s")
            % str(cache_path),
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return 0
        removed = remove_large_image_cache_file(cache_path)
        if isinstance(getattr(self, "otherData", None), dict):
            large_info = dict(self.otherData.get("large_image") or {})
            large_info.pop("optimized_cache_path", None)
            self.otherData["large_image"] = large_info
        self._reloadCurrentLargeImageFromSource()
        if hasattr(self, "status"):
            if removed:
                self.status(self.tr("Cleared optimized cache for the current image"))
            else:
                self.status(
                    self.tr("Removed stale cache reference for the current image")
                )
        if hasattr(self, "setDirty"):
            self.setDirty()
        return 1 if removed else 0

    def clearAllLargeImageCaches(self, _value: bool = False) -> int:
        entries = list_large_image_cache_entries()
        if not entries:
            if hasattr(self, "status"):
                self.status(self.tr("No large image caches are present"))
            return 0
        answer = QtWidgets.QMessageBox.question(
            self,
            self.tr("Clear All Large Image Caches"),
            self.tr("Delete %d optimized TIFF cache file(s) from %s?")
            % (len(entries), str(large_image_cache_root())),
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return 0
        removed = clear_all_large_image_caches()
        if isinstance(getattr(self, "otherData", None), dict):
            large_info = dict(self.otherData.get("large_image") or {})
            large_info.pop("optimized_cache_path", None)
            self.otherData["large_image"] = large_info
        self._reloadCurrentLargeImageFromSource()
        if hasattr(self, "status"):
            self.status(
                self.tr("Cleared %d optimized TIFF cache file(s)") % int(removed)
            )
        if removed and hasattr(self, "setDirty"):
            self.setDirty()
        return int(removed)

    def _activate_canvas_image_view(
        self,
        image: QtGui.QImage,
        *,
        preserve_shapes: bool = False,
        clear_large_image_view: bool = True,
    ) -> None:
        if not hasattr(self, "canvas") or self.canvas is None:
            return
        if clear_large_image_view:
            self.large_image_backend = None
        self._active_image_view = "canvas"
        self.canvas.setEnabled(True)
        if clear_large_image_view and hasattr(self, "large_image_view"):
            self.large_image_view.clear()
        if hasattr(self, "_viewer_stack") and self._viewer_stack is not None:
            self._viewer_stack.setCurrentWidget(self.canvas)
        pixmap = QtGui.QPixmap.fromImage(image)
        self.canvas.loadPixmap(pixmap, clear_shapes=not bool(preserve_shapes))
        self._syncLargeImageDocument()

    def activateLargeImageCanvasEditMode(
        self, *, reason: str = "overlay editing"
    ) -> bool:
        if getattr(self, "_active_image_view", "canvas") != "tiled":
            return False
        image = getattr(self, "image", None)
        if image is None or image.isNull():
            return False
        self._activate_canvas_image_view(
            image,
            preserve_shapes=True,
            clear_large_image_view=False,
        )
        self._large_image_surface_reason = str(reason or self.tr("editing"))
        if hasattr(self.canvas, "setEditing"):
            self.canvas.setEditing(True)
        return True

    def loadFile(self, filename: str) -> None:
        path = Path(filename)
        if not path.exists():
            self.errorMessage(
                self.tr("File Error"), self.tr("Missing file: %s") % filename
            )
            return

        image_path = path
        shapes = []

        initial_large_image_page = None
        if path.suffix.lower() == ".json":
            try:
                label_file = LabelFile(str(path))
            except LabelFileError as exc:
                self.errorMessage(self.tr("Load Error"), str(exc))
                return
            self.labelFile = label_file
            self.otherData = dict(getattr(label_file, "otherData", {}) or {})
            try:
                initial_large_image_page = int(
                    (getattr(label_file, "otherData", {}) or {})
                    .get("large_image_page", {})
                    .get("page_index", 0)
                )
            except Exception:
                initial_large_image_page = None
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
                    self.otherData = dict(getattr(label_file, "otherData", {}) or {})
                    shapes = label_file.shapes
                except Exception:
                    shapes = []
            if not isinstance(getattr(self, "otherData", None), dict):
                self.otherData = {}

        backend_image_path = image_path
        if bool(is_large_tiff_path(image_path)):
            cached_large_image = resolve_fresh_optimized_large_image_path(image_path)
            if cached_large_image is not None:
                backend_image_path = cached_large_image

        image_result = None
        probed_large_image = None
        large_backend = None
        if bool(is_large_tiff_path(backend_image_path)) and hasattr(
            self, "large_image_view"
        ):
            try:
                large_backend = open_large_image(backend_image_path)
                image_result = large_backend.load()
                probed_large_image = image_result.metadata
            except Exception:
                image_result = None
                large_backend = None
        if image_result is None:
            try:
                image_result = load_image_with_backends(backend_image_path)
            except Exception:
                image_result = None

        image = image_result.qimage if image_result is not None else QtGui.QImage()
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
        if probed_large_image is None:
            probed_large_image = probe_large_image(backend_image_path)
        if image_result is not None and image_result.metadata is not None:
            if not isinstance(self.otherData, dict):
                self.otherData = {}
            large_info = image_result.metadata.to_dict()
            if backend_image_path != image_path:
                large_info["optimized_cache_path"] = str(backend_image_path)
                large_info["source_path"] = str(image_path)
            self.otherData["large_image"] = large_info
        elif probed_large_image is not None:
            if not isinstance(self.otherData, dict):
                self.otherData = {}
            large_info = probed_large_image.to_dict()
            if backend_image_path != image_path:
                large_info["optimized_cache_path"] = str(backend_image_path)
                large_info["source_path"] = str(image_path)
            self.otherData["large_image"] = large_info

        if hasattr(self, "canvas") and self.canvas is not None:
            use_tiled_view = False
            handled_page_shapes = False
            if bool(is_large_tiff_path(backend_image_path)) and hasattr(
                self, "large_image_view"
            ):
                try:
                    use_tiled_view = self._activate_tiled_image_view(
                        image_path, backend=large_backend
                    )
                except Exception:
                    self.large_image_backend = None
                    use_tiled_view = False
            if not use_tiled_view:
                self._clear_large_image_stack_navigation()
                self.setLargeImageDocksActive(False)
                self._activate_canvas_image_view(image)
                if hasattr(self, "_restoreLabelImageOverlayFromState"):
                    self._restoreLabelImageOverlayFromState()
                if hasattr(self, "_restoreRasterImageLayersFromState"):
                    self._restoreRasterImageLayersFromState()
            elif large_backend is not None:
                self.setLargeImageDocksActive(True)
                if (
                    initial_large_image_page is not None
                    and self._largeImageSupportsPages(large_backend)
                ):
                    try:
                        page_count = int(
                            getattr(large_backend, "get_page_count", lambda: 1)() or 1
                        )
                        if 0 <= int(initial_large_image_page) < page_count:
                            large_backend.set_page(int(initial_large_image_page))
                            self.large_image_view.set_backend(large_backend)
                    except Exception:
                        pass
                self._setup_large_image_stack_navigation(large_backend)
                if self._largeImageSupportsPages(large_backend) and hasattr(
                    self, "_loadLargeImagePageAnnotations"
                ):
                    handled_page_shapes = True
                    self.frame_number = int(
                        getattr(large_backend, "get_current_page", lambda: 0)() or 0
                    )
                    self.num_frames = int(
                        getattr(large_backend, "get_page_count", lambda: 1)() or 1
                    )
                    self._loadLargeImagePageAnnotations(
                        self.frame_number,
                        fallback_shapes=shapes,
                        fallback_other_data=self.otherData,
                    )
                if hasattr(self, "_restoreLabelImageOverlayFromState"):
                    self._restoreLabelImageOverlayFromState()
                if hasattr(self, "_restoreBaseRasterImageVisibilityFromState"):
                    self._restoreBaseRasterImageVisibilityFromState()
                if hasattr(self, "_restoreRasterImageLayersFromState"):
                    self._restoreRasterImageLayersFromState()
            # AnnolidWindow.loadLabels expects LabelFile JSON dict payloads and
            # materializes Shape objects before forwarding to loadShapes.
            if handled_page_shapes:
                pass
            elif shapes and isinstance(shapes[0], dict) and hasattr(self, "loadLabels"):
                self.loadLabels(shapes)
            elif hasattr(self, "loadShapes"):
                self.loadShapes(shapes, replace=True)
            self._syncLargeImageDocument()

        # Ensure image workflows (Open / Open Dir) activate toolbar actions.
        if hasattr(self, "toggleActions"):
            self.toggleActions(True)

        if (
            bool(is_large_tiff_path(image_path))
            and image_result is not None
            and image_result.metadata is not None
            and hasattr(self, "status")
        ):
            backend_name = str(image_result.metadata.backend_name or "image backend")
            hint = str(image_result.metadata.performance_hint or "").strip()
            message = self.tr("Opened large image with %s backend") % backend_name
            if hint:
                message = f"{message}. {hint}"
            self.status(message)

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
