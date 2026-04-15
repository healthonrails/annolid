from __future__ import annotations

from pathlib import Path
from typing import Optional

from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.label_file import LabelFile, LabelFileError
from annolid.gui.widgets.zone_manager_utils import (
    available_arena_layout_presets,
    build_zone_popup_defaults,
    default_zone_label,
    generate_arena_layout_preset,
    is_zone_shape,
    normalize_zone_flags,
    shape_to_zone_payload,
    zone_file_for_source,
    zone_kind_palette,
    zone_payload_to_shape,
)
from annolid.postprocessing.zone_schema import load_zone_shapes
from annolid.postprocessing.zone_schema import normalize_zone_shape


class ZonePanelWidget(QtWidgets.QWidget):
    """User-facing zone authoring panel for frame-backed zone shapes."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        zone_path: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("zonePanel")

        self._owner_window = parent
        self.canvas = getattr(parent, "canvas", None)
        if self.canvas is None:
            raise ValueError("ZonePanelWidget requires a parent with a canvas.")
        self._zone_file_path = (
            Path(zone_path)
            if zone_path
            else zone_file_for_source(self._current_source())
        )
        self._dirty = False
        self._syncing_selection = False
        self._selected_shape = None
        self._preset_available = False

        self.canvas.newShape.connect(self._handle_new_shape)
        self.canvas.selectionChanged.connect(self._handle_canvas_selection)
        self.canvas.shapeMoved.connect(self._mark_dirty_and_refresh)

        self._build_ui()
        self.refresh_from_current_canvas()
        self._publish_zone_defaults()

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        self.setStyleSheet(
            """
            QWidget#zonePanel {
                background: palette(window);
                color: palette(window-text);
            }
            QFrame[card="true"],
            QFrame#zoneHero,
            QFrame#zoneCard {
                background: palette(base);
                border: 1px solid palette(mid);
                border-radius: 12px;
            }
            QLabel[zoneCardTitle="true"] {
                color: palette(mid);
                font-size: 11px;
                font-weight: 600;
                text-transform: uppercase;
            }
            QLabel[zoneCardValue="true"] {
                color: palette(text);
                font-size: 20px;
                font-weight: 700;
            }
            QLabel[zoneHeroTitle="true"] {
                color: palette(text);
                font-size: 22px;
                font-weight: 700;
            }
            QLabel[zoneHeroSubtitle="true"] {
                color: palette(mid);
            }
            QLabel[zoneHint="true"] {
                color: palette(mid);
            }
            QListWidget {
                border: 1px solid palette(mid);
                border-radius: 10px;
                background: palette(base);
                padding: 4px;
            }
            QTabWidget::pane {
                border: 1px solid palette(mid);
                border-radius: 12px;
                background: palette(base);
                top: -1px;
            }
            QTabBar::tab {
                padding: 8px 12px;
                margin-right: 4px;
                border-top-left-radius: 10px;
                border-top-right-radius: 10px;
                background: palette(button);
                color: palette(window-text);
            }
            QTabBar::tab:selected {
                background: palette(base);
                color: palette(text);
                font-weight: 600;
            }
            QGroupBox {
                border: 1px solid palette(mid);
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 10px;
                font-weight: 600;
                background: palette(base);
            }
            QGroupBox::title {
                left: 10px;
                top: -2px;
                padding: 0 4px;
            }
            QPushButton[primary="true"] {
                font-weight: 700;
            }
            """
        )

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(0)

        scroll = QtWidgets.QScrollArea(self)
        scroll.setObjectName("zonePanelScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        outer.addWidget(scroll)

        content = QtWidgets.QWidget(scroll)
        scroll.setWidget(content)
        content.setObjectName("zonePanelContent")
        root = QtWidgets.QVBoxLayout(content)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        container = content

        hero = QtWidgets.QFrame(container)
        hero.setObjectName("zoneHero")
        hero.setProperty("card", True)
        hero_layout = QtWidgets.QVBoxLayout(hero)
        hero_layout.setContentsMargins(14, 14, 14, 14)
        hero_layout.setSpacing(6)
        title = QtWidgets.QLabel("Zone Studio")
        title.setProperty("zoneHeroTitle", True)
        subtitle = QtWidgets.QLabel(
            "Draw on the live frame, label what each region means, and preview the zone metrics Annolid will export later."
        )
        subtitle.setWordWrap(True)
        subtitle.setProperty("zoneHeroSubtitle", True)
        hero_layout.addWidget(title)
        hero_layout.addWidget(subtitle)
        root.addWidget(hero)

        summary_strip = QtWidgets.QFrame(container)
        summary_strip.setProperty("card", True)
        summary_layout = QtWidgets.QHBoxLayout(summary_strip)
        summary_layout.setContentsMargins(10, 6, 10, 6)
        summary_layout.setSpacing(10)
        self.zone_count_value = self._create_inline_metric(summary_layout, "Zones", "0")
        self.selected_zone_value = self._create_inline_metric(
            summary_layout, "Selected", "None"
        )
        self.metrics_ready_value = self._create_inline_metric(
            summary_layout, "Analysis-ready", "0"
        )
        root.addWidget(summary_strip)

        quick_start = QtWidgets.QFrame(container)
        quick_start.setObjectName("zoneCard")
        quick_start.setProperty("card", True)
        quick_layout = QtWidgets.QVBoxLayout(quick_start)
        quick_layout.setContentsMargins(12, 8, 12, 8)
        quick_layout.setSpacing(6)
        quick_row = QtWidgets.QHBoxLayout()
        quick_row.setSpacing(6)
        quick_label = QtWidgets.QLabel("Quick Start")
        quick_label.setProperty("zoneHeroSubtitle", True)
        quick_row.addWidget(quick_label)
        self.preset_combo = QtWidgets.QComboBox()
        self._preset_definitions = available_arena_layout_presets()
        self._preset_available = bool(self._preset_definitions)
        if self._preset_available:
            for preset in self._preset_definitions:
                self.preset_combo.addItem(
                    str(preset.get("title") or preset.get("key") or "Preset"),
                    userData=str(preset.get("key") or "").strip(),
                )
        else:
            self.preset_combo.addItem("No presets available", userData="")
            self.preset_combo.setEnabled(False)
        self.generate_preset_button = QtWidgets.QPushButton("Generate Preset")
        self.generate_preset_button.clicked.connect(self.generate_selected_preset)
        self.generate_preset_button.setEnabled(self._preset_available)
        self.generate_preset_button.setProperty("primary", True)
        self.refresh_current_button = QtWidgets.QPushButton("Refresh Canvas")
        self.refresh_current_button.clicked.connect(self.refresh_from_current_canvas)
        self.load_zone_button = QtWidgets.QPushButton("Load Zone JSON")
        self.load_zone_button.clicked.connect(self.open_zone_file)
        self.save_zone_button = QtWidgets.QPushButton("Save Zone JSON")
        self.save_zone_button.clicked.connect(self.save_zone_file_as)
        preset_row = QtWidgets.QHBoxLayout()
        preset_row.setSpacing(6)
        preset_row.addWidget(self.preset_combo, 1)
        preset_row.addWidget(self.generate_preset_button)
        refresh_row = QtWidgets.QHBoxLayout()
        refresh_row.setSpacing(6)
        refresh_row.addWidget(self.refresh_current_button)
        file_row = QtWidgets.QHBoxLayout()
        file_row.setSpacing(6)
        file_row.addWidget(self.load_zone_button)
        file_row.addWidget(self.save_zone_button)
        quick_layout.addLayout(quick_row)
        quick_layout.addLayout(preset_row)
        quick_layout.addLayout(refresh_row)
        quick_layout.addLayout(file_row)
        root.addWidget(quick_start)

        self.tabs = QtWidgets.QTabWidget(container)
        self.tabs.addTab(self._build_define_tab(), "Define Zones")
        self.tabs.addTab(self._build_details_tab(), "Zone Details")
        self.tabs.addTab(self._build_metrics_tab(), "Metrics")
        root.addWidget(self.tabs, 1)

        bottom_row = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel(container)
        self.status_label.setWordWrap(True)
        self.close_button = QtWidgets.QPushButton("Close", container)
        self.close_button.clicked.connect(self.close)
        bottom_row.addWidget(self.status_label, 1)
        bottom_row.addWidget(self.close_button)
        root.addLayout(bottom_row)

        self.default_zone_kind_combo.currentTextChanged.connect(
            lambda *_: self._publish_zone_defaults()
        )
        self.default_phase_combo.currentTextChanged.connect(
            lambda *_: self._publish_zone_defaults()
        )
        self.default_occupant_combo.currentTextChanged.connect(
            lambda *_: self._publish_zone_defaults()
        )
        self.default_access_combo.currentTextChanged.connect(
            lambda *_: self._publish_zone_defaults()
        )
        self.default_tags_edit.editingFinished.connect(self._publish_zone_defaults)
        self.default_barrier_checkbox.toggled.connect(
            lambda *_: self._publish_zone_defaults()
        )

    def _build_define_tab(self) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        page = QtWidgets.QWidget(scroll)
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        source_box = QtWidgets.QGroupBox("Frame Context", page)
        source_layout = QtWidgets.QFormLayout(source_box)
        self.source_label = QtWidgets.QLabel("")
        self.source_label.setWordWrap(True)
        self.save_target_label = QtWidgets.QLabel("")
        self.save_target_label.setWordWrap(True)
        self.canvas_status = QtWidgets.QLabel(
            "Use the main Annolid toolbar or the canvas context menu to switch between draw and edit modes."
        )
        self.canvas_status.setWordWrap(True)
        self.canvas_status.setProperty("zoneHint", True)
        source_layout.addRow("Current frame", self.source_label)
        source_layout.addRow("Zone JSON", self.save_target_label)
        source_layout.addRow("Canvas", self.canvas_status)
        layout.addWidget(source_box)

        list_box = QtWidgets.QGroupBox("Zone Inventory", page)
        list_layout = QtWidgets.QVBoxLayout(list_box)
        self.shape_list_label = QtWidgets.QLabel(
            "Inventory shows only explicit zones and keyword-detected zone candidates from your canvas."
        )
        self.shape_list_label.setWordWrap(True)
        self.shape_list_label.setProperty("zoneHint", True)
        filter_row = QtWidgets.QHBoxLayout()
        filter_row.setSpacing(6)
        self.zone_filter_edit = QtWidgets.QLineEdit()
        self.zone_filter_edit.setPlaceholderText(
            "Filter by label, kind, phase, role, tags"
        )
        self.zone_filter_edit.textChanged.connect(
            lambda *_: self._refresh_shape_list(select_shape=self._selected_shape)
        )
        self.zone_sort_combo = QtWidgets.QComboBox()
        self.zone_sort_combo.addItem("Sort: Label", userData="label")
        self.zone_sort_combo.addItem("Sort: Kind", userData="kind")
        self.zone_sort_combo.addItem("Sort: Area (largest)", userData="area_desc")
        self.zone_sort_combo.currentIndexChanged.connect(
            lambda *_: self._refresh_shape_list(select_shape=self._selected_shape)
        )
        filter_row.addWidget(self.zone_filter_edit, 1)
        filter_row.addWidget(self.zone_sort_combo)
        self.inventory_summary_label = QtWidgets.QLabel("Showing 0 of 0 zones")
        self.inventory_summary_label.setProperty("zoneHint", True)
        self.shape_list = QtWidgets.QListWidget(self)
        self.shape_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.shape_list.currentRowChanged.connect(self._on_shape_row_changed)
        self.shape_list.itemSelectionChanged.connect(self._on_shape_selection_changed)
        list_layout.addWidget(self.shape_list_label)
        list_layout.addLayout(filter_row)
        list_layout.addWidget(self.inventory_summary_label)
        list_layout.addWidget(self.shape_list, 1)
        layout.addWidget(list_box, 1)

        scroll.setWidget(page)
        return scroll

    def _build_details_tab(self) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        page = QtWidgets.QWidget(scroll)
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.selected_summary_label = QtWidgets.QLabel(
            "No zone selected. Pick a zone on the canvas or from the list."
        )
        self.selected_summary_label.setWordWrap(True)
        self.selected_summary_label.setProperty("zoneHint", True)
        layout.addWidget(self.selected_summary_label)

        selected_box = QtWidgets.QGroupBox("Selected Zone", page)
        selected_layout = QtWidgets.QFormLayout(selected_box)
        self.zone_label_edit = QtWidgets.QLineEdit()
        self.zone_label_edit.setPlaceholderText("zone label")
        self.zone_description_edit = QtWidgets.QLineEdit()
        self.zone_description_edit.setPlaceholderText("optional description")
        self.zone_kind_combo = self._create_metadata_combo(
            ["chamber", "doorway", "barrier_edge", "interaction_zone", "custom"]
        )
        self.phase_combo = self._create_metadata_combo(["phase_1", "phase_2", "custom"])
        self.occupant_combo = self._create_metadata_combo(
            ["rover", "stim", "neutral", "unknown"]
        )
        self.access_combo = self._create_metadata_combo(
            ["open", "blocked", "tethered", "unknown"]
        )
        self.tags_edit = QtWidgets.QLineEdit()
        self.tags_edit.setPlaceholderText("comma,separated,tags")
        self.barrier_adjacent_checkbox = QtWidgets.QCheckBox(
            "Force barrier-adjacent metric for this zone"
        )

        selected_layout.addRow("Label", self.zone_label_edit)
        selected_layout.addRow("Description", self.zone_description_edit)
        selected_layout.addRow("Zone kind", self.zone_kind_combo)
        selected_layout.addRow("Phase", self.phase_combo)
        selected_layout.addRow("Occupant role", self.occupant_combo)
        selected_layout.addRow("Access state", self.access_combo)
        selected_layout.addRow("Tags", self.tags_edit)
        selected_layout.addRow("", self.barrier_adjacent_checkbox)

        classify_row = QtWidgets.QHBoxLayout()
        classify_row.setSpacing(6)
        self.classify_selected_button = QtWidgets.QPushButton("Use as Zone")
        self.classify_selected_button.clicked.connect(
            self._classify_selected_shape_as_zone
        )
        self.apply_selected_button = QtWidgets.QPushButton("Apply Zone Details")
        self.apply_selected_button.clicked.connect(self._apply_selected_zone_details)
        self.use_as_defaults_button = QtWidgets.QPushButton("Use Selected as Defaults")
        self.use_as_defaults_button.clicked.connect(self._use_selected_as_defaults)
        classify_row.addWidget(self.classify_selected_button)
        classify_row.addWidget(self.apply_selected_button)
        classify_row.addWidget(self.use_as_defaults_button)

        manage_row = QtWidgets.QHBoxLayout()
        manage_row.setSpacing(6)
        self.stroke_button = QtWidgets.QPushButton("Recolor")
        self.stroke_button.clicked.connect(self._recolor_selected_shape)
        self.duplicate_button = QtWidgets.QPushButton("Duplicate")
        self.duplicate_button.clicked.connect(self._duplicate_selected_shape)
        self.delete_button = QtWidgets.QPushButton("Delete Selected")
        self.delete_button.clicked.connect(self._delete_selected_shape)
        manage_row.addWidget(self.stroke_button)
        manage_row.addWidget(self.duplicate_button)
        manage_row.addWidget(self.delete_button)
        selected_layout.addRow("Zone Actions", classify_row)
        selected_layout.addRow("Shape Actions", manage_row)
        layout.addWidget(selected_box)

        defaults_box = QtWidgets.QGroupBox("Defaults for New Zones", page)
        defaults_layout = QtWidgets.QFormLayout(defaults_box)
        self.default_zone_kind_combo = self._create_metadata_combo(
            ["chamber", "doorway", "barrier_edge", "interaction_zone", "custom"]
        )
        self.default_phase_combo = self._create_metadata_combo(
            ["phase_1", "phase_2", "custom"]
        )
        self.default_occupant_combo = self._create_metadata_combo(
            ["rover", "stim", "neutral", "unknown"]
        )
        self.default_access_combo = self._create_metadata_combo(
            ["open", "blocked", "tethered", "unknown"]
        )
        self.default_tags_edit = QtWidgets.QLineEdit()
        self.default_tags_edit.setPlaceholderText("comma,separated,tags")
        self.default_barrier_checkbox = QtWidgets.QCheckBox(
            "Mark new zones as barrier-adjacent"
        )
        defaults_layout.addRow("Zone kind", self.default_zone_kind_combo)
        defaults_layout.addRow("Phase", self.default_phase_combo)
        defaults_layout.addRow("Occupant role", self.default_occupant_combo)
        defaults_layout.addRow("Access state", self.default_access_combo)
        defaults_layout.addRow("Tags", self.default_tags_edit)
        defaults_layout.addRow("", self.default_barrier_checkbox)
        layout.addWidget(defaults_box)

        scroll.setWidget(page)
        return scroll

    def _build_metrics_tab(self) -> QtWidgets.QWidget:
        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        page = QtWidgets.QWidget(scroll)
        layout = QtWidgets.QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        summary_box = QtWidgets.QGroupBox("Selected Zone Preview", page)
        summary_layout = QtWidgets.QFormLayout(summary_box)
        self.selected_area_value = QtWidgets.QLabel("0 px²")
        self.selected_area_value.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        self.selected_metric_summary = QtWidgets.QLabel(
            "Select a zone to preview area and the metrics it will affect."
        )
        self.selected_metric_summary.setWordWrap(True)
        self.selected_metric_summary.setProperty("zoneHint", True)
        summary_layout.addRow("Area", self.selected_area_value)
        summary_layout.addRow("Preview", self.selected_metric_summary)
        layout.addWidget(summary_box)

        self.metric_table = QtWidgets.QTableWidget(0, 3, page)
        self.metric_table.setHorizontalHeaderLabels(
            ["Metric", "How Annolid computes it", "Selected zone impact"]
        )
        self.metric_table.horizontalHeader().setStretchLastSection(True)
        self.metric_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents
        )
        self.metric_table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.Stretch
        )
        self.metric_table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.Stretch
        )
        self.metric_table.verticalHeader().setVisible(False)
        self.metric_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.metric_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.metric_table.setAlternatingRowColors(True)
        layout.addWidget(self.metric_table, 1)

        self.metrics_footer_label = QtWidgets.QLabel(
            "Phase, occupant role, access state, and barrier-adjacent flags influence which zones appear in assay-aware summaries."
        )
        self.metrics_footer_label.setWordWrap(True)
        self.metrics_footer_label.setProperty("zoneHint", True)
        layout.addWidget(self.metrics_footer_label)

        scroll.setWidget(page)
        return scroll

    def _create_inline_metric(
        self,
        parent_layout: QtWidgets.QHBoxLayout,
        title: str,
        value: str,
    ) -> QtWidgets.QLabel:
        box = QtWidgets.QWidget(self)
        box_layout = QtWidgets.QHBoxLayout(box)
        box_layout.setContentsMargins(0, 0, 0, 0)
        box_layout.setSpacing(6)
        title_label = QtWidgets.QLabel(f"{title}:")
        title_label.setProperty("zoneCardTitle", True)
        value_label = QtWidgets.QLabel(value)
        value_label.setProperty("zoneCardValue", True)
        box_layout.addWidget(title_label)
        box_layout.addWidget(value_label)
        box_layout.addStretch(1)
        parent_layout.addWidget(box)
        return value_label

    def _create_metadata_combo(self, options: list[str]) -> QtWidgets.QComboBox:
        combo = QtWidgets.QComboBox()
        combo.setEditable(True)
        combo.addItems(options)
        return combo

    # ------------------------------------------------------------------ #
    # Frame and zone loading
    # ------------------------------------------------------------------ #
    def _set_status(self, text: str) -> None:
        self.status_label.setText(str(text or ""))

    def _confirm_discard_changes(self) -> bool:
        if not self._dirty:
            return True
        reply = QtWidgets.QMessageBox.question(
            self,
            "Discard unsaved changes?",
            "Current zone edits have not been saved. Replace them?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        return reply == QtWidgets.QMessageBox.Yes

    def _update_source_label(self) -> None:
        source = self._current_source() or "(current canvas)"
        self.source_label.setText(source)

    def _update_save_target_label(self) -> None:
        target = str(self._zone_file_path) if self._zone_file_path else "(not set)"
        self.save_target_label.setText(target)

    def _current_source(self) -> str:
        parent = getattr(self, "_owner_window", None) or self.parent()
        if parent is None:
            return ""
        source = getattr(parent, "video_file", None) or getattr(
            parent, "filename", None
        )
        return str(source or "").strip()

    def _current_canvas_pixmap(self) -> QtGui.QPixmap | None:
        pixmap = getattr(self.canvas, "pixmap", None)
        if pixmap is not None and not pixmap.isNull():
            return pixmap
        return None

    def _current_canvas_size(self) -> tuple[int, int] | None:
        pixmap = self._current_canvas_pixmap()
        if pixmap is None:
            return None
        return int(pixmap.width()), int(pixmap.height())

    def _current_canvas_image_path(self) -> str:
        parent = getattr(self, "_owner_window", None) or self.parent()
        if parent is None:
            return ""
        return str(getattr(parent, "filename", "") or "").strip()

    def _publish_zone_defaults(self) -> None:
        parent = getattr(self, "_owner_window", None) or self.parent()
        if parent is None:
            return
        extra_flags = {}
        if self.default_barrier_checkbox.isChecked():
            extra_flags["barrier_adjacent"] = True
        defaults = build_zone_popup_defaults(
            label=default_zone_label(
                self.default_zone_kind_combo.currentText().strip(),
                self._existing_labels(),
            ),
            zone_kind=self.default_zone_kind_combo.currentText().strip() or "custom",
            phase=self.default_phase_combo.currentText().strip() or "custom",
            occupant_role=self.default_occupant_combo.currentText().strip()
            or "unknown",
            access_state=self.default_access_combo.currentText().strip() or "unknown",
            tags=self.default_tags_edit.text().strip(),
            description="",
        )
        if extra_flags:
            defaults["flags"].update(extra_flags)
        setattr(parent, "_zone_authoring_defaults", defaults)
        self._update_metrics_preview()

    def _load_shapes_to_canvas(self, shapes: list, *, replace: bool = True) -> None:
        parent = getattr(self, "_owner_window", None) or self.parent()
        loader = getattr(parent, "loadShapes", None)
        if callable(loader):
            loader(shapes, replace=replace)
        else:
            self.canvas.loadShapes(shapes, replace=replace)
        if hasattr(self.canvas, "setEditing"):
            self.canvas.setEditing(True)

    def _merge_generated_zones(self, generated_shapes: list) -> None:
        current_non_zone = [
            shape for shape in self.canvas.shapes if not is_zone_shape(shape)
        ]
        self._load_shapes_to_canvas(
            current_non_zone + list(generated_shapes), replace=True
        )
        if generated_shapes:
            self._selected_shape = generated_shapes[0]
            self._refresh_shape_list(select_shape=generated_shapes[0])
            self.canvas.selectShapes([generated_shapes[0]])
        self._dirty = True
        self._set_status(
            f"Generated {len(generated_shapes)} zone(s) on the current canvas."
        )

    def generate_selected_preset(self) -> None:
        if not self._preset_available:
            QtWidgets.QMessageBox.information(
                self,
                "No presets available",
                "No arena presets are available. You can still draw and save zones manually.",
            )
            return
        dimensions = self._current_canvas_size()
        if dimensions is None:
            QtWidgets.QMessageBox.warning(
                self,
                "No frame loaded",
                "Load a video frame before generating an arena preset.",
            )
            return

        preset_key = self.preset_combo.currentData() or self.preset_combo.currentText()
        if not preset_key:
            QtWidgets.QMessageBox.warning(
                self, "Missing preset", "Choose an arena preset to generate."
            )
            return

        existing_zone_count = sum(
            1 for shape in self.canvas.shapes if is_zone_shape(shape)
        )
        if existing_zone_count:
            reply = QtWidgets.QMessageBox.question(
                self,
                "Replace existing zones?",
                "Generate a new preset and replace the current zone shapes on this canvas?\n"
                "Non-zone annotations will be preserved.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return

        width, height = dimensions
        try:
            generated_shapes = generate_arena_layout_preset(preset_key, width, height)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Preset generation failed", str(exc))
            return

        if not generated_shapes:
            QtWidgets.QMessageBox.information(
                self,
                "No shapes generated",
                "The selected preset did not generate any shapes.",
            )
            return

        self._merge_generated_zones(generated_shapes)
        self._publish_zone_defaults()
        self.canvas.setFocus()
        self._refresh_shape_list(select_shape=generated_shapes[0])
        self.tabs.setCurrentIndex(1)

    def refresh_from_current_canvas(self) -> None:
        pixmap = self._current_canvas_pixmap()
        if pixmap is None:
            self.canvas_status.setText(
                "Load a video or image in the main canvas before drawing zones."
            )
            self._update_source_label()
            self._update_save_target_label()
            self.shape_list.clear()
            self._selected_shape = None
            self._sync_selected_fields()
            self._update_stats()
            return
        if self._zone_file_path is None:
            self._zone_file_path = zone_file_for_source(self._current_source())
        self._update_source_label()
        self._update_save_target_label()
        self.canvas.setEditing(True)
        self._refresh_shape_list()
        self._set_status("Canvas ready for zone editing.")
        self._publish_zone_defaults()
        self._update_stats()

    def open_zone_file(self) -> None:
        if not self._confirm_discard_changes():
            return
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Zone JSON",
            str(Path.home()),
            "LabelMe JSON (*.json)",
        )
        if not filename:
            return
        self.load_zone_file(filename, replace=True)

    def load_zone_file(self, filename: str, *, replace: bool = True) -> None:
        try:
            lf = LabelFile(filename)
        except LabelFileError as exc:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
            return

        try:
            zone_specs = load_zone_shapes(
                {
                    "shapes": lf.shapes,
                    "imagePath": lf.imagePath,
                    "imageData": lf.imageData,
                    "flags": lf.flags,
                    "caption": lf.caption,
                    **(lf.otherData or {}),
                }
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
            return

        shapes = [zone_payload_to_shape(spec.to_shape_dict()) for spec in zone_specs]
        if replace:
            current = [
                shape for shape in self.canvas.shapes if not is_zone_shape(shape)
            ]
            self.canvas.loadShapes(current + shapes, replace=True)
        else:
            self.canvas.loadShapes(shapes, replace=False)
        self._refresh_shape_list(select_shape=shapes[0] if shapes else None)
        self._set_status(f"Loaded {len(shapes)} zone(s) from {filename}.")
        self._dirty = False
        self._zone_file_path = Path(filename)
        self._update_save_target_label()

    # ------------------------------------------------------------------ #
    # Shape lifecycle
    # ------------------------------------------------------------------ #
    def _existing_labels(self) -> list[str]:
        return [str(getattr(shape, "label", "") or "") for shape in self.canvas.shapes]

    def _shape_matches_inventory_filter(self, shape) -> bool:
        filter_edit = getattr(self, "zone_filter_edit", None)
        token = (
            str(filter_edit.text() or "").strip().lower()
            if filter_edit is not None
            else ""
        )
        if not token:
            return True
        flags = dict(getattr(shape, "flags", {}) or {})
        haystack = " ".join(
            [
                str(getattr(shape, "label", "") or ""),
                str(getattr(shape, "description", "") or ""),
                str(flags.get("zone_kind") or ""),
                str(flags.get("phase") or ""),
                str(flags.get("occupant_role") or ""),
                str(flags.get("access_state") or ""),
                str(getattr(shape, "shape_type", "") or ""),
                ", ".join(str(tag or "") for tag in (flags.get("tags") or [])),
            ]
        ).lower()
        return token in haystack

    def _shape_payload_for_schema(self, shape) -> dict:
        points = []
        for point in list(getattr(shape, "points", []) or []):
            x = point.x() if hasattr(point, "x") else point[0]
            y = point.y() if hasattr(point, "y") else point[1]
            points.append([float(x), float(y)])
        return {
            "label": str(getattr(shape, "label", "") or ""),
            "description": str(getattr(shape, "description", "") or ""),
            "shape_type": str(getattr(shape, "shape_type", "polygon") or "polygon"),
            "points": points,
            "flags": dict(getattr(shape, "flags", {}) or {}),
            "group_id": getattr(shape, "group_id", None),
            "visible": bool(getattr(shape, "visible", True)),
        }

    def _shape_zone_spec(self, shape):
        try:
            return normalize_zone_shape(self._shape_payload_for_schema(shape))
        except Exception:
            return None

    def _shape_zone_state(self, shape) -> str:
        spec = self._shape_zone_spec(shape)
        if spec is None:
            return "shape"
        if spec.is_zone and spec.inferred_from_legacy:
            return "zone (keyword)"
        if spec.is_zone:
            return "zone"
        return "shape"

    def _shape_is_inventory_zone_candidate(self, shape) -> bool:
        return self._shape_zone_state(shape) in {"zone", "zone (keyword)"}

    def _sort_inventory_shapes(self, shapes: list) -> list:
        sort_combo = getattr(self, "zone_sort_combo", None)
        sort_mode = (
            str(sort_combo.currentData() or "label").strip()
            if sort_combo is not None
            else "label"
        )
        if sort_mode == "kind":
            return sorted(
                shapes,
                key=lambda shape: (
                    str((getattr(shape, "flags", {}) or {}).get("zone_kind") or ""),
                    str(getattr(shape, "label", "") or ""),
                ),
            )
        if sort_mode == "area_desc":
            return sorted(
                shapes,
                key=lambda shape: (
                    -float(self._shape_area(shape)),
                    str(getattr(shape, "label", "") or ""),
                ),
            )
        return sorted(shapes, key=lambda shape: str(getattr(shape, "label", "") or ""))

    def _apply_shape_style(self, shape) -> None:
        zone_kind = str(shape.flags.get("zone_kind") or "custom").strip()
        stroke, fill = zone_kind_palette(zone_kind)
        shape.fill = True
        shape.line_color = QtGui.QColor(stroke)
        shape.fill_color = QtGui.QColor(fill)
        shape.select_line_color = QtGui.QColor(255, 255, 255, 255)
        shape.select_fill_color = QtGui.QColor(
            stroke.red(), stroke.green(), stroke.blue(), 160
        )

    def _handle_new_shape(self) -> None:
        if not self.canvas.shapes:
            return
        shape = self.canvas.shapes[-1]
        self._refresh_shape_list(select_shape=shape)
        zone_state = self._shape_zone_state(shape)
        if zone_state == "zone (keyword)":
            self._set_status(
                "Shape added and recognized as a zone from label/description keywords. Review details and click 'Use as Zone' to save explicit metadata."
            )
            return
        self._set_status(
            "Shape added. Select it and click 'Use as Zone' if it should be included in zone analysis."
        )

    def _delete_selected_shape(self) -> None:
        shape = self._selected_shape
        if shape is None:
            return
        self.canvas.selectShapes([shape])
        deleted = self.canvas.deleteSelected() or []
        if deleted:
            self._selected_shape = None
            self._refresh_shape_list()
            self._mark_dirty()
            self._set_status("Deleted selected zone.")

    def _recolor_selected_shape(self) -> None:
        shape = self._selected_shape
        if shape is None:
            return
        current = getattr(shape, "line_color", None)
        current_qcolor = (
            QtGui.QColor(current)
            if current is not None and not isinstance(current, QtGui.QColor)
            else current or QtGui.QColor("#4a90e2")
        )
        color = QtWidgets.QColorDialog.getColor(current_qcolor, self, "Pick Zone Color")
        if not color.isValid():
            return
        shape.line_color = QtGui.QColor(color)
        shape.fill_color = QtGui.QColor(color.red(), color.green(), color.blue(), 85)
        shape.select_line_color = QtGui.QColor(255, 255, 255, 255)
        shape.select_fill_color = QtGui.QColor(
            color.red(), color.green(), color.blue(), 160
        )
        shape.fill = True
        self.canvas.update()
        self._mark_dirty()

    def _duplicate_selected_shape(self) -> None:
        shape = self._selected_shape
        if shape is None:
            return
        flags = dict(getattr(shape, "flags", {}) or {})
        cloned_payload = shape_to_zone_payload(
            shape,
            label=None,
            zone_kind=flags.get("zone_kind"),
            phase=flags.get("phase"),
            occupant_role=flags.get("occupant_role"),
            access_state=flags.get("access_state"),
            tags=flags.get("tags"),
            description=getattr(shape, "description", "") or "",
            extra_flags=flags,
        )
        zone_kind = str(cloned_payload.get("flags", {}).get("zone_kind") or "zone")
        cloned_payload["label"] = default_zone_label(zone_kind, self._existing_labels())
        shifted_points: list[list[float]] = []
        for point in cloned_payload.get("points") or []:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                shifted_points.append([float(point[0]) + 8.0, float(point[1]) + 8.0])
        if shifted_points:
            cloned_payload["points"] = shifted_points
        duplicate = zone_payload_to_shape(cloned_payload)
        all_shapes = list(self.canvas.shapes or [])
        insert_index = (
            all_shapes.index(shape) + 1 if shape in all_shapes else len(all_shapes)
        )
        all_shapes.insert(insert_index, duplicate)
        self._load_shapes_to_canvas(all_shapes, replace=True)
        self._selected_shape = duplicate
        self.canvas.selectShapes([duplicate])
        self._refresh_shape_list(select_shape=duplicate)
        self._mark_dirty()
        self._set_status(f"Duplicated zone as '{duplicate.label}'.")

    def _handle_canvas_selection(self, shapes: list) -> None:
        if self._syncing_selection:
            return
        selected = list(shapes or [])
        shape = selected[0] if selected else None
        self._selected_shape = shape
        self._syncing_selection = True
        try:
            if shape is None:
                self.shape_list.clearSelection()
                self._set_status("No zone selected.")
            else:
                row = self._row_for_shape(shape)
                if row is not None:
                    self.shape_list.setCurrentRow(row)
                self._set_status(f"Selected '{shape.label}'.")
        finally:
            self._syncing_selection = False
        self._sync_selected_fields()
        self._update_stats()

    def _on_shape_selection_changed(self) -> None:
        if self._syncing_selection:
            return
        item = self.shape_list.currentItem()
        shape = item.data(QtCore.Qt.UserRole) if item is not None else None
        if shape is None:
            self.canvas.selectShapes([])
            self._selected_shape = None
            self._sync_selected_fields()
            self._update_stats()
            return
        self._selected_shape = shape
        self._syncing_selection = True
        try:
            self.canvas.selectShapes([shape])
            self._set_status(f"Selected '{shape.label}'.")
        finally:
            self._syncing_selection = False
        self._sync_selected_fields()
        self._update_stats()

    def _on_shape_row_changed(self, row: int) -> None:
        if row < 0:
            return
        item = self.shape_list.item(row)
        if item is None:
            return
        shape = item.data(QtCore.Qt.UserRole)
        if shape is None:
            return
        self._selected_shape = shape
        self._syncing_selection = True
        try:
            self.canvas.selectShapes([shape])
        finally:
            self._syncing_selection = False
        self._sync_selected_fields()
        self._update_stats()

    def _row_for_shape(self, shape) -> int | None:
        for row in range(self.shape_list.count()):
            item = self.shape_list.item(row)
            if item is not None and item.data(QtCore.Qt.UserRole) is shape:
                return row
        return None

    def _zone_area_text(self, shape) -> str:
        area = self._shape_area(shape)
        return f"{area:.1f} px²"

    def _shape_area(self, shape) -> float:
        points = list(getattr(shape, "points", []) or [])
        if len(points) < 2:
            return 0.0
        coords = []
        for point in points:
            x = point.x() if hasattr(point, "x") else point[0]
            y = point.y() if hasattr(point, "y") else point[1]
            coords.append((float(x), float(y)))
        shape_type = str(getattr(shape, "shape_type", "polygon") or "polygon")
        if shape_type == "rectangle" and len(coords) >= 2:
            width = abs(coords[1][0] - coords[0][0])
            height = abs(coords[1][1] - coords[0][1])
            return float(width * height)
        if len(coords) < 3:
            return 0.0
        area = 0.0
        for index, (x1, y1) in enumerate(coords):
            x2, y2 = coords[(index + 1) % len(coords)]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0

    def _metric_rows_for_shape(self, shape) -> list[tuple[str, str, str]]:
        is_zone = bool(shape is not None and is_zone_shape(shape))
        flags = dict(getattr(shape, "flags", {}) or {}) if shape is not None else {}
        zone_kind = str(flags.get("zone_kind") or "custom").strip().lower()
        phase = str(flags.get("phase") or "custom").strip() or "custom"
        occupant_role = (
            str(flags.get("occupant_role") or "unknown").strip() or "unknown"
        )
        access_state = str(flags.get("access_state") or "unknown").strip() or "unknown"
        barrier_enabled = bool(flags.get("barrier_adjacent")) or zone_kind in {
            "barrier_edge",
            "barrier",
            "doorway",
            "passage",
        }
        if shape is None:
            return [
                (
                    "Occupancy + dwell",
                    "Counts how many frames a tracked point stays inside each zone.",
                    "Select a zone to preview its contribution.",
                ),
                (
                    "Entries + first entry",
                    "Starts a new visit when a subject enters a zone from outside or from another zone.",
                    "Select a zone to preview its contribution.",
                ),
                (
                    "Transitions",
                    "Counts moves between consecutive resolved zones.",
                    "Select a zone to preview its contribution.",
                ),
                (
                    "Barrier-adjacent",
                    "Enabled for barrier-edge, doorway, or explicitly flagged zones.",
                    "Select a zone to preview its contribution.",
                ),
            ]
        if not is_zone:
            return [
                (
                    "Not yet a zone",
                    "This shape will not appear in zone metrics until it is marked as a zone.",
                    "Click 'Use as Zone' in Zone Details.",
                ),
                (
                    "Zone metrics",
                    "Occupancy, dwell, entry, transition, and latency are computed only for zone-tagged shapes.",
                    "After classification, apply zone kind/phase/role/access metadata.",
                ),
            ]
        barrier_text = (
            "Enabled for this zone."
            if barrier_enabled
            else "Not enabled unless you mark it barrier-adjacent."
        )
        return [
            (
                "Occupancy + dwell",
                "Counts per-zone presence over time and total dwell duration.",
                f"Included as '{getattr(shape, 'label', '') or zone_kind}'.",
            ),
            (
                "Entries + first entry",
                "Tracks zone visits and latency to first arrival.",
                f"Uses this zone boundary and label for visit segments in {phase}.",
            ),
            (
                "Transitions",
                "Counts movement from one zone label to the next resolved zone.",
                f"Transitions into or out of this zone are tracked whenever access is '{access_state}'.",
            ),
            (
                "Profile filters",
                "Assay summaries can include or exclude zones by phase, role, and access state.",
                f"Phase '{phase}', role '{occupant_role}', access '{access_state}'.",
            ),
            (
                "Barrier-adjacent",
                "Special summary for barrier-edge style regions.",
                barrier_text,
            ),
        ]

    def _update_metrics_table(self) -> None:
        rows = self._metric_rows_for_shape(self._selected_shape)
        self.metric_table.setRowCount(len(rows))
        for row_index, (metric, meaning, impact) in enumerate(rows):
            for column_index, value in enumerate((metric, meaning, impact)):
                item = QtWidgets.QTableWidgetItem(value)
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.metric_table.setItem(row_index, column_index, item)

    def _update_metrics_preview(self) -> None:
        shape = self._selected_shape
        if shape is None:
            self.selected_area_value.setText("0 px²")
            default_text = (
                "New zones will default to "
                f"kind '{self.default_zone_kind_combo.currentText().strip() or 'custom'}', "
                f"phase '{self.default_phase_combo.currentText().strip() or 'custom'}', "
                f"role '{self.default_occupant_combo.currentText().strip() or 'unknown'}', "
                f"access '{self.default_access_combo.currentText().strip() or 'unknown'}'."
            )
            self.selected_metric_summary.setText(default_text)
            self._update_metrics_table()
            return
        if not is_zone_shape(shape):
            preview = (
                f"'{getattr(shape, 'label', '') or '(unnamed)'}' is currently a regular annotation shape. "
                "Use it as a zone to include it in zone metrics and exports."
            )
            self.selected_area_value.setText(self._zone_area_text(shape))
            self.selected_metric_summary.setText(preview)
            self._update_metrics_table()
            return
        flags = dict(getattr(shape, "flags", {}) or {})
        zone_kind = str(flags.get("zone_kind") or "custom").strip()
        barrier_enabled = bool(flags.get("barrier_adjacent")) or zone_kind.lower() in {
            "barrier_edge",
            "barrier",
            "doorway",
            "passage",
        }
        preview = (
            f"'{getattr(shape, 'label', '') or zone_kind}' contributes occupancy, dwell, entry, first-entry, and transition metrics. "
            f"Barrier-adjacent summary is {'enabled' if barrier_enabled else 'disabled'}."
        )
        self.selected_area_value.setText(self._zone_area_text(shape))
        self.selected_metric_summary.setText(preview)
        self._update_metrics_table()

    def _sync_selected_fields(self) -> None:
        shape = self._selected_shape
        widgets = [
            self.zone_label_edit,
            self.zone_description_edit,
            self.tags_edit,
            self.zone_kind_combo,
            self.phase_combo,
            self.occupant_combo,
            self.access_combo,
        ]
        for widget in widgets:
            widget.blockSignals(True)
        self.barrier_adjacent_checkbox.blockSignals(True)
        try:
            if shape is None:
                self.zone_label_edit.setText("")
                self.zone_description_edit.setText("")
                self.tags_edit.setText("")
                self._set_combo_text(self.zone_kind_combo, "custom")
                self._set_combo_text(self.phase_combo, "custom")
                self._set_combo_text(self.occupant_combo, "unknown")
                self._set_combo_text(self.access_combo, "unknown")
                self.barrier_adjacent_checkbox.setChecked(False)
                self.selected_summary_label.setText(
                    "No zone selected. Pick a zone on the canvas or from the list."
                )
            else:
                flags = dict(getattr(shape, "flags", {}) or {})
                spec = self._shape_zone_spec(shape)
                self.zone_label_edit.setText(str(getattr(shape, "label", "") or ""))
                self.zone_description_edit.setText(
                    str(getattr(shape, "description", "") or "")
                )
                if is_zone_shape(shape):
                    self.tags_edit.setText(
                        ", ".join(str(tag) for tag in flags.get("tags") or [])
                    )
                    self._set_combo_text(
                        self.zone_kind_combo,
                        str(
                            flags.get("zone_kind")
                            or (spec.zone_kind if spec is not None else "custom")
                            or "custom"
                        ),
                    )
                    self._set_combo_text(
                        self.phase_combo,
                        str(
                            flags.get("phase")
                            or (spec.phase if spec is not None else "custom")
                            or "custom"
                        ),
                    )
                    self._set_combo_text(
                        self.occupant_combo,
                        str(
                            flags.get("occupant_role")
                            or (spec.occupant_role if spec is not None else "unknown")
                            or "unknown"
                        ),
                    )
                    self._set_combo_text(
                        self.access_combo,
                        str(
                            flags.get("access_state")
                            or (spec.access_state if spec is not None else "unknown")
                            or "unknown"
                        ),
                    )
                    self.barrier_adjacent_checkbox.setChecked(
                        bool(flags.get("barrier_adjacent"))
                    )
                    zone_state = self._shape_zone_state(shape)
                else:
                    self.tags_edit.setText(self.default_tags_edit.text().strip())
                    self._set_combo_text(
                        self.zone_kind_combo,
                        self.default_zone_kind_combo.currentText().strip() or "custom",
                    )
                    self._set_combo_text(
                        self.phase_combo,
                        self.default_phase_combo.currentText().strip() or "custom",
                    )
                    self._set_combo_text(
                        self.occupant_combo,
                        self.default_occupant_combo.currentText().strip() or "unknown",
                    )
                    self._set_combo_text(
                        self.access_combo,
                        self.default_access_combo.currentText().strip() or "unknown",
                    )
                    self.barrier_adjacent_checkbox.setChecked(
                        self.default_barrier_checkbox.isChecked()
                    )
                    zone_state = "not_zone"
                self.selected_summary_label.setText(
                    f"Editing '{shape.label or '(unnamed)'}' | {self._zone_area_text(shape)} | "
                    f"state={zone_state} | kind={flags.get('zone_kind', 'custom')} | phase={flags.get('phase', 'custom')}"
                )
        finally:
            for widget in widgets:
                widget.blockSignals(False)
            self.barrier_adjacent_checkbox.blockSignals(False)
        self._update_action_states()
        self._update_metrics_preview()

    def _set_combo_text(self, combo: QtWidgets.QComboBox, value: str) -> None:
        text = str(value or "").strip()
        index = combo.findText(text)
        if index >= 0:
            combo.setCurrentIndex(index)
        else:
            combo.setEditText(text)

    def _apply_selected_zone_details(self) -> None:
        shape = self._selected_shape
        if shape is None:
            return
        label = self.zone_label_edit.text().strip()
        if not label:
            label = default_zone_label(
                self.zone_kind_combo.currentText().strip(), self._existing_labels()
            )
        extra_flags = dict(getattr(shape, "flags", {}) or {})
        if self.barrier_adjacent_checkbox.isChecked():
            extra_flags["barrier_adjacent"] = True
        else:
            extra_flags.pop("barrier_adjacent", None)
        shape.label = label
        shape.description = self.zone_description_edit.text().strip()
        shape.flags = normalize_zone_flags(
            shape,
            label=label,
            zone_kind=self.zone_kind_combo.currentText().strip() or "custom",
            phase=self.phase_combo.currentText().strip() or "custom",
            occupant_role=self.occupant_combo.currentText().strip() or "unknown",
            access_state=self.access_combo.currentText().strip() or "unknown",
            tags=self.tags_edit.text().strip(),
            extra_flags=extra_flags,
        )
        self._apply_shape_style(shape)
        self.canvas.storeShapes()
        self.canvas.update()
        self._refresh_shape_list(select_shape=shape)
        self._mark_dirty()
        self._set_status(f"Updated zone '{shape.label}'.")
        self._sync_selected_fields()
        self._publish_zone_defaults()

    def _classify_selected_shape_as_zone(self) -> None:
        shape = self._selected_shape
        if shape is None:
            return
        self._apply_selected_zone_details()
        self._set_status(f"'{shape.label}' is now set as a zone.")

    def _use_selected_as_defaults(self) -> None:
        shape = self._selected_shape
        if shape is None:
            return
        flags = dict(getattr(shape, "flags", {}) or {})
        self._set_combo_text(
            self.default_zone_kind_combo,
            str(flags.get("zone_kind") or "custom"),
        )
        self._set_combo_text(
            self.default_phase_combo,
            str(flags.get("phase") or "custom"),
        )
        self._set_combo_text(
            self.default_occupant_combo,
            str(flags.get("occupant_role") or "unknown"),
        )
        self._set_combo_text(
            self.default_access_combo,
            str(flags.get("access_state") or "unknown"),
        )
        self.default_tags_edit.setText(
            ", ".join(str(tag) for tag in flags.get("tags") or [])
        )
        self.default_barrier_checkbox.setChecked(bool(flags.get("barrier_adjacent")))
        self._publish_zone_defaults()
        self._set_status(
            f"Using '{shape.label or '(unnamed)'}' semantics for newly drawn zones."
        )

    def _refresh_shape_list(self, select_shape=None) -> None:
        current = select_shape or self._selected_shape
        all_shapes = list(self.canvas.shapes or [])
        inventory_shapes = [
            shape
            for shape in all_shapes
            if self._shape_is_inventory_zone_candidate(shape)
        ]
        filtered_shapes = [
            shape
            for shape in inventory_shapes
            if self._shape_matches_inventory_filter(shape)
        ]
        shapes = self._sort_inventory_shapes(filtered_shapes)
        self.shape_list.blockSignals(True)
        try:
            self.shape_list.clear()
            for shape in shapes:
                flags = getattr(shape, "flags", {}) or {}
                zone_state = self._shape_zone_state(shape)
                item = QtWidgets.QListWidgetItem(
                    f"{shape.label or '(unnamed)'}  |  {zone_state}  |  {flags.get('zone_kind', '-') if is_zone_shape(shape) else '-'}  |  {self._zone_area_text(shape)}"
                )
                item.setData(QtCore.Qt.UserRole, shape)
                if (
                    is_zone_shape(shape)
                    and getattr(shape, "line_color", None) is not None
                ):
                    color = QtGui.QColor(shape.line_color)
                    item.setBackground(color.lighter(165))
                self.shape_list.addItem(item)
            if current is not None:
                row = self._row_for_shape(current)
                if row is not None:
                    self.shape_list.setCurrentRow(row)
        finally:
            self.shape_list.blockSignals(False)
        if hasattr(self, "inventory_summary_label"):
            zone_count = sum(
                1
                for shape in inventory_shapes
                if self._shape_zone_state(shape) == "zone"
            )
            self.inventory_summary_label.setText(
                f"Showing {len(shapes)} of {len(inventory_shapes)} zone candidates ({zone_count} explicit zones)"
            )
        self._sync_selected_fields()
        self._update_stats()

    def _update_action_states(self) -> None:
        has_selection = self._selected_shape is not None
        for widget_name in (
            "classify_selected_button",
            "apply_selected_button",
            "use_as_defaults_button",
            "stroke_button",
            "duplicate_button",
            "delete_button",
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setEnabled(has_selection)

    # ------------------------------------------------------------------ #
    def _count_analysis_ready_zones(self) -> int:
        count = 0
        for shape in self.canvas.shapes:
            if not is_zone_shape(shape):
                continue
            flags = dict(getattr(shape, "flags", {}) or {})
            if all(
                str(flags.get(key) or "").strip()
                for key in ("zone_kind", "phase", "occupant_role", "access_state")
            ):
                count += 1
        return count

    def _update_stats(self) -> None:
        zone_shapes = [shape for shape in self.canvas.shapes if is_zone_shape(shape)]
        self.zone_count_value.setText(str(len(zone_shapes)))
        selected_label = (
            str(getattr(self._selected_shape, "label", "") or "").strip()
            if self._selected_shape is not None
            else ""
        )
        self.selected_zone_value.setText(selected_label or "None")
        self.metrics_ready_value.setText(str(self._count_analysis_ready_zones()))

    def _mark_dirty(self) -> None:
        self._dirty = True

    def _mark_dirty_and_refresh(self, *args, **kwargs) -> None:
        _ = args, kwargs
        self._mark_dirty()
        self._refresh_shape_list()

    # ------------------------------------------------------------------ #
    # Saving
    # ------------------------------------------------------------------ #
    def _default_save_path(self) -> Path | None:
        if self._zone_file_path is not None:
            return self._zone_file_path
        source = self._current_source() or self._current_canvas_image_path()
        return zone_file_for_source(source)

    def save_zone_file_as(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Zone JSON",
            str(self._default_save_path() or Path.home() / "zones.json"),
            "LabelMe JSON (*.json)",
        )
        if not filename:
            return
        self._zone_file_path = Path(filename)
        self._update_save_target_label()
        self.save_zone_file()

    def save_zone_file(self) -> bool:
        pixmap = self._current_canvas_pixmap()
        if pixmap is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Nothing to save",
                "Load a frame before saving zone annotations.",
            )
            return False
        if self._zone_file_path is None:
            self._zone_file_path = self._default_save_path()
        if self._zone_file_path is None:
            QtWidgets.QMessageBox.warning(
                self, "Missing path", "Choose a file name for the zone JSON."
            )
            return False

        self._zone_file_path.parent.mkdir(parents=True, exist_ok=True)
        zone_shapes = [shape for shape in self.canvas.shapes if is_zone_shape(shape)]
        if not zone_shapes:
            QtWidgets.QMessageBox.information(
                self,
                "No zones found",
                "There are no zone-tagged shapes on the current canvas yet.",
            )
            return False
        shapes = [
            shape_to_zone_payload(
                shape,
                label=getattr(shape, "label", "") or None,
                zone_kind=(getattr(shape, "flags", {}) or {}).get("zone_kind"),
                phase=(getattr(shape, "flags", {}) or {}).get("phase"),
                occupant_role=(getattr(shape, "flags", {}) or {}).get("occupant_role"),
                access_state=(getattr(shape, "flags", {}) or {}).get("access_state"),
                tags=(getattr(shape, "flags", {}) or {}).get("tags"),
                description=getattr(shape, "description", "") or "",
                extra_flags=getattr(shape, "flags", None),
            )
            for shape in zone_shapes
        ]
        try:
            lf = LabelFile()
            lf.save(
                str(self._zone_file_path),
                shapes,
                self._current_canvas_image_path() or self._current_source(),
                int(pixmap.height()),
                int(pixmap.width()),
                imageData=None,
                otherData={"zone_schema_version": 1},
                flags={},
                caption="zone manager export",
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(exc))
            return False

        self._dirty = False
        self._set_status(f"Saved {len(shapes)} zone(s) to {self._zone_file_path}.")
        return True

    # ------------------------------------------------------------------ #
    def _clear_zone_defaults(self) -> None:
        parent = getattr(self, "_owner_window", None) or self.parent()
        if parent is not None and hasattr(parent, "_zone_authoring_defaults"):
            setattr(parent, "_zone_authoring_defaults", None)
