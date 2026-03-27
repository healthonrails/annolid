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
    shape_to_zone_payload,
    zone_file_for_source,
    zone_kind_palette,
    zone_payload_to_shape,
)
from annolid.postprocessing.zone_schema import load_zone_shapes


class ZonePanelWidget(QtWidgets.QWidget):
    """Generic zone authoring panel for frame-backed zone shapes."""

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
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        header = QtWidgets.QLabel(
            "Draw zones on the current frame, tag them with semantics, recolor them, and export a dedicated zone JSON."
        )
        header.setWordWrap(True)
        header.setStyleSheet("font-weight: 600; color: #2d5b88;")
        root.addWidget(header)

        preset_box = QtWidgets.QGroupBox("Arena Presets")
        preset_layout = QtWidgets.QVBoxLayout(preset_box)
        preset_layout.setSpacing(6)
        preset_row = QtWidgets.QHBoxLayout()
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
        self.preset_combo.currentIndexChanged.connect(
            self._update_preset_description_label
        )
        self.generate_preset_button = QtWidgets.QPushButton("Generate Preset")
        self.generate_preset_button.clicked.connect(self.generate_selected_preset)
        self.generate_preset_button.setEnabled(self._preset_available)
        preset_row.addWidget(self.preset_combo, 1)
        preset_row.addWidget(self.generate_preset_button)
        self.preset_description_label = QtWidgets.QLabel("")
        self.preset_description_label.setWordWrap(True)
        self.preset_description_label.setStyleSheet("color: #5d6d7e;")
        preset_layout.addLayout(preset_row)
        preset_layout.addWidget(self.preset_description_label)
        if not self._preset_available:
            self.preset_description_label.setText(
                "No preset generators are available. Draw zones manually on the live canvas."
            )
        root.addWidget(preset_box)

        self.source_label = QtWidgets.QLabel("")
        self.source_label.setWordWrap(True)
        self.save_target_label = QtWidgets.QLabel("")
        self.save_target_label.setWordWrap(True)

        source_box = QtWidgets.QGroupBox("Source")
        source_layout = QtWidgets.QFormLayout(source_box)
        source_layout.addRow("Frame", self.source_label)
        source_layout.addRow("Zone JSON", self.save_target_label)

        self.refresh_current_button = QtWidgets.QPushButton("Refresh Current Canvas")
        self.refresh_current_button.clicked.connect(self.refresh_from_current_canvas)
        self.load_zone_button = QtWidgets.QPushButton("Load Zone JSON")
        self.load_zone_button.clicked.connect(self.open_zone_file)
        self.save_zone_button = QtWidgets.QPushButton("Save Zone JSON")
        self.save_zone_button.clicked.connect(self.save_zone_file_as)

        source_buttons = QtWidgets.QHBoxLayout()
        source_buttons.addWidget(self.refresh_current_button)
        source_buttons.addWidget(self.load_zone_button)
        source_buttons.addWidget(self.save_zone_button)
        source_layout.addRow("Actions", source_buttons)
        root.addWidget(source_box)

        self.shape_list = QtWidgets.QListWidget(self)
        self.shape_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.shape_list.currentRowChanged.connect(self._on_shape_row_changed)
        self.shape_list.itemSelectionChanged.connect(self._on_shape_selection_changed)

        self.zone_kind_combo = QtWidgets.QComboBox()
        self.zone_kind_combo.setEditable(True)
        self.zone_kind_combo.addItems(
            ["chamber", "doorway", "barrier_edge", "interaction_zone", "custom"]
        )

        self.phase_combo = QtWidgets.QComboBox()
        self.phase_combo.setEditable(True)
        self.phase_combo.addItems(["phase_1", "phase_2", "custom"])

        self.occupant_combo = QtWidgets.QComboBox()
        self.occupant_combo.setEditable(True)
        self.occupant_combo.addItems(["rover", "stim", "neutral", "unknown"])

        self.access_combo = QtWidgets.QComboBox()
        self.access_combo.setEditable(True)
        self.access_combo.addItems(["open", "blocked", "tethered", "unknown"])

        self.tags_edit = QtWidgets.QLineEdit()
        self.tags_edit.setPlaceholderText("comma,separated,tags")

        self.stroke_button = QtWidgets.QPushButton("Recolor")
        self.stroke_button.clicked.connect(self._recolor_selected_shape)
        self.delete_button = QtWidgets.QPushButton("Delete Selected")
        self.delete_button.clicked.connect(self._delete_selected_shape)

        self.canvas_status = QtWidgets.QLabel(
            "Use the main Annolid toolbar or canvas context menu to switch draw/edit mode."
        )
        self.canvas_status.setWordWrap(True)
        self.canvas_status.setStyleSheet("color: #5d6d7e;")

        props_box = QtWidgets.QGroupBox("Zone Defaults")
        props_form = QtWidgets.QFormLayout(props_box)
        props_form.addRow("Zone kind", self.zone_kind_combo)
        props_form.addRow("Phase", self.phase_combo)
        props_form.addRow("Occupant role", self.occupant_combo)
        props_form.addRow("Access state", self.access_combo)
        props_form.addRow("Tags", self.tags_edit)

        editor_buttons = QtWidgets.QHBoxLayout()
        editor_buttons.addWidget(self.stroke_button)
        editor_buttons.addWidget(self.delete_button)
        props_form.addRow("Actions", editor_buttons)

        self.shape_list_label = QtWidgets.QLabel("Zones")
        self.shape_list_label.setStyleSheet("font-weight: 600;")

        root.addWidget(self.canvas_status)
        root.addWidget(self.shape_list_label)
        root.addWidget(self.shape_list, 1)
        root.addWidget(props_box)

        bottom_row = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        self.close_button = QtWidgets.QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        bottom_row.addWidget(self.status_label, 1)
        bottom_row.addWidget(self.close_button)
        root.addLayout(bottom_row)
        self._update_preset_description_label()
        self.zone_kind_combo.currentTextChanged.connect(
            lambda *_: self._publish_zone_defaults()
        )
        self.phase_combo.currentTextChanged.connect(
            lambda *_: self._publish_zone_defaults()
        )
        self.occupant_combo.currentTextChanged.connect(
            lambda *_: self._publish_zone_defaults()
        )
        self.access_combo.currentTextChanged.connect(
            lambda *_: self._publish_zone_defaults()
        )
        self.tags_edit.editingFinished.connect(self._publish_zone_defaults)

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
        defaults = build_zone_popup_defaults(
            label=default_zone_label(
                self.zone_kind_combo.currentText().strip(), self._existing_labels()
            ),
            zone_kind=self.zone_kind_combo.currentText().strip() or "custom",
            phase=self.phase_combo.currentText().strip() or "custom",
            occupant_role=self.occupant_combo.currentText().strip() or "unknown",
            access_state=self.access_combo.currentText().strip() or "unknown",
            tags=self.tags_edit.text().strip(),
            description="",
        )
        setattr(parent, "_zone_authoring_defaults", defaults)

    def _update_preset_description_label(self, *args) -> None:
        _ = args
        index = self.preset_combo.currentIndex()
        preset = (
            self._preset_definitions[index]
            if 0 <= index < len(self._preset_definitions)
            else {}
        )
        description = str(preset.get("description") or "").strip()
        self.preset_description_label.setText(description)

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

    def refresh_from_current_canvas(self) -> None:
        pixmap = self._current_canvas_pixmap()
        if pixmap is None:
            self.canvas_status.setText(
                "Load a video or image in the main canvas before drawing zones."
            )
            self._update_source_label()
            self._update_save_target_label()
            self.shape_list.clear()
            return
        if self._zone_file_path is None:
            self._zone_file_path = zone_file_for_source(self._current_source())
        self._update_source_label()
        self._update_save_target_label()
        self.canvas.setEditing(True)
        self._refresh_shape_list()
        self._set_status("Canvas ready for zone editing.")
        self._publish_zone_defaults()

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
        self._refresh_shape_list()
        self._set_status(f"Loaded {len(shapes)} zone(s) from {filename}.")
        self._dirty = False
        self._zone_file_path = Path(filename)
        self._update_save_target_label()

    # ------------------------------------------------------------------ #
    # Shape lifecycle
    # ------------------------------------------------------------------ #
    def _existing_labels(self) -> list[str]:
        return [str(getattr(shape, "label", "") or "") for shape in self.canvas.shapes]

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
        if not getattr(shape, "label", "").strip():
            shape.label = default_zone_label(
                self.zone_kind_combo.currentText().strip(), self._existing_labels()
            )
        self._apply_shape_style(shape)
        self.canvas.storeShapes()
        self._refresh_shape_list(select_shape=shape)
        self._mark_dirty()
        self._set_status(f"Added zone '{shape.label}'.")
        self._publish_zone_defaults()

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

    def _on_shape_selection_changed(self) -> None:
        if self._syncing_selection:
            return
        item = self.shape_list.currentItem()
        shape = item.data(QtCore.Qt.UserRole) if item is not None else None
        if shape is None:
            self.canvas.selectShapes([])
            return
        self._selected_shape = shape
        self._syncing_selection = True
        try:
            self.canvas.selectShapes([shape])
            self._set_status(f"Selected '{shape.label}'.")
        finally:
            self._syncing_selection = False

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

    def _row_for_shape(self, shape) -> int | None:
        for row in range(self.shape_list.count()):
            item = self.shape_list.item(row)
            if item is not None and item.data(QtCore.Qt.UserRole) is shape:
                return row
        return None

    def _refresh_shape_list(self, select_shape=None) -> None:
        current = select_shape or self._selected_shape
        shapes = [shape for shape in self.canvas.shapes if is_zone_shape(shape)]
        self.shape_list.blockSignals(True)
        try:
            self.shape_list.clear()
            for shape in shapes:
                item = QtWidgets.QListWidgetItem(
                    f"{shape.label or '(unnamed)'}  |  {getattr(shape, 'flags', {}).get('zone_kind', 'custom')}"
                )
                item.setData(QtCore.Qt.UserRole, shape)
                if getattr(shape, "line_color", None) is not None:
                    color = QtGui.QColor(shape.line_color)
                    item.setBackground(color.darker(135))
                    item.setForeground(QtGui.QColor(255, 255, 255))
                self.shape_list.addItem(item)
            if current is not None:
                row = self._row_for_shape(current)
                if row is not None:
                    self.shape_list.setCurrentRow(row)
        finally:
            self.shape_list.blockSignals(False)

    # ------------------------------------------------------------------ #
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
        source = self._source_path or self._source_image_path
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
