from qtpy.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QRadioButton,
    QButtonGroup,
    QTableView,
    QHeaderView,
    QHBoxLayout,
    QMessageBox,
    QAbstractItemView,
    QDialogButtonBox,
    QGroupBox,
    QWidget,
)
from qtpy.QtCore import Slot, Signal, QTimer
from pathlib import Path
from typing import List, Optional, Dict

from annolid.jobs.tracking_jobs import TrackingSegment
from annolid.gui.segment_models import SegmentTableModel
from annolid.jobs.tracking_worker import TrackingWorker

from annolid.utils.logger import logger
from annolid.jobs.tracking_jobs import VideoProcessingJob, JobType


class SegmentEditorDialog(QDialog):
    """
    A dialog for defining, editing, and managing tracking segments
    for a single, active video.
    """

    tracking_initiated = Signal(
        TrackingWorker, Path
    )  # worker_instance, video_path_processed
    dialog_status_update = Signal(str, str)

    def __init__(
        self,
        active_video_path: Path,
        active_video_fps: float,
        active_video_total_frames: int,
        current_annolid_frame: int,
        initial_segments_data: Optional[List[Dict]] = None,
        annolid_config: Optional[Dict] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        # Store for worker
        self.annolid_config = annolid_config if annolid_config is not None else {}

        if (
            not active_video_path
            or active_video_fps <= 0
            or active_video_total_frames <= 0
        ):
            QTimer.singleShot(0, self.reject)
            logger.error("SegmentEditorDialog: Initialized with invalid video context.")
            return

        self.setWindowTitle(f"Segment Tracker - {active_video_path.name}")
        self.setMinimumWidth(700)  # Set a reasonable minimum width

        self._active_video_path = active_video_path
        self._active_video_fps = active_video_fps
        self._active_video_total_frames = active_video_total_frames
        # For "Use Current" and display
        self._current_annolid_frame_dynamic = current_annolid_frame

        # Store the segment being edited
        self._editing_segment_obj: Optional[TrackingSegment] = None
        # Store its row for model update
        self._editing_segment_row: Optional[int] = None

        self.internal_status_label = QLabel("")

        self._segment_model = SegmentTableModel(self)
        self._segment_model.set_active_video_context(
            self._active_video_path, self._active_video_fps
        )
        if initial_segments_data:
            self._segment_model.load_segments_from_data(initial_segments_data)

        self._setup_ui()
        self._update_video_info_display()
        self.update_live_annolid_frame_info(
            current_annolid_frame
        )  # Initial display of live frame

        self._reset_form_to_defaults()  # Set initial form state

    def _setup_ui(self):
        dialog_layout = QVBoxLayout(self)

        # --- Static Info Group ---
        info_group = QGroupBox("Active Video Information")
        info_layout = QFormLayout(info_group)
        self.video_name_label = QLabel()
        self.fps_label = QLabel()
        self.total_frames_label = QLabel()
        self.live_annolid_frame_label = QLabel()  # This will be updated dynamically
        info_layout.addRow("Video:", self.video_name_label)
        info_layout.addRow("FPS:", self.fps_label)
        info_layout.addRow("Total Frames:", self.total_frames_label)
        info_layout.addRow("Annolid Frame (Live):", self.live_annolid_frame_label)
        dialog_layout.addWidget(info_group)

        # --- Segment Definition Group ---
        define_group = QGroupBox("Define or Edit Segment")
        form_layout = QFormLayout(define_group)
        form_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        # Annotation Frame
        ann_frame_widget = QWidget()
        ann_frame_layout = QHBoxLayout(ann_frame_widget)
        ann_frame_layout.setContentsMargins(0, 0, 0, 0)
        self.annotation_frame_input = QSpinBox()
        self.annotation_frame_input.setRange(0, self._active_video_total_frames - 1)
        self.use_current_frame_button = QPushButton("Use Annolid Frame")
        self.use_current_frame_button.setToolTip(
            "Use the current frame shown in Annolid's main window."
        )
        self.use_current_frame_button.clicked.connect(
            self._on_use_current_annolid_frame
        )
        ann_frame_layout.addWidget(self.annotation_frame_input)
        ann_frame_layout.addWidget(self.use_current_frame_button)
        form_layout.addRow("Annotation Frame:", ann_frame_widget)

        self.annotation_status_label = QLabel("<i>Check annotation status...</i>")
        self.annotation_status_label.setStyleSheet("font-style: italic;")
        # No label for status
        form_layout.addRow("", self.annotation_status_label)
        self.annotation_frame_input.valueChanged.connect(
            self._on_annotation_frame_input_changed
        )

        # Track Until (End Condition)
        self.end_condition_group = QButtonGroup(self)
        self.duration_radio = QRadioButton("Duration:")
        self.end_time_radio = QRadioButton(
            "End Time (HH:MM:SS):"
        )  # Will add validator later
        self.end_frame_radio = QRadioButton("End Frame #:")
        self.end_condition_group.addButton(self.duration_radio)
        self.end_condition_group.addButton(self.end_time_radio)
        self.end_condition_group.addButton(self.end_frame_radio)
        self.duration_radio.setChecked(True)

        self.duration_spinbox = QSpinBox()
        self.duration_spinbox.setSuffix(" secs")
        self.duration_spinbox.setMinimum(1)
        self.duration_spinbox.setMaximum(3600 * 10)
        self.duration_spinbox.setValue(60)  # Max 10 hours

        self.end_time_input = QLineEdit("00:01:00")  # Example default
        # TODO: Add QRegExpValidator for HH:MM:SS format

        self.end_frame_spinbox = QSpinBox()
        self.end_frame_spinbox.setRange(0, self._active_video_total_frames - 1)

        form_layout.addRow(self.duration_radio, self.duration_spinbox)
        form_layout.addRow(self.end_time_radio, self.end_time_input)
        form_layout.addRow(self.end_frame_radio, self.end_frame_spinbox)
        self._toggle_end_condition_inputs_enabled_state()
        self.end_condition_group.buttonToggled.connect(
            self._toggle_end_condition_inputs_enabled_state
        )  # Use buttonToggled

        self.add_update_button = QPushButton("Add Segment")
        self.add_update_button.clicked.connect(self._on_add_or_update_segment)
        form_layout.addRow("", self.add_update_button)  # Span across columns
        dialog_layout.addWidget(define_group)

        # --- Defined Segments Table Group ---
        table_group = QGroupBox("Defined Segments")
        table_group_layout = QVBoxLayout(table_group)
        self.segments_table = QTableView()
        self.segments_table.setModel(self._segment_model)
        self.segments_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.segments_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.segments_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )  # Stretch last section later if needed
        self.segments_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.segments_table.setAlternatingRowColors(True)
        self.segments_table.selectionModel().selectionChanged.connect(
            self._on_table_selection_changed
        )
        table_group_layout.addWidget(self.segments_table)

        table_actions_layout = QHBoxLayout()
        self.edit_segment_button = QPushButton("Edit Selected")
        self.delete_segment_button = QPushButton("Delete Selected")
        self.edit_segment_button.clicked.connect(
            self._on_edit_selected_segment_button_clicked
        )
        self.delete_segment_button.clicked.connect(
            self._on_delete_selected_segment_button_clicked
        )
        self.edit_segment_button.setEnabled(False)
        self.delete_segment_button.setEnabled(False)
        table_actions_layout.addWidget(self.edit_segment_button)
        table_actions_layout.addWidget(self.delete_segment_button)
        table_actions_layout.addStretch()
        table_group_layout.addLayout(table_actions_layout)
        dialog_layout.addWidget(table_group)

        # --- Main Action Buttons ---
        main_actions_layout = QHBoxLayout()
        self.clear_all_segments_button = QPushButton("Clear All Segments")
        self.track_all_segments_button = QPushButton("Track All Defined Segments")
        self.clear_all_segments_button.clicked.connect(
            self._on_clear_all_segments_button_clicked
        )
        self.track_all_segments_button.clicked.connect(
            self._initiate_tracking_from_dialog
        )

        main_actions_layout.addWidget(self.clear_all_segments_button)
        main_actions_layout.addStretch()
        main_actions_layout.addWidget(self.track_all_segments_button)
        dialog_layout.addLayout(main_actions_layout)

        # --- Standard Dialog Buttons (OK, Cancel) ---
        self.dialog_buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.dialog_buttons.accepted.connect(self.accept)
        self.dialog_buttons.rejected.connect(self.reject)
        dialog_layout.addWidget(self.dialog_buttons)

    @Slot()
    # New handler for "Track All..." button
    def _initiate_tracking_from_dialog(self):
        if not self._active_video_path:
            QMessageBox.warning(self, "No Video", "No active video to track.")
            return

        all_segments = self._segment_model.get_all_segments()
        if not all_segments:
            QMessageBox.information(
                self, "No Segments", "No segments are defined to track."
            )
            return

        valid_segments_to_track = [s for s in all_segments if s.is_annotation_valid()]
        invalid_count = len(all_segments) - len(valid_segments_to_track)

        if invalid_count > 0:
            msg = (
                f"{invalid_count} segment(s) have missing/invalid annotation JSONs and will be skipped. "
                "Proceed with valid segments?"
            )
            reply = QMessageBox.warning(
                self,
                "Invalid Segments",
                msg,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return

        if not valid_segments_to_track:
            QMessageBox.warning(
                self,
                "No Valid Segments",
                "No valid segments (with annotations) to track.",
            )
            return

        # Check if AnnolidWindow (parent) indicates a worker is already busy
        # This requires AnnolidWindow to have a public method like `is_tracking_busy()`
        annolid_main_window = self.parent()
        if (
            hasattr(annolid_main_window, "is_tracking_busy")
            and annolid_main_window.is_tracking_busy()
        ):
            QMessageBox.warning(
                self,
                "Tracking Busy",
                "Annolid is already processing a tracking job. Please wait.",
            )
            return

        segment_data_dicts = [s.to_dict() for s in valid_segments_to_track]
        job = VideoProcessingJob(
            video_path=self._active_video_path,
            job_type=JobType.VIDEO_SEGMENTS,
            segments_data=segment_data_dicts,
            fps=self._active_video_fps,
            video_specific_config=self.annolid_config.get(
                "advanced_tracking_params", {}
            ),  # From Annolid
        )
        jobs_list = [job]

        logger.info(
            f"SegmentEditorDialog: Initiating TrackingWorker for segmented job on {self._active_video_path.name}"
        )
        self.internal_status_label.setText(
            f"Initiating tracking for {self._active_video_path.name}..."
        )

        # Get relevant global config from Annolid's config for the worker
        # This assumes annolid_config is structured to provide these easily.
        # Example: worker_global_config = self.annolid_config.get('tracking_worker_global_config', self.annolid_config)
        worker_global_config = dict(
            self.annolid_config.get("tracking_parameters", self.annolid_config) or {}
        )
        # Prefer the Optical Flow Settings menu values when available.
        of_manager = getattr(annolid_main_window, "optical_flow_manager", None)
        if of_manager is not None:
            from annolid.motion.optical_flow import optical_flow_settings_from

            worker_global_config.update(optical_flow_settings_from(of_manager))

        try:
            # Dialog creates the worker
            # The parent of the worker can be self (the dialog) or None.
            # If parent is self, worker might be auto-deleted when dialog is deleted.
            # If parent is None, AnnolidWindow should manage its lifecycle if it holds a reference.
            # For now, let AnnolidWindow manage it after receiving the instance.
            worker = TrackingWorker(
                processing_jobs=jobs_list,
                global_config=worker_global_config,
                parent=None,  # Worker will be re-parented or managed by AnnolidWindow
            )

            # Emit signal to AnnolidWindow with the worker instance and video path
            self.tracking_initiated.emit(worker, self._active_video_path)

            # Dialog can listen to its own worker's basic signals for internal status
            worker.finished.connect(self._on_dialog_worker_finished)
            worker.error.connect(self._on_dialog_worker_error)

            worker.start()  # Start the worker
            self.internal_status_label.setText(
                f"Tracking started for {self._active_video_path.name}. See main window for progress."
            )
            self.track_all_segments_button.setEnabled(
                False
            )  # Disable while this worker runs
            # Optionally, disable other parts of the dialog here

            # self.accept() # Optionally close dialog once tracking is successfully initiated
        except Exception as e:
            logger.error(
                f"Failed to create or start TrackingWorker from dialog: {e}",
                exc_info=True,
            )
            QMessageBox.critical(
                self, "Worker Error", f"Could not start tracking worker: {e}"
            )
            self.internal_status_label.setText(
                f"<font color='red'>Error starting worker: {e}</font>"
            )

    @Slot(str)
    def _on_dialog_worker_finished(self, message: str):
        self.internal_status_label.setText(
            f"Tracking for {self._active_video_path.name} finished: {message.split('.')[0]}."
        )
        self.track_all_segments_button.setEnabled(True)
        # Make sure to disconnect if the worker is managed by AnnolidWindow and might be deleted
        sender_worker = self.sender()
        if sender_worker:
            try:
                sender_worker.finished.disconnect(self._on_dialog_worker_finished)
                sender_worker.error.disconnect(self._on_dialog_worker_error)
            except (TypeError, RuntimeError):
                pass

    @Slot(str)
    def _on_dialog_worker_error(self, error_message: str):
        self.internal_status_label.setText(
            f"<font color='red'>Tracking error for {self._active_video_path.name}: {error_message}</font>"
        )
        self.track_all_segments_button.setEnabled(True)
        sender_worker = self.sender()
        if sender_worker:
            try:
                sender_worker.finished.disconnect(self._on_dialog_worker_finished)
                sender_worker.error.disconnect(self._on_dialog_worker_error)
            except (TypeError, RuntimeError):
                pass

    # Remains the same
    def get_defined_segments(self) -> List[TrackingSegment]:
        return self._segment_model.get_all_segments()

    def _update_video_info_display(self):
        self.video_name_label.setText(self._active_video_path.name)
        self.fps_label.setText(f"{self._active_video_fps:.2f}")
        self.total_frames_label.setText(str(self._active_video_total_frames))

    @Slot(int)  # Updated by AnnolidWindow's live_annolid_frame_updated signal
    def update_live_annolid_frame_info(self, frame_number: int):
        self._current_annolid_frame_dynamic = frame_number
        time_str = (
            TrackingSegment._format_seconds(frame_number / self._active_video_fps)
            if self._active_video_fps > 0
            else "N/A"
        )
        self.live_annolid_frame_label.setText(f"{frame_number} ({time_str})")

    @Slot()
    def _toggle_end_condition_inputs_enabled_state(self):
        self.duration_spinbox.setEnabled(self.duration_radio.isChecked())
        self.end_time_input.setEnabled(self.end_time_radio.isChecked())
        self.end_frame_spinbox.setEnabled(self.end_frame_radio.isChecked())

    @Slot()
    def _on_use_current_annolid_frame(self):
        if self._current_annolid_frame_dynamic is not None:
            self.annotation_frame_input.setValue(self._current_annolid_frame_dynamic)
            # ValueChanged signal of spinbox will call _validate_annotation_json

    @Slot(int)
    def _on_annotation_frame_input_changed(self, frame_number: int):
        self._validate_annotation_json(frame_number)

    def _validate_annotation_json(self, frame_number: int) -> bool:
        # Create a temporary TrackingSegment just for path construction and validation
        temp_segment = TrackingSegment(
            self._active_video_path, self._active_video_fps, frame_number, 0, 0
        )
        is_valid = temp_segment.is_annotation_valid()
        if is_valid:
            self.annotation_status_label.setText(
                f"<font color='darkgreen'>Annotation for frame {frame_number} found.</font>"
            )
            self.annotation_status_label.setStyleSheet("font-style: normal;")
        else:
            self.annotation_status_label.setText(
                f"< font color='red' > Annotation for frame {frame_number} NOT found! Please save it first. < /font >"
            )
            self.annotation_status_label.setStyleSheet(
                "font-style: normal; font-weight: bold;"
            )
        return is_valid

    def _reset_form_to_defaults(self):
        # Sets the form to a default state for adding a new segment
        self.annotation_frame_input.setValue(self._current_annolid_frame_dynamic or 0)
        self._on_annotation_frame_input_changed(
            self.annotation_frame_input.value()
        )  # Validate

        self.duration_radio.setChecked(True)
        self.duration_spinbox.setValue(60)  # Default 60 seconds
        self.end_time_input.setText(
            TrackingSegment._format_seconds(
                (self.annotation_frame_input.value() / self._active_video_fps) + 60
                if self._active_video_fps > 0
                else 60
            )
        )
        self.end_frame_spinbox.setValue(
            min(
                self.annotation_frame_input.value() + int(60 * self._active_video_fps)
                if self._active_video_fps > 0
                else self.annotation_frame_input.value() + 1800,
                self._active_video_total_frames - 1,
            )
        )
        self._toggle_end_condition_inputs_enabled_state()

        self.add_update_button.setText("Add Segment")
        self._editing_segment_obj = None
        self._editing_segment_row = None
        self.segments_table.clearSelection()

    @Slot()
    def _on_add_or_update_segment(self):
        annotated_frame = self.annotation_frame_input.value()
        if not self._validate_annotation_json(annotated_frame):
            QMessageBox.warning(
                self,
                "Missing Annotation",
                f"Annotation JSON for frame {annotated_frame} is missing. "
                "Please go to this frame in Annolid, annotate, and save (Ctrl+S) before defining a segment with it.",
            )
            return

        # Default: tracking starts from annotated frame
        segment_start_frame = annotated_frame
        segment_end_frame = -1

        try:
            if self.duration_radio.isChecked():
                duration_sec = self.duration_spinbox.value()
                if duration_sec <= 0:
                    raise ValueError("Duration must be positive.")
                num_frames_for_duration = int(duration_sec * self._active_video_fps)
                segment_end_frame = min(
                    segment_start_frame + num_frames_for_duration - 1,
                    self._active_video_total_frames - 1,
                )
            elif self.end_time_radio.isChecked():
                time_str = self.end_time_input.text()
                parts = list(map(int, time_str.split(":")))
                if len(parts) != 3:
                    raise ValueError("End time must be HH:MM:SS format.")
                h, m, s = parts
                if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59):
                    raise ValueError("Invalid time values.")
                total_seconds_for_end_time = h * 3600 + m * 60 + s
                segment_end_frame = min(
                    int(total_seconds_for_end_time * self._active_video_fps),
                    self._active_video_total_frames - 1,
                )
            elif self.end_frame_radio.isChecked():
                segment_end_frame = self.end_frame_spinbox.value()
                if not (0 <= segment_end_frame < self._active_video_total_frames):
                    raise ValueError(
                        f"End frame must be between 0 and {self._active_video_total_frames - 1}."
                    )
            else:
                raise ValueError("No end condition selected for the segment.")

            if segment_end_frame < segment_start_frame:
                raise ValueError(
                    "Segment end cannot be before its annotated/start frame."
                )

        except ValueError as e:
            QMessageBox.critical(self, "Input Error", str(e))
            return
        except Exception as e:  # Catch any other parsing errors
            QMessageBox.critical(
                self, "Input Error", f"Could not parse end condition: {e}"
            )
            return

        new_segment_obj = TrackingSegment(
            video_path=self._active_video_path,
            fps=self._active_video_fps,
            annotated_frame=annotated_frame,
            # For now, assume start tracking from annotated
            segment_start_frame=segment_start_frame,
            segment_end_frame=segment_end_frame,
        )

        if (
            self._editing_segment_obj is not None
            and self._editing_segment_row is not None
        ):
            # Update existing segment in the model
            new_segment_obj.unique_id = (
                self._editing_segment_obj.unique_id
            )  # Preserve ID on update
            self._segment_model.update_segment_at_row(
                self._editing_segment_row, new_segment_obj
            )
        else:
            self._segment_model.add_segment(new_segment_obj)

        self._reset_form_to_defaults()  # Reset form for next addition or after update

    @Slot()
    def _on_table_selection_changed(self):
        selected_indexes = self.segments_table.selectionModel().selectedRows()
        can_edit_delete = bool(selected_indexes)
        self.edit_segment_button.setEnabled(can_edit_delete)
        self.delete_segment_button.setEnabled(can_edit_delete)

        if (
            not can_edit_delete and self._editing_segment_obj is not None
        ):  # Selection cleared while in edit mode
            self._reset_form_to_defaults()

    @Slot()
    def _on_edit_selected_segment_button_clicked(self):
        selected_indexes = self.segments_table.selectionModel().selectedRows()
        if not selected_indexes:
            return

        row_to_edit = selected_indexes[0].row()
        segment_to_edit = self._segment_model.get_segment_at_row(row_to_edit)
        if segment_to_edit:
            self._editing_segment_obj = segment_to_edit
            self._editing_segment_row = row_to_edit

            self.annotation_frame_input.setValue(segment_to_edit.annotated_frame)
            self._on_annotation_frame_input_changed(
                segment_to_edit.annotated_frame
            )  # Validate

            # For simplicity, always set to duration when editing.
            # More complex logic could try to restore the original input method.
            self.duration_radio.setChecked(True)
            self.duration_spinbox.setValue(
                max(1, int(segment_to_edit.duration_sec))
            )  # Ensure at least 1 sec
            self._toggle_end_condition_inputs_enabled_state()

            self.add_update_button.setText("Update Segment")
            self.annotation_frame_input.setFocus()
        else:  # Should not happen if selection is valid
            self._reset_form_to_defaults()

    @Slot()
    def _on_delete_selected_segment_button_clicked(self):
        selected_indexes = self.segments_table.selectionModel().selectedRows()
        if not selected_indexes:
            return

        row_to_delete = selected_indexes[0].row()
        segment_to_delete = self._segment_model.get_segment_at_row(row_to_delete)
        if not segment_to_delete:
            return

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Delete segment: Ann.Frame {segment_to_delete.annotated_frame}, "
            f"Track {segment_to_delete.segment_start_frame}-{segment_to_delete.segment_end_frame}?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._segment_model.remove_segment_at_row(row_to_delete)
            if (
                self._editing_segment_row == row_to_delete
            ):  # If the deleted was being edited
                self._reset_form_to_defaults()

    @Slot()
    def _on_clear_all_segments_button_clicked(self):
        if self._segment_model.rowCount() == 0:
            QMessageBox.information(
                self, "No Segments", "There are no segments to clear."
            )
            return
        reply = QMessageBox.question(
            self,
            "Confirm Clear All",
            "Are you sure you want to clear all defined segments for this video?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._segment_model.clear_all_segments()
            self._reset_form_to_defaults()

    @Slot()
    def _on_track_all_defined_segments_button_clicked(self):
        all_segments = self._segment_model.get_all_segments()
        if not all_segments:
            QMessageBox.information(
                self, "No Segments", "No segments are defined to track."
            )
            return

        valid_segments_to_track = [s for s in all_segments if s.is_annotation_valid()]
        invalid_count = len(all_segments) - len(valid_segments_to_track)

        if invalid_count > 0:
            msg = (
                f"{invalid_count} segment(s) have missing/invalid annotation JSONs and will be skipped. "
                "Do you want to proceed with tracking the valid segments?"
            )
            reply = QMessageBox.warning(
                self,
                "Invalid Segments Found",
                msg,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.No:
                return

        if not valid_segments_to_track:
            QMessageBox.warning(
                self,
                "No Valid Segments",
                "There are no valid segments (with annotations) to track.",
            )
            return

        # Emit signal with the active video path and list of valid TrackingSegment objects
        self.track_segments_for_this_video_requested.emit(
            self._active_video_path, valid_segments_to_track
        )
        # Optionally, close the dialog after emitting, or let AnnolidWindow handle it
        # self.accept() # Example: close on successful emission
