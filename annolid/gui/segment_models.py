import os
from qtpy.QtGui import QColor
from qtpy.QtCore import QAbstractTableModel, Qt, Signal, QModelIndex, QObject
from typing import List, Any, Optional, Dict
from pathlib import Path
from annolid.utils.logger import logger
from annolid.jobs.tracking_jobs import TrackingSegment


class SegmentTableModel(QAbstractTableModel):
    """
    A QAbstractTableModel to manage and display TrackingSegment objects
    for a QTableView in the SegmentEditorDialog.
    """

    # Columns for the table view
    COL_ANNOTATED_FRAME = 0
    COL_START_TIME = 1
    COL_END_TIME = 2
    COL_DURATION_SEC = 3
    COL_ANNOTATION_VALID = 4
    COLUMN_COUNT = 5  # Total number of columns

    model_data_changed = Signal()

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._segments: List[TrackingSegment] = []
        self._headers = [
            "Ann. Frame",
            "Start Time",
            "End Time",
            "Duration (s)",
            "Annotation Valid",
        ]
        self._active_video_path: Optional[Path] = None
        self._active_video_fps: Optional[float] = None

    def set_active_video_context(
        self, video_path: Optional[Path], fps: Optional[float]
    ):
        """
        Sets the context of the video for which segments are being managed.
        Clears existing segments if the video context changes.
        """
        self.beginResetModel()  # Signals that the entire model is changing
        if video_path != self._active_video_path:
            self._segments.clear()  # Clear segments if video is different
        self._active_video_path = video_path
        self._active_video_fps = fps
        self.endResetModel()
        self.model_data_changed.emit()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():  # Should not have children for a flat list model
            return 0
        return len(self._segments)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return self.COLUMN_COUNT

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole
    ) -> Any:
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            if 0 <= section < len(self._headers):
                return self._headers[section]
        return super().headerData(section, orientation, role)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid() or not (0 <= index.row() < len(self._segments)):
            return None

        segment = self._segments[index.row()]
        column = index.column()

        if role == Qt.DisplayRole:
            if column == self.COL_ANNOTATED_FRAME:
                return str(segment.annotated_frame)
            if column == self.COL_START_TIME:
                return segment.start_time_str()
            if column == self.COL_END_TIME:
                return segment.end_time_str()
            if column == self.COL_DURATION_SEC:
                return f"{segment.duration_sec:.2f}"
            if column == self.COL_ANNOTATION_VALID:
                return "Yes" if segment.is_annotation_valid() else "No"

        elif role == Qt.UserRole:  # To retrieve the full Segment object
            return segment

        elif role == Qt.ForegroundRole:
            if column == self.COL_ANNOTATION_VALID:
                return (
                    QColor(Qt.darkGreen)
                    if segment.is_annotation_valid()
                    else QColor(Qt.red)
                )

        elif role == Qt.ToolTipRole:
            if (
                column == self.COL_ANNOTATION_VALID
                and not segment.is_annotation_valid()
            ):
                return f"Annotation JSON not found: {segment.annotation_json_path.name}"
            return f"Segment tracking from frame {segment.segment_start_frame} to {segment.segment_end_frame}"

        return None

    # --- Methods for modifying the model data ---

    def add_segment(self, segment: TrackingSegment) -> bool:
        """Adds a new segment to the model."""
        if not isinstance(segment, TrackingSegment):
            logger.error(
                "SegmentTableModel: Attempted to add non-TrackingSegment object."
            )
            return False
        if (
            segment.video_path != self._active_video_path
            or segment.fps != self._active_video_fps
        ):
            logger.error(
                "SegmentTableModel: Segment video/fps mismatch with model context."
            )
            # This check ensures segments added are for the current video context.
            # return False # Or auto-update segment's path/fps if that's desired behavior

        # Optional: Add validation for overlapping segments here if needed
        # for existing_seg in self._segments:
        #     if max(segment.segment_start_frame, existing_seg.segment_start_frame) <= \
        #        min(segment.segment_end_frame, existing_seg.segment_end_frame):
        #         logger.warning("SegmentTableModel: New segment overlaps with an existing one.")
        #         # return False # Or allow, depending on desired behavior

        row_to_insert = len(self._segments)
        self.beginInsertRows(QModelIndex(), row_to_insert, row_to_insert)
        self._segments.append(segment)
        # Keep segments sorted, e.g., by start frame
        self._segments.sort(key=lambda s: (s.segment_start_frame, s.annotated_frame))
        self.endInsertRows()
        self.model_data_changed.emit()
        return True

    def update_segment_at_row(
        self, row_index: int, updated_segment: TrackingSegment
    ) -> bool:
        """Updates an existing segment at a given row index."""
        if 0 <= row_index < len(self._segments):
            if not isinstance(updated_segment, TrackingSegment):
                return False
            if (
                updated_segment.video_path != self._active_video_path
                or updated_segment.fps != self._active_video_fps
            ):
                logger.error("SegmentTableModel: Updated segment video/fps mismatch.")
                return False

            self._segments[row_index] = updated_segment
            # Re-sort if necessary (if start_frame could change on update)
            self._segments.sort(
                key=lambda s: (s.segment_start_frame, s.annotated_frame)
            )
            # The row_index might change after sorting. Emit layoutChanged for simplicity,
            # or find new index and emit dataChanged for specific rows.
            self.layoutChanged.emit()  # Simpler than finding new index after sort
            # self.dataChanged.emit(self.index(row_index, 0), self.index(row_index, self.COLUMN_COUNT - 1))
            self.model_data_changed.emit()
            return True
        return False

    def remove_segment_at_row(self, row_index: int) -> bool:
        """Removes a segment at a given row index."""
        if 0 <= row_index < len(self._segments):
            self.beginRemoveRows(QModelIndex(), row_index, row_index)
            del self._segments[row_index]
            self.endRemoveRows()
            self.model_data_changed.emit()
            return True
        return False

    def get_segment_at_row(self, row_index: int) -> Optional[TrackingSegment]:
        """Retrieves the TrackingSegment object at a given row."""
        if 0 <= row_index < len(self._segments):
            return self._segments[row_index]
        return None

    def get_all_segments(self) -> List[TrackingSegment]:
        """Returns a copy of all TrackingSegment objects currently in the model."""
        return list(self._segments)

    def clear_all_segments(self):
        """Removes all segments from the model."""
        if not self._segments:
            return
        self.beginRemoveRows(QModelIndex(), 0, len(self._segments) - 1)
        self._segments.clear()
        self.endRemoveRows()
        self.model_data_changed.emit()

    def load_segments_from_data(self, segment_data_list: List[Dict]):
        """
        Clears existing segments and loads new ones from a list of dictionaries.
        Assumes _active_video_path and _active_video_fps are already set.
        """
        if self._active_video_path is None or self._active_video_fps is None:
            logger.error(
                "SegmentTableModel: Cannot load segments, active video context not set."
            )
            return

        self.beginResetModel()
        self._segments.clear()
        for data_dict in segment_data_list:
            try:
                segment = TrackingSegment(
                    video_path=self._active_video_path,  # Use current context
                    fps=self._active_video_fps,  # Use current context
                    annotated_frame=data_dict["annotated_frame"],
                    segment_start_frame=data_dict["segment_start_frame"],
                    segment_end_frame=data_dict["segment_end_frame"],
                    # unique_id can be loaded if present in data_dict
                    unique_id=data_dict.get("unique_id", os.urandom(4).hex()),
                )
                self._segments.append(segment)
            except KeyError as e:
                logger.error(f"Missing key {e} in segment data dict: {data_dict}")
            except Exception as e:
                logger.error(
                    f"Error creating TrackingSegment from dict: {data_dict}, error: {e}"
                )

        self._segments.sort(key=lambda s: (s.segment_start_frame, s.annotated_frame))
        self.endResetModel()
        self.model_data_changed.emit()
