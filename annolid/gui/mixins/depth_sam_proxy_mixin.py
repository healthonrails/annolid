from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from qtpy import QtCore, QtWidgets


class DepthSamProxyMixin:
    """Thin proxy methods delegating SAM3D/depth actions to managers."""

    def _default_sam3_obj_id_from_selection(self) -> int:
        default_obj_id = 1
        canvas = getattr(self, "canvas", None)
        selected = (
            getattr(canvas, "selectedShapes", None) if canvas is not None else None
        )
        if selected:
            shape = selected[0]
            gid = getattr(shape, "group_id", None)
            try:
                gid_int = int(gid)
                if gid_int > 0:
                    return gid_int
            except Exception:
                pass
            try:
                label = str(getattr(shape, "label", "") or "").strip()
                if label:
                    session = getattr(
                        getattr(self, "sam3_manager", None), "sam3_session", None
                    )
                    id_to_labels = (
                        getattr(session, "id_to_labels", {})
                        if session is not None
                        else {}
                    )
                    for obj_id, mapped_label in dict(id_to_labels or {}).items():
                        if str(mapped_label) == label:
                            mapped_id = int(obj_id)
                            if mapped_id > 0:
                                return mapped_id
            except Exception:
                pass
        return default_obj_id

    def close_sam3_session(self) -> None:
        manager = getattr(self, "sam3_manager", None)
        if manager is None:
            return
        if bool(getattr(self, "_prediction_session_is_active", lambda: False)()):
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("SAM3 Session Busy"),
                self.tr("Stop active prediction before closing SAM3 session."),
            )
            return
        had_session = bool(getattr(manager, "sam3_session", None))
        manager.close_session()
        if had_session:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("SAM3 Session"),
                self.tr("SAM3 session has been closed."),
            )
        else:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("SAM3 Session"),
                self.tr("No active SAM3 session to close."),
            )

    def reset_sam3_session(self) -> None:
        manager = getattr(self, "sam3_manager", None)
        if manager is None:
            return
        if bool(getattr(self, "_prediction_session_is_active", lambda: False)()):
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("SAM3 Session Busy"),
                self.tr("Stop active prediction before resetting SAM3 session."),
            )
            return
        ok = manager.reset_active_session()
        if ok:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("SAM3 Session"),
                self.tr("SAM3 session state has been reset."),
            )
        else:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("SAM3 Session"),
                self.tr("No active SAM3 session to reset."),
            )

    def remove_sam3_object(self) -> None:
        manager = getattr(self, "sam3_manager", None)
        if manager is None:
            return
        if bool(getattr(self, "_prediction_session_is_active", lambda: False)()):
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("SAM3 Session Busy"),
                self.tr("Stop active prediction before removing a SAM3 object."),
            )
            return
        obj_id, accepted = QtWidgets.QInputDialog.getInt(
            self,
            self.tr("Remove SAM3 Object"),
            self.tr("Object ID to remove:"),
            int(self._default_sam3_obj_id_from_selection()),
            1,
            10_000,
            1,
        )
        if not accepted:
            return
        frame_idx = max(int(getattr(self, "frame_number", 0) or 0), 0)
        ok = manager.remove_object_from_active_session(
            obj_id=obj_id, frame_idx=frame_idx
        )
        if not ok:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("SAM3 Session"),
                self.tr("No active SAM3 session/object available to remove."),
            )
            return
        try:
            self.loadPredictShapes(frame_idx, getattr(self, "filename", None))
            self.canvas.update()
        except Exception:
            pass
        QtWidgets.QMessageBox.information(
            self,
            self.tr("SAM3 Session"),
            self.tr("Removed SAM3 object ID %s at frame %s.")
            % (int(obj_id), int(frame_idx)),
        )

    def _handle_sam3d_finished(self, result, *, worker_thread: QtCore.QThread):
        if getattr(self, "sam3d_manager", None) is not None:
            self.sam3d_manager._handle_sam3d_finished(
                result, worker_thread=worker_thread
            )

    def configure_sam3d_settings(self):
        manager = getattr(self, "sam3d_manager", None)
        if manager is None and hasattr(self, "ensure_sam3d_manager"):
            try:
                manager = self.ensure_sam3d_manager()
            except Exception:
                manager = None
        if manager is not None:
            manager.configure_sam3d_settings()

    def run_video_depth_anything(self):
        manager = getattr(self, "depth_manager", None)
        if manager is None and hasattr(self, "ensure_depth_manager"):
            try:
                manager = self.ensure_depth_manager()
            except Exception:
                manager = None
        if manager is not None:
            manager.run_video_depth_anything()

    def configure_video_depth_settings(self):
        manager = getattr(self, "depth_manager", None)
        if manager is None and hasattr(self, "ensure_depth_manager"):
            try:
                manager = self.ensure_depth_manager()
            except Exception:
                manager = None
        if manager is not None:
            manager.configure_video_depth_settings()

    def _handle_depth_preview(self, payload: object) -> None:
        if getattr(self, "depth_manager", None) is not None:
            self.depth_manager._handle_depth_preview(payload)

    def _set_depth_preview_frame(self, frame_index: int) -> None:
        if getattr(self, "depth_manager", None) is not None:
            self.depth_manager._set_depth_preview_frame(frame_index)

    def _depth_ndjson_path(self) -> Optional[Path]:
        if getattr(self, "depth_manager", None) is not None:
            return self.depth_manager._depth_ndjson_path()
        return None

    def _load_depth_ndjson_records(self) -> None:
        if getattr(self, "depth_manager", None) is not None:
            self.depth_manager.load_depth_ndjson_records()

    def _build_depth_overlay(
        self, frame_rgb: np.ndarray, depth_map: np.ndarray
    ) -> np.ndarray:
        if getattr(self, "depth_manager", None) is not None:
            return self.depth_manager._build_depth_overlay(frame_rgb, depth_map)
        return depth_map

    def _restore_canvas_frame(self) -> None:
        if getattr(self, "depth_manager", None) is not None:
            self.depth_manager._restore_canvas_frame()

    def _load_depth_overlay_from_json(
        self, json_path: Path, frame_rgb: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        return None

    def _load_depth_overlay_from_record(
        self, record: Dict[str, object], frame_rgb: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        if getattr(self, "depth_manager", None) is not None:
            return self.depth_manager._load_depth_overlay_from_record(record, frame_rgb)
        return None

    def _current_frame_rgb(self) -> Optional[np.ndarray]:
        if getattr(self, "depth_manager", None) is not None:
            return self.depth_manager._current_frame_rgb()
        return None

    def _update_depth_overlay_for_frame(
        self, frame_number: int, frame_rgb: Optional[np.ndarray] = None
    ) -> None:
        if getattr(self, "depth_manager", None) is not None:
            self.depth_manager.update_overlay_for_frame(frame_number, frame_rgb)

    def _handle_video_depth_finished(
        self,
        result,
        *,
        output_dir: str,
        worker_thread: QtCore.QThread,
    ) -> None:
        if getattr(self, "depth_manager", None) is not None:
            self.depth_manager._handle_video_depth_finished(
                result, output_dir=output_dir, worker_thread=worker_thread
            )
