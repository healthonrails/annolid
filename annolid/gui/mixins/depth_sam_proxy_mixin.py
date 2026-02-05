from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from qtpy import QtCore


class DepthSamProxyMixin:
    """Thin proxy methods delegating SAM3D/depth actions to managers."""

    def _handle_sam3d_finished(self, result, *, worker_thread: QtCore.QThread):
        if getattr(self, "sam3d_manager", None) is not None:
            self.sam3d_manager._handle_sam3d_finished(
                result, worker_thread=worker_thread
            )

    def configure_sam3d_settings(self):
        if getattr(self, "sam3d_manager", None) is not None:
            self.sam3d_manager.configure_sam3d_settings()

    def run_video_depth_anything(self):
        if getattr(self, "depth_manager", None) is not None:
            self.depth_manager.run_video_depth_anything()

    def configure_video_depth_settings(self):
        if getattr(self, "depth_manager", None) is not None:
            self.depth_manager.configure_video_depth_settings()

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
