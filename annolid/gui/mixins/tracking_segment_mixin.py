from __future__ import annotations

import json
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

from annolid.jobs.tracking_jobs import TrackingSegment
from annolid.utils.logger import logger


class TrackingSegmentMixin:
    """Tracking segment persistence and tracking UI helpers."""

    def _save_segments_for_active_video(self):
        if not self.video_file or not hasattr(self, "_current_video_defined_segments"):
            return
        segments_as_dicts = [s.to_dict() for s in self._current_video_defined_segments]
        sidecar_path = Path(self.video_file).with_suffix(
            Path(self.video_file).suffix + ".segments.json"
        )
        try:
            with open(sidecar_path, "w") as f:
                json.dump(segments_as_dicts, f, indent=2)
            logger.info(f"Saved {len(segments_as_dicts)} segments to {sidecar_path}")
        except Exception as e:
            logger.error(f"Failed to save segments to {sidecar_path}: {e}")

    def _load_segments_for_active_video(self):
        self._current_video_defined_segments = []
        if not self.video_file or not self.fps:
            return

        sidecar_path = Path(self.video_file).with_suffix(
            Path(self.video_file).suffix + ".segments.json"
        )
        if sidecar_path.exists():
            try:
                with open(sidecar_path, "r") as f:
                    segment_dicts = json.load(f)
                loaded_segments = []
                for s_dict in segment_dicts:
                    try:
                        s_dict["video_path"] = str(self.video_file)
                        s_dict["fps"] = self.fps
                        loaded_segments.append(TrackingSegment.from_dict(s_dict))
                    except Exception as e:
                        logger.error(
                            f"Error creating TrackingSegment from dict {s_dict}: {e}"
                        )
                self._current_video_defined_segments = loaded_segments
                logger.info(
                    f"Loaded {len(self._current_video_defined_segments)} segments from {sidecar_path}"
                )
            except Exception as e:
                logger.error(f"Failed to load segments from {sidecar_path}: {e}")
        if self.caption_widget is not None:
            self.caption_widget.set_video_segments(self._current_video_defined_segments)

    def _get_tracking_device(self):
        if self.config.get("use_cpu_only", False) or torch is None:
            return "cpu" if torch is None else torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def set_tracking_ui_state(self, is_tracking: bool) -> None:
        self.open_segment_editor_action.setEnabled(
            not is_tracking and bool(self.video_file)
        )
        if hasattr(self.actions, "open"):
            self.actions.open.setEnabled(not is_tracking)
        if hasattr(self.actions, "openDir"):
            self.actions.openDir.setEnabled(not is_tracking)
        if hasattr(self.actions, "openVideo"):
            self.actions.openVideo.setEnabled(not is_tracking)
        if hasattr(self, "video_manager_widget") and hasattr(
            self.video_manager_widget, "track_all_button"
        ):
            self.video_manager_widget.track_all_button.setEnabled(not is_tracking)
        logger.info(
            "AnnolidWindow UI state for tracking: %s",
            "ACTIVE" if is_tracking else "IDLE",
        )
