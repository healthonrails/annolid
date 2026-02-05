"""
Video Controller for Annolid GUI Application.

Handles UI interactions and coordinates video operations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from qtpy import QtCore

from ..interfaces.services import IVideoService
from ..services import VideoService

logger = logging.getLogger(__name__)


class VideoController(QtCore.QObject):
    """
    Controller for video-related UI operations.

    Coordinates between the UI and video service, handling
    user interactions and business logic orchestration.
    """

    # Signals
    video_loaded = QtCore.Signal(dict)  # Emitted when video is loaded
    video_error = QtCore.Signal(str)  # Emitted on video errors
    frame_extracted = QtCore.Signal(
        object, int
    )  # Emitted when frame is extracted (frame_data, frame_index)
    video_metadata_updated = QtCore.Signal(dict)  # Emitted when metadata is updated
    progress_updated = QtCore.Signal(int, str)  # Progress updates

    def __init__(
        self,
        video_service: Optional[IVideoService] = None,
        parent: Optional[QtCore.QObject] = None,
    ):
        """
        Initialize the video controller.

        Args:
            video_service: Video service instance
            parent: Parent QObject
        """
        super().__init__(parent)
        self._video_service = video_service or VideoService()
        self._current_video_path: Optional[Path] = None
        self._video_metadata: Optional[Dict[str, Any]] = None
        self._is_video_loaded = False

    def load_video(self, video_path: Union[str, Path]) -> bool:
        """
        Load a video file.

        Args:
            video_path: Path to the video file

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            video_path = Path(video_path)

            # Validate file exists
            if not video_path.exists():
                self.video_error.emit(f"Video file does not exist: {video_path}")
                return False

            # Validate file is a video
            if not self._is_video_file(video_path):
                self.video_error.emit(
                    f"File is not a supported video format: {video_path}"
                )
                return False

            # Load video using service
            success, message = self._video_service.load_video(video_path)
            if not success:
                self.video_error.emit(f"Failed to load video: {message}")
                return False

            # Get video metadata
            metadata = self._video_service.get_video_metadata(video_path)
            if not metadata:
                self.video_error.emit("Failed to retrieve video metadata")
                return False

            self._current_video_path = video_path
            self._video_metadata = metadata
            self._is_video_loaded = True

            self.video_loaded.emit(metadata)
            self.video_metadata_updated.emit(metadata)

            logger.info(f"Video loaded: {video_path}")
            return True

        except Exception as e:
            error_msg = f"Failed to load video: {str(e)}"
            self.video_error.emit(error_msg)
            logger.error(error_msg)
            return False

    def extract_frame_at_index(self, frame_index: int) -> Optional[Any]:
        """
        Extract a specific frame from the current video.

        Args:
            frame_index: Index of the frame to extract

        Returns:
            Frame data or None on failure
        """
        if not self._is_video_loaded or not self._current_video_path:
            self.video_error.emit("No video loaded")
            return None

        try:
            # Validate frame index
            if not self._is_valid_frame_index(frame_index):
                self.video_error.emit(f"Invalid frame index: {frame_index}")
                return None

            # Extract frame using service
            frame_data = self._video_service.extract_frame_at_index(
                self._current_video_path, frame_index
            )

            if frame_data is not None:
                self.frame_extracted.emit(frame_data, frame_index)
                logger.info(f"Frame extracted: {frame_index}")
                return frame_data
            else:
                self.video_error.emit(f"Failed to extract frame: {frame_index}")
                return None

        except Exception as e:
            error_msg = f"Failed to extract frame: {str(e)}"
            self.video_error.emit(error_msg)
            logger.error(error_msg)
            return None

    def extract_frame_at_time(self, time_seconds: float) -> Optional[Any]:
        """
        Extract a frame at a specific time.

        Args:
            time_seconds: Time in seconds

        Returns:
            Frame data or None on failure
        """
        if not self._video_metadata:
            self.video_error.emit("No video metadata available")
            return None

        try:
            fps = self._video_metadata.get("fps", 30.0)
            frame_index = int(time_seconds * fps)

            return self.extract_frame_at_index(frame_index)

        except Exception as e:
            error_msg = f"Failed to extract frame at time {time_seconds}: {str(e)}"
            self.video_error.emit(error_msg)
            logger.error(error_msg)
            return None

    def get_video_thumbnail(self, frame_index: Optional[int] = None) -> Optional[Any]:
        """
        Get a thumbnail for the current video.

        Args:
            frame_index: Optional frame index for thumbnail (default: middle frame)

        Returns:
            Thumbnail image data or None on failure
        """
        if not self._is_video_loaded or not self._current_video_path:
            self.video_error.emit("No video loaded")
            return None

        try:
            # Use middle frame if no index specified
            if frame_index is None and self._video_metadata:
                total_frames = self._video_metadata.get("frame_count", 0)
                frame_index = total_frames // 2
            elif frame_index is None:
                frame_index = 0

            # Extract thumbnail using service
            thumbnail = self._video_service.get_video_thumbnail(
                self._current_video_path, frame_index
            )

            if thumbnail is not None:
                logger.info(f"Video thumbnail generated at frame: {frame_index}")
                return thumbnail
            else:
                self.video_error.emit("Failed to generate video thumbnail")
                return None

        except Exception as e:
            error_msg = f"Failed to get video thumbnail: {str(e)}"
            self.video_error.emit(error_msg)
            logger.error(error_msg)
            return None

    def get_video_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the current video.

        Returns:
            Video information dictionary or None
        """
        if not self._video_metadata:
            return None

        try:
            info = self._video_metadata.copy()

            # Add additional computed information
            if "duration" not in info and "fps" in info and "frame_count" in info:
                info["duration"] = info["frame_count"] / info["fps"]

            if "file_size" in info:
                # Convert bytes to MB
                info["file_size_mb"] = info["file_size"] / (1024 * 1024)

            return info

        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            return None

    def validate_video_file(
        self, video_path: Union[str, Path]
    ) -> Tuple[bool, List[str]]:
        """
        Validate a video file.

        Args:
            video_path: Path to the video file

        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            video_path = Path(video_path)
            errors = []

            # Check if file exists
            if not video_path.exists():
                errors.append("File does not exist")
                return False, errors

            # Check if it's a video file
            if not self._is_video_file(video_path):
                errors.append("File is not a supported video format")
                return False, errors

            # Try to get metadata to validate file integrity
            metadata = self._video_service.get_video_metadata(video_path)
            if not metadata:
                errors.append("Unable to read video metadata")
                return False, errors

            # Check for basic required metadata
            required_fields = ["width", "height", "fps", "frame_count"]
            for field in required_fields:
                if field not in metadata:
                    errors.append(f"Missing required metadata: {field}")

            return len(errors) == 0, errors

        except Exception as e:
            return False, [f"Validation error: {str(e)}"]

    def get_frame_range(
        self,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
        step: int = 1,
    ) -> List[int]:
        """
        Get a range of frame indices.

        Args:
            start_frame: Starting frame index
            end_frame: Ending frame index
            step: Step size

        Returns:
            List of frame indices
        """
        if not self._video_metadata:
            return []

        try:
            total_frames = self._video_metadata.get("frame_count", 0)

            if start_frame is None:
                start_frame = 0
            if end_frame is None:
                end_frame = total_frames - 1

            # Validate ranges
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(0, min(end_frame, total_frames - 1))

            if start_frame > end_frame:
                return []

            return list(range(start_frame, end_frame + 1, step))

        except Exception as e:
            logger.error(f"Failed to get frame range: {e}")
            return []

    def convert_time_to_frame(self, time_seconds: float) -> int:
        """
        Convert time in seconds to frame index.

        Args:
            time_seconds: Time in seconds

        Returns:
            Frame index
        """
        if not self._video_metadata:
            return 0

        try:
            fps = self._video_metadata.get("fps", 30.0)
            frame_index = int(time_seconds * fps)
            total_frames = self._video_metadata.get("frame_count", 0)

            return max(0, min(frame_index, total_frames - 1))

        except Exception as e:
            logger.error(f"Failed to convert time to frame: {e}")
            return 0

    def convert_frame_to_time(self, frame_index: int) -> float:
        """
        Convert frame index to time in seconds.

        Args:
            frame_index: Frame index

        Returns:
            Time in seconds
        """
        if not self._video_metadata:
            return 0.0

        try:
            fps = self._video_metadata.get("fps", 30.0)
            return frame_index / fps

        except Exception as e:
            logger.error(f"Failed to convert frame to time: {e}")
            return 0.0

    def is_video_loaded(self) -> bool:
        """
        Check if a video is currently loaded.

        Returns:
            True if video is loaded, False otherwise
        """
        return self._is_video_loaded

    def get_current_video_path(self) -> Optional[Path]:
        """
        Get the current video file path.

        Returns:
            Current video path or None
        """
        return self._current_video_path

    def unload_video(self) -> None:
        """Unload the current video."""
        self._current_video_path = None
        self._video_metadata = None
        self._is_video_loaded = False

        logger.info("Video unloaded")

    def _is_video_file(self, file_path: Path) -> bool:
        """Check if file is a supported video format."""
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"}
        return file_path.suffix.lower() in video_extensions

    def _is_valid_frame_index(self, frame_index: int) -> bool:
        """Check if frame index is valid for current video."""
        if not self._video_metadata:
            return False

        total_frames = self._video_metadata.get("frame_count", 0)
        return 0 <= frame_index < total_frames
