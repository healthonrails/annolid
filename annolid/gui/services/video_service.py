"""
Video Service for Annolid GUI Application.

Handles video file operations, metadata extraction, and frame processing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    cv2 = None
    HAS_CV2 = False

import numpy as np

logger = logging.getLogger(__name__)


class VideoService:
    """
    Domain service for video operations.

    Provides business logic for loading videos, extracting frames,
    and managing video metadata.
    """

    def __init__(self):
        """Initialize the video service."""
        self._supported_formats = {
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".wmv",
            ".flv",
            ".webm",
            ".m4v",
        }
        self._video_capture: Optional[cv2.VideoCapture] = None
        self._current_video_path: Optional[Path] = None

    def load_video(self, video_path: Union[str, Path]) -> Tuple[bool, str]:
        """
        Load a video file.

        Args:
            video_path: Path to the video file

        Returns:
            Tuple of (success, message)
        """
        if not HAS_CV2:
            return False, "OpenCV not available"

        try:
            video_path = Path(video_path)

            # Validate file exists
            if not video_path.exists():
                return False, f"Video file does not exist: {video_path}"

            # Validate file extension
            if video_path.suffix.lower() not in self._supported_formats:
                return False, f"Unsupported video format: {video_path.suffix}"

            # Close any existing video
            self._close_video()

            # Open new video
            self._video_capture = cv2.VideoCapture(str(video_path))

            if not self._video_capture.isOpened():
                return False, f"Failed to open video file: {video_path}"

            self._current_video_path = video_path

            # Test reading first frame
            ret, frame = self._video_capture.read()
            if not ret:
                self._close_video()
                return False, "Failed to read video frames"

            # Reset to beginning
            self._video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

            return True, f"Video loaded successfully: {video_path.name}"

        except Exception as e:
            self._close_video()
            logger.error(f"Failed to load video: {e}")
            return False, f"Failed to load video: {str(e)}"

    def get_video_metadata(
        self, video_path: Union[str, Path]
    ) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a video file.

        Args:
            video_path: Path to the video file

        Returns:
            Video metadata dictionary or None on failure
        """
        if not HAS_CV2:
            return None

        try:
            # If this is the currently loaded video, use the capture object
            if (
                self._current_video_path
                and Path(video_path) == self._current_video_path
                and self._video_capture
                and self._video_capture.isOpened()
            ):
                return self._get_metadata_from_capture(self._video_capture, video_path)

            # Otherwise, open temporarily
            temp_capture = cv2.VideoCapture(str(video_path))
            if not temp_capture.isOpened():
                return None

            try:
                metadata = self._get_metadata_from_capture(temp_capture, video_path)
                return metadata
            finally:
                temp_capture.release()

        except Exception as e:
            logger.error(f"Failed to get video metadata: {e}")
            return None

    def extract_frame_at_index(
        self, video_path: Union[str, Path], frame_index: int
    ) -> Optional[np.ndarray]:
        """
        Extract a specific frame from a video.

        Args:
            video_path: Path to the video file
            frame_index: Index of the frame to extract

        Returns:
            Frame as numpy array or None on failure
        """
        if not HAS_CV2:
            return None

        try:
            # Use current capture if it's the same video
            if (
                self._current_video_path
                and Path(video_path) == self._current_video_path
                and self._video_capture
                and self._video_capture.isOpened()
            ):
                return self._extract_frame_from_capture(
                    self._video_capture, frame_index
                )

            # Otherwise, open temporarily
            temp_capture = cv2.VideoCapture(str(video_path))
            if not temp_capture.isOpened():
                return None

            try:
                frame = self._extract_frame_from_capture(temp_capture, frame_index)
                return frame
            finally:
                temp_capture.release()

        except Exception as e:
            logger.error(f"Failed to extract frame: {e}")
            return None

    def get_video_thumbnail(
        self, video_path: Union[str, Path], frame_index: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Get a thumbnail image for a video.

        Args:
            video_path: Path to the video file
            frame_index: Optional frame index for thumbnail

        Returns:
            Thumbnail image as numpy array or None on failure
        """
        try:
            # Use middle frame if no index specified
            if frame_index is None:
                metadata = self.get_video_metadata(video_path)
                if metadata:
                    frame_index = metadata.get("frame_count", 0) // 2
                else:
                    frame_index = 0

            frame = self.extract_frame_at_index(video_path, frame_index)
            if frame is None:
                return None

            # Resize to thumbnail size
            height, width = frame.shape[:2]
            max_size = 200
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)

            thumbnail = cv2.resize(
                frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR
            )
            return thumbnail

        except Exception as e:
            logger.error(f"Failed to get video thumbnail: {e}")
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
        errors = []

        try:
            video_path = Path(video_path)

            # Check if file exists
            if not video_path.exists():
                errors.append("File does not exist")
                return False, errors

            # Check file extension
            if video_path.suffix.lower() not in self._supported_formats:
                errors.append(f"Unsupported video format: {video_path.suffix}")

            if not HAS_CV2:
                errors.append("OpenCV not available for video validation")
                return False, errors

            # Try to open and read metadata
            temp_capture = cv2.VideoCapture(str(video_path))
            if not temp_capture.isOpened():
                errors.append("Cannot open video file")
                return False, errors

            try:
                # Check if we can read frames
                ret, frame = temp_capture.read()
                if not ret:
                    errors.append("Cannot read video frames")

                # Check basic properties
                width = temp_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = temp_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
                fps = temp_capture.get(cv2.CAP_PROP_FPS)

                if width <= 0 or height <= 0:
                    errors.append("Invalid video dimensions")

                if fps <= 0:
                    errors.append("Invalid frame rate")

            finally:
                temp_capture.release()

            return len(errors) == 0, errors

        except Exception as e:
            return False, [f"Validation error: {str(e)}"]

    def get_video_duration(self, video_path: Union[str, Path]) -> Optional[float]:
        """
        Get the duration of a video in seconds.

        Args:
            video_path: Path to the video file

        Returns:
            Duration in seconds or None on failure
        """
        try:
            metadata = self.get_video_metadata(video_path)
            if metadata:
                fps = metadata.get("fps", 0)
                frame_count = metadata.get("frame_count", 0)
                if fps > 0:
                    return frame_count / fps
            return None

        except Exception as e:
            logger.error(f"Failed to get video duration: {e}")
            return None

    def get_video_frame_count(self, video_path: Union[str, Path]) -> Optional[int]:
        """
        Get the total number of frames in a video.

        Args:
            video_path: Path to the video file

        Returns:
            Frame count or None on failure
        """
        try:
            metadata = self.get_video_metadata(video_path)
            if metadata:
                return metadata.get("frame_count", 0)
            return None

        except Exception as e:
            logger.error(f"Failed to get video frame count: {e}")
            return None

    def convert_video_format(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        output_format: str = "mp4",
        **kwargs,
    ) -> Tuple[bool, str]:
        """
        Convert video to a different format.

        Args:
            input_path: Input video path
            output_path: Output video path
            output_format: Output format (mp4, avi, etc.)
            **kwargs: Additional conversion parameters

        Returns:
            Tuple of (success, message)
        """
        try:
            input_path = Path(input_path)
            output_path = Path(output_path)

            # This is a placeholder - real implementation would use ffmpeg or similar
            # For now, just copy the file if same format
            if input_path.suffix.lower() == f".{output_format.lower()}":
                import shutil

                shutil.copy2(input_path, output_path)
                return True, f"Video copied to {output_path}"
            else:
                return (
                    False,
                    f"Format conversion not implemented: {input_path.suffix} -> .{output_format}",
                )

        except Exception as e:
            logger.error(f"Failed to convert video format: {e}")
            return False, f"Failed to convert video: {str(e)}"

    def extract_frames_range(
        self,
        video_path: Union[str, Path],
        start_frame: int,
        end_frame: int,
        step: int = 1,
    ) -> List[np.ndarray]:
        """
        Extract a range of frames from a video.

        Args:
            video_path: Path to the video file
            start_frame: Starting frame index
            end_frame: Ending frame index
            step: Step size between frames

        Returns:
            List of frame arrays
        """
        frames = []

        try:
            for frame_idx in range(start_frame, end_frame + 1, step):
                frame = self.extract_frame_at_index(video_path, frame_idx)
                if frame is not None:
                    frames.append(frame)
                else:
                    break  # Stop if we can't read a frame

        except Exception as e:
            logger.error(f"Failed to extract frame range: {e}")

        return frames

    def _get_metadata_from_capture(
        self, capture: cv2.VideoCapture, video_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """Get metadata from an open video capture."""
        metadata = {
            "path": str(video_path),
            "filename": Path(video_path).name,
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": capture.get(cv2.CAP_PROP_FPS),
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fourcc": int(capture.get(cv2.CAP_PROP_FOURCC)),
            "format": self._fourcc_to_format(capture.get(cv2.CAP_PROP_FOURCC)),
        }

        # Calculate duration
        if metadata["fps"] > 0:
            metadata["duration"] = metadata["frame_count"] / metadata["fps"]
        else:
            metadata["duration"] = 0

        # Get file size
        try:
            metadata["file_size"] = Path(video_path).stat().st_size
        except OSError:
            metadata["file_size"] = 0

        return metadata

    def _extract_frame_from_capture(
        self, capture: cv2.VideoCapture, frame_index: int
    ) -> Optional[np.ndarray]:
        """Extract a frame from an open video capture."""
        try:
            # Set frame position
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

            # Read frame
            ret, frame = capture.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame_rgb
            else:
                return None

        except Exception as e:
            logger.error(f"Failed to extract frame from capture: {e}")
            return None

    def _fourcc_to_format(self, fourcc: float) -> str:
        """Convert FOURCC code to format string."""
        try:
            # Convert float to int, then to bytes
            fourcc_int = int(fourcc)
            fourcc_bytes = fourcc_int.to_bytes(4, byteorder="little")
            format_str = fourcc_bytes.decode("ascii", errors="ignore")
            return format_str if format_str else "unknown"
        except (ValueError, OSError, OverflowError):
            return "unknown"

    def _close_video(self) -> None:
        """Close the current video capture."""
        if self._video_capture:
            self._video_capture.release()
            self._video_capture = None
        self._current_video_path = None

    def __del__(self):
        """Cleanup on destruction."""
        self._close_video()
