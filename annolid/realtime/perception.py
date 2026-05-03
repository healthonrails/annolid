import argparse
import re
import json
import sys
import signal
import contextlib
import socket
import statistics
import struct
import time
import warnings
import asyncio
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Union,
    Tuple,
    Protocol,
)
from contextlib import asynccontextmanager

if TYPE_CHECKING:
    from annolid.realtime.mediapipe_engine import MediaPipeEngine

from collections import deque
from enum import Enum, auto
from functools import partial
from itertools import accumulate
from pathlib import Path
from urllib.parse import parse_qs, urlsplit, urlunsplit

import cv2
import numpy as np
import zmq
import zmq.asyncio
from tree_config.utils import (
    yaml_loads as orig_yaml_loads,
    get_yaml,
    yaml_dumps as orig_yaml_dumps,
)
from pycocotools import mask as maskUtils

from ultralytics.engine.results import Masks
from ultralytics import YOLO

from annolid.utils.logger import logger
from annolid.utils.log_paths import resolve_annolid_realtime_logs_root
from annolid.yolo import configure_ultralytics_cache, resolve_weight_path
from annolid.realtime.config import Config
# Late import to avoid dependency issues
# from annolid.realtime.mediapipe_engine import MediaPipeEngine


# --- Configuration and Validation ---


# --- Recording State Management ---
class RecordingState(Enum):
    """Server recording states."""

    UNKNOWN = auto()
    RECORDING = auto()
    STOPPED = auto()
    WAITING_FOR_START = auto()


class RecordingStateManager:
    """Manages server recording state and system behavior."""

    def __init__(self, config: Config):
        self.config = config
        self.state = RecordingState.UNKNOWN
        self.last_state_change = time.time()
        self.state_change_callbacks = []
        self._lock = asyncio.Lock()

    async def update_state(self, new_state: RecordingState):
        """Update recording state and notify callbacks."""
        async with self._lock:
            if self.state != new_state:
                old_state = self.state
                self.state = new_state
                self.last_state_change = time.time()

                logger.info(
                    f"Recording state changed: {old_state.name} -> {new_state.name}"
                )

                # Notify callbacks
                for callback in self.state_change_callbacks:
                    try:
                        await callback(old_state, new_state)
                    except Exception as e:
                        logger.error(f"State change callback error: {e}")

    def add_state_change_callback(self, callback):
        """Add callback for state changes."""
        self.state_change_callbacks.append(callback)

    def should_process_frames(self) -> bool:
        """Determine if frames should be processed based on recording state."""
        if not self.config.pause_on_recording_stop:
            return True

        # Always process when recording or state is unknown (fail-safe)
        should_process = self.state in (
            RecordingState.RECORDING,
            RecordingState.UNKNOWN,
        )
        return should_process

    def get_state_info(self) -> Dict[str, Any]:
        """Get current state information."""
        return {
            "state": self.state.name,
            "time_since_change": time.time() - self.last_state_change,
            "should_process": self.should_process_frames(),
        }


# --- Protocols for Better Type Safety ---


class FrameSource(Protocol):
    """Protocol for frame sources."""

    async def connect(self) -> bool: ...
    async def get_frame(self) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]: ...

    async def disconnect(self) -> None: ...


class DetectionResult:
    """Structured detection result with support for pre-encoded segmentation masks."""

    def __init__(
        self,
        behavior: str,
        confidence: float,
        bbox_normalized: List[float],
        timestamp: float,
        metadata: Dict[str, Any],
        bbox_pixels: Optional[List[float]] = None,
        mask_data: Optional[Dict[str, Any]] = None,
        keypoints_normalized: Optional[List[List[float]]] = None,
        keypoints_pixels: Optional[List[List[float]]] = None,
        keypoint_scores: Optional[List[float]] = None,
        keypoint_labels: Optional[List[str]] = None,
    ):
        self.behavior = behavior
        self.confidence = confidence
        self.bbox_normalized = bbox_normalized
        self.bbox_pixels = bbox_pixels
        self.timestamp = timestamp
        self.metadata = metadata
        self.mask_data = mask_data  # Stores the pre-encoded mask dictionary
        self.keypoints_normalized = keypoints_normalized
        self.keypoints_pixels = keypoints_pixels
        self.keypoint_scores = keypoint_scores
        self.keypoint_labels = keypoint_labels

    def to_dict(self) -> Dict[str, Any]:
        """Converts the result to a dictionary for JSON serialization."""
        result = {
            "behavior": self.behavior,
            "confidence": self.confidence,
            # Normalized coordinates in [0..1] (x1, y1, x2, y2) for easy transport.
            "bbox_normalized": self.bbox_normalized,
            # Pixel coordinates in the input frame space (x1, y1, x2, y2).
            "bbox_pixels": self.bbox_pixels,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "has_mask": self.mask_data is not None,
        }

        if self.mask_data:
            result["mask"] = self.mask_data

        result["has_keypoints"] = (
            self.keypoints_normalized is not None or self.keypoints_pixels is not None
        )

        # Backward compatibility: keep `keypoints` as normalized coordinates.
        if self.keypoints_normalized is not None:
            result["keypoints"] = self.keypoints_normalized
            result["keypoints_normalized"] = self.keypoints_normalized

        if self.keypoints_pixels is not None:
            result["keypoints_pixels"] = self.keypoints_pixels

        if self.keypoint_scores is not None:
            result["keypoint_scores"] = self.keypoint_scores

        if self.keypoint_labels is not None:
            result["keypoint_labels"] = self.keypoint_labels

        return result


# --- Network Protocol Handling ---


class NetworkProtocolHandler:
    """Handles network message encoding/decoding with improved error handling."""

    def __init__(self):
        self.yaml_loads = self._create_yaml_loads_fix()
        self.yaml_dumps = partial(orig_yaml_dumps, get_yaml_obj=get_yaml)

    def _create_yaml_loads_fix(self):
        """Creates a fixed version of yaml_loads for binary data."""

        def _yaml_loads_fixed(value):
            if (
                len(value) >= 12
                and value.startswith("!!binary |\n")
                and value[11] != " "
            ):
                value = value[:11] + " " + value[11:]
            return orig_yaml_loads(value, get_yaml_obj=get_yaml)

        return _yaml_loads_fixed

    def encode_msg(self, msg: str, value) -> bytes:
        """Encodes a message into bytes for network transmission."""
        try:
            bin_data = []
            if msg == "image":
                image, metadata = value
                bin_data = image.to_bytearray()
                data = self.yaml_dumps(
                    (
                        "image",
                        (
                            list(map(len, bin_data)),
                            image.get_pixel_format(),
                            image.get_size(),
                            image.get_linesizes(),
                            metadata,
                        ),
                    )
                )
                data = data.encode("utf8")
            else:
                data = self.yaml_dumps((msg, value))
                data = data.encode("utf8")

            header = struct.pack(">II", len(data), sum(map(len, bin_data)))
            return header + data + b"".join(bin_data)
        except Exception as e:
            logger.error(f"Failed to encode message '{msg}': {e}")
            raise

    def decode_data(self, msg_buff: bytes, msg_len: Tuple[int, int]):
        """Decodes buffer data with validation."""
        n, bin_n = msg_len
        if not (n + bin_n == len(msg_buff)):
            raise ValueError(
                f"Buffer length mismatch: expected {n + bin_n}, got {len(msg_buff)}"
            )

        try:
            data_str = msg_buff[:n].decode("utf8")
            msg, value = self.yaml_loads(data_str)

            if msg == "image":
                bin_data = msg_buff[n:]
                planes_sizes, pix_fmt, size, linesize, metadata = value
                starts = list(accumulate([0] + list(planes_sizes[:-1])))
                ends = accumulate(planes_sizes)
                planes = [bin_data[s:e] for s, e in zip(starts, ends)]
                value = (planes, pix_fmt, size, linesize, metadata)

            return msg, value
        except Exception as e:
            logger.error(f"Failed to decode message data: {e}")
            raise


# --- Enhanced Color Space Converter ---


def _load_ffpyplayer_pic():
    """Lazy-load ffpyplayer.pic to avoid FFmpeg dylib conflicts when unused."""
    try:
        from ffpyplayer.pic import Image as _Image, SWScale as _SWScale  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Remote video decoding requires ffpyplayer. "
            "Install with `pip install ffpyplayer` or use a local camera/video source."
        ) from exc
    return _Image, _SWScale


class ColorSpaceConverter:
    """Handles color space conversion with caching and error handling."""

    def __init__(self):
        self._converter_cache = {}
        self._warned_formats = set()

    def get_converter(self, width: int, height: int, input_format: str):
        """Get or create a converter with caching."""
        cache_key = (width, height, input_format)

        if cache_key not in self._converter_cache:
            try:
                _, SWScale = _load_ffpyplayer_pic()
                # Suppress ffmpeg warnings for known non-accelerated conversions
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    converter = SWScale(width, height, input_format, ofmt="bgr24")
                    self._converter_cache[cache_key] = converter

                    # Log once per format about acceleration
                    if input_format not in self._warned_formats:
                        self._warned_formats.add(input_format)
                        logger.debug(
                            f"Created converter for {input_format} -> BGR24 "
                            f"(hardware acceleration may not be available)"
                        )

            except Exception as e:
                logger.error(f"Failed to create converter for {input_format}: {e}")
                raise

        return self._converter_cache[cache_key]

    def convert_frame(self, img) -> np.ndarray:
        """Convert image to BGR format."""
        try:
            width, height = img.get_size()
            input_format = img.get_pixel_format()

            converter = self.get_converter(width, height, input_format)
            img_bgr = converter.scale(img)

            frame = np.frombuffer(img_bgr.to_bytearray()[0], dtype=np.uint8).reshape(
                (height, width, 3)
            )

            return frame

        except Exception as e:
            logger.error(f"Frame conversion failed: {e}")
            raise


# --- Remote Video Source ---


class AsyncRemoteVideoPlayer(NetworkProtocolHandler):
    """Async remote video client with improved connection and recording state management."""

    def __init__(self, config: Config, recording_manager: RecordingStateManager):
        super().__init__()
        self.config = config
        self.recording_manager = recording_manager
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.frame_queue = asyncio.Queue(maxsize=5)
        self._listener_task: Optional[asyncio.Task] = None
        self._is_active = False
        self._connection_lock = asyncio.Lock()
        self._frame_processor = ColorSpaceConverter()
        self._paused = False
        self.last_error: Optional[str] = None

    async def connect(self) -> bool:
        """Connect to remote server with proper error handling."""
        async with self._connection_lock:
            if self._is_active:
                return True
            self.last_error = None

            logger.info(
                f"Connecting to {self.config.server_address}:{self.config.server_port}"
            )
            try:
                conn_future = asyncio.open_connection(
                    self.config.server_address, self.config.server_port
                )
                self.reader, self.writer = await asyncio.wait_for(
                    conn_future, timeout=self.config.remote_connect_timeout
                )

                self._is_active = True
                self._listener_task = asyncio.create_task(self._listen_for_messages())
                await self._send_message("started_playing", None)
                logger.info("Successfully connected to remote server")

                # Request current recording state
                await self._send_message("get_recording_state", None)
                return True

            except (OSError, socket.gaierror, asyncio.TimeoutError) as e:
                logger.warning(f"Remote connection failed: {e}")
                self.last_error = str(e)
                await self._cleanup_connection()
                return False

    async def _listen_for_messages(self):
        """Listen for incoming messages with robust error handling."""
        try:
            while self._is_active and self.reader and not self.reader.at_eof():
                try:
                    # Read message header
                    header_data = await asyncio.wait_for(
                        self.reader.readexactly(8), timeout=5.0
                    )
                    msg_len = struct.unpack(">II", header_data)

                    # Read message body
                    total_size = sum(msg_len)
                    if total_size > 0:
                        msg_buff = await asyncio.wait_for(
                            self.reader.readexactly(total_size), timeout=10.0
                        )
                    else:
                        msg_buff = b""

                    msg, value = self.decode_data(msg_buff, msg_len)
                    await self._handle_message(msg, value)

                except asyncio.TimeoutError:
                    logger.warning("Message receive timeout")
                    break
                except (asyncio.IncompleteReadError, ConnectionResetError):
                    logger.info("Remote connection closed")
                    break

        except Exception as e:
            logger.error(f"Listener error: {e}", exc_info=True)
        finally:
            await self.disconnect()

    async def _handle_message(self, msg: str, value):
        """Handle different message types with enhanced recording state management."""
        if msg == "image":
            # Always accept frames when not paused, regardless of queue state
            if not self._paused:
                try:
                    # If queue is full, remove oldest frame to make room
                    if self.frame_queue.full():
                        try:
                            self.frame_queue.get_nowait()  # Remove oldest frame
                        except asyncio.QueueEmpty:
                            pass
                    self.frame_queue.put_nowait((value, time.time()))
                except asyncio.QueueFull:
                    pass  # This shouldn't happen now, but keep as safety

        elif msg == "started_recording":
            logger.info("✅ Server started recording - resuming frame processing")
            await self.recording_manager.update_state(RecordingState.RECORDING)
            # Always resume processing when recording starts, regardless of config
            was_paused = self._paused
            self._paused = False
            if was_paused:
                logger.info("🔄 Frame processing resumed - ready to accept new frames")

        elif msg == "stopped_recording":
            logger.info(
                "⏸️  Server stopped recording - processing behavior depends on config"
            )
            await self.recording_manager.update_state(RecordingState.STOPPED)
            if self.config.pause_on_recording_stop:
                self._paused = True
                logger.info("📋 Frame processing paused until recording resumes")
                # Clear frame queue when paused
                await self._clear_frame_queue()
            else:
                logger.info("▶️  Continuing frame processing despite recording stop")

        elif msg == "recording_state":
            # Handle response to get_recording_state request
            current_state = (
                RecordingState.RECORDING if value else RecordingState.STOPPED
            )
            await self.recording_manager.update_state(current_state)
            old_paused = self._paused
            self._paused = self.config.pause_on_recording_stop and not value

            if old_paused and not self._paused:
                logger.info("🔄 Processing resumed based on recording state query")
            elif not old_paused and self._paused:
                logger.info("📋 Processing paused based on recording state query")
                await self._clear_frame_queue()

        else:
            logger.debug(f"Received message: {msg}")

    async def _clear_frame_queue(self):
        """Clear all frames from the queue."""
        cleared_count = 0
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
                cleared_count += 1
            except asyncio.QueueEmpty:
                break
        if cleared_count > 0:
            logger.debug(f"Cleared {cleared_count} frames from queue")

    async def get_frame(self) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Get frame with timeout handling and recording state awareness."""
        if self._paused:
            # When paused, return None but don't timeout immediately
            # Use shorter sleep to be more responsive to state changes
            await asyncio.sleep(0.1)
            return None

        try:
            # Use a shorter timeout when not paused to be more responsive
            raw_frame_data, timestamp = await asyncio.wait_for(
                self.frame_queue.get(), timeout=0.5
            )
            frame, metadata = self._process_raw_frame(raw_frame_data)
            metadata["capture_timestamp"] = timestamp
            metadata["source"] = "remote"
            metadata["recording_state"] = self.recording_manager.state.name
            metadata["paused"] = self._paused
            return frame, metadata
        except asyncio.TimeoutError:
            # Don't log timeouts as they're normal during sparse frame periods
            return None

    def _process_raw_frame(self, raw_frame_data) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process raw frame data into numpy array with enhanced error handling."""
        try:
            Image, _ = _load_ffpyplayer_pic()
            plane_buffers, pix_fmt, size, linesize, metadata = raw_frame_data
            img = Image(
                plane_buffers=plane_buffers,
                pix_fmt=pix_fmt,
                size=size,
                linesize=linesize,
            )

            # Use cached converter to reduce warnings and improve performance
            frame = self._frame_processor.convert_frame(img)
            return frame, metadata

        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            raise

    async def _send_message(self, key: str, value):
        """Send message to server."""
        if self.writer and not self.writer.is_closing():
            try:
                message_bytes = self.encode_msg(key, value)
                self.writer.write(message_bytes)
                await self.writer.drain()
            except Exception as e:
                logger.error(f"Failed to send message '{key}': {e}")

    async def _cleanup_connection(self):
        """Clean up connection resources."""
        if self.writer:
            try:
                self.writer.close()
                await self.writer.wait_closed()
            except Exception:
                pass
        self.writer = self.reader = None

    async def disconnect(self):
        """Disconnect with proper cleanup."""
        async with self._connection_lock:
            if not self._is_active:
                return

            self._is_active = False

            if self._listener_task:
                self._listener_task.cancel()
                try:
                    await self._listener_task
                except asyncio.CancelledError:
                    pass
                self._listener_task = None

            await self._cleanup_connection()
            await self._clear_frame_queue()
            logger.info("Remote video player disconnected")


# --- Local Camera Source ---


class CameraSource:
    """Local camera source with improved error handling."""

    def __init__(self, config: Config, recording_manager: RecordingStateManager):
        self.config = config
        self.recording_manager = recording_manager
        self.cap: Optional[cv2.VideoCapture] = None
        self._lock = asyncio.Lock()
        self.last_error: Optional[str] = None
        self._consecutive_read_failures = 0
        self._stream_reconnect_attempts = 0
        self._stream_reconnect_base_cooldown = max(
            0.5, float(getattr(self.config, "local_reconnect_cooldown", 2.0))
        )
        self._stream_reconnect_max_cooldown = max(
            self._stream_reconnect_base_cooldown,
            float(getattr(self.config, "local_reconnect_max_cooldown", 20.0)),
        )
        self._next_stream_reconnect_time = 0.0

    async def connect(self) -> bool:
        """Connect to local camera."""
        async with self._lock:
            if self.cap and self.cap.isOpened():
                return True

            logger.info(f"Connecting to camera: {self.config.camera_index}")
            self.last_error = None
            try:
                self.cap = await asyncio.to_thread(self._init_camera)
                if self.cap and self.cap.isOpened():
                    logger.info("Successfully connected to local camera")
                    # Local camera is always "recording"
                    await self.recording_manager.update_state(RecordingState.RECORDING)
                    self._consecutive_read_failures = 0
                    self._stream_reconnect_attempts = 0
                    self._next_stream_reconnect_time = 0.0
                    return True
                else:
                    logger.error("Failed to open camera")
                    self.last_error = self._build_camera_open_failure_message()
                    return False
            except Exception as e:
                logger.error(f"Camera connection error: {e}")
                self.last_error = str(e)
                return False

    def _build_camera_open_failure_message(self) -> str:
        source = str(self.config.camera_index or "").strip() or "0"
        if source.lower().startswith("rtp://"):
            return (
                f"Failed to open RTP stream source: {source}. "
                "Dynamic RTP payloads (e.g., type 96) require an SDP description. "
                "Use an RTSP URL when available, or provide an SDP-backed stream."
            )
        if self._is_network_stream_source(source):
            return (
                f"Failed to open stream source: {source}. "
                "For RTSP try transport=tcp, and verify OpenCV was built with FFmpeg/GStreamer."
            )
        if source.isdigit() and sys.platform == "darwin":
            return (
                "Failed to open camera. On macOS, camera permission must be granted to "
                "Annolid/Python in System Settings > Privacy & Security > Camera. "
                "If running from a worker thread, set OPENCV_AVFOUNDATION_SKIP_AUTH=1 "
                "and request permission in the main app process first."
            )
        return f"Failed to open camera source: {source}"

    @staticmethod
    def _is_network_stream_source(source: object) -> bool:
        value = str(source or "").strip().lower()
        return value.startswith(
            (
                "http://",
                "https://",
                "rtsp://",
                "rtsps://",
                "rtp://",
                "udp://",
                "srt://",
                "tcp://",
            )
        )

    def _open_capture(self, candidate: object) -> Optional[cv2.VideoCapture]:
        is_network = self._is_network_stream_source(candidate)
        backends: List[int] = []
        if is_network:
            for attr in ("CAP_FFMPEG", "CAP_GSTREAMER", "CAP_ANY"):
                backend = getattr(cv2, attr, None)
                if isinstance(backend, int) and backend not in backends:
                    backends.append(backend)
        else:
            backend = getattr(cv2, "CAP_ANY", None)
            if isinstance(backend, int):
                backends = [backend]
            else:
                backends = []

        def _open_with_backend(backend: Optional[int]) -> Optional[cv2.VideoCapture]:
            cap_obj: Optional[cv2.VideoCapture] = None
            open_timeout = int(getattr(self.config, "stream_open_timeout_ms", 5000))
            read_timeout = int(getattr(self.config, "stream_read_timeout_ms", 5000))
            params: List[int] = []
            if is_network:
                open_prop = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
                read_prop = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
                if isinstance(open_prop, int):
                    params.extend([open_prop, max(1000, open_timeout)])
                if isinstance(read_prop, int):
                    params.extend([read_prop, max(1000, read_timeout)])
            try:
                if backend is not None and params:
                    cap_obj = cv2.VideoCapture(candidate, backend, params)
                elif backend is not None:
                    cap_obj = cv2.VideoCapture(candidate, backend)
                else:
                    cap_obj = cv2.VideoCapture(candidate)
            except Exception:
                try:
                    if backend is not None:
                        cap_obj = cv2.VideoCapture(candidate, backend)
                    else:
                        cap_obj = cv2.VideoCapture(candidate)
                except Exception:
                    cap_obj = None

            if cap_obj is None or not cap_obj.isOpened():
                if cap_obj is not None:
                    cap_obj.release()
                return None

            try:
                cap_obj.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
                cap_obj.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
                cap_obj.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            return cap_obj

        if backends:
            for backend in backends:
                cap = _open_with_backend(backend)
                if cap is not None:
                    return cap
            return None
        return _open_with_backend(None)

    def _init_camera(self) -> Optional[cv2.VideoCapture]:
        """Initialize camera in thread."""
        try:
            candidates = self._build_camera_candidates()

            seen: set[str] = set()
            for candidate in candidates:
                key = str(candidate)
                if key in seen:
                    continue
                seen.add(key)
                cap = self._open_capture(candidate)
                if cap is not None:
                    return cap
            return None
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return None

    def _build_camera_candidates(self) -> list[object]:
        """
        Build candidate camera sources, always including local camera 0 fallback.
        Priority:
        1) Requested source
        2) Nearby local indices
        3) Guaranteed local fallback [0, 1, 2]
        """
        raw_source = str(self.config.camera_index or "").strip()
        candidates: list[object] = []
        aliases = {"", "default", "camera", "webcam", "cam", "cam0"}

        if raw_source.lower() in aliases:
            candidates.extend([0, 1, 2])
        elif raw_source.isdigit():
            idx = int(raw_source)
            candidates.append(idx)
            if idx != 0:
                candidates.append(0)
            candidates.extend([1, 2])
        elif self._is_network_stream_source(raw_source):
            # Preserve explicit stream source and avoid local fallback unless user requested it.
            candidates.append(raw_source)
        elif raw_source.lower().startswith(("http://", "https://")):
            candidates.extend(self._expand_http_stream_candidates(raw_source))
        else:
            # Explicit path/URL/custom identifier first.
            candidates.append(self.config.camera_index)
            # Then local camera fallbacks.
            candidates.extend([0, 1, 2])

        return candidates

    @staticmethod
    def _expand_http_stream_candidates(source: str) -> list[str]:
        value = str(source or "").strip()
        if not value:
            return []
        candidates: list[str] = [value]
        try:
            parts = urlsplit(value)
            path = (parts.path or "").lower()
            query = parse_qs(parts.query or "", keep_blank_values=True)
        except Exception:
            return candidates

        if path.endswith("/img/main.cgi") and (
            query.get("next_file", [""])[0].lower() == "main.htm" or not parts.query
        ):
            # Common camera control-page URL variants; add likely MJPEG/CGI stream paths.
            for stream_path in (
                "/img/video.mjpeg",
                "/img/mjpeg.cgi",
                "/mjpeg",
                "/video.mjpg",
            ):
                stream_url = urlunsplit(
                    (parts.scheme, parts.netloc, stream_path, "", "")
                )
                if stream_url not in candidates:
                    candidates.append(stream_url)
        return candidates

    async def get_frame(self) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Read frame from camera."""
        if not self.cap or not self.cap.isOpened():
            return None

        try:
            ret, frame = await asyncio.to_thread(self.cap.read)
            if not ret or frame is None:
                # If it's a file, try to loop
                source = self.config.camera_index
                if isinstance(source, str) and Path(source).is_file():
                    logger.debug("Local video EOF reached, looping...")
                    await asyncio.to_thread(self.cap.set, cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = await asyncio.to_thread(self.cap.read)

            if ret and frame is not None:
                self._consecutive_read_failures = 0
                return frame, {
                    "source": "local",
                    "capture_timestamp": time.time(),
                    "recording_state": self.recording_manager.state.name,
                }
            self._consecutive_read_failures += 1
            source = self.config.camera_index
            if self._is_network_stream_source(source):
                # HTTP/RTSP streams can drop transiently; attempt in-place recovery
                # instead of forcing immediate source failover.
                if self._consecutive_read_failures >= 3:
                    await self.recover_connection()
            return None
        except Exception as e:
            logger.error(f"Frame read error: {e}")
            self._consecutive_read_failures += 1
            if self._is_network_stream_source(self.config.camera_index):
                await self.recover_connection()
            return None

    async def recover_connection(self, *, force: bool = False) -> bool:
        """Attempt to recover an unstable network stream capture in-place."""
        if not self._is_network_stream_source(self.config.camera_index):
            return False
        now = time.time()
        if not force and now < self._next_stream_reconnect_time:
            return False
        cooldown = min(
            self._stream_reconnect_max_cooldown,
            self._stream_reconnect_base_cooldown
            * (2.0**self._stream_reconnect_attempts),
        )
        self._next_stream_reconnect_time = now + cooldown
        self._stream_reconnect_attempts = min(self._stream_reconnect_attempts + 1, 8)
        logger.warning(
            "Local stream read failed %d times; attempting reconnect (cooldown %.1fs).",
            self._consecutive_read_failures,
            cooldown,
        )
        async with self._lock:
            try:
                if self.cap is not None:
                    try:
                        await asyncio.to_thread(self.cap.release)
                    except Exception:
                        pass
                self.cap = await asyncio.to_thread(self._init_camera)
                if self.cap is not None and self.cap.isOpened():
                    logger.info("Recovered local network stream connection.")
                    self._consecutive_read_failures = 0
                    self._stream_reconnect_attempts = 0
                    self._next_stream_reconnect_time = 0.0
                    return True
                self.last_error = self._build_camera_open_failure_message()
                return False
            except Exception as exc:
                self.last_error = str(exc)
                return False

    async def disconnect(self):
        """Release camera resources."""
        async with self._lock:
            if self.cap:
                await asyncio.to_thread(self.cap.release)
                self.cap = None
                logger.info("Camera released")
        self._consecutive_read_failures = 0
        self._stream_reconnect_attempts = 0
        self._next_stream_reconnect_time = 0.0


# --- Source State Management ---


class SourceState(Enum):
    DISCONNECTED = auto()
    TRYING_REMOTE = auto()
    USING_REMOTE = auto()
    TRYING_LOCAL = auto()
    USING_LOCAL = auto()


class HybridVideoSource:
    """Hybrid video source with improved state management."""

    def __init__(self, config: Config, recording_manager: RecordingStateManager):
        self.config = config
        self.recording_manager = recording_manager
        self.remote = AsyncRemoteVideoPlayer(config, recording_manager)
        self.local = CameraSource(config, recording_manager)
        self.state = SourceState.DISCONNECTED
        self.last_remote_attempt = 0.0
        self._remote_retry_cooldown = max(
            1.0, float(getattr(self.config, "remote_retry_cooldown", 10.0))
        )
        self._remote_retry_max_cooldown = max(
            self._remote_retry_cooldown,
            float(getattr(self.config, "remote_retry_max_cooldown", 60.0)),
        )
        self._local_no_frame_tolerance = max(
            1, int(getattr(self.config, "local_no_frame_tolerance", 12))
        )
        self._local_reconnect_cooldown = max(
            0.5, float(getattr(self.config, "local_reconnect_cooldown", 2.0))
        )
        self._consecutive_local_misses = 0
        self._next_local_reconnect_time = 0.0
        self._prefer_local_only = self._is_explicit_local_source(config.camera_index)
        self._state_lock = asyncio.Lock()
        self._status_events: deque = deque()

    def _emit_status(self, event: str, **payload: Any) -> None:
        self._status_events.append(
            {
                "event": str(event or "").strip() or "status",
                "timestamp": time.time(),
                **payload,
            }
        )

    def pop_status_events(self) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        while self._status_events:
            events.append(dict(self._status_events.popleft()))
        return events

    @staticmethod
    def _is_network_stream_source(source: object) -> bool:
        value = str(source or "").strip().lower()
        return value.startswith(
            (
                "http://",
                "https://",
                "rtsp://",
                "rtsps://",
                "rtp://",
                "udp://",
                "srt://",
                "tcp://",
            )
        )

    @staticmethod
    def _is_explicit_local_source(source: object) -> bool:
        value = str(source or "").strip()
        lower = value.lower()
        if not value or lower in {"default", "camera", "webcam", "cam", "cam0"}:
            return False
        if value.isdigit():
            return False
        if lower.startswith(
            (
                "http://",
                "https://",
                "rtsp://",
                "rtsps://",
                "rtp://",
                "udp://",
                "srt://",
                "tcp://",
            )
        ):
            return True
        try:
            return (
                Path(value).expanduser().suffix != ""
                or Path(value).expanduser().exists()
            )
        except Exception:
            return True

    async def connect(self, *, raise_on_failure: bool = True) -> bool:
        """Initialize connection with fallback strategy."""
        async with self._state_lock:
            if self._prefer_local_only:
                self.state = SourceState.TRYING_LOCAL
                self._emit_status(
                    "reconnect_local_attempt",
                    source=str(self.config.camera_index),
                    reason="explicit_local_source",
                )
                if await self.local.connect():
                    self.state = SourceState.USING_LOCAL
                    logger.info("Connected to explicit local source")
                    self._emit_status(
                        "reconnect_local_success",
                        source=str(self.config.camera_index),
                    )
                    return True
                self.state = SourceState.DISCONNECTED
                self._emit_status(
                    "reconnect_local_failed",
                    source=str(self.config.camera_index),
                    error=str(self.local.last_error or "connect_failed"),
                )
                detail_text = (
                    f"local={self.local.last_error}" if self.local.last_error else ""
                )
                if detail_text:
                    message = f"Failed to connect to any video source ({detail_text})"
                    if raise_on_failure:
                        raise RuntimeError(message)
                    logger.warning(message)
                    return False
                message = "Failed to connect to any video source"
                if raise_on_failure:
                    raise RuntimeError(message)
                logger.warning(message)
                return False

            # Try remote first
            self.state = SourceState.TRYING_REMOTE
            self._emit_status("reconnect_remote_attempt", source="remote")
            if await self.remote.connect():
                self.state = SourceState.USING_REMOTE
                logger.info("Connected to remote source")
                self._emit_status("reconnect_remote_success", source="remote")
                return True
            self._emit_status(
                "reconnect_remote_failed",
                source="remote",
                error=str(self.remote.last_error or "connect_failed"),
            )

            # Fallback to local
            self.state = SourceState.TRYING_LOCAL
            self._emit_status(
                "reconnect_local_attempt",
                source=str(self.config.camera_index),
                reason="remote_unavailable",
            )
            if await self.local.connect():
                self.state = SourceState.USING_LOCAL
                logger.info("Connected to local source")
                self._emit_status(
                    "reconnect_local_success",
                    source=str(self.config.camera_index),
                )
                return True

            # Last-resort hard fallback: force camera index 0 and retry once.
            if str(
                self.config.camera_index
            ).strip() != "0" and not self._is_network_stream_source(
                self.config.camera_index
            ):
                logger.warning(
                    "Local source '%s' unavailable; retrying with fallback camera 0.",
                    self.config.camera_index,
                )
                self.config.camera_index = 0
                self._emit_status(
                    "reconnect_local_attempt",
                    source="0",
                    reason="fallback_camera_0",
                )
                if await self.local.connect():
                    self.state = SourceState.USING_LOCAL
                    logger.info("Connected to fallback local camera 0")
                    self._emit_status(
                        "reconnect_local_success",
                        source="0",
                    )
                    return True

            self.state = SourceState.DISCONNECTED
            self._emit_status(
                "reconnect_local_failed",
                source=str(self.config.camera_index),
                error=str(self.local.last_error or "connect_failed"),
            )
            details: List[str] = []
            if self.remote.last_error:
                details.append(f"remote={self.remote.last_error}")
            if self.local.last_error:
                details.append(f"local={self.local.last_error}")
            detail_text = "; ".join(details)
            if detail_text:
                message = f"Failed to connect to any video source ({detail_text})"
                if raise_on_failure:
                    raise RuntimeError(message)
                logger.warning(message)
                return False
            message = "Failed to connect to any video source"
            if raise_on_failure:
                raise RuntimeError(message)
            logger.warning(message)
            return False

    async def get_frame(self) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Get frame with automatic fallback and recovery."""
        # Try remote source
        if self.state == SourceState.USING_REMOTE:
            result = await self.remote.get_frame()
            if result:
                # We got a frame, everything is normal.
                return result

            # If result is None, investigate WHY before falling back.
            # Check if the connection is still considered active by the remote player.
            if self.remote._is_active:
                # The connection is still alive. A `None` frame is expected
                # when the remote is paused. We should wait patiently
                # without switching the source.
                return None
            else:
                # The connection is truly inactive. NOW it's correct to fall back.
                logger.warning(
                    "Remote source connection is inactive, falling back to local."
                )
                self._emit_status(
                    "reconnect_remote_failed",
                    source="remote",
                    error="connection_inactive",
                )
                await self.remote.disconnect()
                async with self._state_lock:
                    self.state = SourceState.TRYING_LOCAL

        # Try local source (this part of the logic is now only reached on a true remote failure)
        if self.state == SourceState.USING_LOCAL:
            result = await self.local.get_frame()
            if result:
                self._consecutive_local_misses = 0
                # Periodically try to reconnect to remote
                await self._try_remote_reconnect()
                return result
            else:
                self._consecutive_local_misses += 1
                if self._consecutive_local_misses < self._local_no_frame_tolerance:
                    return None
                if self._is_network_stream_source(self.config.camera_index):
                    self._emit_status(
                        "reconnect_local_attempt",
                        source=str(self.config.camera_index),
                        reason="consecutive_misses",
                        misses=int(self._consecutive_local_misses),
                    )
                    recover_fn = getattr(self.local, "recover_connection", None)
                    recovered = await recover_fn() if callable(recover_fn) else False
                    self._consecutive_local_misses = 0
                    if recovered:
                        self._emit_status(
                            "reconnect_local_success",
                            source=str(self.config.camera_index),
                        )
                        return None
                    self._emit_status(
                        "reconnect_local_failed",
                        source=str(self.config.camera_index),
                        error=str(self.local.last_error or "recover_failed"),
                    )
                logger.warning(
                    "Local source returned no frame %d times; resetting connection state",
                    self._consecutive_local_misses,
                )
                self._consecutive_local_misses = 0
                await self.local.disconnect()
                async with self._state_lock:
                    self.state = SourceState.DISCONNECTED

        # Handle disconnected state
        if self.state in (SourceState.TRYING_LOCAL, SourceState.DISCONNECTED):
            now = time.time()
            if now < self._next_local_reconnect_time:
                return None
            self._next_local_reconnect_time = now + self._local_reconnect_cooldown
            if self.state == SourceState.TRYING_LOCAL:
                async with self._state_lock:
                    if await self.local.connect():
                        self.state = SourceState.USING_LOCAL
                    else:
                        self.state = SourceState.DISCONNECTED
            elif self.state == SourceState.DISCONNECTED:
                logger.info("Attempting to recover connection...")
                self._emit_status(
                    "reconnect_attempt",
                    source=str(self.config.camera_index),
                    reason="disconnected",
                )
                await self.connect(raise_on_failure=False)

        return None

    async def _try_remote_reconnect(self):
        """Periodically attempt remote reconnection."""
        if self._prefer_local_only:
            return
        current_time = time.time()
        if current_time - self.last_remote_attempt > self._remote_retry_cooldown:
            self.last_remote_attempt = current_time
            logger.info("Attempting remote reconnection...")
            self._emit_status(
                "reconnect_remote_attempt",
                source="remote",
                cooldown_sec=float(self._remote_retry_cooldown),
            )

            if await self.remote.connect():
                logger.info("Successfully reconnected to remote source")
                self._remote_retry_cooldown = max(
                    1.0, float(getattr(self.config, "remote_retry_cooldown", 10.0))
                )
                async with self._state_lock:
                    self.state = SourceState.USING_REMOTE
                self._emit_status("reconnect_remote_success", source="remote")
            else:
                self._emit_status(
                    "reconnect_remote_failed",
                    source="remote",
                    error=str(self.remote.last_error or "connect_failed"),
                )
                self._remote_retry_cooldown = min(
                    self._remote_retry_max_cooldown,
                    self._remote_retry_cooldown * 2.0,
                )

    async def cleanup(self):
        """Clean up all resources."""
        await self.remote.disconnect()
        await self.local.disconnect()
        logger.info("Video source cleanup complete")


# --- Performance Metrics ---


class PerformanceMetrics:
    """Enhanced metrics collection and reporting."""

    def __init__(self, report_interval: float = 5.0):
        self.report_interval = report_interval
        self.fps_history = deque(maxlen=100)
        self.latency_history = deque(maxlen=100)
        self.detection_count = 0
        self.frame_count = 0
        self.skipped_frame_count = 0
        self.last_report_time = time.time()
        self.error_count = 0

    def record_frame(self, inference_time: float, detection_count: int):
        """Record metrics for a processed frame."""
        self.latency_history.append(inference_time)
        if detection_count is None:
            detection_count = 0
        self.detection_count += detection_count
        self.frame_count += 1

    def record_skipped_frame(self):
        """Record a skipped frame (due to recording state)."""
        self.skipped_frame_count += 1

    def record_error(self):
        """Record an error occurrence."""
        self.error_count += 1

    def should_report(self) -> bool:
        """Check if it's time to generate a report."""
        return time.time() - self.last_report_time >= self.report_interval

    def generate_report(
        self, source_state: str, recording_state: str
    ) -> Dict[str, Any]:
        """Generate performance report with recording state info."""
        current_time = time.time()
        elapsed = current_time - self.last_report_time

        fps = self.frame_count / elapsed if elapsed > 0 else 0
        self.fps_history.append(fps)

        avg_latency = (
            statistics.mean(self.latency_history) if self.latency_history else 0
        )
        avg_fps = statistics.mean(self.fps_history) if self.fps_history else 0

        report = {
            "current_fps": f"{fps:.2f}",
            "average_fps": f"{avg_fps:.2f}",
            "avg_latency_ms": f"{avg_latency * 1000:.2f}",
            "total_detections": self.detection_count,
            "error_count": self.error_count,
            "source_state": source_state,
            "recording_state": recording_state,
            "frames_processed": self.frame_count,
            "frames_skipped": self.skipped_frame_count,
        }

        # Reset counters
        self.frame_count = 0
        self.skipped_frame_count = 0
        self.last_report_time = current_time

        return report


class DetectionSegmentRecorder:
    """Record MP4 clips around detection events for selected classes."""

    _ANIMAL_CLASSES = {
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
    }

    def __init__(self, config: Config):
        self.enabled = bool(config.save_detection_segments)
        self.targets = [
            str(v).strip().lower() for v in config.detection_segment_targets
        ]
        self.prebuffer_sec = max(0.0, float(config.detection_segment_prebuffer_sec))
        self.postbuffer_sec = max(0.0, float(config.detection_segment_postbuffer_sec))
        self.min_duration_sec = max(
            0.0, float(config.detection_segment_min_duration_sec)
        )
        self.max_duration_sec = max(
            self.min_duration_sec + 1.0,
            float(config.detection_segment_max_duration_sec),
        )
        self.codec = str(config.detection_segment_codec or "mp4v")

        out_dir = str(config.detection_segment_output_dir or "").strip()
        if out_dir:
            self.output_dir = Path(out_dir).expanduser().resolve()
        else:
            self.output_dir = (
                resolve_annolid_realtime_logs_root() / "detection_segments"
            ).resolve()

        self._fps_hint = max(1.0, float(config.max_fps or 30.0))
        self._prebuffer: deque = deque()
        self._writer = None
        self._segment_path: Optional[Path] = None
        self._segment_labels: set[str] = set()
        self._recording = False
        self._segment_start_ts = 0.0
        self._last_trigger_ts = 0.0
        self._consecutive_write_errors = 0
        self._max_consecutive_write_errors = 10
        self._last_write_error_log_ts = 0.0
        self._completed_segments: deque = deque()

    def _matches_target(self, class_name: str) -> bool:
        name = str(class_name or "").strip().lower()
        if not name:
            return False
        if not self.targets or "*" in self.targets:
            return True
        if name in self.targets:
            return True
        if "person" in self.targets and name == "person":
            return True
        if "car" in self.targets and name in {"car", "truck", "bus", "motorcycle"}:
            return True
        if "animal" in self.targets and name in self._ANIMAL_CLASSES:
            return True
        return False

    def _append_prebuffer(self, frame: np.ndarray, timestamp: float) -> None:
        frame_copy = np.ascontiguousarray(frame.copy())
        self._prebuffer.append((float(timestamp), frame_copy))
        cutoff = float(timestamp) - self.prebuffer_sec
        while self._prebuffer and self._prebuffer[0][0] < cutoff:
            self._prebuffer.popleft()

    def _build_segment_path(self, event_ts: float) -> Path:
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(event_ts))
        millis = int((event_ts - int(event_ts)) * 1000)
        return self.output_dir / f"segment_{stamp}_{millis:03d}.mp4"

    @staticmethod
    def _sanitize_label(class_name: str) -> str:
        name = str(class_name or "").strip().lower()
        if not name:
            return ""
        normalized = re.sub(r"[^a-z0-9]+", "_", name)
        return normalized.strip("_")

    def _collect_matched_labels(self, detected_classes: List[str]) -> set[str]:
        labels: set[str] = set()
        for class_name in detected_classes or []:
            if not self._matches_target(class_name):
                continue
            slug = self._sanitize_label(class_name)
            if slug:
                labels.add(slug)
        return labels

    def _segment_path_with_labels(self, path: Path, labels: set[str]) -> Path:
        if not labels:
            return path
        joined = "_".join(sorted(labels))
        stem = path.stem
        if stem.endswith(f"_{joined}") or stem == joined:
            return path
        return path.with_name(f"{stem}_{joined}{path.suffix}")

    def _open_writer(
        self, frame: np.ndarray, event_ts: float, labels: Optional[set[str]] = None
    ) -> bool:
        writer = None
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            h, w = frame.shape[:2]
            self._segment_labels = set(labels or set())
            segment_path = self._segment_path_with_labels(
                self._build_segment_path(event_ts), self._segment_labels
            )
            fourcc = cv2.VideoWriter_fourcc(*self.codec[:4])
            writer = cv2.VideoWriter(
                str(segment_path),
                fourcc,
                float(self._fps_hint),
                (int(w), int(h)),
            )
            if writer is None or not writer.isOpened():
                logger.error("Failed to open segment writer: %s", segment_path)
                return False
            self._writer = writer
            self._segment_path = segment_path
            self._recording = True
            self._segment_start_ts = float(event_ts)
            self._last_trigger_ts = float(event_ts)
            for _ts, buffered in self._prebuffer:
                if not self._safe_write_frame(buffered, event_ts):
                    self._close_writer(event_ts)
                    logger.error(
                        "Aborting detection segment writer due to prebuffer write failures."
                    )
                    return False
            logger.info("Started detection segment recording: %s", segment_path)
            return True
        except Exception as exc:
            logger.error("Failed to start segment recording: %s", exc, exc_info=True)
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
            return False

    def _safe_write_frame(self, frame: np.ndarray, timestamp: float) -> bool:
        writer = self._writer
        if writer is None:
            return False
        try:
            writer.write(frame)
            self._consecutive_write_errors = 0
            return True
        except Exception as exc:
            self._consecutive_write_errors += 1
            now = time.time()
            if (
                self._consecutive_write_errors == 1
                or self._consecutive_write_errors >= self._max_consecutive_write_errors
                or (now - self._last_write_error_log_ts) >= 5.0
            ):
                logger.error(
                    "Detection segment write failed (%d/%d): %s",
                    self._consecutive_write_errors,
                    self._max_consecutive_write_errors,
                    exc,
                    exc_info=self._consecutive_write_errors == 1,
                )
                self._last_write_error_log_ts = now
            if self._consecutive_write_errors >= self._max_consecutive_write_errors:
                logger.error(
                    "Closing detection segment writer after repeated write failures."
                )
                self._close_writer(timestamp)
            return False

    def _close_writer(self, now_ts: float) -> None:
        writer = self._writer
        path = self._segment_path
        duration = max(0.0, float(now_ts) - self._segment_start_ts)
        labels = set(self._segment_labels)
        self._writer = None
        self._segment_path = None
        self._segment_labels = set()
        self._recording = False
        self._segment_start_ts = 0.0
        self._last_trigger_ts = 0.0
        self._consecutive_write_errors = 0

        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        if path is None:
            return

        labeled_path = self._segment_path_with_labels(path, labels)
        if labeled_path != path:
            try:
                path.rename(labeled_path)
                path = labeled_path
            except Exception:
                logger.warning(
                    "Failed to rename segment with labels: %s -> %s",
                    path,
                    labeled_path,
                )

        if duration < self.min_duration_sec:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
            logger.info(
                "Discarded short detection segment (< %.2fs): %s",
                self.min_duration_sec,
                path,
            )
            return

        logger.info("Saved detection segment (%.2fs): %s", duration, path)
        self._completed_segments.append(
            {
                "path": str(path),
                "duration_sec": float(duration),
                "labels": sorted(labels),
                "timestamp": float(now_ts),
            }
        )

    def update(
        self, frame: np.ndarray, timestamp: float, detected_classes: List[str]
    ) -> None:
        if not self.enabled:
            return
        if frame is None or frame.size == 0:
            return

        matched_labels = self._collect_matched_labels(detected_classes or [])
        event_detected = bool(matched_labels)
        self._append_prebuffer(frame, timestamp)

        if not self._recording and event_detected:
            self._open_writer(frame, timestamp, labels=matched_labels)

        if self._recording and self._writer is not None:
            if not self._safe_write_frame(frame, timestamp):
                return
            if event_detected:
                self._last_trigger_ts = float(timestamp)
                self._segment_labels.update(matched_labels)

            silence_sec = float(timestamp) - self._last_trigger_ts
            duration_sec = float(timestamp) - self._segment_start_ts
            if (
                silence_sec >= self.postbuffer_sec
                or duration_sec >= self.max_duration_sec
            ):
                self._close_writer(timestamp)

    def close(self) -> None:
        self._close_writer(time.time())

    def pop_completed_segments(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        while self._completed_segments:
            out.append(dict(self._completed_segments.popleft()))
        return out


# --- Detection Publisher ---


class DetectionPublisher:
    """ZMQ publisher for detection results."""

    def __init__(self, address: str, camera_id: str = "camera0"):
        self.address = address
        self.camera_id = str(camera_id or "camera0")
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.LINGER, 0)
        self._bound = False

    async def bind(self):
        """Bind the publisher to the address with retries."""
        if self._bound:
            return

        last_exc = None
        for attempt in range(3):
            try:
                # Use asyncio to_thread for bind just in case it blocks on some platforms
                await asyncio.to_thread(self.socket.bind, self.address)
                self._bound = True
                logger.info(f"Detection publisher bound to {self.address}")
                return
            except Exception as e:
                last_exc = e
                logger.warning(
                    f"Bind attempt {attempt + 1} failed for {self.address}: {e}"
                )
                if attempt < 2:
                    await asyncio.sleep(0.5)

        logger.error(f"Failed to bind publisher to {self.address} after 3 attempts.")
        if last_exc:
            raise last_exc

    async def publish_detection(self, detection: DetectionResult):
        """Publish a detection result."""
        try:
            payload = detection.to_dict()
            metadata = dict(payload.get("metadata") or {})
            metadata.setdefault("camera_id", self.camera_id)
            payload["metadata"] = metadata
            await self.socket.send_string("detections", flags=zmq.SNDMORE)
            await self.socket.send_json(payload)
        except Exception as e:
            logger.error(f"Failed to publish detection: {e}")

    async def publish_status(self, status: Dict[str, Any]):
        """Publish system status."""
        try:
            payload = dict(status or {})
            payload.setdefault("camera_id", self.camera_id)
            await self.socket.send_string("status", flags=zmq.SNDMORE)
            await self.socket.send_json(payload)
        except Exception as e:
            logger.error(f"Failed to publish status: {e}")

    async def publish_frame(
        self,
        frame: Optional[np.ndarray],
        metadata: Dict[str, Any],
        encoding: str = "jpg",
        quality: int = 80,
    ):
        """Publish an encoded frame with associated metadata."""
        try:
            payload_metadata = dict(metadata or {})
            payload_metadata.setdefault("camera_id", self.camera_id)
            buffer_bytes = b""

            if frame is not None:
                encoding = (encoding or "jpg").lower()
                quality = int(max(1, min(int(quality), 100)))

                encode_params = []
                if encoding in ("jpg", "jpeg"):
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                elif encoding == "png":
                    # Default compression keeps latency low.
                    encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 3]

                success, buffer = await asyncio.to_thread(
                    cv2.imencode, f".{encoding}", frame, encode_params
                )
                if not success:
                    logger.error("Failed to encode frame for publishing")
                    return

                payload_metadata.setdefault(
                    "encoding", "jpeg" if encoding in ("jpg", "jpeg") else encoding
                )
                payload_metadata.setdefault(
                    "shape", [int(frame.shape[0]), int(frame.shape[1])]
                )
                buffer_bytes = buffer.tobytes()
            else:
                payload_metadata["skip_frame"] = True

            await self.socket.send_string("frames", flags=zmq.SNDMORE)
            await self.socket.send_json(payload_metadata, flags=zmq.SNDMORE)
            await self.socket.send(buffer_bytes)
        except Exception as e:
            logger.error(f"Failed to publish frame: {e}")

    async def cleanup(self):
        """Clean up publisher resources."""
        if self.socket is None:
            return
        try:
            if self._bound:
                with contextlib.suppress(Exception):
                    self.socket.unbind(self.address)
            self.socket.setsockopt(zmq.LINGER, 0)
            self.socket.close(0)
            self.socket = None
            if self.context is not None:
                self.context.term()
            self.context = None
            self._bound = False
            logger.info("Detection publisher cleaned up")
        except Exception as e:
            logger.error(f"Publisher cleanup error: {e}")


# --- Main Perception Process ---


class PerceptionProcess:
    """Main perception process orchestrator with recording state management."""

    def __init__(self, config: Config):
        self.config = config
        self.model: Optional[Union[YOLO, "MediaPipeEngine"]] = None
        self.class_names: Optional[List[str]] = None
        self.keypoint_labels: Optional[List[str]] = None

        # Auto-enable pose mode for known pose models
        model_name_lower = str(self.config.model_base_name).lower()
        if not self.config.viewer_only and "pose" in model_name_lower:
            self.config.enable_pose = True

        if self.config.enable_pose and self.config.enable_segmentation:
            logger.info(
                "Pose mode detected; disabling segmentation processing for compatibility."
            )
            self.config.enable_segmentation = False

        # Initialize recording state manager
        self.recording_manager = RecordingStateManager(config)

        # Initialize components with recording manager
        self.video_source = HybridVideoSource(config, self.recording_manager)
        self.metrics = PerformanceMetrics()
        self.segment_recorder = DetectionSegmentRecorder(config)
        self.publisher = DetectionPublisher(
            config.publisher_address,
            camera_id=str(getattr(config, "camera_id", "") or "camera0"),
        )
        self.running = True
        self._shutdown_event = asyncio.Event()
        self._shutdown_complete = False
        self._frame_index = 0

        # Setup recording state callbacks
        self.recording_manager.add_state_change_callback(
            self._on_recording_state_change
        )

    def _extract_keypoint_labels(self, model: YOLO) -> Optional[List[str]]:
        """Attempt to extract keypoint labels from the loaded YOLO model."""

        def _normalize_labels(source) -> Optional[List[str]]:
            if not source:
                return None
            if isinstance(source, dict):
                return [source[key] for key in sorted(source)]
            if isinstance(source, str):
                return [source]
            if isinstance(source, (list, tuple)):
                return list(source)
            try:
                return list(source)
            except TypeError:
                return None

        try:
            overrides = getattr(model, "overrides", None)
            if overrides:
                labels = _normalize_labels(
                    overrides.get("kpt_label") or overrides.get("kpt_labels")
                )
                if labels:
                    return labels

            candidate_attrs = ("kpt_label", "kpt_labels", "names_kpt")
            model_obj = getattr(model, "model", None)

            for attr in candidate_attrs:
                labels = _normalize_labels(getattr(model_obj, attr, None))
                if labels:
                    return labels

            inner_model = getattr(model_obj, "model", None) if model_obj else None
            for attr in candidate_attrs:
                labels = _normalize_labels(getattr(inner_model, attr, None))
                if labels:
                    return labels

        except Exception as exc:
            logger.debug("Unable to extract keypoint labels: %s", exc)

        return None

    async def _on_recording_state_change(
        self, old_state: RecordingState, new_state: RecordingState
    ):
        """Handle recording state changes."""
        state_info = self.recording_manager.get_state_info()

        # Log the state change with processing status
        if new_state == RecordingState.RECORDING:
            logger.info(
                f"🎬 Recording state: {old_state.name} → {new_state.name} (Processing: {'ENABLED' if state_info['should_process'] else 'DISABLED'})"
            )
        elif new_state == RecordingState.STOPPED:
            logger.info(
                f"🛑 Recording state: {old_state.name} → {new_state.name} (Processing: {'ENABLED' if state_info['should_process'] else 'DISABLED'})"
            )
        else:
            logger.info(
                f"📡 Recording state: {old_state.name} → {new_state.name} (Processing: {'ENABLED' if state_info['should_process'] else 'DISABLED'})"
            )

        await self.publisher.publish_status(
            {
                "event": "recording_state_change",
                "old_state": old_state.name,
                "new_state": new_state.name,
                "should_process": state_info["should_process"],
                "timestamp": time.time(),
            }
        )

    @asynccontextmanager
    async def _model_context(self) -> AsyncIterator[Union[YOLO, "MediaPipeEngine"]]:
        """Context manager for perception model."""
        try:
            model_ref = str(self.config.model_base_name)
            if "mediapipe" in model_ref.lower():
                from annolid.realtime.mediapipe_engine import MediaPipeEngine

                if not MediaPipeEngine.is_installed():
                    logger.info("MediaPipe not found. Prompting installation...")
                    if not MediaPipeEngine.install():
                        raise ImportError("Failed to install MediaPipe automatically.")

                logger.info("Loading MediaPipe engine: %s", model_ref)
                model = await asyncio.to_thread(MediaPipeEngine, model_ref)
                self.class_names = list(model.names.values())
                self.keypoint_labels = model.landmark_names
                self.config.enable_pose = True  # Usually true for Mediapipe
                yield model
            else:
                model_ref = str(resolve_weight_path(model_ref))
                logger.info("Loading YOLO model: %s", model_ref)
                model = await asyncio.to_thread(YOLO, model_ref)
                self.class_names = model.names
                keypoint_labels = self._extract_keypoint_labels(model)
                if keypoint_labels:
                    if not self.config.enable_pose:
                        logger.info("Pose metadata detected; enabling pose processing.")
                    self.config.enable_pose = True
                    self.keypoint_labels = keypoint_labels
                    if self.config.enable_segmentation:
                        logger.info(
                            "Disabling segmentation because pose keypoints are present."
                        )
                        self.config.enable_segmentation = False
                logger.info("YOLO model loaded successfully")
                yield model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        finally:
            if "model" in locals() and model and hasattr(model, "close"):
                model.close()
            logger.info("Model context closed")

    async def setup(self):
        """Setup the perception process."""
        logger.info("Setting up perception process...")

        configure_ultralytics_cache()

        if not self.config.viewer_only:
            # Verify model exists or download
            model_ref = str(resolve_weight_path(self.config.model_base_name))
            model_path = Path(model_ref)
            if (
                not (model_path.is_file() or model_path.is_dir())
                and "mediapipe" not in model_ref.lower()
            ):
                logger.info(
                    f"Model '{model_path}' not found locally, attempting Ultralytics resolution..."
                )
                await asyncio.to_thread(YOLO, model_ref)
        else:
            logger.info("Viewer-only mode enabled; skipping model setup.")

        # Initialize publisher
        await self.publisher.bind()

        # Initialize video source. Keep process alive on startup failures so
        # get_frame() can continue reconnection attempts.
        await self.video_source.connect(raise_on_failure=False)
        logger.info("Setup complete")

    async def run(self):
        """Main processing loop with recording state awareness."""
        await self.setup()
        frame_interval = 1.0 / self.config.max_fps
        if self.config.viewer_only:
            await self._run_viewer_only_loop(frame_interval)
            return

        async with self._model_context() as model:
            self.model = model

            while self.running and not self._shutdown_event.is_set():
                loop_start = time.time()

                try:
                    await self._publish_source_status_events()
                    # Get frame
                    frame_data = await self.video_source.get_frame()
                    await self._publish_source_status_events()
                    if not frame_data:
                        await asyncio.sleep(0.1)
                        continue

                    frame, metadata = frame_data
                    metadata = dict(metadata or {})
                    metadata.setdefault("capture_timestamp", time.time())
                    metadata["frame_index"] = self._frame_index
                    metadata["camera_id"] = str(
                        getattr(self.config, "camera_id", "") or "camera0"
                    )

                    # Check if we should process this frame based on recording state
                    processing_active = self.recording_manager.should_process_frames()
                    if not processing_active:
                        self.metrics.record_skipped_frame()
                        # Use shorter sleep when paused to be more responsive to state changes
                        await asyncio.sleep(0.05)
                        continue

                    try:
                        # Run inference
                        results = await self._run_inference(frame)
                        inference_time = time.time() - loop_start

                        # Process results
                        (
                            detection_count,
                            matched_classes,
                        ) = await self._process_detections(
                            results, loop_start, metadata
                        )
                        self.segment_recorder.update(frame, loop_start, matched_classes)
                        completed_segments = (
                            self.segment_recorder.pop_completed_segments()
                        )
                        for segment in completed_segments:
                            await self.publisher.publish_status(
                                {
                                    "event": "detection_segment_saved",
                                    "path": str(segment.get("path") or ""),
                                    "duration_sec": float(
                                        segment.get("duration_sec") or 0.0
                                    ),
                                    "labels": list(segment.get("labels") or []),
                                    "timestamp": float(
                                        segment.get("timestamp") or time.time()
                                    ),
                                }
                            )

                        # Update metrics
                        self.metrics.record_frame(inference_time, detection_count)

                        # Prepare visualization output if requested
                        annotated_frame = None
                        if (
                            self.config.visualize
                            or self.config.publish_annotated_frames
                        ):
                            annotated_frame = self._visualize_results(
                                results, frame, show_window=self.config.visualize
                            )

                        # Report metrics
                        if self.metrics.should_report():
                            report = self.metrics.generate_report(
                                self.video_source.state.name,
                                self.recording_manager.state.name,
                            )
                            logger.info(f"Performance: {json.dumps(report)}")
                            await self.publisher.publish_status(
                                {
                                    "event": "performance_report",
                                    **report,
                                    "timestamp": time.time(),
                                }
                            )

                        if self.config.publish_frames:
                            frame_to_publish = (
                                annotated_frame
                                if (
                                    self.config.publish_annotated_frames
                                    and annotated_frame is not None
                                )
                                else frame
                            )
                        else:
                            frame_to_publish = None

                        await self.publisher.publish_frame(
                            frame_to_publish,
                            {
                                "frame_index": metadata["frame_index"],
                                "capture_timestamp": metadata.get("capture_timestamp"),
                                "camera_id": metadata.get("camera_id"),
                                "source": metadata.get("source"),
                                "recording_state": self.recording_manager.state.name,
                                "processing": processing_active,
                                "detection_count": detection_count,
                                "inference_ms": inference_time * 1000.0,
                            },
                            encoding=self.config.frame_encoding,
                            quality=self.config.frame_quality,
                        )
                    finally:
                        self._frame_index += 1

                except Exception as e:
                    logger.error(f"Processing error: {e}", exc_info=True)
                    self.metrics.record_error()

                # Frame rate limiting
                elapsed = time.time() - loop_start
                await asyncio.sleep(max(0, frame_interval - elapsed))

    async def _run_viewer_only_loop(self, frame_interval: float) -> None:
        """Publish raw frames without loading or running a perception model."""
        while self.running and not self._shutdown_event.is_set():
            loop_start = time.time()
            try:
                await self._publish_source_status_events()
                frame_data = await self.video_source.get_frame()
                await self._publish_source_status_events()
                if not frame_data:
                    await asyncio.sleep(0.1)
                    continue

                frame, metadata = frame_data
                metadata = dict(metadata or {})
                metadata.setdefault("capture_timestamp", time.time())
                metadata["frame_index"] = self._frame_index
                metadata["camera_id"] = str(
                    getattr(self.config, "camera_id", "") or "camera0"
                )

                self.metrics.record_frame(0.0, 0)
                if self.metrics.should_report():
                    report = self.metrics.generate_report(
                        self.video_source.state.name,
                        self.recording_manager.state.name,
                    )
                    logger.info(f"Performance: {json.dumps(report)}")
                    await self.publisher.publish_status(
                        {
                            "event": "performance_report",
                            **report,
                            "timestamp": time.time(),
                        }
                    )

                if self.config.publish_frames:
                    await self.publisher.publish_frame(
                        frame,
                        {
                            "frame_index": metadata["frame_index"],
                            "capture_timestamp": metadata.get("capture_timestamp"),
                            "camera_id": metadata.get("camera_id"),
                            "source": metadata.get("source"),
                            "recording_state": self.recording_manager.state.name,
                            "processing": False,
                            "viewer_only": True,
                            "detection_count": 0,
                            "inference_ms": 0.0,
                        },
                        encoding=self.config.frame_encoding,
                        quality=self.config.frame_quality,
                    )
                self._frame_index += 1
            except Exception as e:
                logger.error(f"Viewer-only processing error: {e}", exc_info=True)
                self.metrics.record_error()

            elapsed = time.time() - loop_start
            await asyncio.sleep(max(0, frame_interval - elapsed))

    async def _publish_source_status_events(self) -> None:
        pop_fn = getattr(self.video_source, "pop_status_events", None)
        if not callable(pop_fn):
            return
        events = pop_fn()
        if not events:
            return
        for event in events:
            payload = dict(event or {})
            payload.setdefault("recording_state", self.recording_manager.state.name)
            await self.publisher.publish_status(payload)

    async def _run_inference(self, frame: np.ndarray):
        """Run YOLO inference on frame."""
        try:
            results = await asyncio.to_thread(
                self.model,
                frame,
                stream=False,
                conf=self.config.confidence_threshold,
                verbose=False,
            )
            return results[0]
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise

    def _encode_mask(
        self, masks_obj: Masks, detection_index: int
    ) -> Optional[Dict[str, Any]]:
        """
        Encodes a single mask from a Masks object based on the configured strategy.

        Args:
            masks_obj: The `ultralytics.engine.results.Masks` object from the result.
            detection_index: The index of the specific mask to encode.

        Returns:
            A dictionary containing the encoded mask data, or None on failure.
        """
        encoding_type = self.config.mask_encoding

        try:
            # --- STRATEGY 1: Polygon (Most Efficient) ---
            # Uses the pre-computed, scaled polygon coordinates directly from the Masks object.
            if encoding_type == "polygon":
                # masks_obj.xy is a list of numpy arrays (N, 2)
                polygon_pixels = masks_obj.xy[detection_index]
                return {
                    "encoding": "polygon",
                    # .tolist() is essential for JSON serialization
                    "points": polygon_pixels.tolist(),
                }

            # --- STRATEGY 2: COCO RLE (Standard & Compressed) ---
            # Operates on the raw mask bitmap.
            elif encoding_type == "rle":
                # Get the raw boolean mask tensor for the specific detection
                mask_tensor = masks_obj.data[detection_index]

                # Convert to a numpy array in the format pycocotools expects
                mask_bitmap = (mask_tensor > 0.5).cpu().numpy().astype(np.uint8)
                target_shape = getattr(masks_obj, "orig_shape", None)
                if target_shape and mask_bitmap.shape[:2] != tuple(target_shape):
                    target_h, target_w = int(target_shape[0]), int(target_shape[1])
                    mask_bitmap = cv2.resize(
                        mask_bitmap,
                        (target_w, target_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                mask_fortran = np.asfortranarray(mask_bitmap)

                # Encode to RLE
                rle = maskUtils.encode(mask_fortran)

                # Decode the byte string for JSON compatibility
                if isinstance(rle["counts"], bytes):
                    rle["counts"] = rle["counts"].decode("utf-8")

                return {
                    "encoding": "coco_rle",
                    "size": rle["size"],
                    "counts": rle["counts"],
                }

            # --- STRATEGY 3: Bitmap (Uncompressed Fallback) ---
            # This is less efficient for transmission but provides raw data.
            elif encoding_type == "bitmap":
                mask_bitmap = masks_obj.data[detection_index].cpu().numpy()
                # Not directly JSON serializable, so we don't implement this by default.
                # If needed, one could use base64 encoding here.
                logger.warning(
                    "Bitmap encoding is not recommended for network transmission."
                )
                return None

            else:
                logger.warning(
                    f"Unknown mask_encoding type: '{encoding_type}'. No mask will be sent."
                )
                return None

        except Exception as e:
            logger.error(f"Failed to encode mask with strategy '{encoding_type}': {e}")
            return None

    async def _process_detections(
        self, result, timestamp: float, metadata: Dict[str, Any]
    ) -> Tuple[int, List[str]]:
        """Process detection results and publish with mask support."""
        # Polymorphic handling of MediaPipe results
        from annolid.realtime.mediapipe_engine import MediaPipeResult

        if isinstance(result, MediaPipeResult):
            return await self._process_mediapipe_detections(result, timestamp, metadata)

        if not result.boxes or len(result.boxes) == 0:
            return 0, []

        def _to_list(data):
            if data is None:
                return None
            processed = data
            if hasattr(processed, "cpu"):
                processed = processed.cpu()
            if hasattr(processed, "numpy"):
                processed = processed.numpy()
            if hasattr(processed, "tolist"):
                return processed.tolist()
            return list(processed)

        detection_count = 0
        matched_classes: List[str] = []
        target_behaviors = self.config.target_behaviors or []
        match_all_targets = (not target_behaviors) or ("*" in target_behaviors)
        boxes = result.boxes

        # Check if segmentation masks are available in the result object
        masks = (
            result.masks
            if self.config.enable_segmentation
            and hasattr(result, "masks")
            and result.masks is not None
            else None
        )
        keypoints_obj = (
            result.keypoints
            if hasattr(result, "keypoints") and result.keypoints is not None
            else None
        )

        for i in range(len(boxes)):
            try:
                class_id = int(boxes.cls[i])
                class_name = self.class_names[class_id]

                if not match_all_targets and class_name not in target_behaviors:
                    continue

                encoded_mask_data = None
                # If masks are available, encode the one for this specific detection
                if masks:
                    encoded_mask_data = self._encode_mask(masks, i)

                # Prepare pose information if available
                keypoints_normalized = None
                keypoints_pixels = None
                keypoint_scores = None

                if keypoints_obj is not None:
                    keypoints_data_norm = getattr(keypoints_obj, "xyn", None)
                    if keypoints_data_norm is not None and i < len(keypoints_data_norm):
                        keypoints_normalized = _to_list(keypoints_data_norm[i])

                    keypoints_data_px = getattr(keypoints_obj, "xy", None)
                    if keypoints_data_px is not None and i < len(keypoints_data_px):
                        keypoints_pixels = _to_list(keypoints_data_px[i])

                    conf_data = getattr(keypoints_obj, "conf", None)
                    if conf_data is not None and i < len(conf_data):
                        keypoint_scores = _to_list(conf_data[i])

                keypoint_labels = (
                    self.keypoint_labels
                    if self.keypoint_labels
                    and (
                        keypoints_pixels is not None or keypoints_normalized is not None
                    )
                    else None
                )

                # Create the DetectionResult with the pre-encoded mask and pose data
                detection = DetectionResult(
                    behavior=class_name,
                    confidence=float(boxes.conf[i]),
                    bbox_normalized=boxes.xyxyn[i].cpu().numpy().tolist(),
                    timestamp=timestamp,
                    metadata=metadata,
                    bbox_pixels=boxes.xyxy[i].cpu().numpy().tolist(),
                    mask_data=encoded_mask_data,  # Pass the encoded dictionary here
                    keypoints_normalized=keypoints_normalized,
                    keypoints_pixels=keypoints_pixels,
                    keypoint_scores=keypoint_scores,
                    keypoint_labels=keypoint_labels,
                )

                await self.publisher.publish_detection(detection)
                detection_count += 1
                matched_classes.append(class_name)

            except Exception as e:
                logger.error(f"Detection processing error: {e}", exc_info=True)
        return detection_count, matched_classes

    async def _process_mediapipe_detections(
        self, result: Any, timestamp: float, metadata: Dict[str, Any]
    ) -> Tuple[int, List[str]]:
        """Process native MediaPipe detections."""
        detection_count = 0
        matched_classes: List[str] = []
        h, w = result.orig_img.shape[:2]

        for i, obj_norm_lms in enumerate(result.norm_landmarks):
            try:
                xs = [p[0] for p in obj_norm_lms]
                ys = [p[1] for p in obj_norm_lms]
                bbox_norm = [min(xs), min(ys), max(xs), max(ys)]
                bbox_pixels = [
                    bbox_norm[0] * w,
                    bbox_norm[1] * h,
                    bbox_norm[2] * w,
                    bbox_norm[3] * h,
                ]

                # MediaPipe classes are currently fixed
                class_name = (
                    self.class_names[0] if self.class_names else result.model_type
                )

                payload_metadata = dict(metadata or {})
                if result.distance_cm is not None:
                    payload_metadata["distance_cm"] = result.distance_cm

                # Merge additional metadata (like gaze) if present
                if hasattr(result, "metadata") and result.metadata:
                    payload_metadata.update(result.metadata)

                detection = DetectionResult(
                    behavior=class_name,
                    confidence=0.9,  # MediaPipe doesn't always expose box confidence in a standard way here
                    bbox_normalized=bbox_norm,
                    timestamp=timestamp,
                    metadata=payload_metadata,
                    bbox_pixels=bbox_pixels,
                    keypoints_normalized=obj_norm_lms,
                    keypoints_pixels=[[p[0] * w, p[1] * h, p[2]] for p in obj_norm_lms],
                    keypoint_labels=self.keypoint_labels,
                )
                await self.publisher.publish_detection(detection)
                detection_count += 1
                matched_classes.append(class_name)
            except Exception as e:
                logger.error(f"MediaPipe detection error: {e}")
        return detection_count, matched_classes

    def _visualize_results(
        self, result, frame: np.ndarray, show_window: bool = True
    ) -> np.ndarray:
        """Create an annotated frame and optionally display it locally."""
        try:
            # Polymorphic plot/annotation
            if hasattr(result, "plot"):
                annotated_frame = result.plot()
            else:
                annotated_frame = frame.copy()

            # Alternative: Manual mask overlay if you want custom styling
            if (
                self.config.enable_segmentation
                and hasattr(result, "masks")
                and result.masks is not None
            ):
                masks = result.masks.data.cpu().numpy()
                boxes = result.boxes

                # Create colored mask overlay
                mask_overlay = np.zeros_like(frame)
                colors = [
                    (255, 0, 0),
                    (0, 255, 0),
                    (0, 0, 255),
                    (255, 255, 0),
                    (255, 0, 255),
                ]

                for i, mask in enumerate(masks):
                    if i < len(boxes):
                        class_id = int(boxes.cls[i])
                        # class_names is a dict: {id: name}
                        class_name = self.class_names[class_id]

                        if class_name in self.config.target_behaviors:
                            color = colors[i % len(colors)]
                            mask_resized = cv2.resize(
                                mask, (frame.shape[1], frame.shape[0])
                            )
                            mask_bool = mask_resized > 0.5
                            mask_overlay[mask_bool] = color

                # Blend with original frame
                alpha = 0.3
                annotated_frame = cv2.addWeighted(
                    annotated_frame, 1 - alpha, mask_overlay, alpha, 0
                )

            # Add performance info (existing code)
            fps_text = (
                f"FPS: {self.metrics.fps_history[-1]:.1f}"
                if self.metrics.fps_history
                else "FPS: N/A"
            )
            source_text = f"Source: {self.video_source.state.name}"
            recording_text = f"Recording: {self.recording_manager.state.name}"

            # Color code recording state
            recording_color = (
                (0, 255, 0)
                if self.recording_manager.state == RecordingState.RECORDING
                else (0, 165, 255)
            )
            processing_text = (
                "PROCESSING"
                if self.recording_manager.should_process_frames()
                else "PAUSED"
            )
            processing_color = (
                (0, 255, 0)
                if self.recording_manager.should_process_frames()
                else (0, 0, 255)
            )

            # Draw text overlays
            cv2.putText(
                annotated_frame,
                fps_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                annotated_frame,
                source_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                annotated_frame,
                recording_text,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                recording_color,
                2,
            )
            cv2.putText(
                annotated_frame,
                processing_text,
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                processing_color,
                2,
            )

            # Add segmentation info
            if (
                self.config.enable_segmentation
                and hasattr(result, "masks")
                and result.masks is not None
            ):
                seg_text = f"Masks: {len(result.masks.data)}"
                cv2.putText(
                    annotated_frame,
                    seg_text,
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

            if show_window:
                cv2.imshow("Perception System", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.running = False
                    self._shutdown_event.set()

            return annotated_frame

        except Exception as e:
            logger.error(f"Visualization error: {e}")
            return frame

    async def shutdown(self):
        """Graceful shutdown."""
        if self._shutdown_complete:
            return

        logger.info("Shutting down perception process...")
        self.running = False
        self._shutdown_event.set()

        cleanup_errors = []
        with contextlib.suppress(Exception):
            self.segment_recorder.close()
        try:
            await self.video_source.cleanup()
        except Exception as e:
            cleanup_errors.append(e)
            logger.error(f"Video source cleanup error: {e}", exc_info=True)
        try:
            await self.publisher.cleanup()
        except Exception as e:
            cleanup_errors.append(e)
            logger.error(f"Publisher cleanup error: {e}", exc_info=True)

        if self.config.visualize:
            # On macOS, destroyAllWindows can crash if called from a thread.
            # Also, we check if it's even available (headless etc).
            try:
                import platform

                if platform.system() == "Darwin":
                    # Be conservative on macOS
                    logger.debug(
                        "Skipping cv2.destroyAllWindows() on macOS to avoid crash."
                    )
                else:
                    cv2.destroyAllWindows()
            except Exception:
                pass

        if cleanup_errors:
            logger.error(
                "Shutdown completed with %d cleanup error(s).", len(cleanup_errors)
            )
        else:
            logger.info("Shutdown complete")
        self._shutdown_complete = True

    def request_stop(self) -> None:
        """Request a prompt stop from another thread without awaiting shutdown."""
        self.running = False
        self._shutdown_event.set()


# --- Configuration and Main ---


def create_config_from_args() -> Config:
    """Create configuration from command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enhanced Computer Vision Perception System"
    )

    parser.add_argument(
        "--camera-index", type=str, default="0", help="Camera index or video file path"
    )
    parser.add_argument(
        "--server-address", type=str, default="localhost", help="Remote server address"
    )
    parser.add_argument(
        "--server-port", type=int, default=5002, help="Remote server port"
    )
    parser.add_argument(
        "--model", type=str, default="yolo11n-seg.pt", help="YOLO model file name"
    )
    parser.add_argument(
        "--publisher", type=str, default="tcp://*:5555", help="ZeroMQ publisher address"
    )
    parser.add_argument(
        "--targets", type=str, nargs="+", default=["mouse"], help="Target class names"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.25, help="Confidence threshold"
    )
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--max-fps", type=float, default=30.0, help="Maximum FPS")
    parser.add_argument("--visualize", action="store_true", help="Enable visualization")
    parser.add_argument(
        "--viewer-only",
        action="store_true",
        help="Publish raw frames without loading a model or running inference.",
    )
    parser.add_argument(
        "--publish-annotated",
        action="store_true",
        help="Publish annotated frames instead of raw frames",
    )
    parser.add_argument(
        "--continue-on-stop",
        action="store_true",
        help="Continue processing when recording stops (default: pause)",
    )
    parser.add_argument(
        "--recording-timeout",
        type=float,
        default=30.0,
        help="Timeout for recording state changes",
    )
    parser.add_argument(
        "--save-detection-segments",
        action="store_true",
        help="Save MP4 segments when target classes are detected.",
    )
    parser.add_argument(
        "--detection-segment-targets",
        type=str,
        nargs="+",
        default=["animal", "car", "person"],
        help="Target labels for segment recording (supports animal/car/person aliases).",
    )
    parser.add_argument(
        "--detection-segment-output-dir",
        type=str,
        default="",
        help="Directory for saved detection segments.",
    )
    parser.add_argument(
        "--detection-segment-prebuffer-sec",
        type=float,
        default=2.0,
        help="Seconds of video to include before first detection.",
    )
    parser.add_argument(
        "--detection-segment-postbuffer-sec",
        type=float,
        default=3.0,
        help="Seconds of video to keep after last detection.",
    )
    parser.add_argument(
        "--detection-segment-min-duration-sec",
        type=float,
        default=1.0,
        help="Minimum clip duration to keep on disk.",
    )
    parser.add_argument(
        "--detection-segment-max-duration-sec",
        type=float,
        default=120.0,
        help="Maximum clip duration before forced rollover.",
    )

    args = parser.parse_args()

    # Convert camera index
    camera_index = (
        int(args.camera_index) if args.camera_index.isdigit() else args.camera_index
    )

    is_pose_model = False if args.viewer_only else "pose" in args.model.lower()

    return Config(
        camera_index=camera_index,
        camera_id="camera0",
        server_address=args.server_address,
        server_port=args.server_port,
        model_base_name=args.model,
        publisher_address=args.publisher,
        target_behaviors=args.targets,
        confidence_threshold=args.confidence,
        frame_width=args.width,
        frame_height=args.height,
        max_fps=args.max_fps,
        visualize=args.visualize,
        publish_annotated_frames=False if args.viewer_only else args.publish_annotated,
        viewer_only=bool(args.viewer_only),
        pause_on_recording_stop=not args.continue_on_stop,
        recording_state_timeout=args.recording_timeout,
        enable_segmentation=not is_pose_model,
        enable_pose=is_pose_model,
        save_detection_segments=False
        if args.viewer_only
        else bool(args.save_detection_segments),
        detection_segment_targets=list(args.detection_segment_targets or []),
        detection_segment_output_dir=str(args.detection_segment_output_dir or ""),
        detection_segment_prebuffer_sec=max(
            0.0, float(args.detection_segment_prebuffer_sec)
        ),
        detection_segment_postbuffer_sec=max(
            0.0, float(args.detection_segment_postbuffer_sec)
        ),
        detection_segment_min_duration_sec=max(
            0.0, float(args.detection_segment_min_duration_sec)
        ),
        detection_segment_max_duration_sec=max(
            1.0, float(args.detection_segment_max_duration_sec)
        ),
    )


async def main():
    """Main entry point with enhanced error handling."""
    perception = None
    try:
        config = create_config_from_args()

        logger.info("🚀 Starting Enhanced Computer Vision Perception System")
        logger.info(
            f"📋 Recording behavior: {'Pause on stop' if config.pause_on_recording_stop else 'Continue on stop'}"
        )

        perception = PerceptionProcess(config)

        # Setup signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig, lambda: asyncio.create_task(perception.shutdown())
            )

        # Run the perception process
        await perception.run()

    except KeyboardInterrupt:
        logger.info("⏹️  Application interrupted by user")
    except Exception as e:
        logger.error(f"❌ Critical error: {e}", exc_info=True)
    finally:
        if perception:
            await perception.shutdown()
        logger.info("👋 Application terminated")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
