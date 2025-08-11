import asyncio
import argparse
import json
import signal
import socket
import statistics
import struct
import time
import traceback
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import partial
from itertools import accumulate
from pathlib import Path
from typing import Optional, List, Union, Tuple, Dict, Any, AsyncIterator, Protocol

import cv2
import numpy as np
import zmq
import zmq.asyncio
from ffpyplayer.pic import Image, SWScale
from tree_config.utils import (yaml_loads as orig_yaml_loads,
                               get_yaml,
                               yaml_dumps as orig_yaml_dumps)
from ultralytics import YOLO

from annolid.utils.logger import logger


# --- Configuration and Validation ---

@dataclass
class Config:
    """Configuration for the PerceptionProcess with validation."""
    camera_index: Union[int, str] = 0
    server_address: str = "localhost"
    server_port: int = 5002
    model_base_name: str = "yolo11n-seg.pt"
    publisher_address: str = "tcp://*:5555"
    target_behaviors: List[str] = field(default_factory=lambda: ['person'])
    confidence_threshold: float = 0.25
    frame_width: int = 640
    frame_height: int = 480
    max_fps: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 5.0
    visualize: bool = False
    remote_connect_timeout: float = 2.0
    remote_retry_cooldown: float = 10.0

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.confidence_threshold < 0 or self.confidence_threshold > 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if self.max_fps <= 0:
            raise ValueError("max_fps must be positive")
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("Frame dimensions must be positive")
        if not self.target_behaviors:
            raise ValueError("At least one target behavior must be specified")


# --- Protocols for Better Type Safety ---

class FrameSource(Protocol):
    """Protocol for frame sources."""

    async def connect(self) -> bool: ...
    async def get_frame(self) -> Optional[Tuple[np.ndarray,
                                                Dict[str, Any]]]: ...

    async def disconnect(self) -> None: ...


class DetectionResult:
    """Structured detection result."""

    def __init__(self, behavior: str,
                 confidence: float,
                 bbox: List[float],
                 timestamp: float,
                 metadata: Dict[str, Any]):
        self.behavior = behavior
        self.confidence = confidence
        self.bbox = bbox
        self.timestamp = timestamp
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        return {
            "behavior": self.behavior,
            "confidence": self.confidence,
            "bbox_normalized": self.bbox,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


# --- Network Protocol Handling ---

class NetworkProtocolHandler:
    """Handles network message encoding/decoding with improved error handling."""

    def __init__(self):
        self.yaml_loads = self._create_yaml_loads_fix()
        self.yaml_dumps = partial(orig_yaml_dumps, get_yaml_obj=get_yaml)

    def _create_yaml_loads_fix(self):
        """Creates a fixed version of yaml_loads for binary data."""
        def _yaml_loads_fixed(value):
            if len(value) >= 12 and value.startswith('!!binary |\n') and value[11] != ' ':
                value = value[:11] + ' ' + value[11:]
            return orig_yaml_loads(value, get_yaml_obj=get_yaml)
        return _yaml_loads_fixed

    def encode_msg(self, msg: str, value) -> bytes:
        """Encodes a message into bytes for network transmission."""
        try:
            bin_data = []
            if msg == 'image':
                image, metadata = value
                bin_data = image.to_bytearray()
                data = self.yaml_dumps((
                    'image', (list(map(len, bin_data)), image.get_pixel_format(),
                              image.get_size(), image.get_linesizes(), metadata)))
                data = data.encode('utf8')
            else:
                data = self.yaml_dumps((msg, value))
                data = data.encode('utf8')

            header = struct.pack('>II', len(data), sum(map(len, bin_data)))
            return header + data + b''.join(bin_data)
        except Exception as e:
            logger.error(f"Failed to encode message '{msg}': {e}")
            raise

    def decode_data(self, msg_buff: bytes, msg_len: Tuple[int, int]):
        """Decodes buffer data with validation."""
        n, bin_n = msg_len
        if not (n + bin_n == len(msg_buff)):
            raise ValueError(
                f"Buffer length mismatch: expected {n + bin_n}, got {len(msg_buff)}")

        try:
            data_str = msg_buff[:n].decode('utf8')
            msg, value = self.yaml_loads(data_str)

            if msg == 'image':
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


# --- Remote Video Source ---

class AsyncRemoteVideoPlayer(NetworkProtocolHandler):
    """Async remote video client with improved state management and error handling."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.frame_queue = asyncio.Queue(maxsize=5)
        self._listener_task: Optional[asyncio.Task] = None
        self._is_active = False
        self._playing = False  # Track play state
        self._recording = False  # Track server recording state
        self._got_exception = False  # Track exception state
        self._connection_lock = asyncio.Lock()
        self._play_state_event = asyncio.Event()  # Event to signal play state changes
        self._play_state_event.set()  # Initially set to allow processing

    async def connect(self) -> bool:
        """Connect to remote server with proper error handling."""
        async with self._connection_lock:
            if self._is_active:
                return True

            logger.info(
                f"Connecting to {self.config.server_address}:{self.config.server_port}")
            try:
                conn_future = asyncio.open_connection(
                    self.config.server_address, self.config.server_port)
                self.reader, self.writer = await asyncio.wait_for(
                    conn_future, timeout=self.config.remote_connect_timeout)

                self._is_active = True
                self._listener_task = asyncio.create_task(
                    self._listen_for_messages())
                # Request to start playing
                await self._send_message('started_playing', None)
                logger.info("Successfully connected to remote server")
                return True

            except (OSError, socket.gaierror, asyncio.TimeoutError) as e:
                await self.process_exception(str(e), traceback.format_exc(), from_thread=False)
                await self._cleanup_connection()
                return False

    async def process_exception(self, error: str, exec_info: str, from_thread: bool = False):
        """Handle exceptions from server or internal operations."""
        self._got_exception = True
        logger.error(
            f"{'Thread' if from_thread else 'Server'} exception: {error}\n{exec_info}")

    async def process_recording_state(self, state: bool):
        """Handle server recording state changes."""
        self._recording = state
        logger.info(f"Server {'started' if state else 'stopped'} recording")

    async def process_image(self, raw_frame_data, timestamp: float):
        """Process incoming image data."""
        if self._playing and not self.frame_queue.full():
            try:
                self.frame_queue.put_nowait((raw_frame_data, timestamp))
                logger.debug(f"Queued image at {timestamp}")
            except asyncio.QueueFull:
                logger.debug("Frame queue full, dropping image")

    async def _listen_for_messages(self):
        """Listen for incoming messages with robust error handling."""
        try:
            while self._is_active and self.reader and not self.reader.at_eof():
                try:
                    # Read message header
                    header_data = await asyncio.wait_for(
                        self.reader.readexactly(8), timeout=5.0)
                    msg_len = struct.unpack('>II', header_data)

                    # Read message body
                    total_size = sum(msg_len)
                    msg_buff = b''
                    if total_size > 0:
                        msg_buff = await asyncio.wait_for(
                            self.reader.readexactly(total_size), timeout=10.0)

                    msg, value = self.decode_data(msg_buff, msg_len)
                    await self._handle_message(msg, value)

                except asyncio.TimeoutError:
                    logger.warning("Message receive timeout")
                    break
                except (asyncio.IncompleteReadError, ConnectionResetError) as e:
                    await self.process_exception(str(e), traceback.format_exc(), from_thread=False)
                    break
                except Exception as e:
                    await self.process_exception(str(e), traceback.format_exc(), from_thread=True)
                    break

        except Exception as e:
            await self.process_exception(str(e), traceback.format_exc(), from_thread=True)
        finally:
            await self.disconnect()

    async def _handle_message(self, msg: str, value):
        """Handle different message types."""
        if msg == 'image':
            await self.process_image(value, time.time())
        elif msg == 'started_recording':
            await self.process_recording_state(True)
        elif msg == 'stopped_recording':
            await self.process_recording_state(False)
        elif msg == 'started_playing':
            logger.info("Received started_playing signal")
            self._playing = True
            self._play_state_event.set()
        elif msg == 'stopped_playing':
            logger.info("Received stopped_playing signal")
            self._playing = False
            self._play_state_event.clear()
        elif msg == 'exception':
            logger.error(f"Server sent exception: {value}")
            await self.process_exception(value, "", from_thread=False)
        else:
            logger.warning(f"Unknown message received: {msg}")

    async def get_frame(self) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Get frame with timeout handling, respecting play state."""
        if not self._playing or not self._recording:
            await self._play_state_event.wait()  # Wait until both playing and recording
            return None

        try:
            raw_frame_data, timestamp = await asyncio.wait_for(
                self.frame_queue.get(), timeout=1.0)
            frame, metadata = self._process_raw_frame(raw_frame_data)
            metadata['capture_timestamp'] = timestamp
            metadata['source'] = 'remote'
            return frame, metadata
        except asyncio.TimeoutError:
            return None

    def _process_raw_frame(self, raw_frame_data) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Process raw frame data into numpy array."""
        try:
            plane_buffers, pix_fmt, size, linesize, metadata = raw_frame_data
            img = Image(plane_buffers=plane_buffers, pix_fmt=pix_fmt,
                        size=size, linesize=linesize)
            sws = SWScale(*img.get_size(),
                          img.get_pixel_format(), ofmt='bgr24')
            img_bgr = sws.scale(img)
            frame = np.frombuffer(
                img_bgr.to_bytearray()[0], dtype=np.uint8
            ).reshape((size[1], size[0], 3))
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
                await self.process_exception(str(e), traceback.format_exc(), from_thread=True)

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

            # Clear frame queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

            logger.info("Remote video player disconnected")


# --- Local Camera Source ---

class CameraSource:
    """Local camera source with improved error handling."""

    def __init__(self, config: Config):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None
        self._lock = asyncio.Lock()

    async def connect(self) -> bool:
        """Connect to local camera."""
        async with self._lock:
            if self.cap and self.cap.isOpened():
                return True

            logger.info(f"Connecting to camera: {self.config.camera_index}")
            try:
                self.cap = await asyncio.to_thread(self._init_camera)
                if self.cap and self.cap.isOpened():
                    logger.info("Successfully connected to local camera")
                    return True
                else:
                    logger.error("Failed to open camera")
                    return False
            except Exception as e:
                logger.error(f"Camera connection error: {e}")
                return False

    def _init_camera(self) -> Optional[cv2.VideoCapture]:
        """Initialize camera in thread."""
        try:
            source = int(self.config.camera_index) if str(
                self.config.camera_index).isdigit() else self.config.camera_index
            cap = cv2.VideoCapture(source)

            if cap.isOpened():
                # Set camera properties
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
                # Reduce buffer for lower latency
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            else:
                cap.release()
                return None
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            return None

    async def get_frame(self) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Read frame from camera."""
        if not self.cap or not self.cap.isOpened():
            return None

        try:
            ret, frame = await asyncio.to_thread(self.cap.read)
            if ret and frame is not None:
                return frame, {"source": "local", "capture_timestamp": time.time()}
            return None
        except Exception as e:
            logger.error(f"Frame read error: {e}")
            return None

    async def disconnect(self):
        """Release camera resources."""
        async with self._lock:
            if self.cap:
                await asyncio.to_thread(self.cap.release)
                self.cap = None
                logger.info("Camera released")


# --- Source State Management ---

class SourceState(Enum):
    DISCONNECTED = auto()
    TRYING_REMOTE = auto()
    USING_REMOTE = auto()
    TRYING_LOCAL = auto()
    USING_LOCAL = auto()


class HybridVideoSource:
    """Hybrid video source with improved state management."""

    def __init__(self, config: Config):
        self.config = config
        self.remote = AsyncRemoteVideoPlayer(config)
        self.local = CameraSource(config)
        self.state = SourceState.DISCONNECTED
        self.last_remote_attempt = 0.0
        self._state_lock = asyncio.Lock()

    async def connect(self):
        """Initialize connection with fallback strategy."""
        async with self._state_lock:
            # Try remote first
            self.state = SourceState.TRYING_REMOTE
            if await self.remote.connect():
                self.state = SourceState.USING_REMOTE
                logger.info("Connected to remote source")
                return

            # Fallback to local
            self.state = SourceState.TRYING_LOCAL
            if await self.local.connect():
                self.state = SourceState.USING_LOCAL
                logger.info("Connected to local source")
                return

            self.state = SourceState.DISCONNECTED
            raise RuntimeError("Failed to connect to any video source")

    async def get_frame(self) -> Optional[Tuple[np.ndarray, Dict[str, Any]]]:
        """Get frame with automatic fallback and recovery."""
        # Try remote source
        if self.state == SourceState.USING_REMOTE:
            result = await self.remote.get_frame()
            if result:
                return result

            # Remote failed, switch to local
            logger.warning("Remote source failed, falling back to local")
            await self.remote.disconnect()
            async with self._state_lock:
                self.state = SourceState.TRYING_LOCAL

        # Try local source
        if self.state == SourceState.USING_LOCAL:
            result = await self.local.get_frame()
            if result:
                # Periodically try to reconnect to remote
                await self._try_remote_reconnect()
                return result
            else:
                logger.error("Local source failed")
                async with self._state_lock:
                    self.state = SourceState.DISCONNECTED

        # Handle disconnected state
        if self.state in (SourceState.TRYING_LOCAL, SourceState.DISCONNECTED):
            async with self._state_lock:
                if self.state == SourceState.TRYING_LOCAL:
                    if await self.local.connect():
                        self.state = SourceState.USING_LOCAL
                    else:
                        self.state = SourceState.DISCONNECTED
                elif self.state == SourceState.DISCONNECTED:
                    logger.info("Attempting to recover connection...")
                    await self.connect()

        return None

    async def _try_remote_reconnect(self):
        """Periodically attempt remote reconnection."""
        current_time = time.time()
        if (current_time - self.last_remote_attempt > self.config.remote_retry_cooldown):
            self.last_remote_attempt = current_time
            logger.info("Attempting remote reconnection...")

            if await self.remote.connect():
                logger.info("Successfully reconnected to remote source")
                async with self._state_lock:
                    self.state = SourceState.USING_REMOTE

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
        self.last_report_time = time.time()
        self.error_count = 0

    def record_frame(self, inference_time: float, detection_count: int):
        """Record metrics for a processed frame."""
        self.latency_history.append(inference_time)
        self.detection_count += detection_count
        self.frame_count += 1

    def record_error(self):
        """Record an error occurrence."""
        self.error_count += 1

    def should_report(self) -> bool:
        """Check if it's time to generate a report."""
        return time.time() - self.last_report_time >= self.report_interval

    def generate_report(self, source_state: str) -> Dict[str, Any]:
        """Generate performance report."""
        current_time = time.time()
        elapsed = current_time - self.last_report_time

        fps = self.frame_count / elapsed if elapsed > 0 else 0
        self.fps_history.append(fps)

        avg_latency = statistics.mean(
            self.latency_history) if self.latency_history else 0
        avg_fps = statistics.mean(self.fps_history) if self.fps_history else 0

        report = {
            "current_fps": f"{fps:.2f}",
            "average_fps": f"{avg_fps:.2f}",
            "avg_latency_ms": f"{avg_latency * 1000:.2f}",
            "total_detections": self.detection_count,
            "error_count": self.error_count,
            "source_state": source_state,
            "frames_processed": self.frame_count
        }

        # Reset counters
        self.frame_count = 0
        self.last_report_time = current_time

        return report


# --- Detection Publisher ---

class DetectionPublisher:
    """ZMQ publisher for detection results."""

    def __init__(self, address: str):
        self.address = address
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(address)
        logger.info(f"Detection publisher bound to {address}")

    async def publish_detection(self, detection: DetectionResult):
        """Publish a detection result."""
        try:
            await self.socket.send_string("detections", flags=zmq.SNDMORE)
            await self.socket.send_json(detection.to_dict())
        except Exception as e:
            logger.error(f"Failed to publish detection: {e}")

    async def cleanup(self):
        """Clean up publisher resources."""
        try:
            self.socket.close()
            self.context.term()
            logger.info("Detection publisher cleaned up")
        except Exception as e:
            logger.error(f"Publisher cleanup error: {e}")


# --- Main Perception Process ---

class PerceptionProcess:
    """Main perception process orchestrator."""

    def __init__(self, config: Config):
        self.config = config
        self.model: Optional[YOLO] = None
        self.class_names: Optional[List[str]] = None
        self.video_source = HybridVideoSource(config)
        self.metrics = PerformanceMetrics()
        self.publisher = DetectionPublisher(config.publisher_address)
        self.running = True
        self._shutdown_event = asyncio.Event()

    @asynccontextmanager
    async def _model_context(self) -> AsyncIterator[YOLO]:
        """Context manager for YOLO model."""
        try:
            logger.info(f"Loading YOLO model: {self.config.model_base_name}")
            model = await asyncio.to_thread(YOLO, self.config.model_base_name)
            self.class_names = model.names
            logger.info("YOLO model loaded successfully")
            yield model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        finally:
            logger.info("Model context closed")

    async def setup(self):
        """Setup the perception process."""
        logger.info("Setting up perception process...")

        # Verify model exists or download
        model_path = Path(self.config.model_base_name)
        if not model_path.is_file():
            logger.info(f"Model '{model_path}' not found, downloading...")
            await asyncio.to_thread(YOLO, self.config.model_base_name)

        # Initialize video source
        await self.video_source.connect()
        logger.info("Setup complete")

    async def run(self):
        """Main processing loop with play state handling."""
        await self.setup()
        frame_interval = 1.0 / self.config.max_fps

        async with self._model_context() as model:
            self.model = model

            while self.running and not self._shutdown_event.is_set():
                loop_start = time.time()
                try:
                    # Get frame
                    frame_data = await self.video_source.get_frame()
                    if not frame_data:
                        await asyncio.sleep(0.1)
                        continue

                    frame, metadata = frame_data
                    results = await self._run_inference(frame)
                    inference_time = time.time() - loop_start
                    detection_count = await self._process_detections(results, loop_start, metadata)
                    self.metrics.record_frame(inference_time, detection_count)

                    if self.metrics.should_report():
                        report = self.metrics.generate_report(
                            self.video_source.state.name)
                        logger.info(f"Performance: {json.dumps(report)}")

                    if self.config.visualize:
                        self._visualize_results(results, frame)

                except Exception as e:
                    logger.error(f"Processing error: {e}", exc_info=True)
                    self.metrics.record_error()

                elapsed = time.time() - loop_start
                await asyncio.sleep(max(0, frame_interval - elapsed))

    async def _run_inference(self, frame: np.ndarray):
        """Run YOLO inference on frame."""
        try:
            results = await asyncio.to_thread(
                self.model, frame,
                stream=False,
                conf=self.config.confidence_threshold,
                verbose=False
            )
            return results[0]
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise

    async def _process_detections(self, result, timestamp: float, metadata: Dict[str, Any]) -> int:
        """Process detection results and publish."""
        if not result.boxes or len(result.boxes) == 0:
            return 0

        detection_count = 0
        boxes = result.boxes

        for i in range(len(boxes)):
            try:
                class_id = int(boxes.cls[i])
                class_name = self.class_names[class_id]

                if class_name in self.config.target_behaviors:
                    detection = DetectionResult(
                        behavior=class_name,
                        confidence=float(boxes.conf[i]),
                        bbox=boxes.xyxyn[i].cpu().numpy().tolist(),
                        timestamp=timestamp,
                        metadata=metadata
                    )

                    await self.publisher.publish_detection(detection)
                    detection_count += 1

            except Exception as e:
                logger.error(f"Detection processing error: {e}")

        return detection_count

    def _visualize_results(self, result, frame: np.ndarray):
        """Visualize detection results."""
        try:
            annotated_frame = result.plot()

            # Add performance info
            fps_text = f"FPS: {self.metrics.fps_history[-1]:.1f}" if self.metrics.fps_history else "FPS: N/A"
            source_text = f"Source: {self.video_source.state.name}"

            cv2.putText(annotated_frame, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, source_text, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Perception System", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                self._shutdown_event.set()

        except Exception as e:
            logger.error(f"Visualization error: {e}")

    async def shutdown(self):
        """Graceful shutdown."""
        if not self.running:
            return

        logger.info("Shutting down perception process...")
        self.running = False
        self._shutdown_event.set()

        try:
            await self.video_source.cleanup()
            await self.publisher.cleanup()

            if self.config.visualize:
                cv2.destroyAllWindows()

            logger.info("Shutdown complete")
        except Exception as e:
            logger.error(f"Shutdown error: {e}")


# --- Configuration and Main ---

def create_config_from_args() -> Config:
    """Create configuration from command line arguments."""
    parser = argparse.ArgumentParser(
        description="Computer Vision Perception System")

    parser.add_argument('--camera-index', type=str, default='0',
                        help="Camera index or video file path")
    parser.add_argument('--server-address', type=str,
                        default="localhost", help="Remote server address")
    parser.add_argument('--server-port', type=int,
                        default=5002, help="Remote server port")
    parser.add_argument('--model', type=str,
                        default="yolo11n-seg.pt", help="YOLO model file name")
    parser.add_argument('--publisher', type=str,
                        default="tcp://*:5555", help="ZeroMQ publisher address")
    parser.add_argument('--targets', type=str, nargs='+',
                        default=['person'], help="Target class names")
    parser.add_argument('--confidence', type=float,
                        default=0.25, help="Confidence threshold")
    parser.add_argument('--width', type=int, default=640, help="Frame width")
    parser.add_argument('--height', type=int, default=480, help="Frame height")
    parser.add_argument('--max-fps', type=float,
                        default=30.0, help="Maximum FPS")
    parser.add_argument('--visualize', action='store_true',
                        help="Enable visualization")

    args = parser.parse_args()

    # Convert camera index
    camera_index = int(
        args.camera_index) if args.camera_index.isdigit() else args.camera_index

    return Config(
        camera_index=camera_index,
        server_address=args.server_address,
        server_port=args.server_port,
        model_base_name=args.model,
        publisher_address=args.publisher,
        target_behaviors=args.targets,
        confidence_threshold=args.confidence,
        frame_width=args.width,
        frame_height=args.height,
        max_fps=args.max_fps,
        visualize=args.visualize
    )


async def main():
    """Main entry point."""
    try:
        config = create_config_from_args()
        perception = PerceptionProcess(config)

        # Setup signal handlers
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig, lambda: asyncio.create_task(perception.shutdown()))

        # Run the perception process
        await perception.run()

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.error(f"Critical error: {e}", exc_info=True)
    finally:
        if 'perception' in locals():
            await perception.shutdown()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
