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
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import partial
from itertools import accumulate
from pathlib import Path
from typing import Optional, List, Union, Tuple

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


# --- Configuration Dataclass ---
@dataclass
class Config:
    """Configuration for the PerceptionProcess."""
    camera_index: Union[int, str] = 0
    server_address: str = "localhost"
    server_port: int = 5002
    model_base_name: str = "yolov11-seg.pt"
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

# --- Network Protocol Handling (Integrated from original RemoteData) ---


class NetworkProtocolHandler:
    """Handles sending and receiving messages according to the custom protocol."""

    def __init__(self):
        # This wrapper includes the bug-fix from the original code.
        self.yaml_loads = self._create_yaml_loads_fix()
        self.yaml_dumps = partial(orig_yaml_dumps, get_yaml_obj=get_yaml)

    def _create_yaml_loads_fix(self):
        def _yaml_loads_fixed(value):
            if len(value) >= 12 and value.startswith('!!binary |\n') and value[11] != ' ':
                value = value[:11] + ' ' + value[11:]
            return orig_yaml_loads(value, get_yaml_obj=get_yaml)
        return _yaml_loads_fixed

    def encode_msg(self, msg: str, value) -> bytes:
        """Encodes a message into bytes for sending over the network."""
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

    def decode_data(self, msg_buff: bytes, msg_len: Tuple[int, int]):
        """Decodes buffer data received from the network."""
        n, bin_n = msg_len
        if not (n + bin_n == len(msg_buff)):
            raise ValueError("Buffer length does not match message length")

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

# --- Native Asyncio Network Client ---


class AsyncRemoteVideoPlayer(NetworkProtocolHandler):
    """A native asyncio client for receiving images."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self.frame_queue = asyncio.Queue(maxsize=5)
        self._listener_task: Optional[asyncio.Task] = None
        self._is_active = False

    async def connect(self) -> bool:
        logger.info(
            f"Connecting to remote server at {self.config.server_address}:{self.config.server_port}")
        try:
            conn_future = asyncio.open_connection(
                self.config.server_address, self.config.server_port)
            self.reader, self.writer = await asyncio.wait_for(conn_future,
                                                              timeout=self.config.remote_connect_timeout)

            self._is_active = True
            self._listener_task = asyncio.create_task(
                self._listen_for_messages())
            await self._send_message_to_server('started_playing', None)
            logger.info("Successfully connected to remote server.")
            return True
        except (OSError, socket.gaierror, asyncio.TimeoutError) as e:
            logger.error(f"Failed to connect to remote server: {e}")
            await self.disconnect()
            return False

    async def _listen_for_messages(self):
        try:
            while self._is_active and self.reader and not self.reader.at_eof():
                header_data = await self.reader.readexactly(8)
                msg_len = struct.unpack('>II', header_data)

                total_size = sum(msg_len)
                msg_buff = await self.reader.readexactly(total_size) if total_size > 0 else b''

                msg, value = self.decode_data(msg_buff, msg_len)

                if msg == 'image':
                    if not self.frame_queue.full():
                        self.frame_queue.put_nowait((value, time.time()))
                elif msg in ('started_recording', 'stopped_recording'):
                    logger.info(f"Server recording state changed: {msg}")
        except (asyncio.IncompleteReadError, ConnectionResetError) as e:
            logger.warning(f"Connection lost to remote server: {e}")
        except Exception as e:
            logger.error(
                f"Unexpected error in network listener: {e}",
                exc_info=traceback.format_exc())
        finally:
            await self.disconnect()

    async def get_frame(self) -> Optional[Tuple[np.ndarray, dict]]:
        try:
            raw_frame_data, timestamp = await asyncio.wait_for(self.frame_queue.get(),
                                                               timeout=1.0)
            frame, metadata = self._process_raw_frame(raw_frame_data)
            metadata['capture_timestamp'] = timestamp
            return frame, metadata
        except asyncio.TimeoutError:
            return None

    def _process_raw_frame(self, raw_frame_data) -> Tuple[np.ndarray, dict]:
        plane_buffers, pix_fmt, size, linesize, metadata = raw_frame_data
        img = Image(plane_buffers=plane_buffers, pix_fmt=pix_fmt,
                    size=size, linesize=linesize)
        sws = SWScale(*img.get_size(), img.get_pixel_format(), ofmt='bgr24')
        img_bgr = sws.scale(img)
        frame = np.frombuffer(img_bgr.to_bytearray()[
                              0], dtype=np.uint8).reshape((size[1], size[0], 3))
        return frame, metadata

    async def _send_message_to_server(self, key, value):
        if self.writer and not self.writer.is_closing():
            message_bytes = self.encode_msg(key, value)
            self.writer.write(message_bytes)
            await self.writer.drain()

    async def disconnect(self):
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
        if self.writer:
            self.writer.close()
            await self.writer.wait_closed()
        self.writer = self.reader = None
        logger.info("Remote video player disconnected.")

# --- Local Camera Handling ---


class CameraSource:
    def __init__(self, config: Config):
        self.config = config
        self.cap: Optional[cv2.VideoCapture] = None

    async def connect(self) -> bool:
        logger.info(f"Connecting to local camera: {self.config.camera_index}")
        self.cap = await asyncio.to_thread(self._init_cap)
        if not self.cap or not self.cap.isOpened():
            logger.error(f"Failed to open camera: {self.config.camera_index}")
            self.cap = None
            return False
        return True

    def _init_cap(self):
        try:
            source = int(self.config.camera_index)
        except ValueError:
            source = self.config.camera_index
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height)
        return cap

    async def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.cap:
            return False, None
        return await asyncio.to_thread(self.cap.read)

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

# --- Hybrid Source with State Machine ---


class SourceState(Enum):
    DISCONNECTED = auto()
    TRYING_REMOTE = auto()
    USING_REMOTE = auto()
    TRYING_LOCAL = auto()
    USING_LOCAL = auto()


class HybridSource:
    def __init__(self, config: Config):
        self.config = config
        self.remote = AsyncRemoteVideoPlayer(config)
        self.local = CameraSource(config)
        self.state = SourceState.DISCONNECTED
        self.last_remote_attempt = 0.0

    async def connect(self):
        self.state = SourceState.TRYING_REMOTE
        if await self.remote.connect():
            self.state = SourceState.USING_REMOTE
            return
        self.state = SourceState.TRYING_LOCAL
        if await self.local.connect():
            self.state = SourceState.USING_LOCAL
            return
        self.state = SourceState.DISCONNECTED
        raise RuntimeError(
            "Failed to connect to both remote and local sources.")

    async def read_frame(self) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        if self.state == SourceState.USING_REMOTE:
            result = await self.remote.get_frame()
            if result:
                return result
            logger.warning("Remote source failed. Falling back to local.")
            await self.remote.disconnect()
            self.state = SourceState.TRYING_LOCAL

        if self.state == SourceState.USING_LOCAL:
            ret, frame = await self.local.read_frame()
            if ret:
                # Check for periodic remote reconnection attempt
                if time.time() - self.last_remote_attempt > self.config.remote_retry_cooldown:
                    logger.info("Attempting to reconnect to remote source...")
                    self.last_remote_attempt = time.time()
                    if await self.remote.connect():
                        logger.info(
                            "Successfully reconnected to remote source. Switching over.")
                        self.state = SourceState.USING_REMOTE
                return frame, {"source": "local"}
            else:
                logger.error("Local source failed. Disconnecting.")
                self.state = SourceState.DISCONNECTED
                return None, None

        if self.state == SourceState.TRYING_LOCAL:
            if await self.local.connect():
                self.state = SourceState.USING_LOCAL
            else:
                self.state = SourceState.DISCONNECTED
            return None, None

        if self.state == SourceState.DISCONNECTED:
            logger.info("Source is disconnected. Attempting to recover...")
            await self.connect()

        return None, None

    async def cleanup(self):
        await self.remote.disconnect()
        self.local.release()

# --- Encapsulated Metrics Manager ---


class MetricsManager:
    def __init__(self):
        self.fps = deque(maxlen=100)
        self.latency = deque(maxlen=100)
        self.detections = 0
        self.frame_count = 0
        self.last_log_time = time.time()

    def update(self, inference_time: float, num_detections: int):
        self.latency.append(inference_time)
        self.detections += num_detections
        self.frame_count += 1

    def report(self, current_source_state: str) -> Optional[dict]:
        current_time = time.time()
        elapsed = current_time - self.last_log_time
        if elapsed < 5.0:
            return None

        fps = self.frame_count / elapsed
        self.fps.append(fps)
        avg_latency = statistics.mean(self.latency) if self.latency else 0

        report_data = {
            "fps": f"{fps:.2f}",
            "avg_latency_ms": f"{avg_latency*1000:.2f}",
            "total_detections": self.detections,
            "source": current_source_state
        }

        self.frame_count = 0
        self.last_log_time = current_time
        return report_data

# --- Main Perception Process Orchestrator ---


class PerceptionProcess:
    def __init__(self, config: Config):
        self.config = config
        self.model: Optional[YOLO] = None
        self.class_names: Optional[List[str]] = None
        self.source = HybridSource(config)
        self.metrics = MetricsManager()
        self.running = True
        self.context = zmq.asyncio.Context()
        self.socket = self.context.socket(zmq.PUB)

    async def setup(self):
        logger.info("Setting up perception process...")
        self.model = YOLO(self.config.model_base_name)
        self.class_names = self.model.names
        logger.info(f"Loaded YOLO model: {self.config.model_base_name}")

        self.socket.bind(self.config.publisher_address)
        logger.info(f"ZMQ publisher bound to {self.config.publisher_address}")

        await self.source.connect()
        logger.info("Setup complete.")

    async def run(self):
        await self.setup()
        frame_interval = 1.0 / self.config.max_fps

        while self.running:
            loop_start_time = time.time()

            frame, metadata = await self.source.read_frame()
            if frame is None:
                await asyncio.sleep(0.1)
                continue

            results = await asyncio.to_thread(
                self.model, frame, stream=False,
                conf=self.config.confidence_threshold,
                verbose=False
            )
            inference_time = time.time() - loop_start_time

            num_detections = await self._process_and_publish(results[0],
                                                             loop_start_time, metadata)
            self.metrics.update(inference_time, num_detections)

            if report := self.metrics.report(self.source.state.name):
                logger.info(json.dumps(report))

            if self.config.visualize:
                # --- FIX IS HERE ---
                # Call _visualize directly without to_thread.
                # The method is now a standard `def` method.
                self._visualize(results[0])

            elapsed = time.time() - loop_start_time
            await asyncio.sleep(max(0, frame_interval - elapsed))

    async def _process_and_publish(self,
                                   result,
                                   timestamp: float,
                                   metadata: Optional[dict]) -> int:
        boxes = result.boxes
        num_detections = 0
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i])
            class_name = self.class_names[class_id]

            if class_name in self.config.target_behaviors:
                num_detections += 1
                detection_message = {
                    "behavior": class_name, "confidence": float(boxes.conf[i]),
                    "bbox_normalized": boxes.xyxyn[i].cpu().numpy().tolist(),
                    "timestamp": timestamp, "metadata": metadata or {}
                }
                await self.socket.send_string("detections", flags=zmq.SNDMORE)
                await self.socket.send_json(detection_message)
        return num_detections

    # --- FIX IS HERE ---
    # This is now a regular method, not an async one, and it's safe to call
    # from the main event loop because waitKey(1) is non-blocking.
    def _visualize(self, result):
        annotated_frame = result.plot()
        fps_text = f"FPS: {self.metrics.fps[-1]:.2f}" if self.metrics.fps else "FPS: N/A"
        cv2.putText(annotated_frame, fps_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("YOLO Segmentation", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.running = False

    async def shutdown(self):
        if not self.running:
            return
        logger.info("Shutting down perception process...")
        self.running = False
        await self.source.cleanup()
        self.socket.close()
        self.context.term()
        if self.config.visualize:
            cv2.destroyAllWindows()
        logger.info("Cleanup complete.")

# --- Argument Parsing and Main Execution ---


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="YOLO Segmentation with Hybrid Source")
    parser.add_argument('--camera-index', type=str, default='0',
                        help="Camera index or video file path")
    parser.add_argument('--server-address', type=str,
                        default="localhost", help="Filers server address")
    parser.add_argument('--server-port', type=int,
                        default=5002, help="Filers server port")
    parser.add_argument('--model', type=str,
                        default="yolo11n-seg.pt", help="YOLO model file name")
    parser.add_argument('--address', type=str,
                        default="tcp://*:5555", help="ZeroMQ publisher address")
    parser.add_argument('--targets', type=str, nargs='+',
                        default=['person'], help="Target class names")
    parser.add_argument('--conf', type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument('--width', type=int, default=640, help="Frame width")
    parser.add_argument('--height', type=int, default=480, help="Frame height")
    parser.add_argument('--max-fps', type=float,
                        default=30.0, help="Maximum FPS")
    parser.add_argument('--visualize', action='store_true',
                        help="Enable visualization")
    args = parser.parse_args()

    # Try to convert camera_index to int if it's a digit
    camera_index = int(
        args.camera_index) if args.camera_index.isdigit() else args.camera_index

    return Config(
        camera_index=camera_index,
        server_address=args.server_address, server_port=args.server_port,
        model_base_name=args.model, publisher_address=args.address,
        target_behaviors=args.targets, confidence_threshold=args.conf,
        frame_width=args.width, frame_height=args.height,
        max_fps=args.max_fps, visualize=args.visualize
    )


async def main():
    config = parse_args()
    model_path = Path(config.model_base_name)
    if not model_path.is_file():
        logger.info(
            f"Model '{model_path}' not found. Attempting to download...")
        try:
            YOLO(config.model_base_name)
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            return

    perception = PerceptionProcess(config)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(
            sig, lambda: asyncio.create_task(perception.shutdown()))

    try:
        await perception.run()
    except Exception as e:
        logger.error(
            f"Critical error in main run loop: {e}", exc_info=traceback.format_exc())
    finally:
        await perception.shutdown()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application interrupted by user.")
