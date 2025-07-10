import zmq
import asyncio
import numpy as np
import cv2
from typing import Optional, List, Union, Tuple
from ultralytics import YOLO
from pathlib import Path
from dataclasses import dataclass
import argparse
from collections import deque
import statistics
import time
from queue import Queue, Empty
import socket
import select
from ffpyplayer.pic import Image, SWScale
from threading import Thread
import traceback
import json
import sys

from annolid.utils.logger import logger
from annolid.realtime.play_network import RemoteData


@dataclass
class Config:
    """Configuration for the PerceptionProcess."""
    camera_index: Union[int, str] = 0
    server_address: str = "localhost"
    server_port: int = 5002
    model_base_name: str = "yolo11n-seg"
    publisher_address: str = "tcp://*:5555"
    target_behaviors: List[str] = None
    confidence_threshold: float = 0.2
    frame_width: int = 640
    frame_height: int = 480
    max_fps: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 2.0
    visualize: bool = False
    timeout: float = 0.01


class RemoteVideoPlayer(RemoteData):
    """Client for receiving images from a Filers server."""

    def __init__(self, server: str, port: int, timeout: float = 0.01):
        super().__init__()
        self.server = server
        self.port = port
        self.timeout = timeout
        self._client_active = False
        self._listener_thread: Thread | None = None
        self._to_main_thread_queue: Queue | None = None
        self._from_main_thread_queue: Queue | None = None

    def process_exception(self, e, exec_info, from_thread: bool = False):
        logger.error(
            f"RemoteVideoPlayer exception (from_thread={from_thread}): {e}", exc_info=exec_info)

    def process_recording_state(self, state: bool):
        logger.info(
            f"RemoteVideoPlayer recording state: {'started' if state else 'stopped'}")

    def process_image(self, image: Image, metadata: dict) -> np.ndarray:
        """Converts ffpyplayer.Image to OpenCV format."""
        sws = SWScale(*image.get_size(),
                      image.get_pixel_format(), ofmt='bgr24')
        img = sws.scale(image)
        img_np = np.frombuffer(img.to_bytearray()[0], dtype=np.uint8)
        img_np = img_np.reshape((image.get_size()[1], image.get_size()[0], 3))
        return img_np, metadata

    def _listener_run(self, _from_main_thread_queue, _to_main_thread_queue):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        server_address = (self.server, self.port)
        logger.info(
            f"RemoteVideoPlayer connecting to {server_address[0]}:{server_address[1]}")
        msg_len, msg_buff = (), b''
        try:
            sock.connect(server_address)
            done = False
            while not done:
                r, _, _ = select.select([sock], [], [], self.timeout)
                if r:
                    msg_len, msg_buff, msg, value = self.read_msg(
                        sock, msg_len, msg_buff)
                    if msg is not None:
                        _to_main_thread_queue.put((msg, value))
                try:
                    while True:
                        msg, value = _from_main_thread_queue.get_nowait()
                        if msg == 'eof':
                            done = True
                            break
                        else:
                            self.send_msg(sock, msg, value)
                except Empty:
                    pass
        except Exception as e:
            exc_info = ''.join(traceback.format_exception(*sys.exc_info()))
            _to_main_thread_queue.put(('exception_exit', (str(e), exc_info)))
        finally:
            logger.info("RemoteVideoPlayer closing socket")
            sock.close()

    def start_listener(self):
        if self._listener_thread is not None:
            return
        self._client_active = True
        self._from_main_thread_queue = Queue()
        self._to_main_thread_queue = Queue()
        self._listener_thread = Thread(
            target=self._listener_run,
            args=(self._from_main_thread_queue, self._to_main_thread_queue)
        )
        self._listener_thread.start()

    def stop_listener(self, join=True):
        self.stop()
        if self._listener_thread is None:
            return
        self._from_main_thread_queue.put(('eof', None))
        if join:
            self._listener_thread.join()
        self._listener_thread = self._to_main_thread_queue = self._from_main_thread_queue = None
        self._client_active = False

    def play(self):
        self._send_message_to_server('started_playing', None)

    def stop(self):
        self._send_message_to_server('stopped_playing', None)

    def _send_message_to_server(self, key, value):
        if self._from_main_thread_queue is None:
            return
        self._from_main_thread_queue.put((key, value))

    async def get_frame(self) -> Tuple[bool, Optional[np.ndarray], Optional[dict]]:
        """Fetches a frame from the queue, returning success, image, and metadata."""
        while self._to_main_thread_queue is not None:
            try:
                msg, value = self._to_main_thread_queue.get(block=False)
                if msg == 'exception' or msg == 'exception_exit':
                    e, exec_info = value
                    self.process_exception(
                        e, exec_info, msg == 'exception_exit')
                    return False, None, None
                elif msg == 'started_recording':
                    self.process_recording_state(True)
                    return False, None, None
                elif msg == 'stopped_recording':
                    self.process_recording_state(False)
                    return False, None, None
                elif msg == 'image':
                    plane_buffers, pix_fmt, size, linesize, metadata = value
                    sws = SWScale(*size, pix_fmt, ofmt='bgr24')
                    img = Image(plane_buffers=plane_buffers,
                                pix_fmt=pix_fmt, size=size, linesize=linesize)
                    frame, metadata = self.process_image(img, metadata)
                    return True, frame, metadata
                else:
                    logger.warning(f"Unknown message: {msg}, {value}")
            except Empty:
                break
        return False, None, None


class CameraSource:
    """Handles local webcam capture."""

    def __init__(self, source: Union[int, str], width: int, height: int):
        self.source = source
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None

    async def connect(self) -> bool:
        logger.info(f"Connecting to local camera: {self.source}")
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            logger.error(f"Failed to open camera: {self.source}")
            return False
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        logger.info(
            f"Connected to camera with {self.width}x{self.height} resolution")
        return True

    async def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.cap or not self.cap.isOpened():
            return False, None
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()


class HybridSource:
    """Manages remote and local sources with automatic fallback."""

    def __init__(self, config: Config):
        self.config = config
        self.remote = RemoteVideoPlayer(
            config.server_address, config.server_port, config.timeout)
        self.local = CameraSource(
            config.camera_index, config.frame_width, config.frame_height)
        self.use_remote = True
        self.last_remote_failure = 0.0
        self.remote_cooldown = 10.0  # Seconds before retrying remote

    async def connect(self) -> bool:
        self.remote.start_listener()
        if await self.local.connect():
            return True
        logger.error("Local camera connection failed")
        return False

    async def read_frame(self) -> Tuple[bool, Optional[np.ndarray], Optional[dict]]:
        if self.use_remote and time.time() - self.last_remote_failure > self.remote_cooldown:
            success, frame, metadata = await self.remote.get_frame()
            if success:
                return True, frame, metadata
            self.use_remote = False
            self.last_remote_failure = time.time()
            logger.warning("Switching to local camera due to remote failure")
        ret, frame = await self.local.read_frame()
        return ret, frame, None

    async def cleanup(self):
        self.remote.stop_listener()
        self.local.release()


class PerceptionProcess:
    """Processes frames from a hybrid source for YOLOv11 segmentation."""

    def __init__(self, config: Config):
        self.config = config
        self.model: Optional[YOLO] = None
        self.source: Optional[HybridSource] = None
        self.context: Optional[zmq.Context] = None
        self.socket: Optional[zmq.Socket] = None
        self.class_names: Optional[List[str]] = None
        self.running = True
        self.metrics = {
            'fps': deque(maxlen=100),
            'latency': deque(maxlen=100),
            'detections': 0
        }

    async def _setup(self):
        for attempt in range(self.config.retry_attempts):
            try:
                self.model = YOLO(f"{self.config.model_base_name}.pt")
                self.class_names = self.model.names
                logger.info(
                    f"Loaded YOLO model: {self.config.model_base_name}.pt")
                break
            except Exception as e:
                logger.error(
                    f"Model load attempt {attempt+1}/{self.config.retry_attempts} failed: {e}")
                if attempt == self.config.retry_attempts - 1:
                    raise
                await asyncio.sleep(self.config.retry_delay)

        for attempt in range(self.config.retry_attempts):
            try:
                self.context = zmq.Context()
                self.socket = self.context.socket(zmq.PUB)
                self.socket.bind(self.config.publisher_address)
                logger.info(
                    f"ZMQ publisher bound to {self.config.publisher_address}")
                break
            except zmq.ZMQError as e:
                logger.error(
                    f"ZMQ bind attempt {attempt+1}/{self.config.retry_attempts} failed: {e}")
                if attempt == self.config.retry_attempts - 1:
                    raise
                await asyncio.sleep(self.config.retry_delay)

        self.source = HybridSource(self.config)
        if not await self.source.connect():
            raise RuntimeError("Failed to initialize source")

    async def run(self):
        await self._setup()
        frame_count = 0
        last_log_time = time.time()
        frame_interval = 1.0 / self.config.max_fps

        while self.running:
            try:
                start_time = time.time()
                ret, frame, metadata = await self.source.read_frame()
                if not ret:
                    logger.warning("Failed to grab frame. Reconnecting...")
                    await self.source.connect()
                    continue

                if time.time() - start_time > frame_interval:
                    logger.debug("Skipping frame to maintain FPS")
                    continue

                inference_start = time.time()
                results = self.model(frame, stream=True,
                                     conf=self.config.confidence_threshold)
                inference_time = time.time() - inference_start
                self.metrics['latency'].append(inference_time)

                for result in results:
                    await self._process_and_publish(result, start_time, metadata)

                if self.config.visualize:
                    annotated_frame = result.plot()
                    cv2.putText(
                        annotated_frame,
                        f"FPS: {int(1/(time.time()-start_time))}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    cv2.imshow("YOLOv11 Segmentation", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.running = False

                frame_count += 1
                current_time = time.time()
                if current_time - last_log_time >= 5.0:
                    fps = frame_count / (current_time - last_log_time)
                    self.metrics['fps'].append(fps)
                    avg_latency = statistics.mean(
                        self.metrics['latency']) if self.metrics['latency'] else 0
                    logger.info(json.dumps({
                        "fps": f"{fps:.2f}",
                        "avg_latency_ms": f"{avg_latency*1000:.2f}",
                        "detections": self.metrics['detections'],
                        "source": "remote" if self.source.use_remote else "local"
                    }))
                    frame_count = 0
                    last_log_time = current_time

                await asyncio.sleep(max(0, frame_interval - (time.time() - start_time)))

            except Exception as e:
                logger.error(f"Error in perception loop: {e}", exc_info=True)
                await asyncio.sleep(1)

        await self._cleanup()

    async def _process_and_publish(self, result, timestamp: float, metadata: Optional[dict]):
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        masks = result.masks.data.cpu().numpy() if result.masks is not None else None

        target_class_ids = [
            self.class_names.index(behavior)
            for behavior in (self.config.target_behaviors or [self.config.target_behavior])
            if behavior in self.class_names
        ]

        for i, class_id in enumerate(class_ids):
            if class_id in target_class_ids and scores[i] >= self.config.confidence_threshold:
                self.metrics['detections'] += 1
                detection_message = {
                    "behavior": self.class_names[class_id],
                    "confidence": float(scores[i]),
                    "bbox": boxes[i].tolist(),
                    "timestamp": timestamp,
                    "mask": masks[i].tolist() if masks is not None and len(masks) > i else [],
                    "metadata": metadata or {}
                }
                self.socket.send_string("detections", flags=zmq.SNDMORE)
                self.socket.send_json(detection_message)
                break

    async def _cleanup(self):
        logger.info("Cleaning up perception process...")
        await self.source.cleanup()
        if self.socket and not self.socket.closed:
            self.socket.close()
        if self.context and not self.context.closed:
            self.context.term()
        if self.config.visualize:
            cv2.destroyAllWindows()
        logger.info("Cleanup complete.")


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="YOLOv11 Segmentation with Hybrid Source")
    parser.add_argument('--camera-index', type=str, default=0,
                        help="Camera index or video file path")
    parser.add_argument('--server-address', type=str,
                        default="localhost", help="Filers server address")
    parser.add_argument('--server-port', type=int,
                        default=5002, help="Filers server port")
    parser.add_argument('--model', type=str,
                        default="yolo11n-seg", help="YOLO model name")
    parser.add_argument('--address', type=str,
                        default="tcp://*:5555", help="ZeroMQ publisher address")
    parser.add_argument('--targets', type=str, nargs='+',
                        default=['person'], help="Target class names")
    parser.add_argument('--conf', type=float, default=0.2,
                        help="Confidence threshold")
    parser.add_argument('--width', type=int, default=640, help="Frame width")
    parser.add_argument('--height', type=int, default=480, help="Frame height")
    parser.add_argument('--max-fps', type=float,
                        default=30.0, help="Maximum FPS")
    parser.add_argument('--visualize', action='store_true',
                        help="Enable visualization")
    args = parser.parse_args()
    return Config(
        camera_index=args.camera_index,
        server_address=args.server_address,
        server_port=args.server_port,
        model_base_name=args.model,
        publisher_address=args.address,
        target_behaviors=args.targets,
        confidence_threshold=args.conf,
        frame_width=args.width,
        frame_height=args.height,
        max_fps=args.max_fps,
        visualize=args.visualize
    )


async def main():
    config = parse_args()
    if not Path(f"{config.model_base_name}.pt").is_file():
        logger.info(f"Downloading {config.model_base_name}.pt")
        YOLO(f"{config.model_base_name}.pt")
    perception = PerceptionProcess(config)
    await perception.run()

if __name__ == '__main__':
    asyncio.run(main())
