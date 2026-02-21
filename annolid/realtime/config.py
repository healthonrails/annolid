from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Union


@dataclass
class Config:
    """Lightweight realtime configuration shared by GUI and perception process."""

    camera_index: Union[int, str] = 0
    server_address: str = "localhost"
    server_port: int = 5002
    model_base_name: str = "yolo11n-seg.pt"
    publisher_address: str = "tcp://*:5555"
    target_behaviors: List[str] = field(default_factory=lambda: ["mouse"])
    confidence_threshold: float = 0.25
    frame_width: int = 640
    frame_height: int = 480
    max_fps: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 5.0
    visualize: bool = False
    remote_connect_timeout: float = 2.0
    remote_retry_cooldown: float = 10.0
    remote_retry_max_cooldown: float = 60.0
    local_no_frame_tolerance: int = 12
    local_reconnect_cooldown: float = 2.0
    pause_on_recording_stop: bool = True
    recording_state_timeout: float = 30.0
    publish_frames: bool = True
    frame_encoding: str = "jpg"
    frame_quality: int = 80
    publish_annotated_frames: bool = False
    enable_segmentation: bool = True
    mask_confidence_threshold: float = 0.5
    mask_encoding: str = "rle"
    enable_pose: bool = False
