from __future__ import annotations

import asyncio

from annolid.gui.realtime_launch import (
    build_multi_camera_realtime_launch_payloads,
    build_realtime_launch_payload,
)
from annolid.realtime.perception import DetectionPublisher, DetectionResult


def test_realtime_launch_payload_includes_gdrive_auto_upload_extras() -> None:
    _cfg, extras = build_realtime_launch_payload(
        camera_source="0",
        model_name="yolo11n",
        gdrive_auto_upload_enabled=True,
        gdrive_auto_upload_delay_sec=12,
        gdrive_auto_upload_remote_folder="annolid/realtime_detect",
        gdrive_auto_upload_skip_if_exists=False,
    )
    assert extras["gdrive_auto_upload_enabled"] is True
    assert extras["gdrive_auto_upload_delay_sec"] == 12.0
    assert extras["gdrive_auto_upload_remote_folder"] == "annolid/realtime_detect"
    assert extras["gdrive_auto_upload_skip_if_exists"] is False


def test_multi_camera_realtime_payloads_assign_isolated_ports_and_outputs() -> None:
    sessions = build_multi_camera_realtime_launch_payloads(
        cameras=[
            {"name": "front cam", "source": "0"},
            {"name": "side-cam", "source": "rtsp://camera.local/live"},
        ],
        base_publisher_address="tcp://*:5600",
        output_root="runs/realtime",
        model_name="yolo11n",
        target_behaviors=["mouse"],
        save_detection_segments=True,
    )

    assert len(sessions) == 2
    first_cfg, first_extras = sessions[0]
    second_cfg, second_extras = sessions[1]
    assert first_cfg.camera_id == "front_cam"
    assert second_cfg.camera_id == "side-cam"
    assert first_cfg.camera_index == 0
    assert second_cfg.camera_index == "rtsp://camera.local/live"
    assert first_cfg.publisher_address == "tcp://*:5600"
    assert second_cfg.publisher_address == "tcp://*:5601"
    assert first_extras["subscriber_address"] == "tcp://127.0.0.1:5600"
    assert second_extras["subscriber_address"] == "tcp://127.0.0.1:5601"
    assert first_cfg.detection_segment_output_dir == "runs/realtime/front_cam"
    assert second_cfg.detection_segment_output_dir == "runs/realtime/side-cam"
    assert first_extras["multi_camera"] is True


def test_multi_camera_realtime_payloads_respect_explicit_endpoints() -> None:
    sessions = build_multi_camera_realtime_launch_payloads(
        cameras=[
            {
                "id": "arena_a",
                "camera_source": 2,
                "publisher_address": "tcp://*:5705",
                "subscriber_address": "tcp://127.0.0.1:5705",
                "detection_segment_output_dir": "custom/arena_a",
            }
        ],
        base_publisher_address="tcp://*:5600",
        output_root="runs/realtime",
    )

    cfg, extras = sessions[0]
    assert cfg.camera_id == "arena_a"
    assert cfg.camera_index == 2
    assert cfg.publisher_address == "tcp://*:5705"
    assert extras["subscriber_address"] == "tcp://127.0.0.1:5705"
    assert cfg.detection_segment_output_dir == "custom/arena_a"


def test_realtime_publisher_adds_camera_id_to_payloads() -> None:
    class _FakeSocket:
        def __init__(self) -> None:
            self.json_payloads = []

        async def send_string(self, *_args, **_kwargs) -> None:
            return None

        async def send_json(self, payload, **_kwargs) -> None:
            self.json_payloads.append(payload)

    publisher = DetectionPublisher("tcp://*:5999", camera_id="arena_a")
    fake_socket = _FakeSocket()
    publisher.socket = fake_socket
    detection = DetectionResult(
        behavior="mouse",
        confidence=0.9,
        bbox_normalized=[0.1, 0.2, 0.3, 0.4],
        bbox_pixels=[10, 20, 30, 40],
        timestamp=1.0,
        metadata={"frame_index": 7},
    )

    try:
        asyncio.run(publisher.publish_detection(detection))
        asyncio.run(publisher.publish_status({"event": "ok"}))
    finally:
        publisher.context.term()

    assert fake_socket.json_payloads[0]["metadata"]["camera_id"] == "arena_a"
    assert fake_socket.json_payloads[1]["camera_id"] == "arena_a"
