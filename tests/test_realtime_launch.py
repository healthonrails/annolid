from __future__ import annotations

from annolid.gui.realtime_launch import (
    build_realtime_launch_payload,
    parse_camera_source,
    parse_label_csv,
)


def test_parse_camera_source_keeps_rtsp_and_rtp_strings() -> None:
    assert (
        parse_camera_source("rtsp://10.0.0.2:554/stream")
        == "rtsp://10.0.0.2:554/stream"
    )
    assert parse_camera_source("rtp://@239.0.0.1:5004") == "rtp://@239.0.0.1:5004"


def test_build_realtime_launch_payload_adds_rtsp_transport_query() -> None:
    cfg, _extras = build_realtime_launch_payload(
        camera_source="rtsp://10.0.0.2:554/stream1",
        model_name="yolo11n",
        rtsp_transport="tcp",
    )
    assert str(cfg.camera_index).endswith("stream1?rtsp_transport=tcp")


def test_build_realtime_launch_payload_preserves_existing_rtsp_transport_query() -> (
    None
):
    cfg, _extras = build_realtime_launch_payload(
        camera_source="rtsp://10.0.0.2:554/stream1?rtsp_transport=udp",
        model_name="yolo11n",
        rtsp_transport="tcp",
    )
    assert str(cfg.camera_index).endswith("rtsp_transport=udp")


def test_parse_camera_source_normalizes_main_cgi_control_page_to_mjpeg() -> None:
    source = "http://camera.local/img/main.cgi?next_file=main.htm"
    parsed = parse_camera_source(source)
    assert parsed == "http://camera.local/img/video.mjpeg"


def test_parse_label_csv_deduplicates_case_insensitively() -> None:
    assert parse_label_csv("person, animal, Person,  animal  ") == ["person", "animal"]


def test_build_realtime_launch_payload_includes_bot_report_extras() -> None:
    _cfg, extras = build_realtime_launch_payload(
        camera_source=0,
        model_name="yolo11n",
        bot_report_enabled=True,
        bot_report_interval_sec=5,
        bot_watch_labels="person,animal,Person",
        bot_email_report=True,
        bot_email_to="lab@example.com",
    )
    assert extras["bot_report_enabled"] is True
    assert extras["bot_report_interval_sec"] == 5.0
    assert extras["bot_watch_labels"] == ["person", "animal"]
    assert extras["bot_email_report"] is True
    assert extras["bot_email_to"] == "lab@example.com"
