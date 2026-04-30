from __future__ import annotations

from annolid.gui.realtime_launch import build_realtime_launch_payload


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
