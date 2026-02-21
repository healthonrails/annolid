---
name: Camera Realtime Control
description: Operate camera streams and realtime inference with model selection, status checks, and logging tools.
---

# Camera Realtime Control

Use these tools for camera/video stream setup and realtime inference automation in Annolid.

## Realtime Tools

- `gui_list_realtime_models`: List available realtime model presets before starting.
- `gui_check_stream_source`: Probe camera/stream connectivity before realtime start.
- `gui_start_realtime_stream`: Start realtime stream with source/model/transport and optional bot-report settings.
- `gui_get_realtime_status`: Verify stream is running and inspect active source/model/viewer.
- `gui_list_realtime_logs`: Retrieve detection and bot-event log file paths.
- `gui_stop_realtime_stream`: Stop stream cleanly when done.

## Recommended Flow

1. Call `gui_list_realtime_models` and choose model.
2. Call `gui_check_stream_source` first.
3. Call `gui_start_realtime_stream` with explicit source and `rtsp_transport` (`tcp` for unstable RTSP).
4. Confirm with `gui_get_realtime_status`.
5. For diagnostics, call `gui_list_realtime_logs`.
6. Call `gui_stop_realtime_stream` after analysis.

## Stream Tips

- Local camera: use `camera_source="0"` (or another index).
- RTSP: use full URL and set `rtsp_transport="tcp"` if frames are unstable.
- RTP multicast: use full `rtp://@host:port` source and validate receiver/network path first.

## Bot Reporting Controls

When starting realtime stream, use:

- `bot_report_enabled`
- `bot_report_interval_sec`
- `bot_watch_labels` (CSV or list)
- `bot_email_report`
- `bot_email_to`

Keep reporting intervals conservative (`>=5s`) for high-FPS streams.
