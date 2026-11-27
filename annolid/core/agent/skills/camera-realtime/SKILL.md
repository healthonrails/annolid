---
name: camera-realtime
description: Operate live camera/RTSP/RTP streams, realtime inference, eye-blink detection, snapshots, and bot stream alerts.
metadata: '{"annolid":{"always":false,"tags":["camera","realtime","stream","rtsp","rtp","webcam","snapshot","blink","live","inference"]}}'
---

# Camera Realtime Control

Use these tools for camera/video stream setup, snapshots, and realtime inference automation in Annolid.

## Realtime Tools

- `gui_list_realtime_models`: List available realtime model presets before starting.
- `gui_check_stream_source`: Probe camera/stream connectivity, optionally save a snapshot, and optionally email it.
- `gui_start_realtime_stream`: Start realtime inference with source, model, viewer, transport, eye-blink options, and bot-report settings.
- `gui_get_realtime_status`: Verify stream is running and inspect active source/model/viewer.
- `gui_list_realtime_logs`: Retrieve detection and bot-event log file paths.
- `gui_stop_realtime_stream`: Stop the active realtime stream cleanly when done.
- `camera_snapshot`: Non-GUI camera probe + snapshot saver for headless/email/chat channels.
- `automation_schedule`: Schedule recurring `camera_check` tasks (`action="add"`, `task_type="camera_check"`).

## Recommended Flow

1. Call `gui_list_realtime_models` and pick the target model preset.
2. Call `gui_check_stream_source` first to verify connectivity.
3. Call `gui_start_realtime_stream` with explicit `camera_source`, `model_name`, `viewer_type` (`"pyqt"` or `"threejs"`), and `rtsp_transport="tcp"` for unstable RTSP links.
   - For face/eye-blink monitoring, pass `classify_eye_blinks=true` (optional `blink_ear_threshold`, `blink_min_consecutive_frames`).
4. Confirm with `gui_get_realtime_status`.
5. For diagnostics, call `gui_list_realtime_logs` (or `gui_list_logs` / `gui_read_log_file`).
6. Call `gui_stop_realtime_stream` when finished.

## Snapshot + Email Workflow

For "check camera, save snapshot, and email it" requests:

1. In GUI sessions, call `gui_check_stream_source` with:
   - `camera_source`
   - `save_snapshot=true`
   - `email_to=<recipient>`
   - optional `email_subject`, `email_content`
2. Confirm returned fields (`ok`, `snapshot_path`, `snapshot_opened_on_canvas`, `email_sent`, `email_result`).
3. If `ok=false` or `snapshot_path` is empty, report the probe failure and do not claim email was sent.

### Headless / Non-GUI Channel Fallback

When GUI tools are unavailable (for example in email or CLI channels):

1. Call `camera_snapshot` with explicit `camera_source`.
2. Pass the returned `snapshot_path` in `attachment_paths` to the `email` tool.
3. Report the saved path, email status, and any actionable error.

## Stream Tips & Bot Reporting

- Local webcam: `camera_source="0"` (or another camera index).
- RTSP: full URL with `rtsp_transport="tcp"` when frames drop.
- RTP multicast: full `rtp://@host:port` URL; verify network path first.
- Periodic bot reporting via `gui_start_realtime_stream`: `bot_report_enabled`, `bot_report_interval_sec` (keep `>= 5.0` for high-FPS streams), `bot_watch_labels`, `bot_email_report`, `bot_email_to`.
