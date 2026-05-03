# Realtime Detection with Wireless Cameras (GUI)

This guide shows how to run realtime YOLO detection from USB cameras, RTSP streams, and MJPEG URLs, then save MP4 event segments when selected objects are detected.

## Open Realtime Control

In Annolid desktop, open the realtime window and use the control panel on the left.

The panel now has two tabs:

1. `Setup`
2. `Run & Output`

## Setup Tab

### 1. Select a model

- Choose a preset YOLO model from **Preset**, or
- Select **Custom…** and browse to your own `*.pt`, `*.onnx`, `*.engine`, or `*.mlpackage` model.
- Enable **View only (no detection or inference)** when you only want to watch a camera or video stream. In this mode Annolid does not load a model, run detections, send bot detection reports, or save detection-triggered clips.

### 2. Enter camera source

In **Camera / Stream**, you can use:

- USB camera index: `0`
- RTSP: `rtsp://camera-host:554/stream1`
- RTP multicast: `rtp://@239.0.0.1:5004`
- MJPEG URL: `http://camera-host/img/video.mjpeg`
- Local video file path

For many IP cameras that expose a control page such as `http://camera.local/img/main.cgi?next_file=main.htm`, Annolid automatically normalizes it to `.../img/video.mjpeg`.

### 3. Configure transport and endpoints

- **RTSP Transport**: `Auto` (default), `TCP`, or `UDP`
- **Publisher Bind** and **Subscriber Address**: keep defaults unless you use a custom pipeline.
- **Target Behaviours**: optional comma-separated labels to focus reporting.

## Run & Output Tab

### 1. Tune realtime performance

- **Frame Width / Height**
- **Max FPS**
- **Confidence** threshold

### 2. Choose output behavior

- **Publish frames to GUI**
- **Send annotated frames**
- Optional bot digest/email reporting

### 3. Save MP4 on detections

Enable **Save MP4 segments on detections** and set:

- **Segment Targets**: default `animal,car,person`
- **Segment Output**: folder path for event clips
- **Prebuffer**: seconds before first detection
- **Postbuffer**: seconds after last detection
- **Min Duration / Max Duration**: clip length bounds

This records only when detections match the target classes.

## Start and Stop

- Click **Start Realtime** to launch
- Check status text for errors or runtime state
- Click **Stop** to end realtime inference or viewing

## Recommended Wireless Camera Preset

Use this as a starting point for IP cameras:

- Source: `http://camera-host/img/video.mjpeg`
- Max FPS: `15`
- Confidence: `0.25`
- Save segments: enabled
- Segment targets: `animal,car,person`
- Pre/Post buffer: `2.0 / 3.0`

## Troubleshooting

- Stream does not open:
  - verify camera URL in browser first,
  - try RTSP with `TCP` transport,
  - check firewall/network route from Annolid host to camera.
- Black or stuttering frames:
  - lower **Frame Width/Height**,
  - reduce **Max FPS**,
  - prefer `TCP` for unstable Wi‑Fi.
- No MP4 clips saved:
  - confirm **Save MP4 segments on detections** is enabled,
  - ensure detected class names match **Segment Targets**,
  - verify output folder is writable.
- High latency:
  - reduce input resolution and FPS,
  - use wired Ethernet for camera uplink when possible.

## Related

- [Video Processing with FFmpeg](video_ffmpeg_processing.md)
- [Tutorials Overview](../tutorials.md)
