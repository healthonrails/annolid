# Multi-Camera Realtime Detection

Annolid realtime detection supports one camera per perception worker. For
multi-camera experiments, launch one worker per camera with a stable `camera_id`,
separate ZeroMQ ports, and separate output folders.

Example configuration shape:

```yaml
realtime:
  cameras:
    - name: front_cam
      source: 0
      publisher: tcp://*:5555
      output_dir: runs/realtime/front_cam
    - name: side_cam
      source: rtsp://user:pass@192.168.1.50:554/stream1
      publisher: tcp://*:5556
      output_dir: runs/realtime/side_cam
```

Each camera session publishes `camera_id` in frame metadata, detection metadata,
and status messages. Treat object IDs as camera-local unless the cameras are
hardware synchronized and a separate cross-camera re-identification step is used.

For command-line use, run one process per camera:

```bash
source .venv/bin/activate
python -m annolid.realtime.perception \
  --camera-index 0 \
  --publisher tcp://*:5555 \
  --model yolo11n-seg.pt \
  --targets mouse \
  --save-detection-segments \
  --detection-segment-output-dir runs/realtime/front_cam
```

```bash
source .venv/bin/activate
python -m annolid.realtime.perception \
  --camera-index 'rtsp://user:pass@192.168.1.50:554/stream1' \
  --publisher tcp://*:5556 \
  --model yolo11n-seg.pt \
  --targets mouse \
  --save-detection-segments \
  --detection-segment-output-dir runs/realtime/side_cam
```
