# Tutorials

This page points to practical tutorial material that exists in the repository today. Use [Getting Started](getting_started.md) first if you have not yet verified that Annolid opens a video and saves annotations in your environment.

## Choose a Tutorial Path

| If you need to... | Use |
| --- | --- |
| Learn the basic GUI tracking loop | [Tracking four interacting mice with one labeled frame](https://youtu.be/PNbPA649r78) and [Workflows](workflows.md) |
| Downsample or preprocess videos | [GUI video downsample workflow](tutorials/video_downsample_workflow.md) and [Video Processing with FFmpeg](tutorials/video_ffmpeg_processing.md) |
| Score behavior events | [Behavior labeling with Timeline, Flags, and Annolid Bot](tutorials/behavior_timeline_flags_bot.md) |
| Define zones and export zone metrics | [Draw zones quickstart](tutorials/draw_zones_quickstart.md) and [Zone analysis workflow](tutorials/zone_analysis_workflow.md) |
| Correct tracking drift or missing frames | [Segment-based batch tracking](tutorials/segment-based-batch-tracking.md) and [Tracking correction with SAM3 Agent and Annolid Bot](tutorials/tracking_correction_with_sam3_bot.md) |
| Train or evaluate models | Notebook tutorials in `docs/tutorials/` and model-specific `annolid-run help` output |
| Run realtime camera workflows | [Realtime Wireless Camera Detection](tutorials/realtime_wireless_camera_detection.md) and [Multi-Camera Realtime Detection](tutorials/multi_camera_realtime.md) |

## Video Walkthroughs

- [Tracking four interacting mice with one labeled frame](https://youtu.be/PNbPA649r78)
- [Annolid YouTube channel](https://www.youtube.com/@annolid)
- [Annolid playlist](https://www.youtube.com/embed/videoseries?list=PLYp4D9Y-8_dRXPOtfGu48W5ENtfKn-Owc)

## Focused Markdown Tutorials

- [GUI video downsample workflow (single video + batch overrides)](tutorials/video_downsample_workflow.md)
- [Behavior labeling with Timeline, Flags, and Annolid Bot](tutorials/behavior_timeline_flags_bot.md)
- [Segment-based batch tracking](tutorials/segment-based-batch-tracking.md)
- [Tracking correction with SAM3 Agent and Annolid Bot](tutorials/tracking_correction_with_sam3_bot.md)
- [Draw zones quickstart (GUI)](tutorials/draw_zones_quickstart.md)
- [Zone analysis workflow](tutorials/zone_analysis_workflow.md)
- [TCN behavior classification from pose features](tutorials/tcn_behavior_pipeline.md)
- [Multi-camera realtime detection](tutorials/multi_camera_realtime.md)
- [LabelMe to YOLO class mapping](tutorials/Labelme_to_YOLO_class_map.md)
- [Sandboxed shell for Annolid Bot](tutorials/sandboxed_shell.md)
- [MCP setup and usage](mcp.md)
- [Video Depth Anything](video_depth_anything.md)
- [SAM 3D integration](sam3d.md)
- [TAPNext ONNX point tracking](tapnext.md)
- [CoWTracker point tracking](cowtracker.md)
- [Large TIFF and atlas overlays](atlas_overlay_workflow.md)

## Notebook Tutorials in `docs/tutorials/`

Representative notebooks currently tracked in the repo:

- `docs/tutorials/Extract_frames_from_a_video.ipynb`
- `docs/tutorials/yolov8_tracking_tutorial.ipynb`
- `docs/tutorials/Annolid_video_batch_inference.ipynb`
- `docs/tutorials/Annolid_model_evaluation.ipynb`
- `docs/tutorials/Annolid_Pose_Estimation_on_YOLO_Tutorial.ipynb`
- `docs/tutorials/Annolid_Instance_Segmentation_on_YOLO_Tutorial.ipynb`
- `docs/tutorials/YOLO_SAHI_inference_for_ultralytics.ipynb`
- `docs/tutorials/zero_shot_object_detection_and_tracking_with_grounding_dino.ipynb`
- `docs/tutorials/RAFT_optical_flow.ipynb`
- `docs/tutorials/Annolid_behavior_video_classification_on_slowfast.ipynb`

## How To Choose

- Start with the GUI/video workflow if you are labeling or reviewing data interactively.
- Use the notebook tutorials when you need training, evaluation, or post-processing examples.
- Use the markdown guides when you need an operational setup such as MCP, sandboxed shell execution, or batch tracking.
- Use `annolid-run help train <model>` or `annolid-run help predict <model>` when a tutorial mentions a model backend and you need the current command-line options.

## Repository-first Note

Some older docs still link to historical book content, but the current maintained tutorial sources for this repo live under `docs/` and `docs/tutorials/`.
