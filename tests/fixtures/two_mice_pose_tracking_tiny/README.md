# Two-Mouse Pose and Tracking Fixture

This deterministic synthetic fixture contains 12 sequential 640x480
JPEG frames at 5 FPS. Every frame has two COCO keypoint annotations with stable
`track_id` values (`1` and `2`) and 13 anatomical landmarks.

Files:

- `annotations/sequence.json`: the uninterrupted sequence for tracking tests.
- `annotations/train.json`: frames 0-7 for pose-loader tests.
- `annotations/val.json`: frames 8-11 for validation tests.
- `coco_spec.yaml`: Annolid COCO keypoint dataset configuration.
- `manifest.json`: deterministic generation parameters.

COCO visibility is `2` for a landmark projected inside the image and `0` outside.
Labels are geometric ground truth and do not attempt pixel-level self-occlusion
classification. The nonstandard `video_id`, `frame_id`, and `track_id` fields provide
the temporal identity contract used by keypoint-tracking tests.

Each image record also carries contact-quality ground truth:
`minimum_subject_clearance`, `subject_overlap`, `minimum_tail_clearance`,
`tail_overlap`, `maximum_foot_slip`, and `maximum_paw_ground_error`. These fields
guard against body/tail intersections and stance sliding in generated sequences.

Regenerate from the repository root:

```bash
source .venv/bin/activate
python scripts/generate_two_mice_pose_dataset.py
```

Generation seed: `20260710`.
