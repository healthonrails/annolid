# Step-by-Step Tutorial: From Annolid Behavior Labeling to a YOLO Pose Dataset

Use this walkthrough to move seamlessly from interactive behavior labeling in Annolid to a YOLO-ready pose dataset with stable class IDs. The process ties directly into Annolid’s project schema, so the behaviors you flag in the GUI end up with matching IDs in your training data and downstream analytics.

## Overview of the Workflow

1. Configure behavior definitions in the Annolid project schema.
2. Label frames or videos in the Annolid GUI.
3. Convert them to Ultralytics YOLO format. The converter reuses the schema automatically, or you can provide a custom class map if needed.
4. Inspect the generated dataset and continue with training or analysis.

---

## Step 0 – Prerequisites

- Annolid installed with GUI and CLI access.
- Video files or extracted frames you plan to label.
- Optional: an existing `project.annolid.json/.yaml` file describing your behaviors, modifiers, and keyboard bindings.

Keep the schema file in (or above) the directory where you store LabelMe annotations—Annolid tools now auto-discover it to lock in class IDs.

---

## Step 1 – Configure Behaviors in the Annolid GUI

1. Launch Annolid and open **File → Project Schema…**.
2. Load your existing schema (`project.annolid.json`) or create a new one. The `code` field of each behavior becomes the primary identifier everywhere (flags, converter, analytics).
3. Assign colors, categories, modifiers, and key bindings if you plan to use keyboard-driven labeling.
4. Save the schema near your project assets (for example, root of the project or the folder that will hold LabelMe annotations).

As soon as a schema is loaded, the **Pinned Flags** panel synchronizes to the same behavior list automatically—no more juggling the legacy `configs/behaviors.yaml` file.

---

## Step 2 – Label Behaviors in Annolid

1. Open a video or an image sequence.
2. Confirm the behavior flags panel reflects the schema from Step 1. If not, reload the schema.
3. Label behaviors using your configured shortcuts. The flags controller enforces exclusivity rules and updates modifier toggles automatically.
4. When satisfied, export to LabelMe:
   - Choose **File → Export → LabelMe JSON…** (or trigger the equivalent CLI command).
   - Annolid writes paired `frame_XXXXX.png` and `frame_XXXXX.json` files per labeled frame.

You now have a LabelMe-compatible dataset with behavior codes already aligned to the project schema.

---

## Step 3 – Optional: Refine in LabelMe

If you want to adjust annotations manually or add pose keypoints:

1. Open the exported dataset in the LabelMe desktop app.
2. For each instance, make sure the **label** (or `flags.display_label`) matches the behavior `code` defined in Step 1.
3. Add polygons or keypoints as needed. The converter groups shapes by instance label and preserves keypoint order.

Keep the folder’s contents clean—only `*.json` + `*.png` pairs belong in this directory.

---

## Step 4 – Convert to YOLO Format

### CLI Path

```bash
python -m annolid.main \
  --labelme2yolo /path/to/labelme_dataset \
  --val_size 0.1 \
  --test_size 0.1
```

- `--labelme2yolo` points at the folder filled with LabelMe annotations.
- The converter walks up from that folder until it finds `project.annolid.json/.yaml`, then uses the schema’s behavior order as the class map. No extra flag required.
- Supply `--labelme2yolo-class-map` (JSON/YAML/TXT) only if you need to override the schema or you are working outside Annolid.
- `--val_size` and `--test_size` are optional fractions for splitting the dataset.

Need a fully automated run? Use the new builder:

```bash
python -m annolid.main \
  --build_pose_dataset /path/to/labelme_dataset \
  --pose_output /path/to/output_dataset \
  --val_size 0.1 \
  --test_size 0.1
```

The builder pulls in your schema automatically, enforces class IDs, writes subject metadata companions, and drops a `dataset_summary.json` so you can review class and subject coverage without opening any files.

### GUI Path

1. In Annolid, open **File → Convert LabelMe to YOLO**.
2. Select the annotation directory.
3. Leave “Optional Class Map” blank to rely on the schema. Provide a file only if you want a custom order.
4. Adjust validation/test split fields if desired.
5. Click **Convert to YOLO Format** and wait for the success message.

Both paths emit the same dataset layout.

---

## Step 5 – Inspect the Output

```
labelme_dataset/
└── YOLO_dataset/
    ├── images/{train,val,test}/
    ├── labels/{train,val,test}/
    └── data.yaml
```

Validate the contents:

1. `labels/train/*.txt`: first number is the class ID defined by the schema, followed by YOLO bbox values and pose keypoints (x, y, visibility).
2. `data.yaml`: the `names` list mirrors your behavior codes; `kpt_shape` and `kpt_labels` reflect the keypoint layout detected from your annotations.
3. Confirm the splits (train/val/test) contain the expected number of files.

Once verified, follow the existing pose training tutorial (`docs/tutorials/Annolid_Pose_Estimation_on_YOLO11_Tutorial.ipynb`) or the IntegraPose workflows to fine-tune YOLO for simultaneous pose and behavior predictions.

---

## Step 6 – Troubleshooting

| Symptom | Likely Cause | Suggested Fix |
| --- | --- | --- |
| `Label 'XYZ' is not defined in the provided class map.` | The annotation references a behavior missing from the schema. | Add the behavior to `project.annolid.json` or adjust the LabelMe label. |
| `Class map indices must be contiguous…` | Manual class map (JSON/YAML/TXT) skips numbers or starts at a non-zero value. | Rewrite the mapping with IDs `0..N-1`. |
| Converter silently adds new classes | No schema or class map was found. | Move `project.annolid.json` into or above the dataset folder, or pass `--labelme2yolo-class-map`. |
| GUI warns about missing PyYAML | You loaded a YAML class map but PyYAML isn’t installed. | `pip install pyyaml`, or use JSON/TXT instead. |
| Keypoints missing or zeroed | Instances in LabelMe lacked those keypoints. | Ensure each instance contains the same keypoint names and re-export. |

---

## Step 7 – Next Steps

- Keep the project schema versioned alongside your dataset—consistency of behavior IDs is crucial for multitask learning and analytics.
- Install pytest and run `python -m pytest tests/test_labelme2yolo_pose.py` to validate conversions in CI.
- Explore downstream analytics (bout smoothing, gait metrics, LSTM refiners) now that the YOLO outputs share the exact class IDs used during labeling.

You’re ready to train multi-task YOLO models (pose + behavior) without ever hand-editing class maps—the schema-driven workflow keeps every tool in sync. Happy labeling!
