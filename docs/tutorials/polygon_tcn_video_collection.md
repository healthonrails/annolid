# Collect Videos for Polygon TCN Training and Testing

This guide explains how to collect Annolid tracking data from videos and use it
to train and test a TCN behavior classifier. It assumes each tracked frame has
the same named body-part polygons, for example `head`, `thorax`, and `abdomen`.

Use at least two videos:

- one or more training videos,
- one held-out test video that was not used for training.

## 1. Record Compatible Videos

Keep acquisition settings as consistent as possible between training and test
videos:

- same camera view and resolution,
- same animal orientation and arena setup,
- same frame rate when possible,
- same tracked body parts,
- same behavior label set.

For a head-fixed fly polygon TCN workflow, every video used in one experiment
should contain the same polygon labels on every tracked frame:

```text
head
thorax
abdomen
```

Do not mix a video with `head`/`thorax`/`abdomen` polygons with a video that only
has a single full-body `fly1` polygon. That produces incompatible feature
columns and invalid test metrics.

## 2. Track Body-Part Polygons

For each video, create or load the first-frame polygons in Annolid and run video
tracking so Annolid writes:

```text
/path/to/videos/<video_name>/
  <video_name>_000000000.json
  <video_name>_annotations.ndjson
  <video_name>_tracking_stats.json
```

The seed JSON should contain one polygon per body part. The NDJSON file should
then contain tracked polygons for the same labels across frames.

Check the annotation schema before training:

```bash
source .venv/bin/activate
python - <<'PY'
from pathlib import Path
import json
from collections import Counter, defaultdict

video_dir = Path("/path/to/videos/2019_08_20_fly2")
name = video_dir.name
counts = Counter()
frames = defaultdict(set)

for path in sorted(video_dir.glob("*.json")):
    if path.name.endswith("_tracking_stats.json"):
        continue
    data = json.loads(path.read_text())
    frame = data.get("frame")
    if frame is None:
        frame = int(path.stem.split("_")[-1])
    for shape in data.get("shapes") or []:
        label = str(shape.get("label"))
        points = shape.get("points") or []
        if str(shape.get("shape_type", "polygon")).lower() == "polygon" and len(points) >= 3:
            counts[label] += 1
            frames[label].add(int(frame))

ndjson = video_dir / f"{name}_annotations.ndjson"
with ndjson.open() as f:
    for line in f:
        if not line.strip():
            continue
        data = json.loads(line)
        frame = int(data.get("frame", -1))
        for shape in data.get("shapes") or []:
            label = str(shape.get("label"))
            points = shape.get("points") or []
            if str(shape.get("shape_type", "polygon")).lower() == "polygon" and len(points) >= 3:
                counts[label] += 1
                frames[label].add(frame)

print(counts)
for label in ("head", "thorax", "abdomen"):
    seen = frames.get(label, set())
    print(label, len(seen), (min(seen), max(seen)) if seen else None)
PY
```

Expected output should show the required labels and frame coverage for each
label. If a test video is missing a required body part, re-run tracking or choose
a different test video.

## 3. Create Behavior Labels

Create one behavior-label CSV per video. The CSV must have one row per frame and
one one-hot column per behavior:

```text
frame,background,still,walk,front_groom,back_groom,abdomen-move
0,1,0,0,0,0,0
1,1,0,0,0,0,0
...
```

Annolid also accepts an index column such as `Unnamed: 0`. Keep the behavior
columns identical across train and test videos. Put unlabeled frames in
`background` unless you intentionally want to exclude them before training.

Check label counts:

```bash
source .venv/bin/activate
python - <<'PY'
from pathlib import Path
import pandas as pd

label_csv = Path("/path/to/labels/2019_08_20_fly2_labels.csv")
df = pd.read_csv(label_csv)
label_cols = [
    c for c in df.columns
    if not str(c).startswith("Unnamed")
    and str(c).lower() not in {"frame", "frames", "frame_number", "index"}
]
print(df.shape)
print({c: int(df[c].sum()) for c in label_cols})
PY
```

## 4. Extract Polygon Feature CSVs

Use the Annolid polygon classifier workflow helper to merge tracked polygons
with manual behavior labels. This writes one feature CSV per video.

```bash
source .venv/bin/activate
python - <<'PY'
from pathlib import Path
from annolid.behavior.polygon_classifier_workflow import generate_polygon_points_csv

base = Path("/path/to/project")
videos = base / "videos"
labels = base / "labels-hand-paper-matched"
out = base / "logs" / "polygon_tcn_train_test"
out.mkdir(parents=True, exist_ok=True)

for name in ("2019_08_20_fly2", "2019_06_26_fly2"):
    result = generate_polygon_points_csv(
        annotation_dir=videos / name,
        label_csv=labels / f"{name}_labels.csv",
        output_csv=out / f"{name}_polygon_points.csv",
        num_points=50,
        include_unlabeled=False,
    )
    print(name, result)
PY
```

The output should list the same polygon columns for every video:

```text
polygon_columns=('abdomen_features', 'head_features', 'thorax_features')
```

## 5. Validate Train/Test Compatibility

Before training, confirm the train and test CSVs have the same feature schema:

```bash
source .venv/bin/activate
python - <<'PY'
from pathlib import Path
import pandas as pd

root = Path("/path/to/project/logs/polygon_tcn_train_test")
train = pd.read_csv(root / "2019_08_20_fly2_polygon_points.csv", nrows=0)
test = pd.read_csv(root / "2019_06_26_fly2_polygon_points.csv", nrows=0)

feature_suffixes = ("_features", "_area", "_centroid", "_perimeter", "_motion_index")
train_features = {c for c in train.columns if c.endswith(feature_suffixes)}
test_features = {c for c in test.columns if c.endswith(feature_suffixes)}

print("missing_from_test", sorted(train_features - test_features))
print("extra_in_test", sorted(test_features - train_features))
PY
```

Both lists should be empty. If not, fix tracking labels or choose a compatible
held-out test video.

## 6. Train TCN and Evaluate on a Held-Out Video

Train on the training CSV and evaluate on the held-out test CSV:

```bash
source .venv/bin/activate
python - <<'PY'
from pathlib import Path
from annolid.behavior.polygon_classifier_workflow import train_polygon_classifier

root = Path("/path/to/project/logs/polygon_tcn_train_test")
outcome = train_polygon_classifier(
    train_csv=root / "2019_08_20_fly2_polygon_points.csv",
    test_csv=root / "2019_06_26_fly2_polygon_points.csv",
    output_dir=Path("/path/to/project/logs/runs/polygon_classifier/train"),
    model_type="tcn",
    run_name="fly2_train_20190626_test_tcn",
    num_epochs=500,
    batch_size=64,
    learning_rate=1e-6,
    window_size=11,
    hidden_dim=128,
    num_residual_blocks=6,
    dropout=0.3,
    device="auto",
)
print(outcome)
PY
```

Use `device="mps"` on Apple Silicon, `device="cuda"` on CUDA machines, or
`device="cpu"` when no accelerator is available.

## 7. Review Metrics and Predictions

Each run writes:

```text
<run_dir>/
  metrics.json
  polygon_tcn_classifier_best.pt
  test_predictions.csv
  tcn_inputs/
```

Check that training did not produce non-finite values and that the prediction
CSV has the expected number of rows:

```bash
source .venv/bin/activate
python - <<'PY'
import json
import math
from pathlib import Path
import pandas as pd

run = Path("/path/to/run_dir")
text = (run / "metrics.json").read_text()
payload = json.loads(text)
history = payload["history"]
losses = [float(row["loss"]) for row in history]
pred = pd.read_csv(run / "test_predictions.csv")

print("contains_NaN_token", "NaN" in text)
print("epochs", len(history))
print("loss_all_finite", all(math.isfinite(v) for v in losses))
print("loss_first", losses[0])
print("loss_last", losses[-1])
print("test_metrics", json.dumps(payload["test_metrics"], indent=2))
print("prediction_rows", len(pred))
print("prediction_missing_cells", int(pred.isna().sum().sum()))
print(pred["predicted_label"].value_counts().to_string())
PY
```

Use `macro_f1`, per-class F1, and non-background accuracy to judge behavior
classification quality. Overall accuracy can be misleading when most frames are
background.

## 8. Reuse the Checkpoint for Inference

After training, run the saved checkpoint on another compatible polygon feature
CSV:

```bash
source .venv/bin/activate
python - <<'PY'
from pathlib import Path
from annolid.behavior.polygon_classifier_workflow import predict_polygon_classifier_csv

run = Path("/path/to/run_dir")
outcome = predict_polygon_classifier_csv(
    feature_csv=Path("/path/to/new_video_polygon_points.csv"),
    checkpoint_path=run / "polygon_tcn_classifier_best.pt",
    output_csv=run / "new_video_predictions.csv",
    device="auto",
)
print(outcome)
PY
```

The input CSV must use the same polygon labels and feature schema used during
training.

## Troubleshooting

### Test video has only one full-body polygon

If the train video has `head`, `thorax`, and `abdomen`, but the test video only
has `fly1` or `fly2`, do not train/evaluate that split. Re-track the test video
with the same body-part labels or train a separate full-body polygon model where
all videos use the same full-body label.

### Metrics are dominated by background

Behavior datasets often contain many background frames. Report:

- macro F1,
- per-class F1,
- non-background accuracy,
- labeled-frame count.

Do not rely only on overall accuracy.

### Training reports NaN or skipped batches

Regenerate polygon feature CSVs and check for missing features or incompatible
label columns. Current Annolid polygon feature extraction replaces missing
optional numeric values with finite defaults before model training, but schema
mismatches still need to be fixed at the annotation level.
