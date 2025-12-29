# Behavior classification

Annolid supports two complementary behavior workflows:
1. **Event scoring in the GUI** (manual or semi-manual), with exports and summaries.
2. **Model-based classification** (optional / experimental), for projects that have labeled training data.

## 1) Event scoring and time budgets (recommended starting point)
### Record events in the GUI
1. Open a video in Annolid.
2. Use your behavior labels (or launch with `annolid --flags "grooming,rearing,freezing"`).
3. Mark events over time (start/end) and save.

### Summarise behavior duration (“time budget”)
After exporting events to CSV, compute summary statistics:

```bash
python -m annolid.behavior.time_budget /path/to/exported_events.csv -o time_budget.csv
```

To get a binned time course (e.g., 60-second bins):

```bash
python -m annolid.behavior.time_budget /path/to/exported_events.csv --bin-size 60 -o time_budget.csv
```

## 2) Train a behavior classifier (optional / experimental)
Annolid includes behavior training utilities under `annolid/behavior/`.

The training entry point is:

```bash
python -m annolid.behavior.training.train --video_folder /path/to/labeled_clips
```

```{note}
The training dataset loader expects per-video CSV annotations (same stem as the video file) with columns such as `Behavior` and `Trial time`. If your data is in a different format, you’ll likely want to adapt `annolid/behavior/data_loading/datasets.py` to your project.
```
