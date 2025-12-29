# Expert mode: CLI tools

Most Annolid users work in the GUI (`annolid`). The CLI is useful when you want to automate conversions, run batch processing, or integrate Annolid into a pipeline.

## The `annolid` command (GUI entry point)
`annolid` launches the GUI, but it also accepts a few helpful CLI flags:

- Print version: `annolid --version`
- Load custom label list: `annolid --labels labels.txt` (or `--labels "mouse_1,mouse_2"`)
- Enable event flags: `annolid --flags flags.txt` (or `--flags "rearing,grooming"`)
- Use a specific config: `annolid --config ~/.labelmerc`
- Auto-save annotations: `annolid --autosave`

Run `annolid --help` to see the full list.

## Batch utilities (run as Python modules)
Annolid also ships “utility” modules you can run with `python -m ...`.

### Convert LabelMe JSON → CSV
Export per-frame segmentation/keypoint annotations into a single tracking-style CSV:

```bash
python -m annolid.annotation.labelme2csv --json_folder /path/to/video_results_folder
```

### Convert LabelMe JSON → YOLO dataset
Create a YOLO dataset (images + labels + `data.yaml`) from LabelMe JSON files:

```bash
python -m annolid.main --labelme2yolo /path/to/labelme_json_folder --val_size 0.1 --test_size 0.1
```

### Behavior time budgets from event exports
If you export behavior events from the GUI, compute a “time budget” summary:

```bash
python -m annolid.behavior.time_budget /path/to/exported_events.csv -o time_budget.csv --bin-size 60
```

If you use a project schema (categories/modifiers), pass it with `--schema`.
