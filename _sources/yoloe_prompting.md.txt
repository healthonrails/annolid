# YOLOE-26 Prompting (Text / Visual / Prompt-free)

Annolid supports Ultralytics YOLOE-26 segmentation models with:

- **Text prompts** (choose classes by name)
- **Visual prompts** (provide exemplar bounding boxes + class IDs)
- **Prompt-free** YOLOE-26 variants (built-in vocabulary)

## CLI (recommended for reproducible runs)

Annolid exports predictions as **LabelMe JSON** via `annolid-run predict yolo_labelme`.

### Text prompt (detect only the prompted classes)

```bash
annolid-run predict yolo_labelme \
  --weights yoloe-26s-seg.pt \
  --source /path/to/image.jpg \
  --classes person,bus
```

Outputs a folder next to the source (default: `/path/to/image/`) containing LabelMe JSON files.

### Visual prompt (JSON file)

Create a JSON file like:

```json
{
  "names": ["person", "glasses"],
  "bboxes": [
    [221.52, 405.8, 344.98, 857.54],
    [120, 425, 160, 445]
  ],
  "cls": [0, 1]
}
```

Then run:

```bash
annolid-run predict yolo_labelme \
  --weights yoloe-26s-seg.pt \
  --source /path/to/image.jpg \
  --visual-prompts /path/to/visual_prompts.json
```

### Visual prompt (LabelMe rectangles)

If you already have a LabelMe JSON with **rectangle** shapes labeled with class names,
you can reuse it as the prompt source:

```bash
annolid-run predict yolo_labelme \
  --weights yoloe-26s-seg.pt \
  --source /path/to/image.jpg \
  --visual-prompts-labelme /path/to/prompts.json
```

### Prompt-free YOLOE-26

Prompt-free weights do not require `--classes` or visual prompts:

```bash
annolid-run predict yolo_labelme \
  --weights yoloe-26s-seg-pf.pt \
  --source /path/to/image.jpg
```

## GUI (video inference)

Annolid’s video inference pipeline uses `annolid/segmentation/yolos.py` under the hood:

- **Selecting YOLOE-26:** pick a YOLOE-26 preset from the model dropdown (for example `YOLOE-26s-seg (Prompted)` or `YOLOE-26s-seg (Prompt-free)`).
- **Text prompting:** put a comma-separated class list in the **Text Prompt** field (e.g. `person,bus`) before running prediction with a YOLOE-26 model.
- **Visual prompting:** draw and label **rectangle** shapes on the canvas; Annolid converts them into YOLOE visual prompts automatically.
- **Prompt-free YOLOE-26:** select a `*-pf.pt` weight; Annolid will not override the internal vocabulary with prompts.
