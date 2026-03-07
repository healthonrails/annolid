# Model Plugin Help

Use this page to understand the current `annolid-run` model help flow as both a
user and a plugin author.

## User Help Flow

Start at the broadest level, then narrow down only when you need more detail.

```bash
annolid-run --help
annolid-run help train
annolid-run help predict
annolid-run help train dino_kpseg
annolid-run help predict yolo
```

Equivalent older forms still work:

```bash
annolid-run train dino_kpseg --help-model
annolid-run predict yolo --help-model
```

## What Built-In Model Help Shows

Built-in model plugins now expose curated quick-reference groups before the full
flag list.

Typical groups include:

- `Required inputs`
- `Outputs and run location`
- `Model and runtime`
- `Training controls`
- `Inference controls`
- `Data and augmentation`
- `Saving and reporting`

That means a command like:

```bash
annolid-run help predict yolo_labelme
```

shows the important flags first, instead of forcing you to scan the full
argparse dump from top to bottom.

## Recommended User Pattern

1. Run `annolid-run list-models` to discover the available model names.
2. Use `annolid-run help train` or `annolid-run help predict` to understand the
   common entry pattern.
3. Use `annolid-run help train <model>` or `annolid-run help predict <model>`
   for the specific plugin you want.
4. Only then run the actual train or predict command.

## Plugin Author Contract

Built-in plugins now use explicit help metadata instead of relying on heuristic
flag grouping.

`annolid.engine.registry.ModelPluginBase` supports:

- `train_help_sections`
- `predict_help_sections`
- `get_help_sections(mode)`

Each section is declared as:

```python
(
    ("Required inputs", ("--data", "--weights")),
    ("Training controls", ("--epochs", "--batch")),
)
```

## Plugin Author Guidelines

- Add explicit help sections for every supported mode.
- Group flags by user intent, not implementation detail.
- Put the minimum viable path first:
  - required paths or inputs,
  - output location,
  - runtime/model choice,
  - train or predict tuning knobs.
- Keep the section names stable and human-readable.
- Treat the full parser output as the detailed reference, not the primary UX.

## Fallback Behavior

Third-party or older plugins that do not declare explicit help sections still
work. The CLI falls back to heuristic grouping for those cases.

Built-in plugins in this repository are expected to define explicit sections,
and the test suite enforces that contract.
