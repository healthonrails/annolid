# Behavior Labeling with Timeline, Flags, and Annolid Bot

This tutorial shows how to use Annolid's behavior workflow end-to-end:

1. define behavior names once,
2. use those names consistently in Flags and Timeline,
3. drag the timeline needle/playhead for fast review,
4. run agent-assisted segment labeling,
5. save outputs for analysis.

## What Is Shared Across UI Panels

Annolid uses a canonical behavior catalog in the project schema (`project.annolid.json`).

- **Flags dock** behavior names are synced into the catalog.
- **Timeline dock** behavior definitions are synced into the catalog.
- **Annolid Bot** behavior labeling uses the same catalog.

This keeps behavior names consistent across manual scoring and agent-generated labels.

## Quick Start (Manual Workflow)

1. Open a video in Annolid.
2. Open the **Flags** dock and add your behavior names (for example: `walking`, `grooming`, `rearing`).
3. Click **Save All** in Flags to persist names/states.
4. Open the **Timeline** dock and verify behaviors appear in the behavior selector.
5. Record events using Start/End in Flags or by creating timeline segments.

## Timeline Needle Dragging

The timeline playhead (red vertical line + red triangle marker) supports direct dragging.

How to use:

1. Move your mouse over the red marker near the timeline header.
2. Press and hold left mouse button.
3. Drag left/right to scrub frames.
4. Release to stop at the target frame.

Behavior notes:

- Frame updates are continuous while dragging.
- The active frame in the main video view follows the playhead.
- Segment editing (dragging behavior bars) still works independently.

## Defining Behaviors from Timeline

You can add behavior names directly from Timeline.

1. Type a behavior name in the behavior combo.
2. Use the define/add action in Timeline.
3. The behavior is added to the shared catalog and saved to schema.
4. Flags and Bot can reuse it immediately.

## Defining Behaviors from Flags

Flags can also serve as a behavior editor.

1. Add or rename rows in the Flags table.
2. Save flags with **Save All** (`Ctrl+S`).
3. Row changes and saves sync behavior names into the shared catalog.

## Agent-Assisted Behavior Labeling (1s Segments)

Once behaviors are defined, you can ask Annolid Bot to label segments.

Recommended pattern:

1. Open **Annolid Bot** from the GUI.
2. Use your defined behavior list from schema/flags/timeline.
3. Ask for segment labeling (for example every 1 second).

Example prompt:

```text
Label behavior in /path/to/video.mp4 from defined list every 1s
```

You can also provide explicit labels:

```text
Label behavior in /path/to/video.mp4 with labels grooming, rearing, walking every 1s
```

What Annolid writes:

- timeline intervals per segment label,
- behavior timestamps CSV,
- segment-level behavior log JSON.

Prompting behavior used by Annolid Bot labeling:

- The model is instructed to act as an animal behavior observer.
- It must describe only observable mouse posture/motion facts (2–4 concise sentences).
- It must classify each segment to exactly one label from your defined behavior list.
- Output is constrained to strict JSON (`label`, `confidence`, `description`) before Annolid writes timeline/log outputs.

## Prompt Cookbook (Copy/Paste)

Use these prompts in **Annolid Bot**.

### 1. Basic: use behavior names already defined in schema/flags/timeline

```text
Label behavior in /path/to/video.mp4 from defined list every 1s
```

### 2. Basic: explicit label list

```text
Label behavior in /path/to/video.mp4 with labels grooming, rearing, walking every 1s
```

### 3. Advanced: explicit structured context fields

Use this when you want to provide experiment metadata and clear behavior definitions.

```text
Label behavior in /path/to/video.mp4 with labels aggression_bout every 1s
video description: Two mice in resident-intruder arena;
instances: 2;
experiment context: Resident intruder social interaction trial;
behavior definitions: Aggression bout includes slap in the face, run away, and initiation of bigger fights;
focus points: Count bouts, identify initiator/responder, and note transition points.
```

### 4. Advanced: full workflow (tracking + behavior labeling together)

This runs end-to-end processing from one prompt.

```text
Process video behaviors in /path/to/video.mp4 with labels aggression_bout
video description: Two mice open field;
instances: 2;
experiment context: Resident intruder assay;
behavior definitions: Aggression bout includes slap in the face, run away, and fight initiation;
focus points: Count bouts and identify initiator and responder.
```

### 5. Free-form prose (no `definitions:` key required)

Annolid Bot can infer definitions and focus from plain language:

```text
Label behavior in /path/to/video.mp4 with labels aggression_bout every 1s
Aggression bout counts of slap in the face, run away, and initiation of bigger fights.
Count bouts and identify initiator.
```

### 6. Uniform windows instead of timeline-derived windows

```text
Label behavior in /path/to/video.mp4 with labels walking, rearing every 1s
```

Tip: when you include `every 1s`, Annolid uses fixed-duration windows (`uniform` mode) with 1-second segments.

## Choosing a Prompt Style

- Use **from defined list** when your team already standardized behavior names.
- Use **explicit labels** for quick experiments on a subset of behaviors.
- Add **structured context fields** for higher consistency across long videos or complex social assays.
- Use **free-form prose** if that is faster; Annolid Bot will still try to infer definitions/focus.
- Prefer short, concrete definitions over broad descriptions.

## Example: Aggression-Bout Study Template

```text
Process video behaviors in /path/to/video.mp4 with labels aggression_bout
video description: Two adult mice in resident-intruder arena;
instances: 2;
experiment context: 10-minute resident-intruder assay, treatment group A;
behavior definitions: Aggression bout includes slap in the face, run away, and initiation of bigger fights;
focus points: Count aggression bouts, identify initiator/responder, and mark uncertain cases due to occlusion.
```

## Caption Widget: Describe Behavior Timeline (Background)

You can also run per-step behavior description directly from the caption panel:

1. In the caption widget, click **Describe behavior**.
2. Set **Run mode** to **Timeline (describe every N seconds)**.
3. Set **Describe every (s)** to `1` for 1-second segments.
4. Pick your start/end range and click **OK**.

The run is queued in background and saves incrementally, so long videos can continue from existing saved points.

Outputs written next to the video:

- `<video_stem>_timestamps.csv`
- `<video_stem>_behavior_segment_labels.json`
- `<video_filename>.behavior_timeline.json`

For bot labeling and process workflow outputs, Annolid also writes:

- `<video_stem>_timestamps.csv`
- `<video_stem>_behavior_segment_labels.json`

When the segment label is `aggression_bout`, the JSON log now includes additive
metadata for scoring:

- `event_schema`: canonical sub-event schema (`slap_in_face`, `run_away`,
  `fight_initiation`)
- `aggression_bout_summary`: stable aggregate counts (`bout_count`,
  `sub_event_counts`, `sub_event_bout_counts`, `bouts_with_initiation`)

Existing CSV exports (`_timestamps.csv`) remain unchanged.

## Recommended Operating Loop

1. Define behavior names first (Flags or Timeline).
2. Save once so schema is canonical.
3. Label a short window manually to validate naming.
4. Run agent segment labeling for a larger window.
5. Review timeline quickly by dragging the playhead and correcting edge cases.
6. Export/report once labels are stable.

## Troubleshooting

If a behavior is missing in one panel:

1. Save flags (`Save All`) or re-define it in Timeline.
2. Ensure a project schema is loaded (or created next to the video).
3. Reopen Timeline popup to refresh behavior choices.

If bot labeling says no labels are available:

1. Define behaviors in Flags/Timeline first.
2. Retry with `from defined list`, or pass labels explicitly in the prompt.

If timeline scrubbing does not move frame:

1. Drag from the red marker area near the header.
2. Confirm a video is loaded and frame range is set.
3. Retry after switching focus back to the Timeline panel.
