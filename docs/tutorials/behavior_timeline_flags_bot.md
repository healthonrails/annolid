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
