# Segment-Based Batch Tracking

## Prerequisites

- Annolid v1.2.2 or later with the Segment Tracker dialog available.
- Videos imported into the Video Manager.
- At least one polygon annotation saved (`Ctrl+S`) for each segment's starting frame.

## 1. Import Videos

1. Click **Import Videos**, browse to the folder that contains your clips, and confirm. Each video now appears as a row in the table.

## 2. Load a Video

1. In the Video Manager table, press **Load** on the first video.
2. The main Annolid window switches to that clip and shows the playback controls in the status bar.

## 3. Open the Segment Editor

1. Choose **Video Tools → Define Video Segments…** or press `Ctrl+Alt+S`.
2. Verify the dialog title matches the current video and review the information displayed in the "Active Video Information" pane.

## 4. Define a Segment

1. Scrub to the frame that contains your saved annotation.
2. In the dialog, press **Use Annolid Frame** to capture that frame number.
3. Pick an end condition (Duration, End Time, or End Frame) and adjust the inputs.
4. Click **Add Segment**. The segment appears in the list below.

## 5. Add More Segments (Optional)

- Repeat step 4 for each additional range you want to track.
- Use **Edit Selected** or **Delete Selected** if you need to revise the list.
- When finished, click **OK**. Annolid writes the segments to `<video>.segments.json`, so they are restored the next time you load the video.

## 6. Move to the Next Video

1. Back in Video Manager, press **Load** on the next video.
2. Define its segments by repeating steps 3–5.
3. Continue until every video has the segments you need.

## 7. Run Track All

1. In Video Manager, click **Track All**.
2. Annolid automatically

   - closes the current video,
   - opens the next video in the queue, and
   - processes each saved segment in order using the Cutie tracker.

3. For videos without saved segments, Annolid falls back to whole-video tracking.

## 8. Review Results

- Each video's output folder (same name as the source video) now contains:

  - Updated JSON annotations for every tracked frame.
  - CSV exports summarising the tracking run.
  - Optional overlay videos when enabled in tracking settings.

## 9. Retrack a Video

If you want to rerun tracking, remove only the auto-generated tracking outputs, then run tracking again.

Delete these generated files next to the video:

- `<video_name>*_tracking.csv`
- `<video_name>*_tracked.csv`
- `<video_name>*_gaps_report.csv`
- `<video_name>*_tracking_gaps_report.md`
- `<video_name>/<video_name>_annotations.ndjson` (if present)

Keep these manual labels inside `<video_name>/`:

- Labeled `.png` frame images
- Matching `.json` annotation files

After deleting generated outputs:

1. Reopen Annolid.
2. Import videos
3. Run **Track All** again to regenerate tracking from your existing manual labels.

If you only need to retrack one video, load that video directly from the toolbar/Video Manager and run tracking for that video only. This avoids reprocessing videos that are already correct.

## 10. Stop When an Animal Is Lost

To make CUTIE tracking stop when an expected animal disappears:

1. Open **Advanced Parameters** before starting tracking.
2. Enable **Automatic Pause on Error Detection**.
3. Start tracking from a frame where all expected instances are labeled.

When this option is enabled, Annolid checks the tracked instances against the
seeded instances. If a seeded animal is missing from the prediction output,
tracking pauses at that frame so you can review the image, correct or add the
missing instance, save the annotation, and continue from that point.

This is a tracking safety check, not a biological event detector. It detects
missing tracked instances in the annotation output. It does not prove the animal
left the arena or became invisible for a specific scientific reason.

Do not use `T_max` for this purpose. `T_max` controls tracker memory/window
behavior; it is not the "stop when lost" setting.

## 11. Repair Missing Sections Without Starting Over

You usually do not need to delete all predictions and start over if only a few
sections are missing.

Use one of these workflows:

- If tracking paused at the first bad frame, correct the missing animal on that
  frame, save the annotation, then continue tracking from there.
- If the saved NDJSON has empty frames, use Annolid Bot tracking correction with
  `replace_only_empty_shapes=true` to fill only frames that currently have no
  shapes.
- If the problem is a short occlusion gap or likely ID switch, use temporal
  repair with `temporal_repair=true`, `expected_instance_count`,
  `max_gap_frames`, and `max_match_distance`.
- If drift or identity corruption is widespread, do a full retrack using the
  cleanup steps in the previous section.

For the SAM3-assisted correction workflow, see
[Tracking Correction with SAM3 Bot](tracking_correction_with_sam3_bot.md). Write
repairs to a new NDJSON first, review the result, then replace the original only
after the corrected file is verified.

## 12. Overnight Runs and Computer Sleep

Annolid does not currently wrap long tracking jobs in an operating-system sleep
prevention command. On macOS or Windows, a sleeping computer pauses CPU/GPU work
and can make an overnight run look like it barely progressed.

Before a long run:

- Keep the computer plugged in.
- On macOS, use System Settings to prevent automatic sleep while the display is
  off when that option is available.
- On Windows, use **Settings → System → Power & battery → Screen and sleep** and
  set plugged-in sleep to **Never** for the duration of the run.
- Do not close the laptop lid unless the machine is configured to keep running
  when the lid is closed.
- Prefer running one GPU-heavy tracking job at a time unless each job has a
  separate GPU or machine.

If launching Annolid from a macOS terminal, you can keep the Mac awake for the
whole process:

```bash
source .venv/bin/activate
caffeinate -dimsu annolid
```

To keep the Mac awake while an already running Annolid process continues, find
the process ID and attach `caffeinate` to it:

```bash
pgrep -fl annolid
caffeinate -dimsu -w <PID>
```

The display may still turn off. The important part is preventing system sleep
while tracking is running.

## Tips

- If the dialog reports a missing annotation, open that frame in the main window, save the polygon (`Ctrl+S`), then re-open the Segment Editor.
- Use **Stop Tracking** in Video Manager to halt the batch job; completed segments remain on disk.
- Reopen the Segment Editor at any time to adjust segments—they are saved as soon as you press **OK**.
