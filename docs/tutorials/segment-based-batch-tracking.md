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

## 10. Repair ID Switches With Manual Seed Frames

For CUTIE ID switches, the recommended workflow is to insert a corrected manual
seed frame and run CUTIE forward from that point, either for a short window or to
the next corrected seed/end of video.

Use this when one animal takes another animal's label after overlap, crossing, or
occlusion:

1. Find the first frame where the IDs are wrong.
2. Correct the polygons/labels on that frame and save the frame. Annolid writes a
   PNG+JSON pair, which CUTIE treats as a manual seed.
3. Rerun CUTIE from that corrected frame, or define a segment/window ending before
   the next already-correct section.
4. Review the next crossing. If another switch occurs, add another corrected seed
   frame and rerun from there.

Manual seed frames supersede automatic prediction. During a rerun, Annolid does
not overwrite frames that already have saved annotations, and CUTIE builds
tracking windows from one seed to the frame before the next seed. This means you
can also pre-label several difficult frames before running tracking; each saved
manual frame becomes a reset point for the following window.

This is the practical "identity repair workflow" for most home-cage CUTIE ID
switches. It is not a separate model or a separate correction algorithm; it is a
productive way to use the existing save/retrack tools.

## 11. Reduce CUTIE Analysis Time

For CUTIE tracking, the largest speed lever is the number of frames sent through
the tracker. A 5-minute, 30 FPS video contains about `9,000` frames; the same
5-minute video at 5 FPS contains about `1,500` frames. If the behavior you need
to measure does not require 30 FPS timing, downsample a copy of the video before
tracking.

Recommended speed workflow:

1. Keep the original video unchanged.
2. Create an analysis copy with **File -> Downsample Video(s)...**.
3. Set **Use specified FPS for all videos** to `5` FPS for ordinary locomotion,
   social position, zone occupancy, and most home-cage tracking.
4. Leave the scale factor at `1.0` for `480 x 270` videos unless a short pilot
   confirms that every mouse remains clearly separable after resizing.
5. Crop only if there is unused space outside the cage. Cropping the arena can
   reduce pixels without shrinking the animals.
6. Track the downsampled copy, then review crossings, occlusions, and any
   behavior event that needs higher temporal precision.

Use higher FPS when the measured behavior is brief or fast, such as grooming
microstructure, rapid attacks, startle responses, jumps, or event timing that
needs sub-200 ms precision. A useful pilot is to track a short difficult segment
at `30`, `10`, and `5` FPS and compare ID continuity and event timing before
processing the full batch.

Hardware and runtime checks:

- CUTIE is usually GPU-limited. Use an NVIDIA CUDA GPU when available; CPU-only
  runs are much slower.
- Confirm Annolid logs show `Running device: cuda` for CUDA inference. If the
  log says `cpu`, install or repair the GPU PyTorch/CUDA environment before
  benchmarking speed.
- Disable **Use CPU Only** in **Settings -> Advanced Parameters** unless you are
  intentionally testing CPU behavior.
- Disable **Save Video with Color Mask** unless you need an overlay movie; saving
  overlay frames adds IO and encoding work.
- Disable optical-flow motion-index calculation when tracking speed is more
  important than motion-index features.
- Run one GPU-heavy tracking job at a time unless each job has a separate GPU or
  enough VRAM.

If a `480 x 270`, 5-minute, 30 FPS video takes about 20 minutes, that is roughly
`7.5` processed frames per second. Whether that is expected depends mostly on
the GPU model, VRAM, CUDA/PyTorch setup, and whether optical flow or overlay
video export is enabled. Share the GPU model, VRAM, CPU, RAM, storage type, OS,
and whether the log says `cuda` or `cpu` when asking for performance estimates.

## 12. Stop When an Animal Is Lost

To make CUTIE tracking stop when an expected animal disappears:

1. Open **Advanced Parameters** before starting tracking.
2. Enable **Automatic Pause on Error Detection**.
3. Start tracking from a frame where all expected instances are labeled.

When this option is enabled, Annolid checks the tracked instances against the
seeded instances. If a seeded animal is missing from the prediction output,
tracking pauses at that frame so you can review the image, correct or add the
missing instance, save the annotation, and continue from that point.

If CUTIE missing-instance recovery is enabled, Annolid may first try to recover
the missing mask from a recent complete frame or a recent instance mask. Recovered
frames are still worth reviewing; the later Temporal continuity repair can audit
those recovery notes and fix label/ID continuity if the mask comes back with the
wrong identity.

This is a tracking safety check, not a biological event detector. It detects
missing tracked instances in the annotation output. It does not prove the animal
left the arena or became invisible for a specific scientific reason.

Do not use `T_max` for this purpose. `T_max` controls tracker memory/window
behavior; it is not the "stop when lost" setting.

## 13. Repair Missing Sections Without Starting Over

You usually do not need to delete all predictions and start over if only a few
sections are missing.

Use one of these workflows:

- If tracking paused at the first bad frame, correct the missing animal on that
  frame, save the annotation, then continue tracking from there.
- If IDs switch after crossing or occlusion, use the manual seed-frame workflow
  above first. It gives CUTIE a new ground-truth starting point and is usually the
  most reliable correction.
- If the saved NDJSON has empty frames, use Annolid Bot tracking correction with
  `replace_only_empty_shapes=true` to fill only frames that currently have no
  shapes.
- If the problem is a short occlusion gap or likely ID switch in CUTIE frame
  JSON output, open **Video Tools -> Identity Governor...**, choose
  **Temporal continuity**, and tune `expected_instance_count`, `max_gap_frames`,
  and `max_match_distance`. This repair pass uses multi-cue temporal assignment:
  centroid motion, constant-velocity prediction, motion-compensated shape
  overlap, area, and orientation when polygon geometry is available. Its report
  also flags duplicate IDs, missing expected IDs, unexpected IDs, implausible
  jumps, and CUTIE recovery/fallback notes.
- If the problem is in an NDJSON correction workflow, use Annolid Bot tracking
  correction with `temporal_repair=true`, `expected_instance_count`,
  `max_gap_frames`, and `max_match_distance`.
- If drift or identity corruption is widespread, do a full retrack using the
  cleanup steps in the previous section.

For the SAM3-assisted correction workflow, see
[Tracking Correction with SAM3 Bot](tracking_correction_with_sam3_bot.md). Write
repairs to a new NDJSON first, review the result, then replace the original only
after the corrected file is verified.

## 14. Overnight Runs and Computer Sleep

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
