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

## Tips
- If the dialog reports a missing annotation, open that frame in the main window, save the polygon (`Ctrl+S`), then re-open the Segment Editor.
- Use **Stop Tracking** in Video Manager to halt the batch job; completed segments remain on disk.
- Reopen the Segment Editor at any time to adjust segments—they are saved as soon as you press **OK**.
