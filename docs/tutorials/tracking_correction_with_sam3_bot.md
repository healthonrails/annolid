# Tracking Correction with SAM3 Agent and Annolid Bot

Use this workflow when a saved Annolid tracking NDJSON needs correction after
missed detections, short occlusions, or likely identity switches.

The correction tool works on LabelMe-style Annolid NDJSON files. It can:

- fill empty frame records from a second tracking NDJSON,
- run the SAM3 agent first and use its output as the correction source,
- repair short temporal gaps and likely ID switches from the existing NDJSON,
- preserve manual annotations and unrelated shapes by default.

## When to Use This

Use tracking correction when:

- an animal or object disappears for a few frames and then reappears,
- the tracker writes empty `shapes` for some frames,
- identities switch after occlusion or close interaction,
- you want to merge better SAM3 agent results into an existing annotation file.

Do not treat this as a replacement for review. Long occlusions, crossings, and
visually identical instances still require manual validation.

## Inputs and Outputs

Required input:

- `ndjson_path`: the target Annolid NDJSON file to correct.

Optional inputs:

- `source_ndjson_path`: a second NDJSON file with better shapes for some frames.
- `video_path`: source video path, used when asking the tool to run SAM3 first.
- `agent_prompt`: text prompt for SAM3, such as `mouse` or `fly`.
- `output_ndjson_path`: where to write the corrected NDJSON.

Recommended output practice:

1. Write to a new `output_ndjson_path` first.
2. Open the corrected file in Annolid and review the changed frames.
3. Replace the original only after the corrected file is verified.

If `output_ndjson_path` is omitted, the tool writes in place.

## Correction Modes

### Mode 1: Fill Empty Frames from a Source NDJSON

Use this when you already have a second NDJSON from SAM3, Cutie, or another
tracking pass.

Example Annolid Bot prompt:

```text
Correct /data/session1/session1_annotations.ndjson using
/data/session1/sam3_predictions.ndjson.
Write /data/session1/session1_corrected.ndjson.
Only fill frames that currently have empty shapes.
```

Equivalent structured tool fields:

```text
gui_correct_tracking_ndjson(
  ndjson_path="/data/session1/session1_annotations.ndjson",
  source_ndjson_path="/data/session1/sam3_predictions.ndjson",
  output_ndjson_path="/data/session1/session1_corrected.ndjson",
  replace_only_empty_shapes=true
)
```

Default behavior is conservative:

- frames with existing shapes are skipped,
- empty target frames are filled from the source,
- frame metadata such as `frame`, `frame_number`, `imagePath`, width, and height
  is preserved when possible.

### Mode 2: Run SAM3 Agent, Then Correct the NDJSON

Use this when you want Annolid Bot to run SAM3 agent tracking and merge the
result into your target NDJSON.

Example prompt:

```text
Run SAM3 Agent on /data/session1/video.mp4 for prompt "mouse",
then correct /data/session1/session1_annotations.ndjson.
Write /data/session1/session1_corrected.ndjson.
Only fill empty frames.
```

Equivalent structured tool fields:

```text
gui_correct_tracking_ndjson(
  ndjson_path="/data/session1/session1_annotations.ndjson",
  video_path="/data/session1/video.mp4",
  agent_prompt="mouse",
  run_sam3_agent=true,
  output_ndjson_path="/data/session1/session1_corrected.ndjson",
  window_size=5,
  stride=5
)
```

Use a dry SAM3 workflow first if you are still choosing output folders or
tracking settings. See [SAM3 Agent Video Tracking](../sam3.md).

### Mode 3: Repair Occlusions and ID Switches

Use temporal repair when the target NDJSON itself has enough stable frames to
infer instance continuity.

Example prompt:

```text
Repair occlusions and ID switches in
/data/session1/session1_annotations.ndjson after frame 500.
There should be 4 instances. Fill gaps up to 20 frames and use max match
distance 80 pixels. Write /data/session1/session1_temporal_repair.ndjson.
```

Equivalent structured tool fields:

```text
gui_correct_tracking_ndjson(
  ndjson_path="/data/session1/session1_annotations.ndjson",
  output_ndjson_path="/data/session1/session1_temporal_repair.ndjson",
  temporal_repair=true,
  start_frame=500,
  expected_instance_count=4,
  max_gap_frames=20,
  max_match_distance=80
)
```

Temporal repair uses centroid continuity. It chooses reference instances from
the first usable frame at or after `start_frame`, then:

- matches later shapes to the nearest previous track within `max_match_distance`,
- restores the canonical label, group ID, and track flags after likely ID
  switches,
- fills short missing tracks by interpolation when a later recovery exists,
- carries the previous shape when no later recovery exists inside the gap limit,
- marks generated or corrected shapes with correction flags.

If duplicate labels do not have usable track IDs, the repair assigns stable
synthetic IDs such as `track_1` and `track_2` in shape flags.

## Important Options

| Option | Default | Use |
| --- | --- | --- |
| `replace_only_empty_shapes` | `true` | Only target records with empty `shapes` are corrected. Set to `false` when you intentionally want non-empty frames considered for update or temporal repair. |
| `replace_all_shapes` | `false` | Source shapes are merged while preserving unrelated/manual target shapes. Set to `true` only when the source should replace every shape in a corrected frame. |
| `allow_append_new_frames` | `false` | When `true`, source frames that do not exist in the target NDJSON are appended. Leave this off if the target file should define the authoritative frame set. |
| `temporal_repair` | `false` | Enables occlusion filling and ID-switch correction. |
| `start_frame` | `0` | First frame to use for temporal repair. Set this to a stable frame after the expected instances are visible. |
| `expected_instance_count` | unset | Number of instances expected after `start_frame`. This helps the repair choose a stable reference frame and detect missing tracks. |
| `max_gap_frames` | `5` | Maximum gap length to fill. Keep this small for conservative correction. |
| `max_match_distance` | `80` | Maximum centroid distance, in pixels, for assigning a current shape to a previous track. Lower values reduce accidental identity changes; higher values tolerate faster movement. |

## Reading the Result Summary

Annolid Bot returns a JSON summary. Important fields include:

- `output_ndjson_path`: corrected file path.
- `target_records`: valid target records read.
- `source_records`: valid source records read.
- `candidate_source_frames`: source frames that had frame metadata.
- `replaced_frames`: target frames corrected from source.
- `appended_frames`: new frames appended from source.
- `skipped_non_empty_frames`: frames skipped because they already had shapes.
- `target_invalid_lines`: malformed target lines preserved but not parsed.
- `source_invalid_lines`: malformed source lines ignored.
- `temporal_reference_instances`: tracks used as temporal references.
- `missing_shapes_filled`: shapes created for short occlusion gaps.
- `id_switches_corrected`: shapes whose identity metadata was corrected.

If `candidate_source_frames` is `0`, the source NDJSON does not contain usable
`frame` or `frame_number` metadata.

## Review Checklist

After correction:

1. Open the corrected NDJSON in Annolid.
2. Jump to frames around each occlusion or interaction.
3. Check corrected shapes with `annolid_correction` flags.
4. Verify that the instance count is correct after `start_frame`.
5. Confirm that manual annotations or behavior labels were not unintentionally
   replaced.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| No source NDJSON was resolved. | Provide `source_ndjson_path`, or set `run_sam3_agent=true` with both `video_path` and `agent_prompt`, or set `temporal_repair=true` to repair the target NDJSON from itself. |
| The tool reports no candidate source frames. | The source records are missing `frame` or `frame_number`, or the file is not an Annolid tracking NDJSON. |
| Too many frames were skipped. | The default only fills empty frames. If you need to correct non-empty frames, set `replace_only_empty_shapes=false` and review the output carefully. |
| ID switches remain after repair. | Start from a more stable `start_frame`, lower or raise `max_match_distance` based on object speed, and verify that `expected_instance_count` matches the visible instances. |
| False ID corrections appear after crossings. | Reduce `max_match_distance`, reduce `max_gap_frames`, or split the repair into shorter ranges around stable intervals. Centroid-based repair is intentionally conservative but cannot prove identity through long ambiguous crossings. |
