# Calculate distances between instances

Once you have tracking results for a video (typically `*_tracking.csv` and `*_tracked.csv` next to your video), you can compute inter-instance distances per frame.

## Prerequisites
- A processed video folder containing Annolid outputs:
  - `<video_stem>_tracking.csv`
  - `<video_stem>_tracked.csv`

If you don’t have CSV outputs yet, generate them in the GUI via *File → Save CSV*.

## Compute distances (Python)
Annolid provides a helper class:

```python
from annolid.postprocessing.tracking_results_analyzer import TrackingResultsAnalyzer

analyzer = TrackingResultsAnalyzer(
    "/path/to/video.mp4",
    zone_file=None,  # optional: a zone JSON for place-preference analyses
    fps=30,          # optional: auto-detected when omitted
)
analyzer.merge_and_calculate_distance()
analyzer.distances_df.to_csv("inter_instance_distances.csv", index=False)
```

The resulting CSV includes `frame_number`, `instance_name_1`, `instance_name_2`, and `distance` (pixel units).

```{note}
If you need distances in real-world units (e.g., mm), calibrate pixels-to-units and multiply the exported distances accordingly.
```
