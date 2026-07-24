# DINOv3 Keypoint Tracking

DINOv3 Keypoint Tracker propagates manually labeled body-part points through a
video without training a pose-estimation model for the target dataset. It uses
frozen DINO-family patch descriptors for appearance matching, a local
motion-prior search, optional mask constraints, stateful track memory, and
Lucas-Kanade pixel refinement for sub-patch motion.

This backend is a sparse-label accelerator. It is useful when you need
reviewable keypoint trajectories from a few manual seed frames. It is not a new
DINO model, and it is not a replacement for supervised pose estimation when you
need a reusable model trained across many videos, camera setups, or animals.

## When To Use It

Use DINOv3 Keypoint Tracker when:

- you have a video-specific tracking job and want to avoid dense manual
  keypoint labeling;
- the same landmarks are visible across nearby frames;
- you can review and correct occasional drift;
- masks or polygons are available, or the local appearance around each
  keypoint is distinctive enough to track without masks.

Use a pose-estimation workflow instead when:

- you need a reusable model for many unseen videos;
- the animal undergoes large viewpoint changes not represented by the seed
  frames;
- you need high-throughput inference after a labeled training set is already
  available;
- you need benchmarked pose-model outputs rather than per-video reviewed
  propagation.

## What It Tracks

The tracker starts from saved LabelMe-compatible Annolid annotations:

- `point` shapes become named keypoints to track;
- `polygon` or mask shapes are optional but recommended because they constrain
  ambiguous points to the instance;
- a plain point without instance metadata is associated with the single polygon
  that contains it, while explicit `instance_label` metadata always wins;
- later manually corrected frames can be used as new seed frames when you rerun
  tracking.

For multi-animal videos, avoid repeated plain point labels such as `nose` with
no instance context. Annolid preserves `instance_label` and `display_label`
metadata when those flags are present, and uses them to keep output labels
reviewable. In scripted LabelMe JSON, set those flags explicitly for each point;
in the GUI, check the saved frame before a long run to confirm the point belongs
to the intended animal. Points inside overlapping polygons, or outside every
polygon, remain unassociated unless their instance metadata is explicit.

Keep display labels stable across frames and animals. Labels such as `nose`,
`leftear`, `rightear`, `tail_base`, `fore_top`, or `abdomen_bot` should mean the
same anatomical point every time.

## How The Backend Works

The implementation separates the DINO-family feature backbone from the tracker
logic:

- `annolid/features/dino_models.py` defines the supported DINOv3, DINOv2, and
  RADIO-compatible feature backbones.
- `annolid/features/dinov3_extractor.py` loads the selected Hugging Face model,
  selects CPU, CUDA, or MPS, snaps image sizes to ViT patch multiples, removes
  special tokens, and returns normalized patch grids.
- `annolid/tracking/dino_keypoint_tracker.py` turns those patch grids into
  stateful point tracks.

Each manual point initializes a track with its current descriptor, reference
descriptor, manual-anchor codebook, appearance codebook, support probes,
velocity, quality state, and optional structural or symmetry state. On each new
frame, Annolid:

1. extracts the current DINO feature grid;
2. optionally debiases coordinate-correlated DINO responses;
3. estimates the current point from valid per-point pyramidal Lucas-Kanade
   flow, otherwise uses dense Farneback flow at the last emitted point, and
   falls back to previous velocity when neither flow observation is valid;
4. searches a bounded candidate region around the motion prior;
5. scores candidates with DINO similarity, appearance/context/support evidence,
   mask or structure priors, backward consistency, and a motion penalty;
6. resolves one-to-one assignments so two tracks do not claim the same best
   candidate;
7. optionally blends the DINO-selected point with a valid Lucas-Kanade point
   estimate for sub-patch refinement;
8. updates descriptors, velocity, quality, support probes, and output
   annotation metadata.

This split matters in practice: DINO descriptors provide robust coarse
appearance correspondence, while optical flow handles small movements inside a
16-pixel ViT patch.

## Requirements

Install Annolid with GUI, model, and tracking dependencies:

```bash
pip install "annolid[gui,ml,tracking]"
```

For a local checkout, use the repository environment:

```bash
source .venv/bin/activate
pip install -e ".[gui,ml,tracking]"
```

The one-line installer default `gui` profile also installs the model and
tracking extras used by this workflow. The first DINOv3 run may need Hugging
Face access. For gated DINOv3 checkpoints, accept the model license on Hugging
Face and authenticate once:

```bash
hf auth login
```

You can also set `HF_TOKEN` in the active environment. To use a shared local
Hugging Face cache:

```bash
export DINOV3_LOCATION=/path/to/hf-cache
```

The optional RADIO backbone requires `open-clip-torch`; choose a DINOv2 or
DINOv3 backbone if that package is not installed.

## Choose A Feature Backbone

Open **Tools -> Advanced Parameters -> Tracker -> DINO feature model** and pick
a model. The default is the small DINOv3 ViT-S/16 checkpoint because it is the
best starting point for interactive tracking.

Common choices:

| Model | Best use |
| --- | --- |
| `facebook/dinov3-vits16-pretrain-lvd1689m` | Default interactive tracking; fastest DINOv3 option. |
| `facebook/dinov3-vits16plus-pretrain-lvd1689m` | Slightly stronger small DINOv3 descriptors. |
| `facebook/dinov3-vitb16-pretrain-lvd1426` | Medium offline runs when ViT-S is not distinctive enough. |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | Higher-memory offline runs. |
| `facebook/dinov2-base` or `facebook/dinov2-large` | Open fallback when DINOv3 access is unavailable. |
| `nvidia/C-RADIOv4-SO400M` | Compatible feature backbone for specialized workflows. |

List the supported catalog from the active environment:

```bash
annolid-run dinov3-models --list
```

Pre-download or verify a model before opening the GUI:

```bash
annolid-run dinov3-models --model facebook/dinov3-vits16-pretrain-lvd1689m
annolid-run dinov3-models --local-files-only --model facebook/dinov3-vits16-pretrain-lvd1689m
```

## GUI Workflow

1. Open a video in Annolid.
2. Navigate to a frame where the landmarks are visible and sharp.
3. Add `point` labels for the keypoints you want to track.
4. For multi-animal videos, verify that each point has the intended instance
   association before starting a long run.
5. Add polygons or masks for the animal instances when available.
6. Save the annotation for that frame.
7. Open **Tools -> Advanced Parameters -> Tracker**.
8. Choose **DINO feature model** and, if useful, a **Tracker preset**.
9. Select **DINOv3 Keypoint Tracker** from the model dropdown.
10. Click the prediction button to track from the current frame.
11. Review the generated JSON files, correct drift, save the corrected frame,
    and rerun from that frame when needed.

For long videos, validate a short difficult segment first. Confirm that the
tracked points survive crossings, occlusions, and fast motion before processing
the full recording.

## Presets And Important Parameters

The tracker exposes presets through **Tools -> Advanced Parameters -> Tracker**:

| Preset | Use |
| --- | --- |
| `fly_70fps_keypoints` | Small, fast motions such as high-frame-rate fly keypoints. Uses tight search, positional debiasing, backward consistency, coherent peak refinement, support probes, context descriptors, and a small Lucas-Kanade window. |
| `rodent_30fps_occlusions` | Larger animal motion and occlusions. Uses wider search, stronger mask/structure support, larger appearance bundles, more support probes, and a larger Lucas-Kanade window. |

Key settings:

- **DINO feature model**: selects the frozen feature backbone.
- **Clamp keypoints to instance mask** and **Mask snap radius**: keep points
  inside the selected instance when masks are available.
- **Search tighten**, **Velocity gain**, **Flow gain**, and **Min/Max radius**:
  control the local motion-prior search.
- **DINOv3 positional debias**: suppresses coordinate-correlated feature
  responses that can pin points to image locations.
- **DINOv3 backward consistency**: prefers candidates that map back to the
  previous feature location.
- **Coherent peak refinement**: refines around the connected similarity peak
  instead of unrelated nearby peaks.
- **Pixel refinement**: the tracker presets configure Lucas-Kanade sub-patch
  correction; programmatic callers can tune its blend weight, window size,
  error threshold, and max jump through the `pixel_refine_*` runtime fields.

## Outputs And Review Contract

DINOv3 Keypoint Tracker writes LabelMe-compatible JSON files into the standard
Annolid result folder:

```text
<video_name>/<video_name>_000000001.json
<video_name>/<video_name>_000000002.json
...
```

Predicted point shapes keep their labels and include tracking metadata such as
backend description, quality, velocity, miss count, display label, and instance
label. The tracker itself does not write a summary CSV; export or summarize the
reviewed JSON files through the usual Annolid analysis tools after corrections
are complete.

The recommended correction loop is:

1. run tracking from a clean seed frame;
2. inspect the first frame where a keypoint drifts or swaps;
3. correct the point manually and save that frame;
4. rerun tracking from the corrected frame;
5. keep corrections as sparse reseed frames rather than relabeling every frame.

## Choosing Among Point And Mask Backends

| Backend | Use it when |
| --- | --- |
| DINOv3 Keypoint Tracker | You want sparse, named keypoint propagation with reviewable LabelMe outputs and optional masks. |
| TAPNext (ONNX) | You want an ONNX point tracker with fixed model assets and no DINO/Hugging Face feature backbone. See [TAPNext ONNX Point Tracking](tapnext.md). |
| CoTracker | You want joint point tracking through a chunk using the official CoTracker model. |
| CoWTracker | You want an alternative dense/windowed point tracker and can install the optional runtime. See [CoWTracker Point Tracking](cowtracker.md). |
| YOLO pose or DinoKPSEG | You want a trainable pose/keypoint model for repeated inference on future videos. |
| INSID3 Video | You want DINOv3 in-context segmentation masks, not direct point tracking. Masks from segmentation workflows can still help constrain keypoints. |
| Cutie or SAM3 | You need masks or polygons rather than point trajectories. See [SAM3](sam3.md) for SAM3 tracking and correction. |

## Programmatic Use

```python
from annolid.tracking.configuration import CutieDinoTrackerConfig
from annolid.tracking.dino_keypoint_tracker import DinoKeypointVideoProcessor

runtime = CutieDinoTrackerConfig.from_preset(
    "rodent_30fps_occlusions",
    patch_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
)

processor = DinoKeypointVideoProcessor(
    video_path="/path/to/video.mp4",
    result_folder=None,
    model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
    short_side=768,
    device=None,
    runtime_config=runtime,
)
processor.process_video()
```

The processor reads the saved seed annotation from the video result folder,
tracks forward, and writes per-frame JSON outputs.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| `No manual PNG+JSON seed pairs found` | Save a labeled seed frame in the video result folder before starting tracking. |
| DINOv3 model download fails | Accept the gated model license, run `hf auth login`, set `HF_TOKEN`, or choose an open DINOv2 fallback. |
| RADIO model import fails | Install `open-clip-torch` or switch back to a DINO-family feature model. |
| Model dropdown uses the wrong DINO backbone | Reopen **Tools -> Advanced Parameters -> Tracker**, confirm **DINO feature model**, and check the log for the resolved model id. |
| Points snap to the same image location | Enable DINOv3 positional debias and reduce search radius if motion is small. |
| Small joints appear stuck inside one patch | Keep pixel refinement enabled and use the fly preset or a smaller Lucas-Kanade window. |
| Left/right landmarks swap | Add masks when possible, keep labels consistent, add more anchor points, and configure symmetry pairs in scripted workflows. |
| Tracking drifts after occlusion | Correct the first bad frame, save it, and rerun from that correction frame. |
| CPU runs are slow | Use CUDA or MPS when available, reduce `short_side`, validate short segments first, and pre-download the model before long runs. |
