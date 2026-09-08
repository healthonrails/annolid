# Two-mouse Three.js simulation

Open **Three.js Examples → Two Mice** in the GUI, or ask Annolid Bot:

```text
open threejs example two mice
```

The scene starts with a black C57BL/6 mouse and a pale BALB/c mouse. Models include
layered curved fur, articulated paws, thin ear pinnae, whiskers, and a continuous
tail. Exploration includes wall following, pauses to sniff, social investigation,
rearing, and grooming. This is a procedural visualization, not a validated animal
biomechanics model. Heart rate and respiration in the HUD are illustrative estimates.

## Viewing and playback

Drag to orbit, scroll to zoom, and use **Simulation Controls → Camera** for an
overview, a view from above, or a camera that follows either strain. The follow
camera preserves your orbit angle while moving with the subject.

- **Pause Simulation** freezes motion and contact solving. The camera remains usable.
- **Playback Speed** supports slow motion down to 0.25×, up to 1.5×.
- **Quality Mode** changes pixel resolution and shadow resolution without changing
  exposure. Post-processing remains a separate choice.
- **Fur Detail** controls coat density for newly spawned subjects. Reduce it before
  spawning more mice on machines with limited graphics memory.
- **Clear All** releases subject-specific GPU resources. **Spawn Subject** uses the
  selected strain and biological traits.

Appearance controls that require a new subject retain the existing respawn notice.
Body and motion settings take effect on simulation updates; resume playback to see
those changes when paused. Physics uses fixed 30 Hz steps independently of rendering.
Long stalls have bounded catch-up, and hidden tabs do not advance the simulation.

The viewer needs WebGL and access to its pinned Three.js and lil-gui imports on
`esm.sh`. A loading error is displayed if the scene cannot initialize.

## Reproducible pose capture

The existing `dataset=1`, `seed`, `frame`, and `fps` query parameters and
`window.annolidPoseDataset.getFrame()` / `setFrame(index)` API remain available.
Frames advance monotonically. Track IDs, 13-keypoint ordering, COCO-style records,
and contact-quality metadata are preserved.

Seeds reproduce captures within this scene version. The corrected motion and new
model change images and trajectories relative to older versions; existing saved
fixtures are not rewritten. Randomness is scene-local and does not replace the
browser's `Math.random` function.

Generate a new dataset in a separate directory for inspection:

```bash
source .venv/bin/activate
python scripts/generate_two_mice_pose_dataset.py --output /tmp/two_mice_preview
```

## Validation and maintenance

The HTML is the presentation shell. `two_mice_scene.js` owns scene assembly,
articulation, controls, and pose export; `two_mice_model.js` owns procedural model
construction and GPU resource ownership; `two_mice_simulation.js` supplies the
independent timing, random-stream, and wall-force helpers. All three modules are
served by Annolid's desktop asset server and included by the package's JS rules.

Run deterministic checks without a browser (Node.js is needed for the JS core):

```bash
source .venv/bin/activate
pytest tests/test_two_mice_simulation.py tests/test_threejs_examples.py
```

Run the optional browser checks with Playwright Chromium installed and CDN access:

```bash
ANNOLID_RUN_THREEJS_BROWSER_TESTS=1 \
ANNOLID_THREEJS_SCREENSHOT_DIR=output/playwright \
.venv/bin/python -m pytest tests/test_two_mice_browser.py
```

These check capture repeatability, contact bounds, deformed fur normals, pause,
camera controls, resource cleanup, and selected anatomy/rearing/grooming poses.
`window.annolidTwoMice.getDiagnostics()` exposes a read-only snapshot of simulation
time, subject positions, and renderer resource counts for diagnostics.
