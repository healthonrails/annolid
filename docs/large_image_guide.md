# Large Image Guide

This guide explains how Annolid works with large TIFF-family images, including opening files, navigating them efficiently, editing annotations, working with atlas overlays, and understanding when the app stays in the tiled viewer versus when it falls back to the canvas workflow.

## What counts as a large image in Annolid

In practice, this guide is for:

- `.tif`
- `.tiff`
- `.ome.tif`
- `.ome.tiff`
- BigTIFF
- large flat TIFF files
- tiled or pyramidal TIFF files

Annolid treats these differently from ordinary images because they may be too large to load comfortably as one full in-memory raster.

Multi-page TIFF stacks are handled on the same large-image path now. Instead of
opening them through the older video-style TIFF stack loader, Annolid opens
them in the tiled large-image viewer and lets you move between pages with the
status-bar slider and the normal next/previous navigation actions. The same
play control used for frame workflows can also step through TIFF pages, and it
stops cleanly at the last page instead of bouncing the viewer out of tiled
mode.

For large multipage TIFFs, the slider now commits the page change when you
release the handle. That keeps dragging responsive even when each page switch
requires a real TIFF decode.

Annolid also recognizes TIFF stacks that are stored as a single series with an
explicit stack axis, so page navigation still appears for those stack-style
files when the TIFF metadata exposes that axis clearly.

Each TIFF page also has its own annotation JSON. Annolid saves and reloads
those page labels from a sibling folder next to the TIFF, using names like:

- `sample_stack/sample_stack_000000000.json`
- `sample_stack/sample_stack_000000001.json`

That means page navigation in a TIFF stack now behaves much more like frame
navigation used to behave for videos, but without leaving the large-image
viewer workflow.

## Large-image support is optional

Annolid still works normally without the large-image extras. A standard GUI install is enough for ordinary image and video annotation workflows.

Install the optional large-image extras when you want better TIFF-family metadata support and better large-image performance:

```bash
pip install "annolid[large_image]"
```

From a source checkout:

```bash
pip install -e ".[large_image]"
```

These extras enable optional backends such as:

- `tifffile`
- `pyvips`
- `openslide-python`

If they are not available, Annolid falls back gracefully to simpler image-loading paths when possible.

## How Annolid opens large TIFF files

When you open a large TIFF-family image, Annolid:

1. Detects whether it should use the large-image path.
2. Chooses the best available backend.
3. Opens the image in the tiled large-image viewer when appropriate.
4. Keeps metadata about the large image in project state.

If an optimized pyramidal cache already exists for that image, Annolid may reuse it automatically for faster viewing.

## Backends and performance

Annolid may use different backends depending on what is installed and what kind of TIFF you open.

### `tifffile`

Best for:

- scientific TIFF metadata
- OME-TIFF
- BigTIFF
- reliable baseline TIFF support

### `pyvips`

Best for:

- efficient region reads
- low-memory navigation
- building optimized pyramidal TIFF caches

### `OpenSlide`

Best for:

- whole-slide style pyramidal images
- pathology-style slide workflows

If Annolid has to fall back to a slower backend, it reports that in the status message when the image opens.

Annolid also tracks backend capabilities explicitly, so the UI can adapt
without guessing. In practice, the important capability differences are:

- page navigation support
- pyramid/level support
- region-read support
- metadata-axis support
- optimized-cache generation support
- label-stack suitability

That is why, for example, TIFF page navigation only appears when the active
backend actually reports paged-stack support.

## Optimizing large TIFF files for fast viewing

If you open a large flat TIFF repeatedly, the source format itself may be the bottleneck. Annolid can generate a pyramidal viewing cache.

Use:

- `File -> Optimize Large TIFF for Fast Viewing...`

This creates an optimized cache file that later opens can reuse while still keeping the original TIFF as the project source image.

For multipage TIFF stacks, the optimized viewing cache affects image loading
speed only. Page-specific annotation JSON files remain in the sibling
annotation folder described above.

## Large TIFF cache management

Annolid includes cache-management tools in the `File` menu:

- `Optimize Large TIFF for Fast Viewing...`
- `Large TIFF Cache Info...`
- `Open Large TIFF Cache Folder`
- `Clear Current TIFF Cache...`
- `Clear All TIFF Caches...`
- `Configure Large TIFF Cache Limits...`

Annolid also prunes old optimized caches automatically when the configured cache size or cache count limits are exceeded.

## Two viewer surfaces

Annolid uses two image-editing surfaces for large-image workflows:

- `Tiled large-image viewer`
- `Canvas`

For large TIFF workflows, the status bar is now kept intentionally minimal so
page navigation remains clear. In practice, it should only show the page
controls such as:

- play/pause
- page slider

Large-image mode/debug information lives in the viewer overlay instead of the
status bar.

### Tiled large-image viewer

Best for:

- very large TIFF navigation
- zooming and panning
- editing supported manual annotations directly on the large image
- editing imported vector overlays

### Canvas

Best for:

- legacy drawing workflows
- AI-assisted creation tools that still depend on canvas-specific logic

For normal manual large-image work, Annolid now keeps you in the tiled viewer as much as possible.

## Docks shown for large TIFF workflows

When you open a large TIFF, Annolid now hides docks that are mainly tied to
video or media playback, including timeline, behavior-log, behavior-controls,
audio, caption, and video-list docks.

It keeps the annotation-focused docks available:

- Files
- Flags
- Layers
- Labels
- Label Instances
- Vector Overlays, when overlays are present

When you leave the large-image workflow and go back to ordinary image or video
work, those hidden docks are restored.

## Layer model and layer dock

Annolid now treats the large-image viewer as a true layer-based workflow even
though it still preserves the existing annotation data model underneath.

The current runtime layer stack includes:

- raster image layer
- label-image overlay layer
- vector overlay layer
- landmark layer
- annotation layer

The `Layers` dock reflects that runtime model directly. Today it gives you a
single place to inspect and control the current layer stack, including:

- hiding/showing label-image overlays
- hiding/showing imported vector overlays
- hiding/showing landmark-pair guide layers
- hiding/showing manual annotation layers
- adjusting opacity for label-image and vector-overlay layers
- manually aligning imported raster overlay layers with drag or the dock's
  `Align / Nudge` controls, which persist `tx/ty/sx/sy/rotation_deg`
  transform values in the project state
- rotating imported raster overlay layers with:
  - quick `Rotate -` / `Rotate +` controls in `Align / Nudge`
  - exact `Rotation` value in `Layer Settings` for reproducible alignment
- resetting a raster overlay layer back to identity alignment from the dock
- keyboard shortcuts in the Layers dock:
  - `Alt+Left` / `Alt+Right` / `Alt+Up` / `Alt+Down` for nudging the selected
    raster overlay layer
  - `Alt+0` for resetting the selected raster overlay alignment
  - the dock also shows a small hint next to the alignment controls

When you align a raster overlay by lowering its opacity or dragging it in the
tiled viewer, Annolid now keeps the selected overlay and the reference raster
layer beneath it visible so the comparison stays usable during alignment.

For cross-modality section alignment (for example slightly angled myelin and
Nissl overlays), use this sequence:

1. Select the overlay layer in `Layers`.
2. Lower opacity so you can see the reference raster beneath it.
3. Use `Rotate -` / `Rotate +` to match angle first.
4. Use nudge + scale (or direct drag in tiled view) to finish registration.
5. Save layer settings so the transform can be reapplied later.

In `Interactive Resize` mode, Annolid draws a transform box that follows the
current overlay rotation plus a rotate handle above the top edge. Drag the
rotate handle directly on-canvas, then use edge/corner handles to resize along
the rotated axes.

Selecting a vector-overlay or landmark layer in the `Layers` dock also keeps
the vector overlay dock aligned with that overlay, so the layer list and the
overlay/alignment tools do not drift out of sync.

That layer dock is intentionally built on top of the shared
`ViewerLayerModel`/`LargeImageDocument` path rather than introducing another
overlay-specific state store.

## What is tile-native

These workflows now work directly in the tiled large-image viewer:

- select shapes
- move shapes
- drag vertices
- edit imported vector overlays

For polygon-heavy workflows such as brain-area mapping, Annolid supports
shared vertices and shared boundary edges between adjacent polygons. When you
drag a polygon vertex onto another polygon vertex, or place a new point close
enough to an existing polygon vertex, Annolid snaps them together and keeps
both polygons on the same vertex coordinate. When two polygons share the same
sequence of linked vertices along a boundary, Annolid also links those edges as
a shared topological boundary while still storing each region as a separate
polygon. The editor keeps this topology in an explicit shared-topology
registry so boundary-linked edits stay synchronized across related polygons.
For already shared borders, the right-click menu includes **Reshape Shared
Boundary**, which lets you drag that boundary segment and update every linked
polygon together.

For sagittal section workflows, Annolid can also infer polygon outlines for
the pages between explicitly annotated sections. When a TIFF page does not yet
have its own annotation JSON, Annolid can load an inferred polygon set from
the nearest annotated pages so you can open the page and tweak the boundaries
instead of redrawing everything from scratch.

- `Infer Page Polygons` loads inferred page shapes for a missing section
- `Collapse Selected Polygons` hides a polygon without deleting it, which is
  useful when a small region should vanish during manual cleanup
- checking a collapsed polygon back on in the `Label Instances` dock restores
  it to normal visibility

- create `point`
- create `line`
- create `linestrip`
- create `polygon`
- create `rectangle`
- create `circle`

Polygon drawing in the tiled viewer follows the same basic interaction as the
normal canvas: as you draw, Annolid shows the in-progress path and point
markers, and when the cursor gets close to the first point it highlights the
close target so you can finish the polygon confidently.

The tiled viewer also mirrors the regular canvas hover feedback more closely:
hovering a polygon vertex, edge, or body shows the same move/create prompts and
cursor changes you get in the standard canvas editor.

When editing an existing polygon or linestrip in the tiled viewer, clicking on
an edge inserts a new point at that location so you can refine the path
directly. At high zoom, Annolid also reduces point-highlight glow and halo
intensity so the image detail stays visible while you edit.

For adjoining regions such as atlas or brain-area mapping, you can right-click
an existing polygon and choose **Start Adjoining Polygon**. Annolid switches
into polygon drawing mode with that polygon as the boundary source, and each
click snaps to the shared boundary so you can trace across multiple touching
edges without having to preselect a single edge first.

This keeps neighboring regions as separate saved polygons while letting their
shared border reuse the same boundary points and coordinates.

This lets you stay in the large-image navigation workflow for common annotation tasks.

## What still switches to canvas

These tools still fall back to the canvas workflow on purpose:

- `AI Polygon`
- `AI Mask`
- `Grounding SAM`
- `Polygon SAM`

Why:

- they depend on canvas-specific preview and model wiring
- they use older tool flows that are not yet ported to the tiled viewer

When this happens, Annolid now shows a status message explaining that it switched to canvas preview mode for that tool.

The fallback reason is reflected in the large-image viewer state and overlay,
without adding more persistent status-bar text.

## What happens during canvas fallback

If you choose an unsupported tool while working on a large TIFF:

1. Annolid keeps the current image and shapes.
2. It switches from the tiled viewer to the canvas preview.
3. It enables the requested tool there.
4. You continue working in the same project.

This is a workflow handoff, not a data conversion.

After you finish the canvas-only step, you can switch back to the tiled
large-image surface without reopening the TIFF.

## Viewer status and debugging overlay

The tiled viewer now shows a compact translucent status overlay in the
upper-left corner. This is useful both for everyday troubleshooting and for
understanding which backend/path Annolid is using.

The overlay reports:

- backend in use
- current page
- current level
- current zoom
- visible raster tile count
- visible label tile count
- outstanding tile requests
- cache hits and misses
- whether you are viewing the source image or an optimized TIFF cache

This is especially useful when comparing behavior between:

- `tifffile`
- `pyvips`
- `OpenSlide`
- source TIFF vs optimized pyramidal cache

## Progressive tile loading

Tile loading is now split into planning and scheduling instead of being treated
as one paint-time action.

In practice, Annolid now uses a progressive strategy:

1. show the thumbnail/preview immediately
2. load a small set of center-priority visible tiles first
3. queue the remaining visible tiles in the background
4. prefetch nearby tiles after that
5. drop stale queued work when the viewport changes

This keeps the viewer responsive during fast pan/zoom/page changes while still
preserving deterministic visible results once the queued work settles.

## Atlas and vector overlay workflows

Large-image work is often paired with atlas overlays.

Annolid supports:

- Illustrator-exported `SVG`
- PDF-compatible Illustrator `.ai`
- supported vector import through the overlay workflow

Imported atlas overlays can now be:

- displayed on top of large TIFF images
- selected in the tiled viewer
- moved and edited
- aligned with landmarks
- exported after correction

Annolid also supports atlas-style label images, similar to a napari labels
layer:

- load an integer-valued label TIFF as a colorized overlay
- keep it tiled and memory-efficient in the large-image viewer
- hover to inspect the region id under the cursor
- click a region to select and emphasize that label
- optionally import a CSV/TSV mapping so ids resolve to acronyms and region names

Use the `File -> Overlays` menu:

- `Import Label Image Overlay...`
- `Import Label Mapping...`
- `Show Vector Overlays`
- `Show Label Image Overlay`
- `Set Label Overlay Opacity...`
- `Clear Label Image Overlay`

For atlas CSV mappings, Annolid looks for flexible column names such as:

- `id`, `region_id`, `structure_id`, `label_id`
- `acronym`, `abbreviation`
- `name`, `region_name`, `structure_name`

This works well for mouse brain annotation TIFFs where each pixel stores a
brain-region id instead of RGB color values.

Annolid keeps the label-overlay visibility state in project metadata. When you
move through a multi-page TIFF with the page slider, the label layer now stays
available on the next page instead of disappearing. If the label stack matches
the TIFF page count, the label overlay follows the active TIFF page
automatically.

For the atlas-specific step-by-step workflow, see [Large TIFF and Atlas Overlay Workflow](atlas_overlay_workflow.md).

## Recommended workflow

For most large-image users:

1. Open the TIFF-family image.
2. Stay in the tiled viewer for navigation.
3. Use tile-native manual tools for ordinary annotations.
4. Import atlas overlays if needed.
5. Use the overlay dock for visibility, opacity, transform, and alignment.
6. Only switch to AI/SAM tools when you actually need them.
7. Build an optimized TIFF cache if the same flat TIFF will be opened often.

For multipage TIFF stacks:

1. Open the TIFF stack directly.
2. Drag the page slider and release it to jump smoothly, or use the
   next/previous actions for stepwise browsing.
3. Save annotations normally; Annolid writes them to the page-specific JSON for
   the current page.
4. Reopen the TIFF later and Annolid restores each page’s shapes as you browse.

## Troubleshooting

### The image opens slowly

Check:

- whether the file is a flat non-pyramidal TIFF
- whether `annolid[large_image]` is installed
- whether `pyvips` is available
- whether an optimized TIFF cache has been built

Annolid now reuses the first large-image preview when switching into tiled
view, so it avoids rebuilding the same thumbnail twice during open. On very
large TIFFs this should noticeably reduce startup latency.

### The viewer changed unexpectedly

If you selected an AI/SAM tool, that is expected. Those tools still use the canvas fallback path.

### My annotations do not follow the TIFF page

Check that you opened the `.tif` or `.tiff` stack itself, not just one of its
page JSON files. Annolid stores page labels in the sibling annotation folder,
and page changes in the large-image viewer reload the matching JSON
automatically.

### I can see a shape but cannot edit it

Check:

- the shape is visible
- the overlay is not locked
- you are in edit mode, not a create mode
- the shape is not a mask-only object

### I imported an atlas overlay and want to edit vertices

Stay in the tiled viewer and use normal shape selection. Imported overlays are editable there.

### I want the fastest repeated viewing workflow

Use:

1. local SSD storage
2. `annolid[large_image]`
3. optimized pyramidal TIFF caches
4. tiled or pyramidal TIFF exports when you control the source pipeline

## Current scope

Annolid’s tiled large-image workflow is now strongest for manual geometry editing and atlas-style overlay work. AI-assisted large-image creation tools still rely on the older canvas path, which is the main remaining functional split.
