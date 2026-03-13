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

## Optimizing large TIFF files for fast viewing

If you open a large flat TIFF repeatedly, the source format itself may be the bottleneck. Annolid can generate a pyramidal viewing cache.

Use:

- `File -> Optimize Large TIFF for Fast Viewing...`

This creates an optimized cache file that later opens can reuse while still keeping the original TIFF as the project source image.

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

## What is tile-native

These workflows now work directly in the tiled large-image viewer:

- select shapes
- move shapes
- drag vertices
- edit imported vector overlays
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

When editing an existing polygon or linestrip in the tiled viewer, clicking on
an edge inserts a new point at that location so you can refine the path
directly. At high zoom, Annolid also reduces point-highlight glow and halo
intensity so the image detail stays visible while you edit.

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

## What happens during canvas fallback

If you choose an unsupported tool while working on a large TIFF:

1. Annolid keeps the current image and shapes.
2. It switches from the tiled viewer to the canvas preview.
3. It enables the requested tool there.
4. You continue working in the same project.

This is a workflow handoff, not a data conversion.

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

## Troubleshooting

### The image opens slowly

Check:

- whether the file is a flat non-pyramidal TIFF
- whether `annolid[large_image]` is installed
- whether `pyvips` is available
- whether an optimized TIFF cache has been built

### The viewer changed unexpectedly

If you selected an AI/SAM tool, that is expected. Those tools still use the canvas fallback path.

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
