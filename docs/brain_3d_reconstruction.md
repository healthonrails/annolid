# Brain 3D Reconstruction Workflow

This guide explains how to build and edit a Brain 3D model from sagittal polygon annotations, generate coronal planes, and keep everything synchronized with normal page-based Annolid annotations.

## What This Workflow Is For

Use this when you have sagittal section polygons and want to:

- reconstruct a canonical 3D region model,
- generate candidate coronal contours automatically,
- edit coronal contours directly in Annolid,
- preview and inspect the 3D result,
- keep the result compatible with existing LabelMe-style page workflows.

This is an additive workflow. Existing 2D annotation projects still work as before.

## Quick Start

1. Open your sagittal stack and annotations in Annolid.
2. Go to `View -> Build Brain 3D Model...`.
3. Configure output and smoothing/snapping settings.
4. Click OK to build the canonical model.
5. Open the dock with `View -> Open Brain 3D Session`.
6. Generate or refresh coronal planes with `View -> Regenerate Coronal Planes`.
7. Optionally open the 3D preview with `View -> Open Brain 3D Preview`.

## Main Actions

These actions are available in `View`:

- `Build Brain 3D Model...`
- `Open Brain 3D Session`
- `Regenerate Coronal Planes`
- `Apply Current Coronal Edits to Brain 3D`
- `Open Brain 3D Preview`
- `Create Region On Plane`
- `Hide Region On Plane`
- `Restore Region On Plane`

The Brain 3D Session dock also includes Create/Hide/Restore buttons.

## Build Settings Explained

When you run `Build Brain 3D Model...`, the dialog includes:

- `Coronal output`:
  choose spacing-based or plane-count-based reslicing.
- `Coronal spacing`:
  distance between generated coronal planes.
- `Coronal plane count`:
  total number of generated coronal planes.
- `Contour point count`:
  resampling density for each polygon contour.
- `Interpolation density`:
  how many virtual interpolation steps are inserted between sagittal source sections.
- `Longitudinal smoothing`:
  smoothing strength along the sagittal-to-coronal reconstruction path.
- `Coronal in-plane smoothing`:
  contour smoothing applied after coronal extraction.
- `Enable reference snapping`:
  turns guided snapping on/off for coronal edit application.
- `Snapping strength`:
  blend amount toward guide/reference points.
- `Snapping max distance`:
  distance threshold for snapping influence.

## Typical Editing Loop

1. Select a coronal plane in the Brain 3D Session dock.
2. Edit polygons on that plane.
3. Click `Apply Current Edits` (or save, which also synchronizes generated coronal pages).
4. Annolid updates the canonical Brain 3D model first.
5. Annolid regenerates affected nearby coronal pages.
6. Review in both 2D and 3D preview.

## Region Presence Controls

For each region on a plane, choose one:

- `Create Region On Plane`: region should exist here.
- `Hide Region On Plane`: region should not appear on this plane.
- `Restore Region On Plane`: return to present/default visible state.

These states are persisted in the Brain 3D artifact and reflected during regeneration.

## Synchronization and Persistence Rules

Annolid persists both:

- the canonical Brain 3D artifact (`otherData["brain_3d_model"]`),
- generated coronal page polygons (normal page annotations).

Rules:

- Coronal edits update the 3D artifact first, then regenerate affected coronal pages.
- Manual edits on original sagittal source pages invalidate the Brain 3D artifact and require rebuild/regeneration.
- Existing LabelMe-style data is not silently reinterpreted.

## 3D Preview Selection Sync

When you click a region in the Three.js Brain 3D preview:

- Annolid resolves the region id,
- selects the region in the Brain 3D Session dock,
- moves to the nearest plane where that region is available,
- highlights/selects matching polygons on the active view.

## Tips for Reference TIFF Alignment

When comparing generated coronal contours to reference coronal TIFF overlays:

- keep overlay and annotations visible together,
- lower top-layer opacity to inspect alignment,
- apply snapping conservatively first (`0.2` to `0.5`),
- use small local edits and regenerate iteratively.

## Troubleshooting

### “No Brain 3D model found”

Run `View -> Build Brain 3D Model...` first.

### Coronal edits do not seem to persist

Use `Apply Current Coronal Edits to Brain 3D`, then regenerate planes.
Saving generated coronal pages also triggers pre-save synchronization.

### Model invalidated after manual edits

If you changed sagittal source polygons, this is expected. Rebuild the Brain 3D model from the updated sagittal pages.

### Region not visible on a plane

Check state controls:

- it may be hidden or zero-area on that plane,
- use `Restore Region On Plane` or `Create Region On Plane`.

## Developer API Reference

Core internal interfaces:

```python
build_brain_3d_model(sagittal_pages, config) -> Brain3DModel

reslice_brain_model(
    model,
    orientation="coronal",
    spacing=...,
    plane_count=...,
) -> list[PlanePolygonSet]

apply_coronal_polygon_edit(
    model,
    plane_index,
    region_id,
    edited_shape,
) -> Brain3DModel

export_brain_model_mesh(model, smoothing=...) -> MeshPayload
```
