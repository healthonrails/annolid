# Large TIFF and Atlas Overlay Workflow

This workflow is for users who want to open a large TIFF-family image in Annolid and align an atlas or anatomy drawing exported from Adobe Illustrator.

## What Annolid supports today

- Large TIFF-family image loading with metadata-aware backends.
- SVG overlay import for Illustrator-exported atlas drawings.
- Manual overlay visibility, opacity, translation, scale, rotation, and z-order controls.
- Tile-native overlay editing for large TIFF sessions.
- Explicit landmark pairing between overlay points and image points.
- Affine alignment from landmark pairs.
- Immutable import provenance for SVG / AI / PDF overlays.
- Editable correction layers derived from the imported source overlay.
- Export of corrected overlays as `SVG`, overlay `JSON`, or LabelMe-like `JSON`.

## Recommended source formats

For raster images:

- `.tif`
- `.tiff`
- `.ome.tif`
- `.ome.tiff`

For vector overlays:

- `SVG` exported from Illustrator
- PDF-compatible Illustrator `.ai` files

Annolid can now import PDF-compatible Illustrator `.ai` files directly by converting the embedded PDF page to SVG during import. Exporting `SVG` from Illustrator is still the cleanest path when you want the most predictable interchange.

On import, Annolid ignores non-rendered SVG definition content such as clip paths in `<defs>`, converts PDF-compatible `.ai` files through an SVG import path, and can automatically fit a small atlas drawing to the currently open image when the document coordinates are clearly not already in image space. For PDF-compatible `.ai` and `.pdf` files, Annolid preserves the source art box when available, then the page box during import and the auto-fit path uses that box first, so imported geometry stays aligned to the underlying raster coordinate system instead of being re-centered in the view. Annolid also extracts visible text from the source page and uses it as region labels when the text can be matched to nearby imported shapes.

Internally, Annolid now normalizes `SVG`, PDF-compatible `.ai`, and `.pdf` imports into the same overlay model. The imported source geometry is preserved as immutable provenance data, and the shapes you edit in the viewer are treated as the derived correction layer for that overlay.

## Install the optional large-image dependencies

This workflow is optional. Annolid's regular GUI install and normal image/video annotation workflows do not require any of the large-image packages below.

If you work with large TIFF-family images, install the optional extra:

```bash
pip install "annolid[large_image]"
```

From a source checkout:

```bash
pip install -e ".[large_image]"
```

This enables the optional `tifffile`, `pyvips`, and `openslide-python` backends when available.

If you do not install `large_image`, Annolid still launches and works as usual. Large TIFF-family files may fall back to the standard image loader path, which is acceptable for smaller TIFFs but less capable for metadata-rich or very large whole-slide workflows.

## Performance tips for very large TIFF files

If your source image is a very large non-pyramidal TIFF, Annolid now opens it more efficiently than before by avoiding duplicate preview loads and by using lower-cost preview generation where possible.

For the best interactive performance:

- install the `large_image` extra,
- keep the image on a fast local SSD,
- prefer tiled or pyramidal TIFF when you control the export pipeline,
- use `SVG` overlays instead of embedding anatomy drawings into the raster image.

Large, flat TIFF files still work, but pyramidal or tiled TIFF files remain the best format for repeated zoom-and-pan workflows.

If `pyvips` is available, you can now choose `File -> Optimize Large TIFF for Fast Viewing...` to build a pyramidal cached TIFF. Annolid can reuse that optimized cache on later opens while keeping your original image path as the project source.

Annolid also includes basic cache management for this workflow:

- `File -> Large TIFF Cache Info...` shows the cache folder, current cache path, and total disk usage.
- `File -> Configure Large TIFF Cache Limits...` lets you set how many optimized TIFF caches Annolid keeps and how much total disk space they may use.
- `File -> Open Large TIFF Cache Folder` opens the cache location in your file manager.
- `File -> Clear Current TIFF Cache...` removes only the optimized cache for the image you are viewing.
- `File -> Clear All TIFF Caches...` removes every optimized TIFF cache created by Annolid.

Annolid also prunes old optimized TIFF caches automatically after creating a new one. The newest cache is kept, and older caches are removed only when the cache grows beyond the configured size/count limit.

## Quick workflow

1. Open Annolid.
2. Open a TIFF-family image with `File -> Open`.
3. Import your Illustrator `SVG`, `.ai`, or `.pdf` file with `File -> Import Vector Overlay...`.
4. For very large flat TIFF files, optionally run `File -> Optimize Large TIFF for Fast Viewing...`.
5. If you need to inspect or clean up disk usage later, use the TIFF cache actions in the `File` menu.
6. If the image is open in the tiled large-image viewer, polygons and other supported vector shapes remain editable directly in that viewer. You can select shapes, drag them, drag their vertices, and create native point/line/linestrip/polygon/rectangle/circle annotations without leaving the large-image view.
7. Use the `Vector Overlays` dock to adjust visibility, opacity, transform, and z-order.
8. Create or select corresponding point landmarks on the image and overlay.
9. Pair landmarks in the dock with `Pair Selected`, or click existing pair guide lines to inspect them.
10. Run `Align Points` once you have at least 3 valid matches.
11. Export the corrected overlay with `File -> Export Corrected Overlay...`.

## Landmark pairing options

Annolid supports two pairing modes.

Explicit landmark pairs are now stored as overlay-layer state, not only as temporary point-shape flags. That means the overlay model keeps the pair list, the pair labels, and the current alignment inputs together with the overlay record itself.

### Explicit pairing

This is the recommended mode for atlas alignment.

1. Select one overlay point and one image point.
2. Click `Pair Selected` in the `Vector Overlays` dock.
3. Repeat until you have at least 3 pairs.
4. Click `Align Points`.

You can also:

- remove a single bad pair,
- clear all explicit pairs for the active overlay,
- click a pair entry in the dock,
- click a pair guide line in the viewer.

When a pair is active, Annolid now:

- highlights the guide line,
- highlights the two endpoint points,
- selects the corresponding point shapes in the canvas and label list.

### Automatic pair activation from normal selection

You do not have to use the pair list every time.

If your current selection clearly identifies one explicit pair, Annolid activates it automatically. This works when you select:

- one paired overlay point,
- one paired image point,
- both paired points,
- the paired points plus additional non-point shapes.

If you switch to an unrelated selection, Annolid clears the active pair highlight.

## Notes for Illustrator users

Recommended export settings:

- Export each atlas layer or artboard as `SVG`.
- If you keep `.ai`, save with PDF compatibility enabled so Annolid can import it directly.
- Keep important anatomical boundaries as vector paths, not text.
- Convert critical labels to geometry only if you need direct shape editing.
- Preserve layer names where possible so overlay provenance remains readable after import.

## Current limitations

- Tile-native creation currently covers point, line, linestrip, polygon, rectangle, and circle tools. AI-assisted create modes still fall back to the existing canvas workflow.
- Exported corrected SVG preserves corrected geometry, but not full Illustrator Bézier editing semantics.
- Landmark alignment is affine only in the current GUI workflow.

For a broader overview of large TIFF support, viewing backends, caches, tile-native editing, and canvas fallback behavior, see [Large Image Guide](large_image_guide.md).

## Output formats

You can export corrected overlays as:

- `SVG`
- overlay `JSON`
- `*.labelme.json`

Use `*.labelme.json` when you want a compatibility-friendly polygon/point export that fits LabelMe-style downstream tooling.
