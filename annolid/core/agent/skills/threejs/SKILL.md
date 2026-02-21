---
name: threejs
description: Open and control Three.js views/examples in Annolid Bot sessions.
---

# Three.js

Use this skill when users ask to open Three.js content, switch to 3D canvas, or load built-in 3D examples.

## Preferred tools

- `gui_open_threejs(path_or_url)`
- `gui_open_threejs_example(example_id)`

## Example IDs

- `two_mice_html` (default)
- `brain_viewer_html`
- `helix_points_csv`
- `wave_surface_obj`
- `sphere_points_ply`

## Natural-language command patterns

- `open threejs example two mice`
- `open threejs example brain`
- `open threejs html /path/to/viewer.html`
- `open threejs https://example.org/viewer.html`
- `open threejs /path/to/model.ply`

## Notes

- If no example is provided, default to `two_mice_html`.
- For local `.html/.htm/.xhtml`, open in Three.js URL viewer mode.
- For local model files, use Three.js model viewer mode.
