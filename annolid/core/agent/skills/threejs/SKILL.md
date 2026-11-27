---
name: threejs
description: Open and control Three.js 3D scenes, point clouds, meshes, and built-in neuroscience/animal 3D examples in Annolid GUI sessions.
metadata: '{"annolid":{"always":false,"tags":["threejs","3d","viewer","mesh","point cloud","ply","obj","glb","brain","two mice","fly"]}}'
---

# Three.js 3D Viewer

Use this skill when the user asks to open Three.js HTML scenes, switch to the 3D canvas, inspect 3D models/point clouds (`.ply`, `.obj`, `.csv`, `.glb`), or load built-in 3D examples.

## Preferred Tools

- `gui_open_threejs(path_or_url)` — open a local `.html`/`.htm`/`.xhtml` scene, remote viewer URL, or local 3D model/point-cloud file.
- `gui_open_threejs_example(example_id?)` — open a bundled Annolid 3D example.

## Built-In Example IDs

- `two_mice_html` (default when no example is specified)
- `fly_brain_mouse_html`
- `brain_viewer_html`
- `helix_points_csv`
- `wave_surface_obj`
- `sphere_points_ply`

## Natural-Language Examples

- `open threejs example two mice` -> `gui_open_threejs_example(example_id="two_mice_html")`
- `open 3D fly brain mouse viewer` -> `gui_open_threejs_example(example_id="fly_brain_mouse_html")`
- `open 3D brain viewer` -> `gui_open_threejs_example(example_id="brain_viewer_html")`
- `open threejs /path/to/model.ply` -> `gui_open_threejs(path_or_url="/path/to/model.ply")`
- `open threejs https://example.org/viewer.html` -> `gui_open_threejs(path_or_url="https://example.org/viewer.html")`

## Notes

- Local `.html`/`.htm`/`.xhtml` files open in Three.js URL viewer mode; local mesh/point files open in 3D model viewer mode.
- Keep local paths inside configured workspace/read roots.
