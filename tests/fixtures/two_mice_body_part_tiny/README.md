# Two-Mouse 3D Body-Part Segmentation Fixture

Deterministic 320x240 synthetic fixture generated from Annolid's bundled Three.js
Two-Mice simulation (`annolid/gui/assets/threejs/two_mice.html`) with exact
WebGL 3D mesh ground-truth instance and body-part ID masks (`masks/*_ids.png`)
and surface-occlusion-classified 13-keypoint LabelMe annotations (`labelme/*.json`).

Scenarios included:
- `social_huddle` (4 sequential frames from separated approach into close contact)
- `curled_grooming` (2 frames with C-curved flank-grooming posture)
- `rearing_oblique` (2 oblique 3D frames with upright rearing and visible fore/hindlimbs)
