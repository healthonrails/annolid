# Merge canvas shapes

Use **Edit Polygons** mode on the standard image/video canvas. Select two or
more matching shapes in the Label Instances list (use Ctrl/Cmd-click for
multiple selection), then choose **Edit → Merge Selected Shapes**, or
right-click on the selected shapes and choose **Merge Selected Shapes**.

Two merge modes are supported:

- **Vector shapes (`polygon`, `rectangle`, `oriented_rectangle`)**: the
  selected shapes become one polygon covering their exact combined area.
  Overlapping shapes, contained shapes, and shapes sharing an edge are
  supported.
- **Mask fragments (`mask`)**: two or more selected mask shapes are unioned
  into a single tight-cropped `mask` shape.

The merged shape stays selected; **Undo** (`Ctrl+Z`) restores the original
shapes. Save the annotation normally to keep the result.

Shapes must be unlocked, visible, and share matching labels, group IDs, flags,
descriptions, and custom metadata. Edit conflicting values or unlock locked
shapes (`Ctrl+Shift+L`) before merging. Mixed mask and vector selections,
custom vertex labels, invalid polygons, disconnected polygon unions, and
polygon unions containing holes are rejected with an explanation; the originals
remain unchanged. Points, lines, circles, and the tiled large-image editor are
not supported by this action.
